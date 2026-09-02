// A minimal IndexedDB layer for caching model files.
//
// IndexedDB rather than localStorage or the Cache API:
//   * localStorage caps at ~5-10 MB and is synchronous. A 469 MB model is three
//     orders of magnitude past it.
//   * the Cache API would work, but it is keyed on Request/Response and would
//     re-run the fetch pipeline; IndexedDB stores the bytes and nothing else.
//
// Models are stored as Blobs, not ArrayBuffers. A Blob can stay backed by disk
// until it is read, so writing one does not require holding a second 469 MB
// copy in the JS heap.

/** A cached model file, as stored. */
export interface StoredModel {
  url: string;
  blob: Blob;
  bytes: number;
  storedAt: number;
}

/** Metadata only, without pulling the blob into memory. */
export type StoredModelMeta = Omit<StoredModel, 'blob'>;

const DB_NAME = 'nano-infer';
const STORE = 'models';
const VERSION = 1;

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: 'url' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

// The return type is what makes the bug below impossible to reintroduce: a
// `get` that misses resolves to `undefined`, and the caller must handle it.
function tx<T>(
  db: IDBDatabase,
  mode: IDBTransactionMode,
  fn: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T | undefined> {
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const store = t.objectStore(STORE);
    let req: IDBRequest<T>;
    try {
      req = fn(store);
    } catch (e) {
      reject(e);
      return;
    }
    // Always resolve the *request's* result, never the request object.
    //
    // A `get` that misses leaves `result` undefined, and an earlier version
    // fell back to returning the IDBRequest itself in that case -- which is
    // truthy, so every cache miss looked like a hit and the caller then read
    // `.bytes` off an IDBRequest. Found on the very first load against an empty
    // database.
    t.oncomplete = () => resolve(req.result);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error || new Error('transaction aborted'));
  });
}

/** Metadata for every cached model, without reading the blobs. */
export async function list(): Promise<StoredModelMeta[]> {
  const db = await openDb();
  const rows = await tx<StoredModel[]>(db, 'readonly', (s) => s.getAll());
  db.close();
  return (rows ?? []).map(({ url, bytes, storedAt }) => ({ url, bytes, storedAt }));
}

export async function get(url: string): Promise<StoredModel | null> {
  const db = await openDb();
  const row = await tx<StoredModel>(db, 'readonly', (s) => s.get(url));
  db.close();
  return row ?? null;
}

/**
 * Store a model. Returns false when the browser refuses on quota, which is a
 * normal outcome rather than an error: the model still loads, it just will not
 * be there next time.
 */
export async function put(url: string, blob: Blob): Promise<boolean> {
  try {
    const db = await openDb();
    const row: StoredModel = { url, blob, bytes: blob.size, storedAt: Date.now() };
    await tx(db, 'readwrite', (s) => s.put(row));
    db.close();
    return true;
  } catch (e) {
    const name = e instanceof DOMException ? e.name : '';
    if (name === 'QuotaExceededError' || name === 'AbortError') return false;
    throw e;
  }
}

export async function remove(url: string): Promise<void> {
  const db = await openDb();
  await tx(db, 'readwrite', (s) => s.delete(url));
  db.close();
}

export async function clear(): Promise<void> {
  const db = await openDb();
  await tx(db, 'readwrite', (s) => s.clear());
  db.close();
}

/**
 * Ask the browser not to evict this origin's storage.
 *
 * Without it a 469 MB cache is "best effort" and can be cleared under disk
 * pressure, which turns an instant revisit back into a download. Chrome grants
 * it silently for engaged sites; Firefox prompts. A refusal is not fatal.
 */
export async function requestPersistence(): Promise<boolean> {
  if (!navigator.storage?.persist) return false;
  if (await navigator.storage.persisted()) return true;
  return navigator.storage.persist();
}

export async function quota(): Promise<{ usage: number; quota: number } | null> {
  if (!navigator.storage?.estimate) return null;
  const { usage, quota } = await navigator.storage.estimate();
  return { usage: usage ?? 0, quota: quota ?? 0 };
}
