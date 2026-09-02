#!/usr/bin/env node
// Assert the wasm-bindgen externref table is still growable after wasm-opt.
//
// The generated glue initialises itself with:
//
//     const t = wasm.__wbindgen_externrefs; const o = t.grow(4);
//
// so that export must resolve to the externref table, which has no declared
// maximum. The module also contains a funcref table whose min equals its max.
// Point the export at that one and `grow` throws RangeError on every load, in
// every browser, before a single token is generated.
//
// That is exactly what binaryen 108 did: it renumbered the exported table from
// index 1 to index 0 while leaving both table definitions intact, so the wasm
// still validates, still instantiates in Node, and still passes a kernel
// self-test. Only a browser actually running the glue sees it. Structure is
// therefore the only thing worth asserting here.
//
//   node tools/check_wasm_tables.js web/pkg/nano_infer_wasm_bg.wasm

import fs from 'node:fs';

const FUNCREF = 0x70;
const EXTERNREF = 0x6f;

function uleb(b, i) {
  let r = 0, s = 0, x;
  do { x = b[i++]; r |= (x & 0x7f) << s; s += 7; } while (x & 0x80);
  return [r >>> 0, i];
}

// Returns { reftype, min, max } — max is null when the table can grow freely.
function limits(b, i) {
  const flag = b[i++];
  let min, max = null;
  [min, i] = uleb(b, i);
  if (flag & 1) [max, i] = uleb(b, i);
  return [{ min, max }, i];
}

function parse(bytes) {
  const tables = [];          // table index space: imports first, then definitions
  const exportedTables = new Map();
  let i = 8;                  // skip magic + version
  while (i < bytes.length) {
    const id = bytes[i++];
    let size; [size, i] = uleb(bytes, i);
    const end = i + size;
    let j = i;

    if (id === 2) {           // imports may occupy the low table indices
      let n; [n, j] = uleb(bytes, j);
      for (let k = 0; k < n; k++) {
        let l; [l, j] = uleb(bytes, j); j += l;
        [l, j] = uleb(bytes, j); j += l;
        const kind = bytes[j++];
        if (kind === 0) { [, j] = uleb(bytes, j); }
        else if (kind === 1) { const rt = bytes[j++]; let lim; [lim, j] = limits(bytes, j); tables.push({ reftype: rt, ...lim, imported: true }); }
        else if (kind === 2) { [, j] = limits(bytes, j); }
        else if (kind === 3) { j += 2; }
      }
    } else if (id === 4) {
      let n; [n, j] = uleb(bytes, j);
      for (let k = 0; k < n; k++) {
        const rt = bytes[j++];
        let lim; [lim, j] = limits(bytes, j);
        tables.push({ reftype: rt, ...lim, imported: false });
      }
    } else if (id === 7) {
      let n; [n, j] = uleb(bytes, j);
      for (let k = 0; k < n; k++) {
        let l; [l, j] = uleb(bytes, j);
        const name = Buffer.from(bytes.subarray(j, j + l)).toString(); j += l;
        const kind = bytes[j++];
        let idx; [idx, j] = uleb(bytes, j);
        if (kind === 1) exportedTables.set(name, idx);
      }
    }
    i = end;
  }
  return { tables, exportedTables };
}

const path = process.argv[2] ?? 'web/pkg/nano_infer_wasm_bg.wasm';
const { tables, exportedTables } = parse(fs.readFileSync(path));
const name = '__wbindgen_externrefs';
const fail = (msg) => { console.error(`ERROR: ${path}: ${msg}`); process.exit(1); };

const idx = exportedTables.get(name);
if (idx === undefined) fail(`no exported table named ${name} (found: ${[...exportedTables.keys()].join(', ') || 'none'})`);
const t = tables[idx];
if (!t) fail(`${name} exports table ${idx}, but the module defines ${tables.length}`);

const kind = t.reftype === EXTERNREF ? 'externref' : t.reftype === FUNCREF ? 'funcref' : `0x${t.reftype.toString(16)}`;
if (t.reftype !== EXTERNREF)
  fail(`${name} -> table ${idx}, which is ${kind}, not externref. ` +
       `The glue calls .grow(4) on it at startup; this build cannot initialise in a browser. ` +
       `Almost certainly a wasm-opt that predates multi-table support — check the binaryen version.`);
if (t.max !== null)
  fail(`${name} -> table ${idx} is externref but capped at max=${t.max} (min=${t.min}); .grow(4) will throw.`);

console.log(`    ${name} -> table ${idx}: ${kind}, min=${t.min}, growable`);
