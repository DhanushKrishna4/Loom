#!/usr/bin/env bash
# Build the deployable site into dist/.
#
# This is the ONE build path: CI runs this exact script, so a locally-working
# build and a deployed build cannot diverge.
#
#   1. wasm-pack  -> web/pkg  (wasm + JS bindings + TypeScript definitions)
#   2. tsc        -> typecheck only, no emit
#   3. vite       -> dist/    (bundled, minified, relative base)
#
# SIMD128 comes from .cargo/config.toml; wasm-opt flags come from the
# [package.metadata.wasm-pack] block in crates/wasm/Cargo.toml. Both live with
# the thing they configure rather than here.
set -euo pipefail
cd "$(dirname "$0")/.."

# Enforced, not aspirational: a size target nobody checks stops holding.
BUDGET_GZIP=$((500 * 1024))

echo "==> wasm-pack (release, wasm32, simd128, wasm-opt -O3)"
wasm-pack build crates/wasm --target web --out-dir ../../web/pkg --release --no-pack

RAW=web/pkg/nano_infer_wasm_bg.wasm

# wasm-pack has just run wasm-opt over the module. An old wasm-opt rewrites the
# exported externref table to point at the funcref table instead, and the result
# validates, instantiates under Node, and dies on load in every browser. Nothing
# else in this pipeline looks at the binary's structure, so check it here.
node tools/check_wasm_tables.js "$RAW"

GZ=$(gzip -9 -c "$RAW" | wc -c | tr -d ' ')
printf '    %s: %d bytes raw, %d gzipped (budget %d, %d%% used)\n' \
    "$RAW" "$(wc -c < "$RAW")" "$GZ" "$BUDGET_GZIP" "$(( GZ * 100 / BUDGET_GZIP ))"
if [ "$GZ" -gt "$BUDGET_GZIP" ]; then
    echo "ERROR: over the ${BUDGET_GZIP}-byte gzipped budget" >&2
    exit 1
fi

if [ ! -d node_modules ]; then
    echo "==> npm install"
    npm install --silent
fi

echo "==> tsc --noEmit"
npm run --silent typecheck

echo "==> vite build"
npm run --silent build 2>&1 | sed 's/^/    /'

# GitHub Pages runs Jekyll by default, which skips files beginning with _ .
touch dist/.nojekyll
# Pages serves 404.html for any unmatched path. This is not a single-page app,
# so refreshes already work; a mistyped URL landing somewhere useful is the point.
cp dist/index.html dist/404.html

echo "==> dist/ ready"
du -sh dist | sed 's/^/    /'
echo "==> preview with:  python3 tools/serve.py   (serves the repo root, so"
echo "    /models/*.gguf is reachable alongside /dist/index.html)"
