#!/usr/bin/env bash
# Render the Open Graph card to web/public/.
#
# The card is a real 1200x630 PNG because LinkedIn, Slack and iMessage will not
# render SVG. It is drawn by headless Chrome from card.html -- the same fonts,
# palette and attention field as the site itself -- at 2x and downsampled, which
# is the cheapest way to get clean type without a design tool in the loop.
#
# The filename is VERSIONED on purpose. Crawlers cache og:image against its URL
# and will not re-fetch a changed file, so a new card means a new name and a
# matching bump in the <meta> tags of both pages.
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=${1:-web/public/og-v1.png}
CHROME=${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}
[ -x "$CHROME" ] || { echo "ERROR: Chrome not found at $CHROME (set CHROME=...)" >&2; exit 1; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

"$CHROME" --headless=new --disable-gpu --hide-scrollbars \
  --force-device-scale-factor=2 --window-size=1200,630 \
  --virtual-time-budget=8000 \
  --screenshot="$TMP/2x.png" "file://$PWD/tools/og/card.html" 2>/dev/null

python3 - "$TMP/2x.png" "$OUT" <<'PY'
import sys
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
im = Image.open(src).convert("RGB")
assert im.size == (2400, 1260), f"unexpected render size {im.size}"
im.resize((1200, 630), Image.LANCZOS).save(dst, "PNG", optimize=True)
print(f"    {dst}: 1200x630")
PY
