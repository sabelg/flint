#!/usr/bin/env bash
# Exercise the public reader against delayed GPU events, a missing zone and repeated CPU scopes.
set -euo pipefail
flint="$(cd "$(dirname "$0")/.." && pwd)/tools/flint.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"
cat > before.json <<'JSON'
[
 {"name":"skin","ph":"X","ts":100,"dur":500},
 {"name":"skin","ph":"X","ts":700,"dur":500},
 {"name":"pace","ph":"X","ts":2500,"dur":3000},
 {"name":"GPU frame","ph":"X","ts":300,"dur":1000},
 {"name":"frame less pace","ph":"X","ts":400,"dur":500},
 {"name":"frame","ph":"X","ts":100,"dur":6000,"args":{"frame":10}},
 {"name":"frame","ph":"X","ts":6200,"dur":9000,"args":{"frame":11}},
 {"name":"gpu_ms","ph":"C","args":{"frame":10,"GPU frame":2}}
]
JSON
cat > after.json <<'JSON'
{"traceEvents":[
 {"name":"skin","ph":"X","ts":400,"dur":2000},
 {"name":"frame","ph":"X","ts":200,"dur":10000,"args":{"frame":0}}
]}
JSON
bash "$flint" studio --compare before.json after.json --text > report
grep -Fx 'frame | 7.50 10.00 | 9.00 10.00 | +1.00 | 2 1' report
grep -Fx 'skin | 1.00 2.00 | 1.00 2.00 | +1.00 | 1 1' report
# GPU counters are namespaced, so a CPU scope with the same name is a separate column, and a
# CPU scope named like the derived "frame less pace" column is renamed instead of overwriting it.
grep -Fx 'gpu:GPU frame | 2.00 -- | 2.00 -- | -- | 1 0' report
grep -Fx 'GPU frame | 1.00 -- | 1.00 -- | -- | 1 0' report
grep -Fx 'cpu:frame less pace | 0.50 -- | 0.50 -- | -- | 1 0' report
grep -Fx 'frame less pace | 6.00 -- | 9.00 -- | -- | 2 0' report
echo '[' > truncated.json
if bash "$flint" studio --compare truncated.json after.json --text > error.log 2>&1; then
    echo 'Accepted an incomplete capture' >&2
    exit 1
fi
grep -Fq 'incomplete trace' error.log
