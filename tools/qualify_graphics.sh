#!/usr/bin/env bash
# Qualify the large-scene renderer on a pinned Flint revision using local scene fixtures.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
rev=$(git rev-parse --verify "${1:-e9869ad}^{commit}")
mkdir -p build/graphics-qualification
out=$(mktemp -d "$root/build/graphics-qualification/${rev:0:10}-XXXXXX")
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
git archive "$rev" src labs | tar -x -C "$scratch"
prefix="$root/.pixi/envs/default"
export MODULAR_HOME="$prefix/share/max" PATH="$prefix/bin:$PATH" LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
version=$(mojo --version | awk '{print $2}')
printf 'comptime BUILD_COMMIT: String = "%s"\ncomptime BUILD_DIRTY: Bool = False\ncomptime BUILD_MOJO: String = "%s"\n' "$rev" "$version" > "$scratch/src/flint/core/build_stamp.mojo"
(cd "$scratch" && mojo build -I src -I tests -O3 --fp-mode contract=off --Werror labs/lab_bench.mojo -o "$scratch/bench")
printf 'revision=%s\nmojo=%s\n' "$rev" "$version" >"$out/run.txt"
if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=name,driver_version --format=csv >>"$out/run.txt"; fi
asset_dir="${FLINT_SCENES_DIR:-$HOME/flint-models/_scenes}"
(cd "$asset_dir" && find sponza_khronos bistro forest characters -type f -print0 | sort -z | xargs -0 -r sha256sum) >"$out/scene-assets.sha256"
printf 'Camera variants and pass timings are recorded per scene. Shader cache starts in a new temporary directory.\n' >>"$out/run.txt"
printf 'Writing graphics qualification to %s\n' "$out"
for scene in sponza bistro forest; do
    mkdir "$out/$scene"
    (cd "$scratch" && "$scratch/bench" --scenes "$scene" --res 1080 --frames 40 --ref-frames 8 --preset high --out "$out/$scene") >"$out/$scene/run.log" 2>&1
done
python3 - "$out" <<'PY'
import math
import pathlib
import struct
import sys
root = pathlib.Path(sys.argv[1])
for scene in ("sponza", "bistro", "forest"):
    directory = root / scene
    report = (directory / "bench_high.md").read_text()
    for variant in ("static", "moving_cam", "moving_light", "skinned"):
        image = directory / f"{scene}_1080p_high_{variant}.png"
        with image.open("rb") as stream:
            header = stream.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II", header[16:24]) != (1920, 1080):
            sys.exit(f"Invalid benchmark capture {image}")
        rows = [line.split("|") for line in report.splitlines() if f"| {variant} |" in line]
        if len(rows) != 1 or any(not math.isfinite(float(v)) or float(v) <= 0 for v in rows[0][4:7]):
            sys.exit(f"Missing valid timing row for {scene}/{variant}")
print("Qualified 12 scene/variant pairs at 1080p. Reports and images:", root)
PY
