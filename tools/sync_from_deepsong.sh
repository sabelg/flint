#!/usr/bin/env bash
# Harvests pure-Mojo, game-agnostic engine modules from a local Deepsong checkout
# into Flint (`src/flint/*`), rewrites module imports to `flint.*`, and enforces
# the Zero-Game-Leak Gate (no `gameplay.*` or `world.*` imports allowed).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERIFY_ONLY=0
FORCE_OVERWRITE=0
DEEPSONG_DIR="${DEEPSONG_DIR:-}"

for arg in "$@"; do
    case "$arg" in
        --verify|--check)
            VERIFY_ONLY=1
            ;;
        --force-overwrite)
            FORCE_OVERWRITE=1
            ;;
        *)
            DEEPSONG_DIR="$arg"
            ;;
    esac
done

if [[ "$VERIFY_ONLY" -eq 0 ]]; then
    if [[ -z "$DEEPSONG_DIR" ]]; then
        if [[ -d "../Deepsong/src" ]]; then
            DEEPSONG_DIR="$(cd ../Deepsong && pwd)"
        elif [[ -d "/tmp/Deepsong/src" ]]; then
            DEEPSONG_DIR="/tmp/Deepsong"
        else
            echo "error: Deepsong source not found. Set DEEPSONG_DIR=/path/to/Deepsong" >&2
            exit 1
        fi
    fi

    echo "[Flint Sync] Harvesting pure-Mojo libraries from: $DEEPSONG_DIR"

    python3 - "$DEEPSONG_DIR/src" "$ROOT/src/flint" "$FORCE_OVERWRITE" <<'PY'
import os
import re
import sys

src_root, dst_root, force_flag = sys.argv[1], sys.argv[2], int(sys.argv[3])

# Modules where Flint carries ahead-of-upstream capabilities (aarch64 syscalls,
# Vulkan/AMDGPU GGX PBR + analytical IBL + AgX Inset, .flpk bundler, RGBA PNG,
# xdg_toplevel resize/maximize, TTF SDF fonts, and concert BUS_MUSIC).
# Protected by default unless --force-overwrite is explicitly passed.
FLINT_ENHANCED = {
    ("core", "anchor.mojo"),
    ("core", "crash.mojo"),
    ("core", "cvars.mojo"),
    ("core", "detmath.mojo"),
    ("core", "jobs.mojo"),
    ("core", "perf_history.mojo"),
    ("core", "profiler.mojo"),
    ("platform", "audio.mojo"),
    ("platform", "pulse.mojo"),
    ("platform", "vulkan.mojo"),
    ("platform", "wayland.mojo"),
    ("assets", "glb_native.mojo"),
    ("assets", "image.mojo"),
    ("assets", "jpeg.mojo"),
    ("assets", "model.mojo"),
    ("assets", "vfs.mojo"),
    ("render", "font.mojo"),
    ("render", "fx_atlas.mojo"),
    ("render", "gpu_layout.mojo"),
    ("render", "gpu_raster.mojo"),
    ("render", "grading.mojo"),
    ("render", "lighting.mojo"),
    ("render", "rasterizer.mojo"),
    ("render", "spirv.mojo"),
    ("render", "vulkan_shaders.mojo"),
}

MODULES = {
    "core": [
        "anchor.mojo", "build_info.mojo", "check.mojo", "console.mojo", "crash.mojo",
        "crc32.mojo", "cvars.mojo", "detmath.mojo", "detsort.mojo", "entity.mojo",
        "ik.mojo", "jobs.mojo", "log.mojo", "lz4.mojo", "math.mojo", "memory.mojo",
        "offset_allocator.mojo", "perf_history.mojo", "profiler.mojo", "replay.mojo",
        "serial.mojo", "session_replay.mojo", "springs.mojo", "test_history.mojo",
        "transform.mojo",
    ],
    "platform": [
        "amd_isa.mojo", "amdgpu.mojo", "audio.mojo", "input.mojo", "narration.mojo",
        "os.mojo", "pulse.mojo", "vulkan.mojo", "wayland.mojo", "wayland_bridge.mojo",
    ],
    "assets": [
        "asset_check.mojo", "clip_names.mojo", "cook.mojo", "cooked.mojo", "cooked_model.mojo",
        "glb.mojo", "glb_native.mojo", "hot_reload.mojo", "image.mojo", "jpeg.mojo", "json.mojo",
        "json_write.mojo", "manifest.mojo", "model.mojo", "vfs.mojo",
    ],
    "text": [
        "lettering.mojo", "line_break.mojo", "rich_text.mojo", "sdf_bake.mojo",
        "ttf.mojo", "utf8.mojo",
    ],
    "ui": [
        "button_glyphs.mojo", "layout.mojo", "nine_slice.mojo", "widgets.mojo",
    ],
    "render": [
        "debug_draw.mojo", "dither.mojo", "flip.mojo", "font.mojo", "frame_passes.mojo",
        "fx_atlas.mojo", "gpu_layout.mojo", "gpu_raster.mojo", "gpu_skinning.mojo",
        "grading.mojo", "image_diff.mojo", "kernel.mojo", "lighting.mojo",
        "livelink.mojo", "particle_atlas.mojo", "particle_layout.mojo",
        "rasterizer.mojo", "spirv.mojo", "vulkan_shaders.mojo",
    ],
    "physics": [
        "broadphase.mojo", "density_table.mojo", "events.mojo", "forces.mojo",
        "handling.mojo", "islands.mojo", "joints.mojo", "mass.mojo", "ropes.mojo",
        "solver.mojo", "solver_bodies.mojo", "water.mojo", "wind.mojo",
        "wind_mirror.mojo", "xpbd.mojo",
    ],
    "anim": [
        "animator.mojo", "cloth_proxy.mojo", "correctives.mojo", "graph.mojo",
        "helper_joints.mojo", "pose.mojo", "procedural.mojo", "ragdoll.mojo",
        "skin_limb.mojo", "skinning.mojo", "spring_bones.mojo", "sync.mojo",
    ],
}

SUBPKGS = ("core", "platform", "assets", "text", "ui", "render", "physics", "anim")

copied = 0
for pkg, files in MODULES.items():
    out_dir = os.path.join(dst_root, pkg)
    os.makedirs(out_dir, exist_ok=True)
    for fname in files:
        in_path = os.path.join(src_root, pkg, fname)
        if not os.path.isfile(in_path):
            print(f"  [skip missing] {pkg}/{fname}")
            continue
        with open(in_path, "r", encoding="utf-8") as fh:
            text = fh.read()

        # Decouple specific shims before generic namespace rewrite
        text = re.sub(
            r"^from gameplay\.tunables import",
            "from flint.physics.defaults import",
            text,
            flags=re.MULTILINE,
        )
        text = re.sub(
            r"^from physics\.collision import",
            "from flint.physics.filter import",
            text,
            flags=re.MULTILINE,
        )
        text = re.sub(
            r"^from render\.occupancy import",
            "from flint.render.occupancy_layout import",
            text,
            flags=re.MULTILINE,
        )

        # Rewrite top-level engine package imports to `flint.<pkg>`
        for sp in SUBPKGS:
            text = re.sub(
                rf"^from {sp}\.",
                f"from flint.{sp}.",
                text,
                flags=re.MULTILINE,
            )
            text = re.sub(
                rf"^from {sp} import ",
                f"from flint.{sp} import ",
                text,
                flags=re.MULTILINE,
            )

        # Enrich Vec3 with ergonomic .dot() and .cross() methods
        if pkg == "core" and fname == "math.mojo":
            target = "    def distance_to(self, other: Self) -> Float32:\n        return (other - self).length()"
            extra = (
                "    def distance_to(self, other: Self) -> Float32:\n        return (other - self).length()\n\n"
                "    @always_inline\n"
                "    def dot(self, other: Self) -> Float32:\n"
                "        return self.x * other.x + self.y * other.y + self.z * other.z\n\n"
                "    @always_inline\n"
                "    def cross(self, other: Self) -> Self:\n"
                "        return Self(\n"
                "            self.y * other.z - self.z * other.y,\n"
                "            self.z * other.x - self.x * other.z,\n"
                "            self.x * other.y - self.y * other.x,\n"
                "        )"
            )
            if "def dot(self," not in text:
                text = text.replace(target, extra)

        # Upgrade VFS pack magic & extensions to Flint defaults while preserving env overrides
        if pkg == "assets" and fname == "vfs.mojo":
            text = text.replace('PACK_PATH = "build/deepsong.sdpack"', 'PACK_PATH = "build/assets.flpk"')
            text = text.replace('0x4B504453  # "SDPK"', '0x4B504C46  # "FLPK"')
            text = text.replace('env_now("SD_COOKED")', 'env_now("FLINT_COOKED")')
            text = text.replace('env_now("SD_PACK")', 'env_now("FLINT_PACK")')

        if pkg == "assets" and fname == "asset_check.mojo":
            text = text.replace("re-export with bash tools/run.sh assets ", "re-export asset ")

        if pkg == "assets" and fname == "hot_reload.mojo":
            text = text.replace("assets/deepsong/layers/layer5.json", "assets/layers/layer5.json")

        # Enrich Wayland evdev key table with full US-QWERTY punctuation & Flint window title
        if pkg == "platform" and fname == "wayland.mojo":
            text = text.replace('"SHARDDEPTH"', '"FLINT ENGINE"')
            text = text.replace('"sharddepth"', '"flint"')
            text = text.replace('"sharddepth-framebuffer"', '"flint-framebuffer"')
            target = "    if key == 88:\n        return 301"
            extra = (
                "    if key == 88:\n        return 301\n"
                "    if key == 13:\n        return 61\n"
                "    if key == 26:\n        return 91\n"
                "    if key == 27:\n        return 93\n"
                "    if key == 39:\n        return 59\n"
                "    if key == 40:\n        return 39\n"
                "    if key == 43:\n        return 92\n"
                "    if key == 51:\n        return 44\n"
                "    if key == 53:\n        return 47"
            )
            if "if key == 26:" not in text:
                text = text.replace(target, extra)

        # Enrich 5x7 bitmap font with '$', '#', '[', ']', and quotes
        if pkg == "render" and fname == "font.mojo":
            target = "        case 60: return 0x410411110"
            extra = (
                "        case 60: return 0x410411110\n"
                "        case 35: return 0x15F55F54A\n"
                "        case 36: return 0x11F4717C4\n"
                "        case 39: return 0x000001084\n"
                "        case 91: return 0x38421084E\n"
                "        case 93: return 0x39084210E"
            )
            if "case 36:" not in text:
                text = text.replace(target, extra)

        out_path = os.path.join(out_dir, fname)
        if os.path.isfile(out_path) and (pkg, fname) in FLINT_ENHANCED and not force_flag:
            continue
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        copied += 1

print(f"[Flint Sync] Synced and namespaced {copied} pure-Mojo modules into src/flint/ (protected {len(FLINT_ENHANCED)} Flint-enhanced modules).")
PY
fi

# Zero-Game-Leak Gate: ensure zero gameplay.* or world.* imports exist anywhere in src/flint/
if grep -rnE "^from (gameplay|world)(\.| )" "$ROOT/src/flint/"; then
    echo "[FAIL] Game-specific imports leaked into src/flint/!" >&2
    exit 1
fi

echo "[PASS] Zero-Game-Leak Gate: 0 gameplay/voxel imports across $(find "$ROOT/src/flint" -name '*.mojo' | wc -l) Flint modules."
