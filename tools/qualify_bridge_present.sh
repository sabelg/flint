#!/usr/bin/env bash
# Six windowed Vulkan traversals, using separate engine and driver cache roots per resolution.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
out_parent=${FLINT_QUALIFY_DIR:-"$root/build/production-qualification"}
mkdir -p "$out_parent"
output=$(realpath "$(mktemp -d "$out_parent/presented-XXXXXX")")
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
prefix="$root/.pixi/envs/default"
export MODULAR_HOME="$prefix/share/max" PATH="$prefix/bin:$PATH" LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
flags=(-I src -I tests -O3 --fp-mode contract=off --Werror)
printf 'Writing native presentation logs to %s\n' "$output"
mojo build "${flags[@]}" examples/open_world/main.mojo -o "$scratch/world"
mojo build "${flags[@]}" examples/pack_cli/main.mojo -o "$scratch/pack"
"$scratch/pack" --cook examples/character_course/assets "$scratch/bridge.flpk" --cache-dir "$scratch/cooker" > "$output/cook.log"
{
    mojo --version
    git rev-parse HEAD
    git status --porcelain -- src examples tools
    sha256sum "$scratch/world" "$scratch/bridge.flpk"
    if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=name,driver_version --format=csv; fi
    printf 'session=%s Wayland=%s X11=%s\n' "${XDG_SESSION_TYPE:-}" "${WAYLAND_DISPLAY:-}" "${DISPLAY:-}"
} > "$output/environment.txt"
target=1800
for height in 1080 800; do
    width=1920
    if [[ $height == 800 ]]; then width=1280; fi
    frames=$target
    for phase in cold warm1 warm2; do
        attempt=0
        while :; do
            ((attempt += 1))
            # Every cold retry starts from empty caches; the qualified cold run seeds the two warm runs.
            if [[ $phase == cold ]]; then
                profile=$(mktemp -d "$scratch/profile-$height-XXXXXX")
                mkdir -p "$profile/build" "$profile/driver/mesa" "$profile/save"
            fi
            log="$output/${height}p-$phase-attempt-$attempt.log"
            (
                cd "$profile"
                export FLINT_WINDOWED=1 FLINT_SAVE_DIR="$profile/save"
                export XDG_CACHE_HOME="$profile/driver" MESA_SHADER_CACHE_DIR="$profile/driver/mesa"
                export __GL_SHADER_DISK_CACHE=1 __GL_SHADER_DISK_CACHE_PATH="$profile/driver"
                date --iso-8601=seconds
                printf 'requested=%sx%s cache=%s frames=%s driver_bytes_before=' "$width" "$height" "$phase" "$frames"
                du -sb "$profile/driver" | cut -f1
                if [[ -s build/vk_pso.cache ]]; then sha256sum build/vk_pso.cache; else echo 'Engine pipeline cache absent'; fi
                "$scratch/world" --bridge --frames "$frames" --pack "$scratch/bridge.flpk" --width "$width" --height "$height"
                printf 'driver_bytes_after='
                du -sb "$profile/driver" | cut -f1
                if [[ -s build/vk_pso.cache ]]; then sha256sum build/vk_pso.cache; fi
            ) > "$log" 2>&1
            if ! grep -Fq "offscreen False width $width height $height RT " "$log" ||
               ! grep -Fq 'zero_copy True' "$log"; then
                echo "Native swapchain extent or session unavailable. See $log" >&2
                exit 1
            fi
            visible=$(awk '/^\[Bridge\] presentation timestamps / {print $6}' "$log")
            if [[ -z $visible ]]; then
                grep -Fq 'actualPresentTime unavailable' "$log" || { echo "No timing result in $log" >&2; exit 1; }
                echo "Presentation feedback unavailable. Only submitted frames are qualified." >> "$log"
                break
            fi
            if (( visible >= target )); then break; fi
            if (( visible == 0 || attempt >= 3 )); then
                echo "Only $visible visible presentations. See $log" >&2
                exit 1
            fi
            # Allow a margin for changes in compositor cadence and preserve the evidence from this trial.
            frames=$(( (frames * (target + target / 6) + visible - 1) / visible ))
        done
        cp "$log" "$output/${height}p-$phase.log"
        grep -E '^\[Bridge\] (frames|GPU timestamp|presentation timestamps|actualPresentTime|measured refresh|Wayland feedback)' "$log"
    done
done
