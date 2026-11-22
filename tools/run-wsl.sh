#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
kit="$here/wsl-source"
if [[ "$(uname -m)" == "$(cat "$kit/build-arch")" && "${FLINT_WSL_REBUILD:-0}" != 1 ]]; then
    exec bash "$here/run.sh" "$@"
fi
# Pixi installs the locked runtime for the receiving machine's Linux architecture.
pixi_bin="$(command -v pixi || true)"
[[ -n "$pixi_bin" ]] || pixi_bin="$HOME/.pixi/bin/pixi"
[[ -x "$pixi_bin" ]] || { echo "Install Pixi in the WSL distribution to build for $(uname -m)." >&2; exit 1; }
cache="${XDG_CACHE_HOME:-$HOME/.cache}/flint-wsl"
mkdir -p "$cache"
# Contents, not filenames or timestamps, identify a revision of this kit.
key="$(cd "$kit" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
build="$cache/$key"
mkdir -p "$build"
exec 9>"$build/build.lock"
flock 9
if [[ ! -x "$build/player" ]]; then
    cp "$kit/engine/pixi.toml" "$kit/engine/pixi.lock" "$build/"
    "$pixi_bin" install --locked --manifest-path "$build/pixi.toml"
    mapfile -t flags < "$kit/flags"
    "$pixi_bin" run --manifest-path "$build/pixi.toml" mojo build "$kit/game/$(cat "$kit/entry")" \
        -I "$kit/game/src" -I "$kit/engine/src" "$(cat "$kit/optimization")" "${flags[@]}" -o "$build/player.next"
    mv "$build/player.next" "$build/player"
fi
flock -u 9
cd "$here"
export LD_LIBRARY_PATH="$build/.pixi/envs/default/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$build/player" "$@"
