#!/usr/bin/env bash
# Exercise the launcher with a fake desktop terminal so this test also runs without a display.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
first=""
cleanup() {
    # Never leave a held pane behind if an assertion fails while it waits for release.
    [[ -n "$first" ]] && kill "$first" 2>/dev/null || true
    rm -rf "$work"
}
trap cleanup EXIT
project="$work/game with spaces"
mkdir -p "$project/code" "$work/bin"
printf '[project]\nentry = "code/entry.mojo"\n' > "$project/flint.toml"
touch "$project/code/entry.mojo" "$project/code/source:123" "$project/code/a file.mojo"
cat > "$work/bin/gnome-terminal" <<'TERMINAL'
#!/usr/bin/env bash
[[ "$1" == "--working-directory=$PWD" && "$2" == -- ]] || exit 2
shift 2
exec "$@"
TERMINAL
cat > "$work/bin/test ide" <<'IDE'
#!/usr/bin/env bash
[[ "$#" == 2 && "$1" == --reuse && -x "$TURBOMOJO_COMPILER" ]] || exit 3
printf '%s\n%s\n%s\n' "$PWD" "$2" "$XDG_STATE_HOME" > "$IDE_TEST_LOG"
IDE
chmod +x "$work/bin/gnome-terminal" "$work/bin/test ide"
export PATH="$work/bin:$PATH" DISPLAY=:test FLINT_TURBOMOJO="$work/bin/test ide"
export FLINT_IDE_STATE_BASE="$work/state" IDE_TEST_LOG="$work/launch"
cd "$project/code"
open_ide() { bash "$root/tools/flint.sh" studio --ide-window "$@"; }
open_ide
mapfile -t launch < "$IDE_TEST_LOG"
[[ "${launch[0]}" == "$project" && "${launch[1]}" == "$project/code/entry.mojo" ]]
[[ -d "${launch[2]}" && "$(readlink "$project/build/flint-ide/engine-src")" == "$root/src" ]]
for arg in 'a file.mojo:12:3' 'source:123:5' 'source:123'; do
    open_ide "$arg"
    [[ "$(sed -n '2p' "$IDE_TEST_LOG")" == "$project/code/$arg" ]]
done
if open_ide 'entry.mojo:0' 2>/dev/null || open_ide 'missing.mojo:12' 2>/dev/null; then
    exit 1
fi
echo 'Studio IDE project discovery, source locations and launcher validation passed.'

# A pane owns its session. Two Studio windows for one game must keep separate
# tmide IPC endpoints and open the requested files without --reuse.
cat > "$work/bin/pane ide" <<'IDE'
#!/usr/bin/env bash
set -eu
[[ "$#" == 1 && "$TERM" == xterm-256color && "$COLORTERM" == truecolor ]]
printf '%s\n%s\n' "$1" "$XDG_STATE_HOME" > "$IDE_TEST_LOG"
if [[ "${TEST_HOLD:-}" == 1 ]]; then
    # Handshake: tell the test this pane is up (and holds its session lock), then keep the
    # lock until the test releases it. No timeouts: the test decides when the pane goes away.
    echo ready > "$TEST_READY"
    read -r _ < "$TEST_RELEASE"
fi
IDE
chmod +x "$work/bin/pane ide"
export FLINT_TURBOMOJO="$work/bin/pane ide"
mkfifo "$work/ready" "$work/release"
# Open both FIFOs read-write here, so neither side's open() blocks and a write never waits
# for a reader that has not started yet.
exec 7<> "$work/ready" 8<> "$work/release"
TEST_HOLD=1 TEST_READY="$work/ready" TEST_RELEASE="$work/release" IDE_TEST_LOG="$work/first" \
    bash "$root/tools/flint.sh" studio --ide-embedded &
first=$!
until read -r -t 1 -u 7 _; do
    # Waiting on the pane's own signal; only bail out if it died before sending it.
    kill -0 "$first" 2>/dev/null || { echo 'First pane exited before its ready handshake' >&2; exit 1; }
done
[[ -s "$work/first" ]]
IDE_TEST_LOG="$work/second" bash "$root/tools/flint.sh" studio --ide-embedded 'a file.mojo'
echo done >&8
wait "$first"
first=""
exec 7>&- 8>&-
mapfile -t one < "$work/first"
mapfile -t two < "$work/second"
[[ "${one[0]}" == "$project/code/entry.mojo" && "${two[0]}" == "$project/code/a file.mojo" ]]
[[ "${one[1]}" == */embedded-1 && "${two[1]}" == */embedded-2 ]]
IDE_TEST_LOG="$work/third" bash "$root/tools/flint.sh" studio --ide-embedded
[[ "$(sed -n '2p' "$work/third")" == "${one[1]}" ]]
echo 'Embedded pane launch, isolation and session slot reuse passed.'

# Both graphical entry points share the project directory and isolated GUI session.
cat > "$work/bin/gui ide" <<'IDE'
#!/usr/bin/env bash
set -eu
printf '%s\n' "$PWD" "$XDG_STATE_HOME" "$@" > "$IDE_TEST_LOG"
IDE
chmod +x "$work/bin/gui ide"
export FLINT_TURBOMOJO_GUI="$work/bin/gui ide"
bash "$root/tools/flint.sh" studio --moves
mapfile -t moves < "$IDE_TEST_LOG"
[[ "${#moves[@]}" == 4 && "${moves[0]}" == "$project" && "${moves[1]}" == */gui-1 ]]
[[ "${moves[2]}" == --moves && "${moves[3]}" == "$project/code/entry.mojo" ]]
bash "$root/tools/flint.sh" studio --ide-gui 'a file.mojo:3'
mapfile -t gui < "$IDE_TEST_LOG"
[[ "${#gui[@]}" == 3 && "${gui[1]}" == "${moves[1]}" && "${gui[2]}" == "$project/code/a file.mojo:3" ]]
echo 'Graphical editor and Move Library launch arguments passed.'
