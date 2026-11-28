# Contributing to Flint

Flint is a pure Mojo 1.2 engine with no C/C++ wrapper libraries. Read [AGENTS.md](AGENTS.md) (the
rules below come from it) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before a larger change.

## Setup

```bash
pixi install                    # Mojo 1.2 toolchain in .pixi/envs/default
bash tools/flint.sh doctor      # what this machine can run (Vulkan, display, audio, evdev)
bash tools/ci.sh                # the check CI runs (~10 min cold on 20 cores, ~1.5 min warm)
```

## Ground rules

- **No foreign code.** No SDL, GLFW, OpenGL, shaderc, miniaudio, freetype, ENet, Box2D/Jolt or other
  C/C++ libraries. Talk to the OS and drivers directly (syscalls, Wayland/X11/DRM protocols,
  Vulkan through the system loader).
- **Engine/game separation.** Nothing under `src/flint/` may import `gameplay.*`, `world.*` or
  other game modules, or mention a game by name. `bash tools/sync_from_deepsong.sh --verify`
  checks this.
- **Zero warnings.** Everything builds with `-I src -I tests -O3 --fp-mode contract=off --Werror`
  and no external `-I` paths. A warning is a build failure.
- **Determinism.** Simulation code must give the same result on 1 and N threads: structural ECS
  changes from parallel code go through `ParallelCommands`, transcendental math in simulation uses
  `flint.core.detmath`, and parallel work is dispatched with `flint.core.ecs.run_parallel_ctx`.

## Mojo 1.2 rules

- `def` for every function and method (`fn` is gone); `raises` goes before `->`.
- `comptime` instead of `alias` and `@parameter`; `var` instead of `let`.
- Argument conventions: immutable borrow by default, `mut`, `var` (owned), `out self` in
  `__init__`, `deinit self` in `__deinit__`.
- Structs: `@fieldwise_init` with explicit `Copyable, Movable`; move non-`ImplicitlyCopyable`
  values with `^`. Inside a generic `struct Foo[T: Trait]`, write `Self.T`.
- Standard library imports use the `std.` prefix (`from std.testing import assert_equal`).
  Strings are UTF-8: `s.byte_length()` and `s[byte=i]`.
- **ASAP destruction.** A local dies after its last use, even if a worker thread still holds its
  address. Dispatch `JobPool` work through `run_parallel_ctx`, call `.keep_alive()` on a
  `SimdColumn` (or `World`) whose `base_addr()` you passed to workers, and prefer
  `hash_list(bytes)` over `block_hash(address_of(bytes), ...)` on a `List[UInt8]` at its last use.

## Tests and gates

| Command | What it runs | Needs |
| :--- | :--- | :--- |
| `bash tools/ci.sh` | Leak gate, `--Werror` builds of the unit/e2e/cycles/v5 test binaries, all CPU-safe suites in parallel; `test_flint` and `test_e2e` on Mesa lavapipe if installed | Mojo (+ `mesa-vulkan-drivers` for the Vulkan pair) |
| `bash tools/flint.sh test unit\|e2e\|v5\|studio` | One group of suites | some suites need Vulkan |
| `bash tools/flint.sh lab <physics\|anim\|render\|audio_ui\|net\|all> [--fast]` | Subsystem benchmarks | - |
| `bash tools/flint.sh gate` | Everything: sync verify, all builds, all suites, Studio suites, headless golden shots, labs, FLIP and p99 perf checks (~50 min, serial) | Vulkan GPU |

Run `tools/ci.sh` before every push and `tools/flint.sh gate` before merging anything that
touches rendering, platform code or Studio. If you change a golden image, say so in the commit
message.

### Adding a test suite

- **Subsystem suite (preferred):** create `tests/test_flint_v5_<area>.mojo` with
  `def main() raises:` that calls your `test_*` functions and raises (or fails an `assert_*` from
  `std.testing`) on failure. `flint.sh build v5` and `flint.sh test v5` pick up every
  `tests/test_flint_v5_*.mojo` automatically, and so does `tools/ci.sh`. If the suite needs a
  Vulkan device, either skip the GPU part when the renderer is inactive (print `[SKIP] ...`, as
  `test_flint_v5_render.mojo` does) or list it in `V5_EXCLUDE` / `VULKAN_SUITES` in
  `tools/ci.sh`.
- **E2E case:** add a module under `tests/e2e/tier*/` exposing `run_all(mut ctx: TestContext)`,
  then import it and call it from `tests/test_e2e.mojo`.
- **Studio editor suite:** create `tests/test_studio_<topic>.mojo` and add its name to
  `STUDIO_TEST_SUITES` in `tools/flint.sh` (built at `-O1`, each run gets a fresh
  `XDG_STATE_HOME`).
- **Shell-level Studio/shipping checks:** `tests/e2e_studio/tier*/` scripts, run by
  `tests/e2e_studio/run_studio_e2e.sh`. The gate runs tier 1 only (`.mojo` tests compiled in
  parallel at `-O1`). Tiers 2-4 (boundary, cross-feature, scenario) take well over the gate's
  budget (tier 2 alone is about 11 minutes), so they run in the `e2e-studio` CI job and locally
  with `bash tools/flint.sh test e2e-studio --extended` (`--all` runs tiers 1-4;
  `FLINT_GATE_E2E_TIERS="1 2"` widens the gate itself).

Write temporary files under a per-process path (for example include `getpid()`), because
`tools/ci.sh` runs suites in parallel.

## Branches and commits

- `main` is the default branch; work happens on topic branches (`studio-redesign`, `gaps-*`,
  `claude/<topic>`), merged after review. Do not commit directly to `main`.
- Commit subjects follow Conventional Commits with the subsystem as the scope:
  `feat(render): ...`, `fix(studio): ...`, `docs(v5): ...`; several scopes are comma-separated
  (`feat(ui,studio): ...`). Older history also uses `physics/nav: ...`. Use the imperative mood and
  keep the subject under about 100 characters.
- The body explains what changed and why, grouped by area for larger commits, and names new
  tests and updated goldens. AI-assisted commits end with a `Co-Authored-By:` trailer.
- Add a line to the `[Unreleased]` section of [CHANGELOG.md](CHANGELOG.md) for user-visible
  changes, and follow [docs/API_STABILITY.md](docs/API_STABILITY.md) for anything that changes a
  stable API or file format.
