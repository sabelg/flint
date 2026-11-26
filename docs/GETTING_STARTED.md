# Getting Started

From a fresh checkout to a running game, a Studio session and a release bundle. Every command
below is a `tools/flint.sh` subcommand; run `bash tools/flint.sh` with no arguments for the list.

## 1. Requirements

- **Linux, x86_64 or aarch64.** `pixi.toml` only lists `linux-64` and `linux-aarch64`. Windows is
  reached through a WSL2 bundle (see [Packaging](#7-packaging)); the macOS and Win32 backends in
  `src/flint/platform/` are not wired into `WindowPlugin` and are untested.
- **[Pixi](https://pixi.sh)**, which installs Mojo `>=1.2,<1.3` and `max-core` (used by `flint.ml`
  for GPU matrix products). A `mojo` 1.2 on `PATH`, or `MOJO=/path/to/mojo`, also works.
- **Optional:** a Wayland or X11 session (or a text console for DRM/KMS) for windows, a Vulkan
  driver for the GPU renderer, PulseAudio/PipeWire or ALSA for sound. Without them Flint runs
  headless and renders with the CPU `SimdCanvas` (render tier 0).

## 2. Install and check the toolchain

```bash
git clone <flint-repo> ~/flint && cd ~/flint
pixi install                      # creates .pixi/envs/default with Mojo 1.2
bash tools/flint.sh doctor        # Mojo, Vulkan ICDs, Wayland/X11/KMS, audio, evdev, inotify
bash tools/flint.sh play          # build + run examples/minimal_spark (build/flint-spark)
bash tools/flint.sh shot          # headless PNG -> build/shots/flint_spark.png
```

`flint.sh` finds Mojo in `.pixi/envs/default`, then `../.pixi/envs/default`, then
`~/.pixi/envs/turbomojo`, then `PATH`. Binaries go to `build/` and are rebuilt only when a `.mojo`
file under `src/` (or the program's own tree) is newer; `FLINT_FORCE_BUILD=1` forces a rebuild.

Every build uses `-O3 --fp-mode contract=off --Werror`. `FLINT_WERROR=0` and
`FLINT_DETERMINISTIC_FP=0` turn the last two off for a local experiment; the gate does not.

## 3. Create a game

```bash
bash ~/flint/tools/flint.sh new ../my-game
```

`cmd_new` writes this layout (the game lives outside the engine repository):

| Path | Contents |
| :--- | :--- |
| `flint.toml` | Project manifest (keys below). `[engine] flint_version` is pinned to `5.0.0+<engine commit>`. |
| `src/main.mojo` | A complete ~600-line game: reads `flint.toml`, builds an `EngineApp` with `TransformPlugin`, `WindowPlugin` and `InputPlugin`, loads the entry scene and draws it with `SimdCanvas`. A/D or the arrow keys orbit, Esc quits. |
| `assets/main.flscn` | JSON scene: ground, cube, camera, point light, environment. |
| `tools/run.sh` | Standalone build-and-run script (finds Flint via `FLINT_ROOT`, `../flint` or `~/flint`). |
| `pixi.toml` | Mojo 1.2 dependency and a `play` task that runs `tools/run.sh`. |
| `README.md`, `.gitignore` | Quick-start notes; ignores `build/`, `dist/`, `.pixi/`, `*.flpk`, `*.png`. |

### `flint.toml` keys

Some keys are read by `tools/flint.sh`, others only by the generated `src/main.mojo` (which has
its own small TOML reader); the engine library itself only reads `[engine] flint_version`
(`flint.core.build_info.check_flint_version_pin`).

| Key | Read by | Meaning |
| :--- | :--- | :--- |
| `[project] name` | flint.sh, main.mojo | Binary name `build/<name>`; defaults to the directory name. |
| `[project] entry` | flint.sh | Entry source; default `src/main.mojo`. |
| `[project] entry_scene` | main.mojo | Scene loaded at start (`.flscn` JSON or `.flscnb` binary). |
| `[project] version` | `package` | Written to the bundle manifest. |
| `[project] gameplay_entry`, `gameplay_output` | `gameplay-so` | Hot-reload plugin source / output; defaults `src/gameplay.mojo`, `build/gameplay.so`. |
| `[engine] flint_version` | flint.sh, `build_info` | Engine pin, e.g. `"5.0.0+1f500fd"`, `">=5.0"`, `"^5"`. See [API_STABILITY.md](API_STABILITY.md). |
| `[engine] werror`, `deterministic_fp` | flint.sh | `false` drops `--Werror` / `--fp-mode contract=off` for this project. |
| `[window] title`, `width`, `height`, `vsync` | main.mojo | Passed to `AppConfig` / `WindowPlugin` (`vsync` is parsed but not used by the template). |
| `[physics] fixed_hz` | main.mojo | `AppConfig.fixed_hz` (the `STAGE_FIXED_UPDATE` rate). |
| `[net] tick_hz`, `max_rollback_ticks`, `mtu_bytes` | main.mojo | Parsed and printed only; the template does not add `NetPlugin`. |
| `[assets] pack_output` | flint.sh, main.mojo | `.flpk` written by `pack` and mounted at start; default `build/assets.flpk`. `[project] pak_path` / `[assets] pak_path` are legacy fallbacks. |
| `[studio] motion_entry`, `combat_entry`, `room_entry` | `studio --motion/--combat/--room` | Game-specific editor entry points (see README). |
| `[build_profiles.<name>] platform`, `optimization`, `loose_assets` | `package --profile` | `platform` is `linux` or `windows-wsl2`, `optimization` `O0`..`O3`. |

## 4. Run it

From the game directory:

```bash
bash ~/flint/tools/flint.sh run                    # build build/my-game with the engine's toolchain, then exec it
bash ~/flint/tools/flint.sh run --headless-smoke   # 4 headless frames, prints "Headless v5 frame loop OK"
bash ~/flint/tools/flint.sh run --shots --out build/shots/my-game.png
bash ~/flint/tools/flint.sh build                  # build only
bash ~/flint/tools/flint.sh preview .              # -O1 rebuild on every .mojo save, swaps the window
FLINT_ROOT=~/flint bash tools/run.sh               # the same without flint.sh (or: pixi run play)
```

The template goes headless when `FLINT_HEADLESS=1` is set, or when none of `FLINT_BACKEND`,
`WAYLAND_DISPLAY`, `DISPLAY` is set and it is not on a virtual console. A freshly scaffolded game
builds and passes `--headless-smoke` in about 15 s on a warm cache.

Next steps for the code: [ECS_GUIDE.md](ECS_GUIDE.md) (World, systems, stages) and
[GAMEPLAY_GUIDE.md](GAMEPLAY_GUIDE.md) (scenes, assets, input, plugins, hot reload).
`examples/test_harness_game/` is a fuller game that uses every plugin and a hot-reloaded
`gameplay.so`.

## 5. Flint Studio

```bash
cd ../my-game
bash ~/flint/tools/flint.sh studio                         # editor for the current project
bash ~/flint/tools/flint.sh studio --project ../my-game    # or name the project explicitly
bash ~/flint/tools/flint.sh studio --headless-smoke --out build/studio.png
bash ~/flint/tools/flint.sh mcp                            # Studio as a headless MCP v2 stdio server
```

`studio` builds `examples/studio/main.mojo` into `~/flint/build/flint-studio` and exports the
calling directory as `FLINT_STUDIO_PROJECT`. The pin in the project's `flint.toml` is checked first.
The motion, combat, room, profile-compare and TurboMojo IDE modes are documented in the README.

## 6. Hot-reloaded gameplay

Put gameplay logic in `src/gameplay.mojo` (C-ABI exports, see
[GAMEPLAY_GUIDE.md](GAMEPLAY_GUIDE.md#hot-reloading-gameplay-so)) and build it with

```bash
bash ~/flint/tools/flint.sh gameplay-so            # -O1; --fast/--debug = -O0, --release = -O3, --out PATH
```

The `.so` is written atomically to `[project] gameplay_output`; the game loads or swaps it with
`HotReloadHost.load_or_reload`.

## 7. Packaging

```bash
bash ~/flint/tools/flint.sh pack            # assets/ -> [assets] pack_output (.flpk)
bash ~/flint/tools/flint.sh package         # release bundle in dist/<name>/
bash ~/flint/tools/flint.sh package --list-profiles
bash ~/flint/tools/flint.sh package --profile linux-development   # -> build/packages/<profile>/
```

`package` builds the entry, copies it to `bin/<name>`, rewrites its RUNPATH to `$ORIGIN/../lib`
(needs `chrpath` and `readelf`), copies the Mojo runtime libraries it links (`ldd`) into `lib/`,
cooks `assets/` into `assets.flpk` (or copies them with `--loose-assets`), and writes `flint.toml`,
`run.sh` and `MANIFEST.txt` (architecture, platform, optimization, binary SHA-256). A
`windows-wsl2` profile also bundles `Play-WSL.cmd`, `run-wsl.sh` and the sources for a first-run
build inside WSL. Engine examples can be bundled with `--target spark|world|studio|cycles|kms|kms3d`.

## 8. Troubleshooting

| Symptom | Fix |
| :--- | :--- |
| `Mojo 1.2 compiler not found` | Run `pixi install` in the Flint checkout, or export `MOJO=/path/to/mojo`. |
| `flint_version mismatch` / `commit mismatch` | The project pins a different engine. Update the pin or check out the pinned commit. |
| `No flint.toml found` | `run`, `gameplay-so` and project `build`/`package` must be started from the game directory. |
| Black window or CPU-only rendering | `bash tools/flint.sh doctor`; without a Vulkan ICD the renderer falls back to tier 0. |
| `Failed to initialize Crashpad` on stderr | Harmless message from the Mojo runtime. |
