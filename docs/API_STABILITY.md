# API Stability Policy

Flint is pre-1.0 in practice: the engine reports version **5.0.0** for every commit on the v5 line
and there are no release tags yet. This page defines what games may rely on, how the version and
pin work today, and how breaking changes are introduced.

## 1. Versioning

The engine version lives in `src/flint/core/build_info.mojo`:

```mojo
comptime FLINT_VERSION_MAJOR: Int = 5
comptime FLINT_VERSION_MINOR: Int = 0
comptime FLINT_VERSION_PATCH: Int = 0
comptime FLINT_ENGINE_VERSION: String = "5.0.0"
```

The full identity of a build is `5.0.0+<commit>` (`flint_version_pin()`), where `<commit>` is
`BUILD_COMMIT` from the generated, git-ignored `src/flint/core/build_stamp.mojo`. `tools/flint.sh`
writes that stamp before every build (`write_build_stamp`: short commit hash, dirty flag, Mojo
version); outside a git checkout the commit is `dev`.

`tools/flint.sh` repeats the version as `eng_maj=5 eng_min=0 eng_pat=0` in
`validate_project_flint_version` and in the scaffolded `tools/run.sh` template. **A version bump
must change all three places in one commit.**

From now on:

- **MAJOR** changes when a stable API (section 2) or a stable file format breaks.
- **MINOR** changes when stable API is added or a deprecation starts.
- **PATCH** changes for fixes that do not change stable API.
- Every bump gets a `CHANGELOG.md` section and a `v<version>` git tag.

## 2. What is public

**Stable** (covered by the version rules and the deprecation process):

| Area | Surface |
| :--- | :--- |
| ECS | Everything re-exported by `flint/ecs/__init__.mojo` (`World`, `Entity`, `Component`, `ComponentBitSet`, `ParallelCommands`, `Events`, `comp_hash64`, storage modes). |
| App | Everything re-exported by `flint/app/__init__.mojo`: `EngineApp`, `AppConfig`, `EnginePlugin`, `Resource`/`Resources`, the `SystemFn` signature, the `STAGE_*` constants and their order, `FixedTime`/`FrameTime`, flow helpers, `HotReloadHost`. |
| Plugin ABI | `FLINT_PLUGIN_API_VERSION` (currently 2), `FlintPluginApiV2` layout and the four `flint_plugin_*` export names. Any layout change bumps the ABI version; the host keeps accepting the previous version behind `allow_v1`-style opt-ins for at least one MINOR release. |
| Scene | Everything re-exported by `flint/scene/__init__.mojo`, the `SerializableComponent` trait. |
| Plugins | The plugin structs and their component types: `TransformPlugin`, `WindowPlugin`, `InputPlugin` + `InputMap`, `PhysicsPlugin`, `NavPlugin`, `AiPlugin`, `AnimPlugin`, `AudioPlugin` + `AudioServer` play/register methods, `RenderPlugin` + `ScalableRenderer` register methods, `NetPlugin`, `AssetPlugin` + `AssetServer` load/mount methods. Engine system names (`"flint.physics.fixed_step"`, ...) are stable because games order against them. |
| Data formats | `.flscn` (`SCENE_SCHEMA_VERSION` = 2) and `.flscnb` (`SCENE_BIN_MAGIC`), `.flprefab`, `.flpk` v3 (`FLPK_V3_VERSION`), world snapshots (`WORLD_BLOB_VERSION` = 3). Readers keep loading the previous version for at least one MAJOR release; scene JSON upgrades go through `SceneMigrationRegistry`. Component `COMPONENT_ID` hashes and `SCHEMA_NAME`s of engine components never change. |
| Tooling | `tools/flint.sh` subcommands and flags listed by `bash tools/flint.sh` (no arguments), and the `flint.toml` keys in [GETTING_STARTED.md](GETTING_STARTED.md#flinttoml-keys). |

**Provisional** (may change in any MINOR release; changes are noted in the changelog): `flint.render`
internals beyond the plugin surface (`FlintAaaVulkanRenderer`, render graph, shader DSL),
`flint.studio`, `flint.devtools` (including MCP tool names), `flint.ml`, `flint.cycles`,
`flint.procedural`, `flint.lab`, the `flint.net` wire format and crypto, and everything marked
**[Exp]** in [ARCHITECTURE.md](ARCHITECTURE.md).

**Internal** (no guarantees): names starting with `_`, `flint.platform.*` (use the plugins), the
legacy `flint.core.ecs` tables (except `run_parallel_ctx`), `build_stamp.mojo`, test helpers under
`tests/`, and the layout of `build/`.

## 3. Pinning the engine from a game

`[engine] flint_version` in a game's `flint.toml` is checked by `tools/flint.sh` (`build`, `run`,
`package`, `studio`, `gameplay-so`), by the scaffolded `tools/run.sh`, and at runtime by
`check_flint_version_pin` (the scaffolded `main.mojo` calls it on start).

| Pin | Accepts |
| :--- | :--- |
| `"5.0.0+1f500fd"` | exactly 5.0.0, and only when the engine checkout is at commit `1f500fd` (`flint.sh new` writes this form) |
| `"5.0.0"`, `"=5.0.0"`, `"==5.0.0"` | exactly 5.0.0, any commit |
| `">=5.0"`, `">5"`, `"<=5.1"`, `"<6"` | the comparison against `MAJOR.MINOR.PATCH` (missing parts are 0) |
| `"^5.0"`, `"~5.0"`, `"~=5.0"` | same MAJOR and not older than the pin |
| `"5.0.0+dirty"`, empty, or omitted | version check only / no check |

A leading `v` and a `-prerelease` suffix are ignored. The commit check is skipped when the engine
commit is `dev`. The commit must match the engine's `git rev-parse --short HEAD` output exactly.
Because every v5 commit currently reports 5.0.0, **a commit pin is the only way to lock a game to a
known engine API**; projects that track the engine loosely should use `^5.0` and run their own
tests after each engine update.

## 4. Deprecation process

1. **Announce.** Add the replacement first. Mark the old API in its docstring with
   `Deprecated since 5.N: use X. Removal in 6.0.` and add a `Deprecated` entry to `CHANGELOG.md`.
   (Mojo 1.2 has no deprecation attribute that works with `--Werror`-clean builds, so the docstring
   and changelog are the record.)
2. **Keep it working** for at least one MINOR release, with tests for both the old and the new
   path. Studio, examples and the scaffold template move to the new API in the same change.
3. **Remove** only in the next MAJOR release, with a `Removed` changelog entry and a migration
   note (code snippet or scene migration hook).
4. **Exception:** security fixes (for example in the MCP server or `flint.net`) may change
   behaviour without a deprecation period; the changelog entry is marked `Security`.

File formats follow the same steps, with readers for the old version kept for a full MAJOR release
and `SCENE_SCHEMA_VERSION` / `WORLD_BLOB_VERSION` / `FLPK_V3_VERSION` bumped on any layout change.
