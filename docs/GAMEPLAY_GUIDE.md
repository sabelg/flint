# Gameplay Guide

Building a game on `EngineApp`: plugins, scenes and prefabs, assets, input, physics, audio,
rendering and hot-reloaded gameplay code. Read [ECS_GUIDE.md](ECS_GUIDE.md) first for World,
systems and stages. Snippets are copied or trimmed from the cited files and use Mojo 1.2 syntax.

The two reference programs are the scaffold that `tools/flint.sh new` writes (template inside
`cmd_new` in `tools/flint.sh`) and `examples/test_harness_game/`, which uses every plugin below.

## App skeleton

```mojo
# examples/test_harness_game/src/main.mojo (_create_configured_app, trimmed)
var app = EngineApp(AppConfig(
    title=cfg.window_title,
    headless=effective_headless,
    fixed_hz=Float32(cfg.fixed_hz),
    worker_threads=0,            # 0 = run systems inline; default -1 = one worker per core
    enable_crash_handler=True,
))
app.add_plugin(TransformPlugin(snapshot_prev_in_fixed=True))
app.add_plugin(WindowPlugin(headless=effective_headless, width=cfg.window_width,
                            height=cfg.window_height, title=cfg.window_title))
app.add_plugin(InputPlugin(enable_evdev=not effective_headless))
app.add_plugin(PhysicsPlugin(gravity=Float32(0.0)))
app.add_plugin(NavPlugin(draw_debug=True))
app.add_plugin(AiPlugin())
app.add_plugin(AudioPlugin(headless=effective_headless))
app.add_plugin(AnimPlugin())
setup_harness_input(app)
register_harness_systems(app)    # app.add_system(...) calls
```

| Plugin | Module | Adds |
| :--- | :--- | :--- |
| `TransformPlugin` | `flint.scene` | `LocalTransform` -> `GlobalTransform` propagation in `STAGE_POST_UPDATE`; with `snapshot_prev_in_fixed=True`, `PrevGlobalTransform` for interpolation |
| `WindowPlugin` | `flint.app.window_plugin` | `WindowStateResource`: Wayland, else X11, DRM/KMS on a text console (`FLINT_BACKEND=wayland/x11/kms` forces one); no window when headless |
| `InputPlugin` | `flint.input.plugin` | `InputStateResource`, `InputMap`, evdev gamepads |
| `PhysicsPlugin` | `flint.physics.plugin` | `PhysicsWorld`, body/collider/character components, trigger and contact events |
| `NavPlugin`, `AiPlugin` | `flint.nav.plugin`, `flint.ai.behavior_tree` | navmesh agents and ORCA crowds; behavior-tree ticking |
| `AnimPlugin` | `flint.anim.plugin` | skeletal animation, foot-plant IK, ragdolls, bone sockets |
| `AudioPlugin` | `flint.audio.plugin` | `AudioServer` (null output when headless), emitter/listener sync |
| `RenderPlugin` | `flint.render.plugin` | `ScalableRenderer`, `RenderWorld`, mesh/camera/light components |
| `NetPlugin` | `flint.net.plugin` | transport poll, fixed-step rollback/replication |
| `AssetPlugin` | `flint.assets.asset_plugin` | asset tick and handle resolution |

The scaffold and the harness draw with their own `SimdCanvas` systems instead of `RenderPlugin`;
`examples/minimal_spark/main.mojo` shows `RenderPlugin`.

## Scenes

Scenes are JSON `.flscn` (deterministic key and entity order, diff-friendly) or LZ4 binary
`.flscnb`. A `ComponentRegistry` maps component names to codecs; each plugin family has a
`register_*_codecs` function.

```mojo
# examples/test_harness_game/src/main.mojo
var reg = ComponentRegistry.with_builtins()      # transforms, hierarchy, Studio metadata
register_render_codecs(reg)
register_physics_codecs(reg)
register_audio_codecs(reg)
reg.ensure_registered_in_world(app.world)

var scene = Scene(cfg.entry_scene)
try:
    if cfg.entry_scene.endswith(".flscnb") or not _looks_like_json_scene(scene_bytes):
        _ = scene.load_flscnb_bytes(app.world, reg, scene_bytes, clear_existing=True)
    else:
        _ = scene.load_flscn_bytes(app.world, reg, scene_bytes, clear_existing=True)
except err:
    return (False, app^, "failed to load entry_scene: " + String(err))
```

An entity in `.flscn` lists components by registered name (from the scaffold's `assets/main.flscn`):

```json
{"slot": 1, "parent": -1, "components": {
  "LocalTransform": {"translation": [0.0, 1.2, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [1.1, 1.1, 1.1]},
  "RigidBodyComponent": {"motion_mode": 2, "mass": 1.0, "can_sleep": true},
  "ColliderComponent": {"shape": 2, "half_extents": [0.55, 0.55, 0.55], "is_trigger": false},
  "MeshRendererComponent": {"mesh_id": 0, "material_id": 0, "cast_shadows": true, "visible": true}
}}
```

Unknown component names are kept in `UnknownComponentsComponent` and written back on save, so a
scene survives a build that lacks a plugin. Saving: `scene.save_flscn_file(path, world, reg)`,
`scene.save_flscnb_file(...)`. `scene.enter_pie(world)` / `scene.exit_pie(world)` snapshot and
restore the whole world (Studio's Play-In-Editor).

To make your own component serializable, implement `SerializableComponent`:

```mojo
# tests/test_flint_v5_core_ecs_scene.mojo (ArmorStats)
@fieldwise_init
struct ArmorStats(SerializableComponent, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.ArmorStats")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "ArmorStats"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("ARMR")
    comptime SCHEMA_VERSION: Int = 1
    var armor: Float32
    var tier: Int32

    def write_bin(self, mut out: BlobWriter):
        out.f32(self.armor)
        out.u32(UInt32(self.tier))

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        return Self(reader.f32(), Int32(reader.u32()))

    def write_json(self) -> String:
        return "{\"armor\": " + String(Float64(self.armor)) + ", \"tier\": " + String(self.tier) + "}"

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        return Self(Float32(doc.number(doc.object_get(node, "armor"), 0.0)),
                    Int32(doc.integer(doc.object_get(node, "tier"), 0)))

var reg = ComponentRegistry.with_builtins()
reg.register[ArmorStats]()
```

Bump `SCHEMA_VERSION` when the layout changes and handle old versions in `read_bin`; JSON-level
upgrades go through `SceneMigrationRegistry` (`load_flscn_bytes_with_migrations`).

## Prefabs

A `Prefab` is a tree of nodes with component values; prefabs nest through `NestedPrefabRef`, and
`PrefabOverride` replaces a component on a named node per instance.

```mojo
# tests/test_flint_v5_core_ecs_scene.mojo (test_d_scene_prefabs_and_pie)
var weapon_prefab = Prefab("PlasmaWeapon")
var w_mount = weapon_prefab.add_node("WeaponMount", -1)
weapon_prefab.set_component[ArmorStats](w_mount, ArmorStats(25.0, 1))
var lib = List[Prefab]()
lib.append(weapon_prefab^)

var mech_prefab = Prefab("ValkyrieMech")
var m_hull = mech_prefab.add_node("Hull", -1)
mech_prefab.set_component[ArmorStats](m_hull, ArmorStats(250.0, 2))
var top_overrides = List[PrefabOverride]()
top_overrides.append(PrefabOverride.for_name[ArmorStats]("Hull", ArmorStats(900.0, 5)))
var spawned = mech_prefab.instantiate(w, reg, lib, top_overrides)   # raises; returns List[Entity]
propagate_transforms_parallel(w, pool)
```

On disk prefabs are JSON `.flprefab` files: `save_flprefab_file(path, prefab)`,
`load_flprefab_file(path)` (see `examples/test_harness_game/assets/prefabs/`).

## Assets

`AssetServer` serves files from mounted `.flpk` packs and a loose directory, with generational
`Handle[T]`s, reference counting, LRU eviction and (when `enable_watcher=True`) inotify hot reload
of loose files.

```mojo
# tools/flint.sh (scaffolded src/main.mojo)
var assets = AssetServer(enable_watcher=not effective_headless)
if file_size(cfg.pack_output) > 0:
    _ = assets.mount_flpk(cfg.pack_output)
if is_directory("assets", follow_symlinks=True):
    assets.loose_root = "assets"
    assets.index_loose_directory()

var rh = assets.load_raw(cfg.entry_scene)
var rptr = assets.raws.get_ptr(rh)
if rptr and len(rptr.value()[unsafe_offset=0].bytes) > 0:
    scene_bytes = rptr.value()[unsafe_offset=0].bytes.copy()
```

Typed loaders: `load_mesh`, `load_texture`, `load_hdr`, `load_font`, `load_audio`, `load_raw`, plus
async variants (`load_mesh_async`, `load_texture_async`, then `poll_async_loads`). Call
`assets.tick()` once per frame (or add `AssetPlugin`) and `assets.close()` at shutdown. The harness
inserts the server as a resource (`app.insert_resource[AssetServer](assets^)`) so systems can
reach it. Cook loose files into a pack with `bash tools/flint.sh pack`.

## Input

`InputMap` turns keys, mouse and gamepad input into named actions inside prioritized contexts.

```mojo
# tools/flint.sh (scaffolded src/main.mojo, _setup_input)
var imap_opt = app.resources.get_ptr_mut[InputMap]()
ref imap = imap_opt.value()[unsafe_offset=0]
_ = imap.register_action("orbit_left", ACTION_BOOL)
_ = imap.register_action("quit", ACTION_BOOL)
var ctx = InputContext("gameplay", priority=10)
ctx.bind_key("orbit_left", 65)    # A
ctx.bind_key("orbit_left", 263)   # Left
ctx.bind_key("quit", 256)         # Esc
imap.add_context(ctx^)
imap.push_context("gameplay")

# in a system:
if imap.just_pressed("quit"):
    win_opt.value()[unsafe_offset=0].request_close()
if imap.is_down("orbit_left"):
    state.cam_yaw -= Float32(1.5) * dt
```

Other bindings: `bind_mouse_button`, `bind_mouse_look`, `bind_pad_button`, `bind_pad_axis_1d`,
`bind_pad_stick_2d` (`ACTION_AXIS1D` / `ACTION_AXIS2D`, read with `axis_1d` / `axis_2d`).
Gameplay helpers: `consume_buffered` (input buffering), `set_grounded` / `coyote_valid`. Runtime
rebinding: `rebind_key`, `save_json` / `load_json`. `pack_deterministic_tick` /
`unpack_deterministic_tick` serialize one tick of actions for replays and rollback.

## Physics

Add `PhysicsPlugin(gravity=9.81)` and give entities `RigidBodyComponent` + `ColliderComponent`
(in code or in the scene, as above). Motion modes are `BODY_MOTION_STATIC` (0), `KINEMATIC` (1)
and `DYNAMIC` (2); shapes are sphere, capsule, box, convex hull, BVH8 mesh, heightfield and
compound (`BODY_SHAPE_*` in `flint.physics.world`). `flint.physics.plugin.PhysicsPlugin` steps the
world in `STAGE_FIXED_UPDATE` and syncs transforms in `STAGE_PRE_UPDATE`.
`CharacterControllerComponent` gives kinematic character movement; `PhysicsWorld` is a resource
for raycasts and direct access.

Triggers arrive through `TriggerEventsResource` (see [ECS_GUIDE.md](ECS_GUIDE.md#events));
`OnEnter(ev, entity, world, resources)` and `entity_from_body(body_id, resources)` map them back
to entities (`examples/test_harness_game/src/gameplay_core.mojo`).

## Audio

```mojo
# examples/minimal_spark/main.mojo (_setup_spark_audio, trimmed)
ref audio_server = app.resources.get_ptr_mut[AudioServer]().value()[unsafe_offset=0]
var aid = audio_server.register_pcm16_asset(pcm, channels=1, sample_rate=48000)
var em_e = app.world.spawn()
_ = app.world.insert[LocalTransform](em_e, LocalTransform())
_ = app.world.insert[GlobalTransform](em_e, GlobalTransform())
_ = app.world.insert[AudioEmitterComponent](em_e, AudioEmitterComponent(
    asset_id=aid, volume=Float32(0.55), bus_id=BUS_AMBIENCE,
    looping=True, spatial=False, play_on_spawn=True,
))
```

Put an `AudioListenerComponent(active=True)` on the camera entity for spatial sound. One-shots:
`audio_server.play_2d(asset_id, volume=..., bus_id=BUS_SFX)` or
`play_3d(asset_id, pos, vel=...)`, both returning a `SoundHandle`. Assets can also come from
`register_wav_asset` and `register_qoa_asset`. The output backend is PulseAudio/PipeWire with an
ALSA fallback (Linux only); headless apps get a null output.

## Rendering

`RenderPlugin(tier=..., width=..., height=...)` inserts a `ScalableRenderer`:

| Tier | Constant | What runs |
| :--- | :--- | :--- |
| 0 | `RENDER_TIER_0_HEADLESS_SIMD` | CPU `SimdCanvas`; no GPU needed |
| 1 | `RENDER_TIER_1_FORWARD_CLUSTERED` | the Vulkan renderer with ray tracing, SVGF, DoF and motion blur turned off |
| 2 | `RENDER_TIER_2_AAA_VISBUFFER_RT` (default) | Vulkan meshlet visibility buffer, virtual shadow maps, TSR; RT shadows/reflections/GI when the device has `ray_query` |

If Vulkan cannot be initialized, tiers 1 and 2 fall back to tier 0 (`ScalableRenderer.tier_name()`
reports the active tier). Register materials and meshes on the renderer, then reference them from
components:

```mojo
# examples/minimal_spark/main.mojo (_populate_spark_scene, trimmed)
ref renderer = app.resources.get_ptr_mut[ScalableRenderer]().value()[unsafe_offset=0]
var mat = renderer.register_material(MaterialAsset(base_color=_srgb_hex_to_linear_vec3(0xFFB040), roughness=0.20, metallic=0.10))
var mesh = renderer.register_mesh(_build_spark_diamond_mesh(), mat)
var se = app.world.spawn()
_ = app.world.insert[LocalTransform](se, LocalTransform(Vec3(0.0, 0.65, 0.0)))
_ = app.world.insert[GlobalTransform](se, GlobalTransform())
_ = app.world.insert[MeshRendererComponent](se, MeshRendererComponent(mesh, mat))
```

`CameraComponent`, `PointLightComponent`, `EnvironmentComponent` and `MaterialOverrideComponent`
complete the set; `flint.render.plugin.register_render_codecs` makes them loadable from scenes.
For a fully custom 2D/3D software path, draw into a `SimdCanvas` yourself and present it with
`win_res.windows[0].present(canvas.pixels, canvas.width, canvas.height)` (the scaffold does this).

## Hot-reloading gameplay (`.so`)

Gameplay code can live in a shared library that is swapped while the game runs. The host keeps
`World` and `Resources`; the plugin talks to them only through the C-ABI table
`FlintPluginApiV2` (`src/flint/app/hot_reload.mojo`), so components shared across the boundary
must be plain-old-data (`register_pod_component` / `register_schema`).

The plugin exports four functions:

```mojo
# examples/test_harness_game/src/gameplay.mojo (trimmed)
from flint.app.hot_reload import FLINT_PLUGIN_API_VERSION, FlintPluginApiV2

@export("flint_plugin_api_version")
def flint_plugin_api_version() abi("C") -> UInt32:
    return FLINT_PLUGIN_API_VERSION          # 2; a mismatch is rejected

@export("flint_plugin_on_load")
def flint_plugin_on_load(api_ptr: Int, world_ptr: Int, res_ptr: Int, is_reload: Int32) abi("C") -> Int32:
    ref api = Pointer[FlintPluginApiV2, ImmutAnyOrigin](unsafe_from_address=api_ptr)[unsafe_offset=0]
    if not api.is_valid():
        return Int32(-2)
    var tid = api.register_schema(world_ptr, res_ptr, build_harness_stats_schema())
    return Int32(0) if tid >= 0 else Int32(-3)

@export("flint_plugin_on_step")
def flint_plugin_on_step(api_ptr: Int, world_ptr: Int, res_ptr: Int, dt: Float32) abi("C") -> Int32:
    ref api = Pointer[FlintPluginApiV2, ImmutAnyOrigin](unsafe_from_address=api_ptr)[unsafe_offset=0]
    var tid = api.register_pod_component(world_ptr, HarnessStats.COMPONENT_ID, size_of[HarnessStats](), 4, 0)
    _ = api.for_each_pod1(world_ptr, tid, _on_harness_stats_row, dt)   # per-row C callback
    return Int32(1)

@export("flint_plugin_on_unload")
def flint_plugin_on_unload(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi("C") -> None:
    pass
```

Build it with `bash ~/flint/tools/flint.sh gameplay-so` (uses `[project] gameplay_entry` /
`gameplay_output`). The host side:

```mojo
# examples/test_harness_game/src/main.mojo, tests/test_flint_v5_core_ecs_scene.mojo (test_f)
var host = HotReloadHost()                 # optional staging dir; allow_v1=True accepts ABI v1
if host.load_or_reload("build/gameplay.so", app.world, app.resources):
    _ = host.step(app.world, app.resources, dt)     # calls flint_plugin_on_step
else:
    print(host.last_error)
_ = host.unload(app.world, app.resources, is_reload=False)
```

`load_or_reload` copies the `.so` to a staging path before `dlopen`, so rebuilding in place is
safe. A failed reload (corrupt file, wrong ABI version, missing `flint_plugin_on_step`) leaves the
previous plugin loaded; entities and resource contents survive a successful swap (`test_f`
checks a swap completes in under 300 ms). Keep state in POD components and resources
(`upsert_pod_resource` / `get_resource_ptr`), never in globals inside the `.so`.

## Game flow helpers

`flint.app.flow` has `GameStateMachine` / `GameFlow` (allowed transitions), `GameTimer`, `Tween`
and `SaveSlotStore`:

```mojo
# examples/test_harness_game/src/main.mojo
var flow = GameFlow(initial_state="Menu")
_ = flow.fsm.allow_transition("Menu", "Play")
_ = flow.fsm.allow_transition("Play", "Pause")
_ = flow.fsm.transition_to("Play")
app.insert_resource[GameFlow](flow^)
```
