# ECS Guide

How to model game state with `flint.ecs` and run it with `flint.app.EngineApp`. Every snippet is
copied or trimmed from a file in this repository (cited above each block) and uses Mojo 1.2 syntax.

There are two ECS layers in the tree:

- **`flint.ecs` (ECS v2, `src/flint/ecs/world.mojo`)**: the `World` used by `EngineApp`, scenes,
  plugins and Studio. Use this one.
- **`flint.core.ecs`**: the older `SimdColumn` struct-of-arrays tables plus `run_parallel_ctx`, the
  helper that every parallel dispatch in the engine (including ECS v2) goes through.

## World and entities

`Entity` is a 64-bit `(index, generation)` handle; a despawned slot gets a new generation, so stale
handles fail `world.is_alive(e)`. Adding or removing components moves the entity to another
archetype but never changes its handle.

```mojo
# tests/test_flint_v5_core_ecs_scene.mojo (test_b_ecs_v2_hybrid_events_determinism)
var w = World()
var hero = w.spawn_empty()
_ = w.insert[CompPos](hero, CompPos(1.0, 2.0, 3.0))   # registers CompPos on first use
_ = w.insert[CompVel](hero, CompVel(4.0, 5.0, 6.0))
assert_equal(w.get[CompPos](hero).x, Float32(1.0))
_ = w.remove[CompVel](hero)
_ = w.despawn(hero)
```

Accessors: `has[C](e)`, `get[C](e)` (copy), `get_ptr[C](e)` / `get_ptr_mut[C](e)` (returns
`Optional[Pointer]`; dereference with `[unsafe_offset=0]`), `entity_count()`, `state_checksum()`.

## Components

A component is a struct implementing `Component` (`Copyable, Movable, Deinitable`) with two
compile-time constants: a stable 64-bit id and a storage mode.

```mojo
# tests/test_flint_v5_core_ecs_scene.mojo
from flint.ecs import Component, STORAGE_SPARSE, STORAGE_TABLE, comp_hash64

@fieldwise_init
struct CompPos(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.CompPos")
    comptime STORAGE_MODE: Int = STORAGE_TABLE   # archetype chunk column (default choice)
    var x: Float32
    var y: Float32
    var z: Float32

@fieldwise_init
struct TagBurningSparse(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("test.TagBurningSparse")
    comptime STORAGE_MODE: Int = STORAGE_SPARSE  # paged sparse set; no archetype move
    var dps: Float32
```

- `STORAGE_TABLE` components live in 64-byte aligned, 64-row `ArchetypeChunk` columns and are what
  the parallel iterators are fast on. Use it for anything iterated every frame.
- `STORAGE_SPARSE` components attach and detach without moving the entity between archetypes. Use
  it for short-lived tags and markers.
- The id is the hash of a name string. Keep the name unique and stable: it is the key used by
  scene files, snapshots and the hot-reload ABI.
- Components may own heap data (`String`, `List`); the World runs their destructors. To persist
  them in scenes, implement `SerializableComponent` (see [GAMEPLAY_GUIDE.md](GAMEPLAY_GUIDE.md#scenes)).

`app.register_component[C]()` (or `world.register_component[C]()`) registers up front, which you
need before building a `ComponentBitSet` for a schedule or query; `insert` also registers lazily.

## Queries

Queries take a `ComponentBitSet`; `world.bit_for[C]()` gives the bit for one type.

```mojo
# examples/minimal_spark/main.mojo (_sys_spark_update)
var sat_mask = world.bit_for[CrystalSatComponent]().union(world.bit_for[LocalTransform]())
var sat_entities = world.query_entities(sat_mask)
for i in range(len(sat_entities)):
    var sc_opt = world.get_ptr[CrystalSatComponent](sat_entities[i])
```

`query_entities(with_mask, without_mask)` returns a `List[Entity]`. For throughput, iterate the
chunks in parallel with `par_for_each1/2/3`. The kernel receives the worker lane, a deterministic
row order, the entity, one pointer per component and your context address:

```mojo
# tests/test_flint_v5_core_ecs_scene.mojo (_integrate_local_vel_kernel / _fixed_integrator_sys)
def _integrate_local_vel_kernel(
    _lane: Int,
    _order: Int,
    _e: Entity,
    lt: Pointer[LocalTransform, MutAnyOrigin],
    vel: Pointer[CompVel, MutAnyOrigin],
    ft_addr: Int,
):
    var dt = Pointer[FixedTime, ImmutAnyOrigin](unsafe_from_address=ft_addr)[unsafe_offset=0].dt
    lt[unsafe_offset=0].translation.x += vel[unsafe_offset=0].vx * dt

# inside a system:
world_ptr[unsafe_offset=0].par_for_each2[LocalTransform, CompVel](
    pool_ptr[unsafe_offset=0],
    _integrate_local_vel_kernel,
    Int(Pointer(to=ft_opt.value()[unsafe_offset=0])),
)
```

Optional arguments: `without_mask=`, `read_only=True` (does not mark rows changed),
`changed_since=tick` (skips chunks and 8-row groups untouched since `tick`).

**Context lifetime.** The context is passed as a raw address, and Mojo 1.2 destroys locals after
their last use, so keep the context alive past the call. The test does it with a one-element list:

```mojo
# tests/test_flint_v5_core_ecs_scene.mojo (_run_deterministic_ecs_sim)
var ctx_box = List[_SimStepCtx](capacity=1)
ctx_box.append(_SimStepCtx(Int(Pointer(to=cmds)), step))
w.par_for_each2[CompPos, CompVel](pool, _det_sim_kernel, Int(ctx_box.unsafe_ptr()))
var keep_ctx = len(ctx_box)   # last use is after the parallel call
_ = keep_ctx
```

For your own `JobPool` work use `flint.core.ecs.run_parallel_ctx`, which anchors the context for
you (AGENTS.md section 2).

## Change detection

Every table write through `get_ptr_mut` or a non-`read_only` iterator stamps the chunk with
`world.current_tick`. Ask for what changed since a tick you saved:

```mojo
# tests/test_flint_v5_core_ecs_scene.mojo (Wave 2 regressions)
var tick_before = w.current_tick
_ = w.advance_tick()
w.get_ptr_mut[CompPos](ents[0]).value()[unsafe_offset=0].y = Float32(7.0)
var changed = w.query_changed_entities[CompPos](tick_before)
assert_true(len(changed) > 0 and len(changed) <= 8)
```

Granularity is one 8-row SIMD group for table components (neighbours of a changed row are
reported too) and one entity for sparse components. `EngineApp` advances the tick once per fixed
step and once at the end of each frame.

## Commands (deferred structural changes)

Parallel kernels must not spawn, despawn, insert or remove directly. Record into
`ParallelCommands` with the kernel's `lane`; `world.flush_commands(cmds)` applies them in a
deterministic order (sorted by `order_key`, then lane and sequence), so 1 and 7 worker threads
produce the same `state_checksum()`.

```mojo
# tests/test_flint_v5_core_ecs_scene.mojo (_det_sim_kernel)
ref cmds = Pointer[ParallelCommands, MutAnyOrigin](unsafe_from_address=ctx.cmds_addr)[unsafe_offset=0]
if (order + ctx.step_idx) % 29 == 0:
    var spawned = cmds.spawn_empty(lane, order_key=order)   # provisional entity, remapped on flush
    cmds.insert[CompPos](lane, spawned, CompPos(0.0, 0.0, 0.0), order_key=order)
elif (order + ctx.step_idx) % 47 == 0:
    cmds.despawn(lane, entity, order_key=order)
```

Inside `EngineApp` the commands buffer is the fourth system argument and is flushed after every
stage (and after every fixed step).

## Resources

Resources are typed singletons stored in `Resources` (`app.resources`), keyed by `RESOURCE_ID`.

```mojo
# tools/flint.sh (scaffolded src/main.mojo, GameStateResource)
struct GameStateResource(Resource, Movable):
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.game.GameStateResource")
    var headless: Bool
    var canvas: SimdCanvas
    ...

app.insert_resource[GameStateResource](GameStateResource(effective_headless, 1280, 720))
var st_opt = resources.get_ptr_mut[GameStateResource]()   # Optional[Pointer]
if st_opt:
    ref state = st_opt.value()[unsafe_offset=0]
```

`EngineApp` inserts `FrameTime` (`delta_seconds`, `elapsed_seconds`, `frame_index`) and
`FixedTime` (`dt`, `tick_count`, `overstep_alpha`) for you. Plugins insert their own
(`WindowStateResource`, `InputMap`, `AudioServer`, `ScalableRenderer`, ...).

## Systems and stages

A system is a plain function with this signature (`flint.app.SystemFn`):

```mojo
# tools/flint.sh (scaffolded src/main.mojo, _sys_game_update)
def _sys_game_update(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,                      # the `user_ctx` given to add_system
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    ...

_ = app.add_system(STAGE_UPDATE, "game.update", _sys_game_update, exclusive=True)
_ = app.add_system(STAGE_LAST, "game.present", _sys_game_present, exclusive=True)
```

`add_system(stage, name, fn, reads=, writes=, exclusive=, user_ctx=, after_system_id=)` returns a
system id. Within a stage the schedule builds a dependency graph and runs independent systems in
parallel on the `JobPool`:

- Two systems conflict if either is `exclusive`, or their `writes` overlap each other's `reads` or
  `writes`. Conflicting systems run in registration order.
- **Resources are not tracked.** A system that is not `exclusive` and declares no `reads`/`writes`
  conflicts with nothing and may run concurrently with every other system in its stage. Mark
  systems that touch shared resources `exclusive=True` (every system in the examples does), or
  declare component masks:

```mojo
# tests/test_flint_v5_core_ecs_scene.mojo (_run_app_at_display_hz)
var b_lt = app.world.bit_for[LocalTransform]()
var b_vel = app.world.bit_for[CompVel]()
_ = app.add_system(STAGE_FIXED_UPDATE, "fixed_integrator", _fixed_integrator_sys, reads=b_vel, writes=b_lt)
```

- `after_system_id=` (or `app.schedule.add_dependency`) forces an order across non-conflicting
  systems. More than 16 dependents are split through relay nodes automatically.

### Frame order (`EngineApp.step_one_frame`)

| Stage | When | Engine systems registered there |
| :--- | :--- | :--- |
| `STAGE_STARTUP` | once, in `startup()` | window open, audio start, asset resolve |
| `STAGE_FIRST` | every frame | `flint.app.window_poll`, then `flint.input.update`; `flint.net.poll` |
| `STAGE_PRE_UPDATE` | every frame | `flint.physics.pre_sync`, asset tick, ragdoll cleanup |
| `STAGE_FIXED_UPDATE` | 0..`max_fixed_steps_per_frame` times at `fixed_hz` | `flint.physics.fixed_step`, `flint.nav.crowd_step`, `flint.net.fixed_step`, previous-transform snapshot |
| `STAGE_UPDATE` | every frame | `flint.ai.behavior_tree_tick`, `flint.anim.update` -> `foot_plant` -> `ragdoll` -> `bone_sockets` |
| `STAGE_POST_UPDATE` | every frame | `flint.scene.propagate_transforms`, then `flint.net.replicate`; nav debug |
| `STAGE_LAST` | every frame | `flint.render.extract_and_draw`, `flint.audio.sync` |
| `STAGE_SHUTDOWN` | once, in `shutdown()` | window, audio, render, net teardown |

After every stage the command buffer is flushed. After `STAGE_LAST` the event-swap hooks run and
the world tick advances. Each fixed step also advances the tick. `FrameTime.delta_seconds` is
clamped to `AppConfig.max_frame_dt` (0.1 s by default).

Drive the loop yourself with `app.startup()`, `app.step_one_frame(dt)` and `app.shutdown()` (as
the template does), or use `app.run(max_frames)` (fixed `default_frame_dt`, headless tests) or
`app.run_realtime(max_frames, max_seconds)`.

## Events

`Events[T]` is a double-buffered channel: an event stays readable for two frames, and each reader
keeps its own cursor.

```mojo
# tests/test_flint_v5_core_ecs_scene.mojo (test_b)
var dmg_events = Events[DamageEvent]()
var cursor = 0
_ = dmg_events.send(DamageEvent(hero, 12.0))
var fresh = dmg_events.read_since(cursor)   # returns new events and advances cursor
dmg_events.swap_buffers()                   # frame boundary
```

Put the channel in a resource and register a swap hook with `app.add_event_swap_hook(fn)` so it
swaps after `STAGE_LAST`; this is how `PhysicsPlugin` publishes `TriggerEventsResource` and
`PhysicsEventsResource`. Reading trigger events from a system:

```mojo
# examples/test_harness_game/src/gameplay_core.mojo (_sys_harness_update)
var trig_opt = resources.get_ptr[TriggerEventsResource]()
if trig_opt:
    var tevs = trig_opt.value()[unsafe_offset=0].events.read_since(state.trigger_cursor)
    for ti in range(len(tevs)):
        ref ev = tevs[ti]
        var actor = entity_from_body(ev.body_id, resources)
```

## Plugins

A plugin is any struct with `build(self, mut app: EngineApp)` that registers components, inserts
resources and adds systems. Engine plugins check for their own system names first, so adding one
twice is harmless.

```mojo
# src/flint/app/engine_app.mojo
trait EnginePlugin:
    def build(self, mut app: EngineApp):
        ...
```

## Snapshots and determinism

- `world.state_checksum()` hashes all component data. The ECS tests compare it across 0, 1 and 7
  worker threads, and across 30 Hz and 144 Hz display rates with a 60 Hz fixed step.
- `world.snapshot_blob()` / `world.restore_blob(bytes)` serialize the whole world (used for
  Play-In-Editor, rollback and save games).
- Builds use `--fp-mode contract=off`; use `flint.core.detmath` (`det_sin`, `det_cos`, ...) in
  simulation code that must match across machines.
