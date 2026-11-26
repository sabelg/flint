"""ECS Perception Plugin (`flint.ai.perception_plugin`, v5).

- `PerceiverComponent`: sight/hearing/memory settings for an entity (eye height, fov, range, team).
- `StimulusComponent`: makes an entity visible to perceivers (team, optional physics body id).
- `PerceptionResource`: owns the `PerceptionSystem`; games call `report_noise(...)` on it and read
  memories via `memories_of(entity)` / `system.take_events()`.
- System `flint.ai.perception` (`STAGE_UPDATE`): syncs entity poses (`GlobalTransform`, falling back
  to `LocalTransform`; forward = rotation * +Z) into the system and updates it with the
  `PhysicsWorld` resource for line of sight (no occlusion when no physics world is present).
"""
from std.collections import Dict, List
from std.math import isfinite, max
from flint.app.engine_app import (
    EngineApp,
    EnginePlugin,
    FrameTime,
    Resource,
    Resources,
    STAGE_UPDATE,
)
from flint.core.jobs import JobPool
from flint.core.math import Vec3, Quat
from flint.ecs.world import Component, Entity, ParallelCommands, STORAGE_TABLE, World, comp_hash64
from flint.physics.world import PhysicsWorld
from flint.scene.transform_hierarchy import GlobalTransform, LocalTransform
from flint.ai.perception import PerceptionSystem, Perceiver, PerceptionMemory


struct PerceiverComponent(Component, Copyable, ImplicitlyCopyable, Movable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.ai.PerceiverComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var team: Int
    var eye_height: Float32
    var fov_deg: Float32
    var sight_range: Float32
    var hearing_scale: Float32
    var memory_duration: Float32
    var body_id: Int
    var enabled: Bool
    var perceiver_index: Int  # runtime slot in `PerceptionResource.system` (-1 until registered)

    def __init__(out self, team: Int = 0, fov_deg: Float32 = 110.0, sight_range: Float32 = 25.0, memory_duration: Float32 = 5.0):
        self.team = team
        self.eye_height = 1.6
        self.fov_deg = fov_deg
        self.sight_range = sight_range
        self.hearing_scale = 1.0
        self.memory_duration = memory_duration
        self.body_id = -1
        self.enabled = True
        self.perceiver_index = -1


struct StimulusComponent(Component, Copyable, ImplicitlyCopyable, Movable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.ai.StimulusComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var team: Int
    var height: Float32  # aim point above the entity origin
    var body_id: Int
    var active: Bool

    def __init__(out self, team: Int = 0, height: Float32 = 1.0, body_id: Int = -1):
        self.team = team
        self.height = height
        self.body_id = body_id
        self.active = True


struct PerceptionResource(Resource, Movable):
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.ai.PerceptionResource")
    var system: PerceptionSystem
    var entity_to_perceiver: Dict[UInt64, Int]
    var fallback_world: PhysicsWorld

    def __init__(out self):
        self.system = PerceptionSystem()
        self.entity_to_perceiver = Dict[UInt64, Int]()
        self.fallback_world = PhysicsWorld()

    def report_noise(mut self, pos: Vec3, radius: Float32, source: UInt64 = 0, team: Int = -1, ttl: Float32 = 0.0):
        self.system.report_noise(pos, radius, source, team, ttl)

    def memories_of(self, entity: UInt64) -> List[PerceptionMemory]:
        var found = self.entity_to_perceiver.get(entity)
        if not found:
            return List[PerceptionMemory]()
        var idx = found.value()
        if idx < 0 or idx >= len(self.system.perceivers):
            return List[PerceptionMemory]()
        return self.system.perceivers[idx].memories.copy()


def _pose_of(world: World, e: Entity) -> Tuple[Bool, Vec3, Quat]:
    var gt = world.get_ptr[GlobalTransform](e)
    if gt:
        return (True, gt.value()[unsafe_offset=0].translation, gt.value()[unsafe_offset=0].rotation)
    var lt = world.get_ptr[LocalTransform](e)
    if lt:
        return (True, lt.value()[unsafe_offset=0].translation, lt.value()[unsafe_offset=0].rotation)
    return (False, Vec3(), Quat())


def _sys_ai_perception(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var st_opt = resources.get_ptr_mut[PerceptionResource]()
    if not st_opt:
        return
    ref st = st_opt.value()[unsafe_offset=0]
    var dt = Float32(1.0 / 60.0)
    var ft_opt = resources.get_ptr[FrameTime]()
    if ft_opt:
        var raw = ft_opt.value()[unsafe_offset=0].delta_seconds
        dt = max(Float32(0.0), raw) if isfinite(raw) else Float32(0.0)

    # Stimuli
    var s_ents = world.query_entities(world.bit_for[StimulusComponent]())
    for i in range(len(s_ents)):
        var e = s_ents[i]
        var sc_opt = world.get_ptr[StimulusComponent](e)
        if not sc_opt:
            continue
        var sc = sc_opt.value()[unsafe_offset=0]
        var pose = _pose_of(world, e)
        if not pose[0]:
            continue
        var si = st.system.register_stimulus(e.to_bits(), pose[1] + Vec3(0.0, sc.height, 0.0), sc.team, sc.body_id)
        st.system.set_stimulus_active(si, sc.active)

    # Perceivers
    var p_ents = world.query_entities(world.bit_for[PerceiverComponent]())
    for i in range(len(p_ents)):
        var e = p_ents[i]
        var pc_opt = world.get_ptr_mut[PerceiverComponent](e)
        if not pc_opt:
            continue
        ref pc = pc_opt.value()[unsafe_offset=0]
        var bits = e.to_bits()
        var found = st.entity_to_perceiver.get(bits)
        var idx: Int
        if found:
            idx = found.value()
        else:
            var p = Perceiver(bits, pc.team, pc.fov_deg, pc.sight_range, pc.memory_duration)
            idx = st.system.add_perceiver(p^)
            st.entity_to_perceiver[bits] = idx
        pc.perceiver_index = idx
        ref pr = st.system.perceivers[idx]
        pr.team = pc.team
        pr.fov_deg = pc.fov_deg
        pr.sight_range = pc.sight_range
        pr.hearing_scale = pc.hearing_scale
        pr.memory_duration = pc.memory_duration
        pr.body_id = pc.body_id
        pr.enabled = pc.enabled
        var pose = _pose_of(world, e)
        if pose[0]:
            st.system.set_perceiver_pose(idx, pose[1] + Vec3(0.0, pc.eye_height, 0.0), pose[2].rotate(Vec3(0.0, 0.0, 1.0)))

    var pw_opt = resources.get_ptr[PhysicsWorld]()
    if pw_opt:
        st.system.update(pw_opt.value()[unsafe_offset=0], dt)
    else:
        st.system.update(st.fallback_world, dt)


def _has_perception_system(app: EngineApp, stage: Int, name: StaticString) -> Bool:
    ref sys = app.schedule.stages[stage].systems
    for i in range(len(sys)):
        if sys[i].name == name:
            return True
    return False


def register_perception_components(mut world: World):
    _ = world.register_component[PerceiverComponent]()
    _ = world.register_component[StimulusComponent]()


struct PerceptionPlugin(EnginePlugin, Copyable, ImplicitlyCopyable, Movable):
    """Registers perception components, `PerceptionResource` and the `flint.ai.perception` system."""

    def __init__(out self):
        pass

    def build(self, mut app: EngineApp):
        register_perception_components(app.world)
        if not app.resources.has[PerceptionResource]():
            app.insert_resource[PerceptionResource](PerceptionResource())
        if _has_perception_system(app, STAGE_UPDATE, "flint.ai.perception"):
            return
        var b_p = app.world.bit_for[PerceiverComponent]()
        var b_s = app.world.bit_for[StimulusComponent]()
        var b_gt = app.world.bit_for[GlobalTransform]()
        var b_lt = app.world.bit_for[LocalTransform]()
        _ = app.add_system(
            STAGE_UPDATE,
            "flint.ai.perception",
            _sys_ai_perception,
            reads=b_s.union(b_gt).union(b_lt),
            writes=b_p,
            exclusive=True,
        )
