"""ECS Camera Plugin: Virtual Camera Components + Camera Brain Resource (`flint.camera.plugin`, v5).

Games add `CameraPlugin()` to an `EngineApp`, spawn entities with `VirtualCameraComponent`
(priority, mode, follow / lock-on target entities, blend time, orbit params) and read the blended
pose from `CameraBrainResource.brain` (`pos`, `rot`, `fov_deg`, `world_matrix()`). Shake is
event-driven: `CameraBrainResource.add_trauma(x)` or `shake_event(pos, strength, radius)`.

System `flint.camera.brain` (`STAGE_POST_UPDATE`, after transforms) syncs component settings and
target entity `GlobalTransform`s (falling back to `LocalTransform`) into the brain, uses the
`PhysicsWorld` resource for spring-arm occluders when present, and updates the brain with
`FrameTime.delta_seconds`.
"""
from std.collections import Dict, List
from std.math import isfinite
from flint.app.engine_app import (
    EngineApp,
    EnginePlugin,
    FrameTime,
    Resource,
    Resources,
    STAGE_POST_UPDATE,
)
from flint.core.jobs import JobPool
from flint.core.math import Vec3, Quat
from flint.core.detmath import det_atan2
from flint.ecs.world import Component, Entity, ParallelCommands, STORAGE_TABLE, World, comp_hash64
from flint.physics.world import PhysicsWorld
from flint.scene.transform_hierarchy import GlobalTransform, LocalTransform
from flint.camera.vcam import (
    VirtualCamera,
    CameraBrain,
    VCAM_FOLLOW,
    VCAM_FIXED,
    VCAM_LOCK_ON,
)


struct VirtualCameraComponent(Component, Copyable, ImplicitlyCopyable, Movable):
    """Virtual camera shot attached to an entity (FIXED vcams use the entity's own transform)."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.camera.VirtualCameraComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var priority: Int
    var mode: UInt8
    var enabled: Bool
    var follow_entity: UInt64  # Entity.to_bits(), 0 = none
    var lock_entity: UInt64  # lock-on target, 0 = none
    var blend_in_time: Float32
    var fov_deg: Float32
    var distance: Float32
    var pitch_deg: Float32
    var yaw_deg: Float32
    var look_offset: Vec3
    var follow_yaw_from_target: Bool  # VCAM_FOLLOW: chase the follow entity's facing (rotation yaw)
    var vcam_id: Int  # brain id (runtime, -1 until registered)

    def __init__(out self, mode: UInt8 = VCAM_FOLLOW, priority: Int = 0, follow_entity: UInt64 = 0):
        self.priority = priority
        self.mode = mode
        self.enabled = True
        self.follow_entity = follow_entity
        self.lock_entity = 0
        self.blend_in_time = 0.5
        self.fov_deg = 65.0
        self.distance = 4.5
        self.pitch_deg = 15.0
        self.yaw_deg = 0.0
        self.look_offset = Vec3(0.0, 1.5, 0.0)
        self.follow_yaw_from_target = True
        self.vcam_id = -1


struct CameraBrainResource(Resource, Movable):
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.camera.CameraBrainResource")
    var brain: CameraBrain
    var entity_to_vcam: Dict[UInt64, Int]
    var fallback_world: PhysicsWorld

    def __init__(out self):
        self.brain = CameraBrain()
        self.entity_to_vcam = Dict[UInt64, Int]()
        self.fallback_world = PhysicsWorld()

    def add_trauma(mut self, amount: Float32):
        self.brain.add_trauma(amount)

    def shake_event(mut self, pos: Vec3, strength: Float32, radius: Float32):
        self.brain.shake_event(pos, strength, radius)


def _entity_pose(world: World, bits: UInt64) -> Tuple[Bool, Vec3, Quat]:
    if bits == 0:
        return (False, Vec3(), Quat())
    var e = Entity.from_bits(bits)
    if not world.is_alive(e):
        return (False, Vec3(), Quat())
    var gt = world.get_ptr[GlobalTransform](e)
    if gt:
        return (True, gt.value()[unsafe_offset=0].translation, gt.value()[unsafe_offset=0].rotation)
    var lt = world.get_ptr[LocalTransform](e)
    if lt:
        return (True, lt.value()[unsafe_offset=0].translation, lt.value()[unsafe_offset=0].rotation)
    return (False, Vec3(), Quat())


def _sys_camera_brain(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var st_opt = resources.get_ptr_mut[CameraBrainResource]()
    if not st_opt:
        return
    ref st = st_opt.value()[unsafe_offset=0]

    var dt = Float32(1.0 / 60.0)
    var ft_opt = resources.get_ptr[FrameTime]()
    if ft_opt:
        var d = ft_opt.value()[unsafe_offset=0].delta_seconds
        if isfinite(d) and d >= Float32(0.0):
            dt = d

    var mask = world.bit_for[VirtualCameraComponent]()
    var ents = world.query_entities(mask)
    var seen = List[UInt64]()
    for i in range(len(ents)):
        var e = ents[i]
        var vc_opt = world.get_ptr_mut[VirtualCameraComponent](e)
        if not vc_opt:
            continue
        ref vc = vc_opt.value()[unsafe_offset=0]
        var bits = e.to_bits()
        seen.append(bits)
        var found = st.entity_to_vcam.get(bits)
        var vid: Int
        if found:
            vid = found.value()
        else:
            var nv = VirtualCamera(vc.mode, vc.priority, vc.blend_in_time)
            nv.rig.yaw_deg = vc.yaw_deg
            vid = st.brain.add_vcam(nv^)
            st.entity_to_vcam[bits] = vid
        vc.vcam_id = vid
        var bi = st.brain.index_of(vid)
        if bi < 0:
            continue
        st.brain.set_priority(vid, vc.priority)
        st.brain.set_enabled(vid, vc.enabled)
        ref cam = st.brain.vcams[bi]
        cam.mode = vc.mode
        cam.blend_in_time = vc.blend_in_time
        cam.fov_deg = vc.fov_deg
        cam.rig.spring_arm.target_length = vc.distance
        cam.rig.pitch_deg = vc.pitch_deg
        cam.rig.look_offset = vc.look_offset
        cam.lock_distance = vc.distance
        var own = _entity_pose(world, bits)
        if vc.mode == VCAM_FIXED and own[0]:
            cam.fixed_pos = own[1]
            cam.fixed_rot = own[2]
        var fol = _entity_pose(world, vc.follow_entity)
        if fol[0]:
            cam.follow_target = fol[1]
            if vc.follow_yaw_from_target:
                var f = fol[2].rotate(Vec3(0.0, 0.0, 1.0))
                cam.follow_yaw_deg = det_atan2(f.x, f.z) * Float32(57.29577951)
        var lk = _entity_pose(world, vc.lock_entity)
        cam.has_lock_target = lk[0] and vc.mode == VCAM_LOCK_ON
        if lk[0]:
            cam.lock_target = lk[1]

    # Drop vcams whose entity lost the component / despawned
    var stale = List[UInt64]()
    for item in st.entity_to_vcam.items():
        var keep = False
        for k in range(len(seen)):
            if seen[k] == item.key:
                keep = True
                break
        if not keep:
            stale.append(item.key)
    for k in range(len(stale)):
        try:
            var vid = st.entity_to_vcam.pop(stale[k])
            st.brain.remove_vcam(vid)
        except:
            pass

    var pw_opt = resources.get_ptr[PhysicsWorld]()
    if pw_opt:
        st.brain.update(pw_opt.value()[unsafe_offset=0], dt)
    else:
        st.brain.update(st.fallback_world, dt)


def _has_camera_system(app: EngineApp, stage: Int, name: StaticString) -> Bool:
    ref sys = app.schedule.stages[stage].systems
    for i in range(len(sys)):
        if sys[i].name == name:
            return True
    return False


def register_camera_components(mut world: World):
    _ = world.register_component[VirtualCameraComponent]()


struct CameraPlugin(EnginePlugin, Copyable, ImplicitlyCopyable, Movable):
    """Registers `VirtualCameraComponent`, `CameraBrainResource`, and the `flint.camera.brain` system."""

    def __init__(out self):
        pass

    def build(self, mut app: EngineApp):
        register_camera_components(app.world)
        if not app.resources.has[CameraBrainResource]():
            app.insert_resource[CameraBrainResource](CameraBrainResource())
        var b_vc = app.world.bit_for[VirtualCameraComponent]()
        var b_gt = app.world.bit_for[GlobalTransform]()
        var b_lt = app.world.bit_for[LocalTransform]()
        if not _has_camera_system(app, STAGE_POST_UPDATE, "flint.camera.brain"):
            _ = app.add_system(
                STAGE_POST_UPDATE,
                "flint.camera.brain",
                _sys_camera_brain,
                reads=b_gt.union(b_lt),
                writes=b_vc,
                exclusive=True,
            )
