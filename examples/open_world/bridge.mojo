"""Playable bridge route shared by the interactive world and its frame benchmark."""
from std.math import atan2, isfinite
from bridge_captions import BridgeCaptions
from bridge_vfx import load_bridge_dust
from bridge_cells import advance_bridge_cells, setup_bridge_cells
from flint.audio.server import AudioServer
from flint.app.flow import SaveSlotStore
from flint.assets.json import parse_json
from flint.platform.os import player_save_directory_for
from flint.anim.flanim_ecs import RAGDOLL_SM_ANIMATED, RAGDOLL_SM_LIMP
from flint.anim.plugin import AnimatorComponent, AnimAssetCache, RagdollComponent, SkeletonComponent, register_anim_codecs
from flint.anim.procedural import Placement
from flint.physics.filter import CAT_ALL, CAT_STATIC, CollisionFilter
from flint.physics.world import PhysicsWorld
from flint.app import EngineApp, FrameTime, Resource, Resources
from flint.assets.server import AssetServer
from flint.core.jobs import JobPool
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.ecs import Entity, ParallelCommands, World, comp_hash64
from flint.input.input_map import InputMap
from flint.render.plugin import CameraComponent, register_render_codecs
from flint.scene import ComponentRegistry, LocalTransform, Scene


comptime DECK_HALF_WIDTH = Float32(2.85)
comptime RECOVERY_EDGE_CLEARANCE = Float32(0.8)
comptime ROUTE_START = Float32(-25.5)
comptime ROUTE_END = Float32(14)


def valid_bridge_position(p: Vec3) -> Bool:
    return (isfinite(p.x) and isfinite(p.y) and isfinite(p.z) and abs(p.x) <= DECK_HALF_WIDTH
            and p.y >= 1.6 and p.y <= 4 and p.z >= -26 and p.z <= 15)


struct BridgeRoute(Resource, Copyable, Movable):
    comptime RESOURCE_ID: UInt64 = comp_hash64("example.sunspire.BridgeRoute")
    var camera: Entity
    var actor: Entity
    var guide: Entity
    var interactions: Int
    var pivots: Int
    var pivot_active: Bool
    var checkpoint_ready: Bool
    var deck_body_id: Int
    var guide_near: Bool
    var falls: Int
    var has_actor: Bool
    var replay: Bool
    var position: Vec3
    var replay_direction: Float32
    var walking: Bool

    def __init__(out self, camera: Entity, replay: Bool):
        self.camera = camera
        self.actor = Entity()
        self.guide = Entity()
        self.interactions = 0
        self.pivots = 0
        self.pivot_active = False
        self.checkpoint_ready = True
        self.deck_body_id = -1
        self.guide_near = False
        self.falls = 0
        self.has_actor = False
        self.replay = replay
        self.position = Vec3(0, 2.49, -24)
        self.replay_direction = 1
        self.walking = False


    def recover_on_deck(self, world: PhysicsWorld, p: Vec3, foot_offset: Float32) -> Tuple[Bool, Vec3]:
        # Leave a standing performer's width between the recovered origin and the outside edge.
        var limit = DECK_HALF_WIDTH - RECOVERY_EDGE_CLEARANCE
        var x = max(-limit, min(limit, p.x)) if isfinite(p.x) else Float32(0)
        var z = max(ROUTE_START, min(ROUTE_END, p.z)) if isfinite(p.z) else ROUTE_START
        var support = world.raycast(Vec3(x, 4, z), Vec3(0, -1, 0), 4, CollisionFilter(CAT_ALL, CAT_STATIC))
        if not support.hit or support.body_id != self.deck_body_id or support.normal.y < 0.7:
            return (False, p)
        var grounded = Vec3(x, support.point.y - foot_offset, z)
        if not valid_bridge_position(grounded):
            return (False, p)
        if x == p.x and z == p.z and valid_bridge_position(p) and abs(p.y - grounded.y) <= 0.1:
            return (True, p)
        return (True, grounded)

    def save_checkpoint(self) -> Bool:
        if not self.checkpoint_ready or not valid_bridge_position(self.position):
            return False
        var payload = String("{\"x\":") + String(self.position.x) + ",\"y\":" + String(self.position.y)
        payload += ",\"z\":" + String(self.position.z) + ",\"interactions\":" + String(self.interactions) + "}"
        return SaveSlotStore(player_save_directory_for("sunspire")).save_versioned("bridge", payload, 1, "Sunspire bridge")

    def load_checkpoint(mut self) -> Bool:
        var slot = SaveSlotStore(player_save_directory_for("sunspire")).load_versioned("bridge")
        if not slot.found or slot.version > 1:
            return False
        var bytes = List[UInt8]()
        for b in slot.payload.as_bytes():
            bytes.append(b)
        var doc = parse_json(bytes)
        if not doc.valid:
            return False
        var z = Float32(doc.number(doc.object_get(doc.root, "z"), 10000))
        var x = Float32(doc.number(doc.object_get(doc.root, "x"), 0))
        var y = Float32(doc.number(doc.object_get(doc.root, "y"), 2.49))
        var p = Vec3(x, y, z)
        if not valid_bridge_position(p):
            return False
        self.position = p
        self.interactions = max(0, doc.integer(doc.object_get(doc.root, "interactions"), 0))
        return True


def load_bridge(mut app: EngineApp, camera: Entity, pack: String, baseline: Bool, replay: Bool) raises:
    var route = BridgeRoute(camera, replay)
    if not baseline:
        ref assets = app.resources.get_ptr_mut[AssetServer]().value()[]
        if pack and not assets.mount_flpk(pack):
            raise Error("Could not mount bridge content: " + pack)
        load_bridge_dust(app)
        var handle = assets.load_raw("bridge.flscn")
        var raw = assets.raws.get_ptr(handle)
        if not raw:
            raise Error("Missing bridge.flscn in bridge content")
        var registry = ComponentRegistry.with_builtins()
        register_anim_codecs(registry)
        register_render_codecs(registry)
        var scene = Scene()
        var spawned = scene.load_flscn_bytes(app.world, registry, raw.value()[].bytes, clear_existing=False)
        _ = assets.raws.release(handle)
        for entity in spawned:
            if app.world.has[AnimatorComponent](entity):
                if route.has_actor:
                    raise Error("Bridge scene has multiple player animators")
                route.actor = entity
                route.position = app.world.get[LocalTransform](entity).translation
                route.has_actor = True
                app.world.get_ptr_mut[AnimatorComponent](entity).value()[].animator.extract_root_yaw = True
        if not route.has_actor:
            raise Error("Bridge scene needs an animated performer")
        var physical = assets.load_raw("bridge_rig.flscnb")
        var bytes = assets.raws.get_ptr(physical)
        if not bytes:
            raise Error("Bridge performer is missing its physical rig")
        var rig_world = World()
        var rig_scene = Scene()
        var rig_entities = rig_scene.load_flscnb_bytes(rig_world, registry, bytes.value()[].bytes)
        _ = assets.raws.release(physical)
        if len(rig_entities) != 1 or not rig_world.has[RagdollComponent](rig_entities[0]):
            raise Error("Invalid bridge performer rig")
        _ = app.world.insert[RagdollComponent](route.actor, rig_world.get[RagdollComponent](rig_entities[0]))
        ref physics = app.resources.get_ptr_mut[PhysicsWorld]().value()[]
        var deck = physics.add_box(Vec3(0, 1.47, -5.5), Vec3(DECK_HALF_WIDTH, 0.2, 21.5), mass=0)
        route.deck_body_id = physics.bodies[deck].id
        setup_bridge_cells(app, registry^)
    app.insert_resource(route^)


def update_bridge(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool: Pointer[JobPool, MutAnyOrigin],
    _commands: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[]
    ref resources = res_ptr[]
    ref route = resources.get_ptr_mut[BridgeRoute]().value()[]
    ref clock = resources.get_ptr[FrameTime]().value()[]
    ref input = resources.get_ptr[InputMap]().value()[]
    var forward = Float32(1 if route.replay or input.is_down("move_forward") else 0)
    if route.replay:
        if route.position.z >= ROUTE_END:
            route.replay_direction = -1
        elif route.position.z <= ROUTE_START:
            route.replay_direction = 1
        forward = route.replay_direction
    if not route.replay and input.is_down("move_back"):
        forward -= 1
    var captions = resources.get_ptr_mut[BridgeCaptions]()
    if captions:
        captions.value()[].seconds -= clock.delta_seconds
    var old_z = route.position.z
    route.position.z = max(ROUTE_START, min(ROUTE_END, old_z + 1.3 * forward * clock.delta_seconds))
    route.walking = route.position.z != old_z
    var pivot_pressed = input.just_pressed("bridge_pivot") or (route.replay and clock.frame_index % 1800 == 300)
    if route.has_actor:
        ref physical = resources.get_ptr_mut[PhysicsWorld]().value()[]
        ref machine = world.get_ptr_mut[RagdollComponent](route.actor).value()[].state_machine
        ref skeleton = world.get_ptr_mut[SkeletonComponent](route.actor).value()[]
        ref cache = resources.get_ptr[AnimAssetCache]().value()[]
        ref local = world.get_ptr_mut[LocalTransform](route.actor).value()[]
        var slot = cache.find_model(skeleton.model_id)
        var fall = input.just_pressed("bridge_fall") or (route.replay and clock.frame_index % 3600 == 830)
        var rise = input.just_pressed("bridge_rise") or (route.replay and clock.frame_index % 3600 == 930)
        var heading = local.rotation.rotate(Vec3(0, 0, 1))
        var place = Placement(local.translation, local.scale, atan2(heading.x, heading.z) * Float32(57.2957795))
        if slot >= 0 and fall and machine.state == RAGDOLL_SM_ANIMATED:
            ref model = cache.models[slot]
            machine.enter_limp(skeleton.skeleton, place, model, Float32(1.0 / 60.0))
            machine.ragdoll.apply_impulse_at_bone(model.find_node("Chest"), Vec3(4, 0, -12))
            route.falls += 1
            if captions:
                captions.value()[].show(2)
        if slot >= 0 and rise and machine.state == RAGDOLL_SM_LIMP:
            ref actor_anim = world.get_ptr[AnimatorComponent](route.actor).value()[]
            machine.begin_getup_world(physical, skeleton.skeleton, place, cache.models[slot], actor_anim.animator.pose)
            local.translation = place.position
            local.rotation = place.world_yaw_quat()
            if captions:
                captions.value()[].show(3)
        var supported = True
        if machine.state == RAGDOLL_SM_ANIMATED:
            if not route.checkpoint_ready:
                var landing = route.recover_on_deck(physical, local.translation, machine.authored_floor_y * machine.ragdoll.scale)
                supported = landing[0]
                route.position = landing[1]
                route.walking = False
            if input.just_pressed("bridge_save"):
                print("[Bridge] checkpoint saved", route.save_checkpoint())
            if input.just_pressed("bridge_load"):
                print("[Bridge] checkpoint loaded", route.load_checkpoint())
        if machine.state != RAGDOLL_SM_ANIMATED:
            route.position = local.translation
            route.walking = False
        route.guide = advance_bridge_cells(world, resources, route.position)
        var near = False
        if world.is_alive(route.guide):
            var guide_pos = world.get[LocalTransform](route.guide).translation
            near = (guide_pos - route.position).length() < 3.0
            if near and (input.just_pressed("bridge_interact") or (route.replay and not route.guide_near)):
                ref guide = world.get_ptr_mut[AnimatorComponent](route.guide).value()[]
                guide.clip_name = "Pivot"
                guide.time = 0
                route.interactions += 1
                if captions:
                    captions.value()[].show(1)
                    _ = resources.get_ptr_mut[AudioServer]().value()[].play_3d(captions.value()[].bell_id, guide_pos)
        route.guide_near = near
        ref component = world.get_ptr_mut[AnimatorComponent](route.actor).value()[]
        ref anim = component.animator
        route.checkpoint_ready = machine.state == RAGDOLL_SM_ANIMATED and supported
        if anim.configured():
            var state = anim.graph.state("Pivot")
            var clip = anim.graph.states[state].clip if state >= 0 else -1
            if route.pivot_active and clip >= 0 and anim.clip_of_state() == clip and anim.clip_time >= anim.clips[clip].duration - Float32(1e-6):
                route.pivot_active = False
                route.pivots += 1
            route.pivot_active = route.checkpoint_ready and clip >= 0 and (pivot_pressed or route.pivot_active)
            component.apply_root_motion = route.pivot_active
            if route.pivot_active:
                route.walking = False
                route.position.z = old_z
            elif route.walking:
                var facing = Quat() if forward >= 0 else Quat.from_axis_angle(Vec3(0, 1, 0), Float32(3.14159265))
                local.rotation = local.rotation.slerp(facing, min(Float32(1), 8 * clock.delta_seconds))
            if len(component.foot_in_stance) == 2:
                for leg in range(2):
                    component.foot_in_stance[leg] = ((leg == 1) if route.pivot_active else
                        (not route.walking or ((anim.clip_time < 0.5) == (leg == 0))))
            anim.set(anim.graph.param("action"), Float32(3 if route.pivot_active else (1 if route.walking else 0)))
        world.get_ptr_mut[LocalTransform](route.actor).value()[].translation = route.position
    var close_up = input.is_down("bridge_closeup") or (route.replay and clock.frame_index % 600 >= 480)
    var eye = route.position + (Vec3(0.54, 0.78, 1.60) if close_up else Vec3(1.35, 1.6, -3.8))
    var target = route.position + Vec3(0, Float32(0.44) if close_up else Float32(0.18), 0)
    ref cam = world.get_ptr_mut[LocalTransform](route.camera).value()[]
    cam.translation = eye
    cam.rotation = Quat()
    world.get_ptr_mut[CameraComponent](route.camera).value()[].look_target_offset = target - eye
