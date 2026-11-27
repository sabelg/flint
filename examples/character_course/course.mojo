"""One actor, one input stream, and the standard Flint asset, animation and physics plugins."""
from std.math import atan2, isfinite
from flint.anim.flanim_ecs import RAGDOLL_SM_ANIMATED, RAGDOLL_SM_LIMP, RagdollStateMachine
from flint.anim.motion_matching import MotionDatabase, MotionFeature
from flint.anim.motion_debug import MotionMatchHistory
from flint.anim.plugin import AnimAssetCache, AnimPlugin, AnimatorComponent, RagdollComponent, SkeletonComponent
from flint.anim.procedural import Placement
from flint.app import AppConfig, EngineApp
from flint.assets.asset_plugin import AssetPlugin
from flint.assets.asset_ref import AssetRef
from flint.assets.flpk_v3 import ASSET_KIND_MESH, ASSET_KIND_RAW
from flint.assets.model import NativeModel
from flint.core.math import Vec3
from flint.ecs import Entity
from flint.physics.character import CharacterControllerConfig
from flint.physics.filter import CAT_ALL, CAT_STATIC, CollisionFilter
from flint.physics.plugin import CharacterControllerComponent, PhysicsPlugin
from flint.physics.world import PhysicsWorld
from flint.render.plugin import MeshRendererComponent
from flint.scene import GlobalTransform, LocalTransform, TransformPlugin
from performer import physical_performer

comptime DT = Float32(1.0 / 60.0)
comptime ASSETS = "examples/character_course/assets"


def actor_placement(app: EngineApp, actor: Entity) -> Placement:
    ref lt = app.world.get_ptr[LocalTransform](actor).value()[]
    var fwd = lt.rotation.rotate(Vec3(0, 0, 1))
    return Placement(lt.translation, lt.scale, atan2(fwd.x, fwd.z) * Float32(57.2957795))


def make_ragdoll(model: NativeModel) raises -> RagdollStateMachine:
    var machine = RagdollStateMachine()
    machine.ragdoll = physical_performer(model)
    machine.authored_floor_y = -0.90
    machine.chest_limb_index = machine.ragdoll.limb_for_bone(model.find_node("Chest"))
    for i in range(len(model.animations)):
        if model.animations[i].name == "GetupSupine":
            machine.supine_getup_clip = i
        if model.animations[i].name == "GetupProne":
            machine.prone_getup_clip = i
    return machine^


struct Course(Movable):
    var app: EngineApp
    var actor: Entity
    var model: NativeModel
    var database: MotionDatabase
    var match_history: MotionMatchHistory
    var deck: Int
    var wall: Int
    var tick: Int
    var matching: Bool
    var previous_action: Int
    var selection: Int
    var intent: MotionFeature
    var left_prev: Vec3
    var right_prev: Vec3
    var stop_ticks: Int
    var action_clips: List[Int]
    var pivot_active: Bool
    var was_walking: Bool

    def __init__(out self, matching: Bool = False, history_capacity: Int = 0) raises:
        self.match_history = MotionMatchHistory(history_capacity)
        self.app = EngineApp(AppConfig(title="Flint Character Course", headless=True, worker_threads=0))
        self.app.add_plugin(TransformPlugin())
        self.app.add_plugin(AssetPlugin(loose_root=ASSETS))
        self.app.add_plugin(PhysicsPlugin())
        self.app.add_plugin(AnimPlugin())
        _ = self.app.register_component[MeshRendererComponent]()
        self.app.startup()
        self.actor = self.app.world.spawn()
        self.matching = matching
        self.tick = 0
        self.previous_action = -1
        self.selection = 0
        self.intent = MotionFeature()
        self.left_prev = Vec3()
        self.right_prev = Vec3()
        self.stop_ticks = 0
        self.action_clips = List[Int]()
        self.pivot_active = False
        self.was_walking = False
        var start = Vec3(0, 0.93, -1.7)
        var config = CharacterControllerConfig()
        config.half_height = 0.57
        config.radius = 0.33
        _ = self.app.world.insert[LocalTransform](self.actor, LocalTransform(start))
        _ = self.app.world.insert[GlobalTransform](self.actor, GlobalTransform())
        _ = self.app.world.insert[CharacterControllerComponent](self.actor, CharacterControllerComponent(start, config))
        var mesh = AssetRef.from_path("performer.glb", ASSET_KIND_MESH)
        _ = self.app.world.insert[MeshRendererComponent](self.actor, MeshRendererComponent(-1, mesh_ref=mesh^))
        _ = self.app.world.insert[SkeletonComponent](self.actor, SkeletonComponent())
        var graph = AssetRef.from_path("locomotion.flanim", ASSET_KIND_RAW)
        var anim = AnimatorComponent(apply_root_motion=True, enable_foot_plant=True, graph_ref=graph^)
        anim.animator.extract_root_yaw = True
        _ = self.app.world.insert[AnimatorComponent](self.actor, anim^)
        ref world = self.app.resources.get_ptr_mut[PhysicsWorld]().value()[]
        var floor_filter = CollisionFilter(CAT_STATIC, CAT_ALL, 0)
        _ = world.add_box(Vec3(0, -0.2, 0), Vec3(12, 0.2, 12), mass=0, filter=floor_filter)
        self.deck = world.add_kinematic_box(Vec3(0, 0.09, 1.25), Vec3(1.25, 0.09, 1.25))
        self.wall = world.add_box(Vec3(0, 0.55, 3.1), Vec3(2, 0.55, 0.10), mass=0, filter=floor_filter)
        world.rebuild_broadphase()
        self.app.step_one_frame(DT)
        ref cache = self.app.resources.get_ptr[AnimAssetCache]().value()[]
        var mid = self.app.world.get_ptr[SkeletonComponent](self.actor).value()[].model_id
        var slot = cache.find_model(mid)
        if slot < 0:
            raise Error("Character course could not resolve performer.glb")
        self.model = cache.models[slot].copy()
        ref ac = self.app.world.get_ptr_mut[AnimatorComponent](self.actor).value()[]
        if not ac.animator.configured():
            raise Error("Character course could not bind locomotion.flanim to performer.glb")
        for side in [String(".L"), String(".R")]:
            ac.foot_plant.add_leg(self.model.find_node("Hip" + side), self.model.find_node("Knee" + side), self.model.find_node("Foot" + side), ankle_height=0.081)
            ac.foot_in_stance.append(True)
        self.database = MotionDatabase()
        for name in ["Idle", "Walk", "Stop", "Pivot"]:
            var state = ac.animator.graph.state(name)
            if state < 0 or ac.animator.graph.states[state].clip < 0:
                raise Error("Character course needs a single clip for " + name)
            var clip = ac.animator.graph.states[state].clip
            self.action_clips.append(clip)
            ref info = ac.animator.clips[clip]
            if name == "Pivot" and (info.loop or not isfinite(info.duration) or info.duration <= 0):
                raise Error("Character course needs a finite, non-looping pivot")
            self.database.bake_clip_features(self.model, clip, self.model.find_node("Foot.L"), self.model.find_node("Foot.R"), loop=info.loop)
        self.database.finalize_and_normalize()
        var machine = make_ragdoll(self.model)
        _ = self.app.world.insert[RagdollComponent](self.actor, RagdollComponent(machine^))

    def set_locomotion(mut self, action: Int):
        ref ac = self.app.world.get_ptr_mut[AnimatorComponent](self.actor).value()[]
        ref skel = self.app.world.get_ptr[SkeletonComponent](self.actor).value()[].skeleton
        var hip = skel.global_position(0)
        var l = skel.global_position(ac.foot_plant.legs[0].ankle) - hip
        var r = skel.global_position(ac.foot_plant.legs[1].ankle) - hip
        self.intent.left_foot_pos = l
        self.intent.right_foot_pos = r
        self.intent.left_foot_vel = (l - self.left_prev) * (Float32(1) / DT) if self.previous_action >= 0 else Vec3()
        self.intent.right_foot_vel = (r - self.right_prev) * (Float32(1) / DT) if self.previous_action >= 0 else Vec3()
        self.left_prev = l
        self.right_prev = r
        var vel = Vec3(0, 0, Float32(1.3) if action == 1 else Float32(0))
        self.intent.hip_vel = vel
        self.intent.traj_pos_200 = vel * Float32(0.2)
        self.intent.traj_pos_400 = vel * Float32(0.4)
        self.intent.traj_pos_800 = vel * Float32(0.8)
        if action == 2:
            var remaining = Float32(self.stop_ticks) / 30
            self.intent.traj_pos_200.z = Float32(0.14) * remaining
            self.intent.traj_pos_400.z = Float32(0.23) * remaining
            self.intent.traj_pos_800.z = Float32(0.25) * remaining
        var turn_rate = Float32(1.57079633) / ac.animator.clips[self.action_clips[3]].duration
        var remaining_turn = Float32(0)
        if self.pivot_active:
            var time = ac.animator.clip_time if ac.animator.clip_of_state() == self.action_clips[3] else Float32(0)
            remaining_turn = max(Float32(0), Float32(1.57079633) - time * turn_rate)
        self.intent.traj_facing_200 = min(remaining_turn, turn_rate * Float32(0.2))
        self.intent.traj_facing_400 = min(remaining_turn, turn_rate * Float32(0.4))
        self.intent.traj_facing_800 = min(remaining_turn, turn_rate * Float32(0.8))
        if self.pivot_active or not self.matching:
            self.selection = action
        elif self.tick % 10 == 0 or action != self.previous_action:
            var matched = self.match_history.search(
                self.database, self.tick + 1, self.intent, ac.animator.clip_of_state(), ac.animator.clip_time
            )
            if matched >= 0:
                ref choice = self.database.frames[matched]
                # The library includes turns for trajectory normalization. Only
                # the explicit pivot command can initiate the full quarter-turn.
                if choice.clip_index == self.action_clips[3]:
                    self.selection = action
                else:
                    for index in range(len(self.action_clips)):
                        if self.action_clips[index] == choice.clip_index:
                            self.selection = index
                            break
                    if choice.clip_index != ac.animator.clip_of_state() or abs(choice.time_seconds - ac.animator.clip_time) > 0.12:
                        ac.animator.request_match_frame(choice.clip_index, choice.time_seconds)
        ac.animator.set(0, Float32(self.selection))
        self.previous_action = action
        for leg in range(2):
            var phase = ac.animator.clip_time
            ac.foot_in_stance[leg] = self.selection == 0 or (self.selection == 1 and ((phase < 0.5) == (leg == 0))) or (self.selection == 3 and leg == 1)

    def step(mut self, walk: Bool, steer: Float32 = 0, pivot: Bool = False, fall: Bool = False, rise: Bool = False):
        ref world = self.app.resources.get_ptr_mut[PhysicsWorld]().value()[]
        # One metre of translation and 30 degrees of yaw over two seconds.
        var moving = self.tick % 720 >= 108 and self.tick % 720 < 228
        var returning = self.tick % 720 >= 468 and self.tick % 720 < 588
        var direction = Float32(Int(moving) - Int(returning))
        world.bodies[self.deck].linear_vel = Vec3(0.5 * direction, 0, 0)
        world.bodies[self.deck].angular_vel = Vec3(0, 0.2617994 * direction, 0)
        ref rc = self.app.world.get_ptr_mut[RagdollComponent](self.actor).value()[]
        ref ac = self.app.world.get_ptr_mut[AnimatorComponent](self.actor).value()[]
        ref kc = self.app.world.get_ptr_mut[CharacterControllerComponent](self.actor).value()[]
        var place = actor_placement(self.app, self.actor)
        ref skel = self.app.world.get_ptr_mut[SkeletonComponent](self.actor).value()[].skeleton
        if fall and rc.state_machine.state == RAGDOLL_SM_ANIMATED:
            rc.state_machine.enter_limp(skel, place, self.model, DT)
            rc.state_machine.ragdoll.apply_impulse_at_bone(self.model.find_node("Chest"), Vec3(8, 0, -28))
        if rise and rc.state_machine.state == RAGDOLL_SM_LIMP:
            rc.state_machine.begin_getup_world(world, skel, place, self.model, ac.animator.pose)
            ref lt = self.app.world.get_ptr_mut[LocalTransform](self.actor).value()[]
            lt.translation = place.position
            lt.rotation = place.world_yaw_quat()
        var animated = rc.state_machine.state == RAGDOLL_SM_ANIMATED
        if animated and not kc.active:
            kc.controller.pos = place.position
            kc.controller.grounded = False
            kc.controller.has_ground_frame = False
        kc.active = animated
        ac.apply_root_motion = animated
        if animated and steer != 0:
            ref lt = self.app.world.get_ptr_mut[LocalTransform](self.actor).value()[]
            place.yaw += steer * Float32(90) * DT
            lt.rotation = place.world_yaw_quat()
        self.pivot_active = animated and (pivot or self.pivot_active)
        if not walk and self.was_walking:
            self.stop_ticks = 30
        self.was_walking = walk
        var action = 3 if self.pivot_active else (1 if walk else (2 if self.stop_ticks > 0 else 0))
        if animated:
            self.set_locomotion(action)
        else:
            # AnimPlugin releases IK locks while the ragdoll owns the pose.
            ac.animator.set(0, 0)
            self.previous_action = -1
        self.stop_ticks = max(0, self.stop_ticks - 1)
        self.app.step_one_frame(DT)
        if self.pivot_active:
            ref result = self.app.world.get_ptr[AnimatorComponent](self.actor).value()[].animator
            var clip = self.action_clips[3]
            if result.clip_of_state() == clip and result.clip_time >= result.clips[clip].duration - Float32(1e-6):
                self.pivot_active = False
        self.tick += 1
