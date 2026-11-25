"""Flint v5 physics-driven animation blending suite (`test_flint_v5_physical_anim.mojo`).

A small biped walks forward (swinging legs and arm, root motion through the placement):
  1. Idle: the drawn pose is the animation exactly, frame after frame.
  2. Tracking: with every limb fully simulated, the world drives keep each limb within a few
     centimetres of its animated transform while walking, and the body does not fall.
  3. Hit reaction: a punch to the hand loosens the arm most, the chest less and the legs
     least (by joint hops); the arm visibly swings, then eases back and the controller
     returns to idle with the exact animated pose.
  4. Knockdown: a heavy hit to the chest falls, settles, gets up through
     `RagdollStateMachine` and resumes the animation.
  5. Determinism: the same hit twice gives bit-identical poses.
"""
from std.math import abs
from std.testing import assert_true, assert_equal
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.core.detmath import det_sin
from flint.core.ik import angle_axis
from flint.assets.model import NativeModel, NativeNode, NativeAnimation, NativeChannel
from flint.anim.pose import Pose, SkeletonInstance
from flint.anim.procedural import Placement
from flint.anim.physical_anim import PhysicalAnimator, PA_IDLE, PA_BLEND, PA_FALLEN
from flint.physics.world import PhysicsWorld

comptime DT: Float32 = 1.0 / 60.0
comptime SPEED: Float32 = 1.1


def _biped() -> NativeModel:
    var m = NativeModel()
    var names = [
        String("Pelvis"), String("Chest"), String("Head"),
        String("UpperArm.R"), String("Hand.R"),
        String("Thigh.L"), String("Shin.L"), String("Foot.L"),
        String("Thigh.R"), String("Shin.R"), String("Foot.R"),
    ]
    var parents = [-1, 0, 1, 1, 3, 0, 5, 6, 0, 8, 9]
    var offsets = [
        Vec3(0.0, 0.92, 0.0), Vec3(0.0, 0.42, 0.0), Vec3(0.0, 0.32, 0.0),
        Vec3(0.26, 0.24, 0.0), Vec3(0.0, -0.48, 0.0),
        Vec3(-0.14, 0.0, 0.0), Vec3(0.0, -0.44, 0.0), Vec3(0.0, -0.40, 0.0),
        Vec3(0.14, 0.0, 0.0), Vec3(0.0, -0.44, 0.0), Vec3(0.0, -0.40, 0.0),
    ]
    for i in range(len(names)):
        var nd = NativeNode()
        nd.name = names[i]
        nd.translation = offsets[i]
        nd.base_translation = offsets[i]
        nd.rotation = Quat()
        nd.base_rotation = Quat()
        nd.scale = Vec3(1, 1, 1)
        nd.base_scale = Vec3(1, 1, 1)
        m.nodes.append(nd^)
        var p = parents[i]
        if p >= 0:
            m.nodes[p].children.append(i)
        else:
            m.roots.append(i)
    for cname in [String("GetupSupine"), String("GetupProne")]:
        var anim = NativeAnimation()
        anim.name = cname
        anim.duration = 0.40
        var ch = NativeChannel()
        ch.node = 0
        ch.path = 0
        ch.times = [Float32(0.0), Float32(0.40)]
        ch.values = [Float32(0.0), Float32(0.92), Float32(0.0), Float32(0.0), Float32(0.92), Float32(0.0)]
        anim.channels.append(ch^)
        m.animations.append(anim^)
    return m^


def _walk(model: NativeModel, t: Float32) -> Pose:
    """A walk cycle: legs swing +-0.45 rad about x, the shins bend, the arm swings."""
    var p = Pose()
    p.rest(model)
    var ph = det_sin(Float32(6.2831853) * t)
    var x = Vec3(1, 0, 0)
    p.r[5] = angle_axis(Float32(0.45) * ph, x)
    p.r[8] = angle_axis(Float32(-0.45) * ph, x)
    p.r[6] = angle_axis(Float32(-0.25) * (ph * ph), x)
    p.r[9] = angle_axis(Float32(-0.25) * (ph * ph), x)
    p.r[3] = angle_axis(Float32(0.35) * ph, x)
    p.r[1] = angle_axis(Float32(0.06) * ph, Vec3(0, 1, 0))
    return p^


def _animator(model: NativeModel) -> PhysicalAnimator:
    var pa = PhysicalAnimator()
    pa.rsm.supine_getup_clip = 0
    pa.rsm.prone_getup_clip = 1
    _ = pa.rsm.ragdoll.add_limb_from_model(model, 0, -1, 15.0, 0.14, 0.30)          # pelvis
    _ = pa.rsm.ragdoll.add_limb_from_model(model, 1, 0, 12.0, 0.15, 0.32)           # chest
    _ = pa.rsm.ragdoll.add_limb_from_model(model, 3, 1, 3.5, 0.06, 0.48)            # arm
    _ = pa.rsm.ragdoll.add_limb_from_model(model, 5, 0, 7.0, 0.08, 0.44)            # thigh L
    _ = pa.rsm.ragdoll.add_limb_from_model(model, 6, 3, 4.0, 0.05, 0.40)            # shin L
    _ = pa.rsm.ragdoll.add_limb_from_model(model, 8, 0, 7.0, 0.08, 0.44)            # thigh R
    _ = pa.rsm.ragdoll.add_limb_from_model(model, 9, 5, 4.0, 0.05, 0.40)            # shin R
    return pa^


def _world() -> PhysicsWorld:
    var w = PhysicsWorld()
    _ = w.add_box(Vec3(0.0, -0.5, 0.0), Vec3(40.0, 0.5, 40.0), mass=0.0)
    w.rebuild_broadphase()
    return w^


def _pose_diff(a: Pose, b: Pose) -> Float32:
    var d: Float32 = 0
    for i in range(min(a.count(), b.count())):
        d = max(d, (a.t[i] - b.t[i]).length())
        d = max(d, abs(a.r[i].x - b.r[i].x) + abs(a.r[i].y - b.r[i].y) + abs(a.r[i].z - b.r[i].z) + abs(a.r[i].w - b.r[i].w))
    return d


def _angle(a: Quat, b: Quat) -> Float32:
    var d = abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w)
    return Float32(1.0) - min(Float32(1.0), d)


def test_idle_exact() raises:
    var model = _biped()
    var world = _world()
    var pa = _animator(model)
    var skel = SkeletonInstance(len(model.nodes))
    skel.reset_to_rest(model)
    var place = Placement()
    for f in range(120):
        var anim = _walk(model, Float32(f) * DT)
        place.position.z += SPEED * DT
        pa.update(world, skel, place, model, anim, DT)
        assert_equal(pa.state, PA_IDLE)
        assert_true(_pose_diff(skel.pose, anim) == 0.0, "idle output is the animation")
    print("  idle: animation passes through exactly")


def test_tracking() raises:
    var model = _biped()
    var world = _world()
    var pa = _animator(model)
    for i in range(pa.limb_count()):
        pa.set_base_weight(i, 1.0)
    var skel = SkeletonInstance(len(model.nodes))
    skel.reset_to_rest(model)
    var place = Placement()
    var worst: Float32 = 0
    for f in range(150):
        var anim = _walk(model, Float32(f) * DT)
        place.position.z += SPEED * DT
        pa.update(world, skel, place, model, anim, DT)
        assert_equal(pa.state, PA_BLEND)
        if f >= 20:
            for i in range(pa.limb_count()):
                worst = max(worst, (pa.rsm.ragdoll.states[i].com_pos - pa.targets_pos[i]).length())
    print("  tracking: worst limb error while walking", worst, "m")
    assert_true(worst < 0.06, "fully simulated limbs follow the walk")
    assert_equal(pa.falls, 0)


def _hit_run(mut pa: PhysicalAnimator, model: NativeModel, world: PhysicsWorld, frames: Int,
             mut swing: Float32, mut back_idle: Int) raises -> Pose:
    var skel = SkeletonInstance(len(model.nodes))
    skel.reset_to_rest(model)
    var place = Placement()
    var mid = Pose()
    for f in range(frames):
        var anim = _walk(model, Float32(f) * DT)
        place.position.z += SPEED * DT
        if f == 30:
            var hand = place.to_world(skel.global_position(4))
            pa.hit(4, Vec3(-22.0, 0.0, 0.0), hand)
            assert_equal(pa.state, PA_BLEND)
        pa.update(world, skel, place, model, anim, DT)
        if f > 30 and f < 60:
            swing = max(swing, _angle(skel.pose.r[3], anim.r[3]))
        if f == 40:
            mid.assign(skel.pose)
        if f > 30 and back_idle < 0 and pa.state == PA_IDLE:
            back_idle = f
        if back_idle >= 0 and f > back_idle:
            assert_true(_pose_diff(skel.pose, anim) == 0.0, "back on the animation after the hit")
    return mid^


def test_hit_reaction() raises:
    var model = _biped()
    var world = _world()
    var pa = _animator(model)
    var swing: Float32 = 0
    var back_idle = -1
    # weights straight after the hit come from the hit levels
    var skel = SkeletonInstance(len(model.nodes))
    skel.reset_to_rest(model)
    _ = _hit_run(pa, model, world, 240, swing, back_idle)
    assert_equal(pa.falls, 0)
    assert_true(swing > 0.01, "the arm swings away from the animation")
    assert_true(back_idle > 30 and back_idle < 240, "the hit wears off")
    # spread of a fresh hit: arm > chest > pelvis > shin
    var pb = _animator(model)
    var place = Placement()
    pb.update(world, skel, place, model, _walk(model, 0.0), DT)
    pb.hit(4, Vec3(-22.0, 0.0, 0.0), place.to_world(skel.global_position(4)))
    var w_arm = pb.physics_weight(2)
    var w_chest = pb.physics_weight(1)
    var w_pelvis = pb.physics_weight(0)
    var w_shin = pb.physics_weight(4)
    print("  hit: arm", w_arm, "chest", w_chest, "pelvis", w_pelvis, "shin", w_shin, "| swing", swing,
          "| back to animation at frame", back_idle)
    assert_true(w_arm > w_chest and w_chest > w_pelvis and w_pelvis > w_shin and w_shin > 0.0)


def test_knockdown() raises:
    var model = _biped()
    var world = _world()
    var pa = _animator(model)
    var skel = SkeletonInstance(len(model.nodes))
    skel.reset_to_rest(model)
    var place = Placement()
    var saw_fallen = False
    var resumed = -1
    for f in range(900):
        var anim = _walk(model, Float32(f) * DT)
        if pa.state != PA_FALLEN:
            place.position.z += SPEED * DT
        if f == 20:
            var chest = place.to_world(skel.global_position(1))
            pa.hit(1, Vec3(0.0, 10.0, -220.0), chest)
            assert_equal(pa.state, PA_FALLEN)
        pa.update(world, skel, place, model, anim, DT)
        if pa.state == PA_FALLEN:
            saw_fallen = True
        if saw_fallen and pa.state == PA_IDLE and resumed < 0:
            resumed = f
        if resumed >= 0 and f > resumed:
            assert_true(_pose_diff(skel.pose, anim) == 0.0, "walking again after the get-up")
    print("  knockdown: fell, got up (recoveries", pa.rsm.recovery_count, ") and resumed at frame", resumed,
          "| pelvis snap", pa.rsm.last_pelvis_snap_m, "m")
    assert_true(saw_fallen)
    assert_equal(pa.falls, 1)
    assert_equal(pa.rsm.recovery_count, 1)
    assert_true(resumed > 20)
    assert_true(pa.rsm.last_pelvis_snap_m < 1.0e-4)


def test_determinism() raises:
    var model = _biped()
    var world = _world()
    var a = _animator(model)
    var b = _animator(model)
    var s1: Float32 = 0
    var s2: Float32 = 0
    var i1 = -1
    var i2 = -1
    var m1 = _hit_run(a, model, world, 60, s1, i1)
    var m2 = _hit_run(b, model, world, 60, s2, i2)
    assert_true(_pose_diff(m1, m2) == 0.0)
    assert_true(_pose_diff(m1, _walk(model, 40.0 * DT)) > 0.0)
    print("  determinism: identical poses")


def main() raises:
    print("=== Flint v5 Physical Animation Suite ===")
    test_idle_exact()
    test_tracking()
    test_hit_reaction()
    test_knockdown()
    test_determinism()
    print("=== ALL FLINT v5 PHYSICAL ANIMATION TESTS PASSED ===")
