"""One supine recovery, one chest impulse, two controllers on the same physical performer.

Run from the repository root with:
    pixi run mojo run -I src -O3 --fp-mode contract=off --Werror examples/character_course/recovery_bench.mojo

Output is CSV. Positions are metres. Pose RMS uses pelvis-relative joint pivots over
the complete trajectory. Stance drift is the largest horizontal excursion of either
ankle from its simulated position at the start of the final 0.5 s double-stance hold.
Support ticks count hold ticks with an actual foot capsule within 2 cm of the floor.
Success requires support, pelvis height at least 80% of the authored final height,
and torso up-dot at least 0.85 throughout all 30 final ticks. Both modes start
at rest in the first authored pose and receive a 12 N·s impulse along +X at
the chest on tick zero. The floor is level. The motor can apply a world-space
pelvis attitude torque. It has no root translation actuator or balance controller.
The original ball-joint elbow rig is replayed alongside its corrected counterpart.
The recovery verdict retains the original 2 cm proximity criterion. Separate
impulse columns report ticks when one or both feet received an upward normal
impulse from the floor. Elbow-angle RMS uses the simulated local rotation relative
to the requested animation, in radians, on both elbows across all ticks.
The foot load share is the fraction of total upward floor impulse applied to the
two foot capsules over the hold. The final floor load column gives its denominator
divided by the 0.5 s hold duration, in newtons. Each mode repeats the identical
deterministic shove three times. Pass --check for the calibration checks alone.
CPU time uses Linux process clock microseconds around target sampling, controller
evaluation and PhysicsWorld.step, excluding asset loading, scoring and printing.
"""
from std.collections import List
from std.ffi import external_call
from std.sys import argv
from std.math import abs, ceil, isfinite, sqrt
from std.testing import assert_true
from flint.anim.pose import Pose, SkeletonInstance, det_quat_to_scaled_angle, sample_pose
from flint.anim.procedural import Placement
from flint.anim.ragdoll import ArticulatedRagdoll, clamp_joint_relative_rotation
from flint.assets.glb_native import load_native_glb
from flint.assets.model import NativeModel
from flint.core.math import Quat, Vec3
from flint.physics.events import static_other
from flint.physics.world import PhysicsWorld
from performer import physical_performer


comptime DT = Float32(1.0 / 60.0)
comptime HOLD_TICKS = 30
comptime SHOVE_NS = Float32(12.0)



def pivot(rig: ArticulatedRagdoll, index: Int) -> Vec3:
    return rig.states[index].joint_pivot_world(rig.limbs[index].local_com_offset, rig.scale)


def horizontal_distance(a: Vec3, b: Vec3) -> Float32:
    var dx = a.x - b.x
    var dz = a.z - b.z
    return sqrt(dx * dx + dz * dz)


def elbow_rig(model: NativeModel, legacy: Bool) raises -> ArticulatedRagdoll:
    var rig = physical_performer(model)
    if legacy:
        for name in ["Elbow.L", "Elbow.R"]:
            var j = rig.limb_for_bone(model.find_node(String(name)))
            rig.limbs[j].is_hinge = False
            rig.limbs[j].hinge_axis_local = Vec3(1, 0, 0)
    return rig^


def foot_floor_impulse(world: PhysicsWorld, floor_slot: Int, body_id: Int) -> Float32:
    # Static collision partners use a slot-derived key in this tick's contact cache.
    var vertical = Float32(0)
    for pair in world.cache.tick_pairs:
        if pair.a == body_id and pair.b == static_other(floor_slot) and pair.normal.y > Float32(0):
            vertical += pair.impulse * pair.normal.y
    return vertical


def check_load_measure() raises:
    var air = PhysicsWorld(gravity=0)
    var ground = air.add_box(Vec3(0, -0.25, 0), Vec3(5, 0.25, 5), mass=0)
    # The capsule bottom is exactly 1 cm above the floor, inside the old proximity band.
    var foot = air.add_capsule(Vec3(0, 0.21, 0), 0.1, 0.1, 1.0)
    air.step(DT)
    assert_true(foot_floor_impulse(air, ground, air.body_id(foot)) == Float32(0), "air gap must carry no load")
    var loaded = PhysicsWorld()
    var floor = loaded.add_box(Vec3(0, -0.25, 0), Vec3(5, 0.25, 5), mass=0)
    var body = loaded.add_capsule(Vec3(0, 0.21, 0), 0.1, 0.1, 1.0)
    loaded.bodies[body].can_sleep = False
    for _ in range(90):
        loaded.step(DT)
    assert_true(foot_floor_impulse(loaded, floor, loaded.body_id(body)) > Float32(0), "resting foot must transmit load")


def check_elbow_targets(model: NativeModel, clip: Int, report: Bool) raises:
    var rig = elbow_rig(model, False)
    var legacy = elbow_rig(model, True)
    var pose = Pose()
    var max_old = Float32(0)
    var max_new = Float32(0)
    var incompatible = 0
    var samples = 0
    var frames = Int(ceil(Float64(model.animations[clip].duration) * 60.0 - 1.0e-5))
    for tick in range(frames):
        sample_pose(model, clip, Float32(tick + 1) * DT, pose)
        rig.set_target_pose(pose)
        for side in ["Elbow.L", "Elbow.R"]:
            var j = rig.limb_for_bone(model.find_node(String(side)))
            var target = rig.states[j].target_local_rot
            var old_limit = clamp_joint_relative_rotation(legacy.limbs[j], target)[0]
            var new_limit = clamp_joint_relative_rotation(rig.limbs[j], target)[0]
            var old_error = det_quat_to_scaled_angle(old_limit * target.conjugate()).length()
            var new_error = det_quat_to_scaled_angle(new_limit * target.conjugate()).length()
            max_old = max(max_old, old_error)
            max_new = max(max_new, new_error)
            incompatible += Int(old_error > Float32(0.02))
            samples += 1
    assert_true(incompatible > 0, "legacy rig must reproduce its limit conflict")
    assert_true(max_new < Float32(0.02), "authored elbow targets must fit the calibrated hinge")
    if report:
        print(model.animations[clip].name, "legacy_conflicts", incompatible, "of", samples,
              "max_old_rad", max_old, "max_new_rad", max_new)


def run_take(model: NativeModel, clip: Int, active: Bool, repeat: Int, legacy: Bool) raises:
    var rig = elbow_rig(model, legacy)
    var root = model.find_node("Root")
    var chest = rig.limb_for_bone(model.find_node("Chest"))
    var feet: List[Int] = [
        rig.limb_for_bone(model.find_node("Foot.L")), rig.limb_for_bone(model.find_node("Foot.R")),
    ]
    var elbows: List[Int] = [
        rig.limb_for_bone(model.find_node("Elbow.L")), rig.limb_for_bone(model.find_node("Elbow.R")),
    ]
    var pose = Pose()
    var reference = SkeletonInstance()
    var placement = Placement(Vec3(0, 0.93, 0))
    sample_pose(model, clip, 0, pose)
    reference.apply_pose(pose, model)
    rig.handoff_from_poses(reference, placement, reference, placement, model, DT)
    if active:
        rig.set_active_pd()
    else:
        rig.go_limp()
    var world = PhysicsWorld()
    var floor_slot = world.add_box(Vec3(0, -0.25, 0), Vec3(20, 0.25, 20), mass=0)
    var slot = world.add_ragdoll(rig.copy())
    world.apply_ragdoll_impulse(slot, chest, Vec3(SHOVE_NS, 0, 0), rig.states[chest].com_pos)

    var duration = model.animations[clip].duration
    var clip_ticks = Int(ceil(Float64(duration) * 60.0 - 1.0e-5))
    var total_ticks = clip_ticks + HOLD_TICKS
    var stance_start = List[Vec3](length=2, fill=Vec3())
    var target_rotations = List[Quat](length=len(rig.limbs), fill=Quat())
    var sq_error = Float64(0)
    var max_drift = Float32(0)
    var support_ticks = 0
    var loaded_foot_ticks = 0
    var both_loaded_ticks = 0
    var total_load = Float64(0)
    var foot_load = Float64(0)
    var elbow_error_sq = Float64(0)
    var success_ticks = 0
    var final_height = Float32(0)
    var cpu_us = Int64(0)
    for tick in range(total_ticks):
        var start = external_call["clock", Int64]()
        sample_pose(model, clip, min(duration, Float32(tick + 1) * DT), pose)
        if active:
            rig.set_target_pose(pose)
            for j in range(len(rig.limbs)):
                target_rotations[j] = rig.states[j].target_local_rot
            world.set_ragdoll_targets(slot, target_rotations, rig.target_root_world_rot, 1, False)
        world.step(DT)
        var stop = external_call["clock", Int64]()
        if start < 0 or stop < start:
            raise Error("Process CPU clock unavailable")
        cpu_us += stop - start

        reference.apply_pose(pose, model)
        ref physical = world.ragdolls[slot].rag
        var pelvis = physical.pelvis_world_position()
        var expected_pelvis = reference.global_position(root)
        for j in range(1, len(physical.limbs)):
            var actual_offset = pivot(physical, j) - pelvis
            var expected_offset = reference.global_position(physical.limbs[j].bone_node) - expected_pelvis
            sq_error += Float64((actual_offset - expected_offset).length_squared())
        for e in elbows:
            var p = physical.limbs[e].parent_limb
            var actual = physical.states[p].rotation.conjugate() * physical.states[e].rotation
            var expected = reference.global_rotation(physical.limbs[p].bone_node).conjugate() * reference.global_rotation(physical.limbs[e].bone_node)
            elbow_error_sq += Float64(det_quat_to_scaled_angle(actual * expected.conjugate()).length_squared())
        if tick >= clip_ticks:
            var supported = False
            var loaded_feet = 0
            for foot in range(2):
                var foot_pos = pivot(physical, feet[foot])
                if tick == clip_ticks:
                    stance_start[foot] = foot_pos
                max_drift = max(max_drift, horizontal_distance(foot_pos, stance_start[foot]))
                var body = world.find_slot_by_id(world.ragdolls[slot].limb_body_ids[feet[foot]])
                var upward = foot_floor_impulse(world, floor_slot, world.body_id(body))
                foot_load += Float64(upward)
                loaded_feet += Int(upward > Float32(0))
                var endpoints = world.bodies[body].capsule_endpoints()
                var bottom = min(endpoints[0].y, endpoints[1].y) - world.bodies[body].radius
                supported = supported or abs(bottom) <= Float32(0.02)
            for pair in world.cache.tick_pairs:
                if pair.b == static_other(floor_slot) and pair.normal.y > Float32(0):
                    total_load += Float64(pair.impulse * pair.normal.y)
            loaded_foot_ticks += Int(loaded_feet > 0)
            both_loaded_ticks += Int(loaded_feet == 2)
            if supported:
                support_ticks += 1
            var torso_up = (pivot(physical, chest) - pelvis).normalized().y
            final_height = pelvis.y
            if supported and pelvis.y >= (placement.position.y + expected_pelvis.y) * Float32(0.8) and torso_up >= Float32(0.85):
                success_ticks += 1

    var rms = sqrt(sq_error / Float64(total_ticks * (len(rig.limbs) - 1)))
    if not isfinite(rms) or not isfinite(max_drift) or not isfinite(final_height):
        raise Error("Non-finite trajectory")
    print(
        String("motor" if active else "passive") + "," + String(repeat) + "," + String(total_ticks)
        + "," + String(Int(success_ticks == HOLD_TICKS)) + "," + String(support_ticks)
        + "," + String(final_height) + "," + String(max_drift) + "," + String(rms)
        + "," + String(Float64(cpu_us) / Float64(total_ticks))
        + "," + ("legacy" if legacy else "corrected")
        + "," + String(loaded_foot_ticks) + "," + String(both_loaded_ticks)
        + "," + String(sqrt(elbow_error_sq / Float64(total_ticks * 2)))
        + "," + String(foot_load / total_load if total_load > 0 else Float64(0))
        + "," + String(total_load / (Float64(HOLD_TICKS) * Float64(DT))),
    )


def main() raises:
    var model = load_native_glb("examples/character_course/assets/performer.glb")
    var clip = -1
    for i in range(len(model.animations)):
        if model.animations[i].name == "GetupSupine":
            clip = i
    if clip < 0:
        raise Error("GetupSupine clip missing. Run from the Flint repository root.")
    var args = argv()
    var check_only = len(args) == 2 and args[1] == "--check"
    check_load_measure()
    for i in range(len(model.animations)):
        if model.animations[i].name == "GetupSupine" or model.animations[i].name == "GetupProne":
            check_elbow_targets(model, i, check_only)
    if check_only:
        return
    print("controller,repeat,ticks,recovered,supported_hold_ticks,final_pelvis_y_m,stance_window_drift_m,pose_rms_m,mean_cpu_us_tick,rig,loaded_foot_hold_ticks,both_loaded_hold_ticks,elbow_tracking_rms_rad,foot_floor_impulse_share,mean_all_floor_load_N")
    for repeat in range(3):
        var first_motor = repeat % 2 == 0
        for legacy in [True, False]:
            run_take(model, clip, first_motor, repeat + 1, legacy)
            run_take(model, clip, not first_motor, repeat + 1, legacy)
