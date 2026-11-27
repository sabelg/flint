"""Flint's playable character and contact review course."""
from std.ffi import external_call
from std.sys import argv
from flint.anim.flanim_ecs import RAGDOLL_SM_ANIMATED, RAGDOLL_SM_LIMP, RAGDOLL_SM_GETUP_SUPINE
from flint.anim.plugin import AnimatorComponent, RagdollComponent, SkeletonComponent
from flint.anim.pose import write_pose
from flint.anim.skinning import SKIN_LBS, skin_mesh
from flint.assets.image import write_png
from flint.core.math import Vec3
from flint.physics.plugin import CharacterControllerComponent
from flint.physics.world import PhysicsWorld
from flint.platform.input import InputState
from flint.platform.os import get_time, make_dirs, write_bytes
from flint.platform.wayland import WaylandWindow
from flint.render.simd_canvas import Camera3D, SimdCanvas
from course import Course, DT, actor_placement
from flint.anim.motion_debug import MotionMatchCandidate


def clip_label(c: Course) -> String:
    ref rc = c.app.world.get_ptr[RagdollComponent](c.actor).value()[].state_machine
    if rc.state == RAGDOLL_SM_LIMP:
        return "Fall"
    if rc.state != RAGDOLL_SM_ANIMATED:
        return "GetupSupine" if rc.state == RAGDOLL_SM_GETUP_SUPINE else "GetupProne"
    return c.app.world.get_ptr[AnimatorComponent](c.actor).value()[].animator.state_name()


def source_frame(c: Course) -> Int:
    ref rc = c.app.world.get_ptr[RagdollComponent](c.actor).value()[].state_machine
    if rc.state == RAGDOLL_SM_LIMP:
        return -1
    if rc.state != RAGDOLL_SM_ANIMATED:
        return Int(rc.getup_time * 30)
    return Int(c.app.world.get_ptr[AnimatorComponent](c.actor).value()[].animator.clip_time * 30)


def draw_box(mut canvas: SimdCanvas, cam: Camera3D, world: PhysicsWorld, slot: Int, color: UInt32):
    ref body = world.bodies[slot]
    var corners = List[Vec3]()
    for i in range(8):
        var h = body.half_extents
        var corner = Vec3(h.x if i & 1 else -h.x, h.y if i & 2 else -h.y, h.z if i & 4 else -h.z)
        corners.append(body.pos + body.rot.rotate(corner))
    for face in [[0,1,3,2], [4,6,7,5], [2,3,7,6], [0,4,5,1], [0,2,6,4], [1,5,7,3]]:
        canvas.draw_tri_3d_lit(cam, corners[face[0]], corners[face[1]], corners[face[2]], color)
        canvas.draw_tri_3d_lit(cam, corners[face[0]], corners[face[2]], corners[face[3]], color)


def draw(mut c: Course, mut canvas: SimdCanvas, side_view: Bool):
    canvas.clear(0x19303B, 0x526975)
    var place = actor_placement(c.app, c.actor)
    var target = Vec3(place.position.x, 0.9, place.position.z)
    var offset = Vec3(3.2, 0.30, 0) if side_view else Vec3(2.4, 0.95, 2.8)
    var cam = Camera3D(canvas.width, canvas.height, target + offset, target, 41)
    ref world = c.app.resources.get_ptr[PhysicsWorld]().value()[]
    for i in range(-8, 9):
        var f = Float32(i) * 0.5
        canvas.draw_line_3d(cam, Vec3(f, 0.002, -4), Vec3(f, 0.002, 4), 0x677B80)
        canvas.draw_line_3d(cam, Vec3(-4, 0.002, f), Vec3(4, 0.002, f), 0x677B80)
    draw_box(canvas, cam, world, c.deck, 0x698381)
    draw_box(canvas, cam, world, c.wall, 0x756A5D)
    ref skel = c.app.world.get_ptr[SkeletonComponent](c.actor).value()[].skeleton
    write_pose(c.model, skel.pose)
    for mi in range(len(c.model.meshes)):
        var verts = skin_mesh(c.model, mi, SKIN_LBS)
        ref mesh = c.model.meshes[mi]
        var points = List[Vec3](capacity=len(verts) // 8)
        for i in range(0, len(verts), 8):
            points.append(place.to_world(Vec3(verts[i], verts[i+1], verts[i+2])))
        for i in range(0, len(mesh.indices) - 2, 3):
            canvas.draw_tri_3d_lit(cam, points[mesh.indices[i]], points[mesh.indices[i+1]], points[mesh.indices[i+2]], mesh.base_color)
    ref ac = c.app.world.get_ptr[AnimatorComponent](c.actor).value()[]
    for leg in ac.foot_plant.legs:
        if ac.enable_foot_plant and leg.locked:
            var contact = leg.locked_world_pos
            var a = Vec3(0.065, 0, 0)
            var b = Vec3(0, 0, 0.065)
            canvas.draw_line_3d(cam, contact-a, contact+a, 0xFFC165, 2)
            canvas.draw_line_3d(cam, contact-b, contact+b, 0xFFC165, 2)
    var start = place.position - Vec3(0, 0.74, 0)
    canvas.draw_line_3d(cam, start, start + place.vector_to_world(c.intent.traj_pos_800), 0x73E8D3, 2)
    canvas.draw_text(12, 12, String("Match") if c.matching else String("Graph"), 0xC2DFD9, 2)
    canvas.draw_text(12, 35, clip_label(c) + "   tick " + String(c.tick), 0xFFFFFF, 2)
    canvas.draw_text(12, 58, "Clip frame " + String(source_frame(c)) + "   stance L/R " + String(Int(ac.enable_foot_plant and ac.foot_plant.legs[0].locked)) + "/" + String(Int(ac.enable_foot_plant and ac.foot_plant.legs[1].locked)), 0xF0D3A1, 1)


def _step_replay(mut c: Course, i: Int):
    c.step(i >= 12 and i < 114, pivot=i == 232, fall=i == 279, rise=i == 325)


def _print_candidate(c: Course, role: String, candidate: MotionMatchCandidate):
    if candidate.index < 0:
        print(role, "(none)")
        return
    print(role, c.model.animations[candidate.clip].name, candidate.source_frame, candidate.index,
          candidate.terms[0], candidate.terms[1], candidate.terms[2], candidate.terms[3],
          candidate.terms[4], candidate.terms[5], candidate.bias, candidate.total, sep=",")


def inspect_match(tick: Int) raises:
    if tick < 1 or tick > 465:
        raise Error("Course replay tick must be between 1 and 465")
    var c = Course(matching=True, history_capacity=465)
    var playback = String("")
    for i in range(465):
        _step_replay(c, i)
        if c.tick == tick:
            playback = clip_label(c)
    var decision = c.match_history.rewind(tick)
    if not decision:
        raise Error("No motion search recorded at or before that tick")
    ref d = decision.value()
    print("Replay tick", tick, "plays", playback, "with latest motion search at tick", d.tick)
    print("role,clip,source_frame,database_index,trajectory,facing,feet,linear_velocity,angular_velocity,stance,bias,total")
    _print_candidate(c, "winner", d.winner)
    _print_candidate(c, "continuation", d.continuation)
    _print_candidate(c, "alternative", d.alternative)
    c.app.shutdown()


def capture(folder: String, matching: Bool, every_frame: Bool) raises:
    if not make_dirs(folder):
        raise Error("Could not create the course capture directory")
    var c = Course(matching)
    var tile = SimdCanvas(480, 320)
    var sheet = SimdCanvas(480 * 4, 320 * 2)
    var label = String("match") if matching else String("graph")
    var trace = String("tick,clip,clip_frame,x,y,z,desired_forward_800m,left_lock,right_lock,left_x,left_y,left_z,right_x,right_y,right_z,ground,deck_x,deck_z,deck_qy,deck_qw,actor_yaw_deg,desired_facing_800_rad\n")
    var sample = 0
    var checkpoint_ticks = [105, 132, 244, 316]
    var saw_fall = False
    var saw_getup = False
    var walk_end = Vec3()
    var stop_origin = Vec3()
    var stop_distance = Float32(0)
    var response_ticks = -1
    var drift = Float32(0)
    var deck_anchors = [Vec3(), Vec3()]
    var supported_ticks = 0
    var pivot_start_yaw = Float32(0)
    var pivot_net_yaw = Float32(0)
    var pivot_first_time = Float32(-1)
    var pivot_last_time = Float32(-1)
    var pivot_rewinds = 0
    var pivot_anchor = Vec3()
    var pivot_drift = Float32(0)
    var pivot_ticks = 0
    var pivot_duration = c.model.animation_duration(c.action_clips[3])
    var start = actor_placement(c.app, c.actor).position
    for i in range(465):
        if i == 232:
            pivot_start_yaw = actor_placement(c.app, c.actor).yaw
        _step_replay(c, i)
        var place = actor_placement(c.app, c.actor)
        ref ac = c.app.world.get_ptr[AnimatorComponent](c.actor).value()[]
        ref kc = c.app.world.get_ptr[CharacterControllerComponent](c.actor).value()[].controller
        ref sk = c.app.world.get_ptr[SkeletonComponent](c.actor).value()[].skeleton
        ref deck = c.app.resources.get_ptr[PhysicsWorld]().value()[].bodies[c.deck]
        ref rc = c.app.world.get_ptr[RagdollComponent](c.actor).value()[].state_machine
        saw_fall = saw_fall or rc.state == RAGDOLL_SM_LIMP
        saw_getup = saw_getup or rc.recovery_count > 0
        if i >= 232 and i < 279:
            # Read the actor's achieved world heading, including any yaw on exit.
            pivot_net_yaw = place.yaw - pivot_start_yaw
            if pivot_net_yaw > 180:
                pivot_net_yaw -= 360
            elif pivot_net_yaw < -180:
                pivot_net_yaw += 360
            if ac.animator.clip_of_state() == c.action_clips[3]:
                if pivot_first_time < 0:
                    pivot_first_time = ac.animator.clip_time
                if ac.animator.clip_time < pivot_last_time - Float32(1e-6):
                    pivot_rewinds += 1
                pivot_last_time = ac.animator.clip_time
        if i == 113:
            walk_end = place.position
        var left = place.to_world(sk.global_position(ac.foot_plant.legs[0].ankle))
        var right = place.to_world(sk.global_position(ac.foot_plant.legs[1].ankle))
        var deck_inverse = deck.rot.conjugate()
        if i >= 232 and i < 279 and ac.animator.clip_of_state() == c.action_clips[3]:
            if not ac.foot_plant.legs[1].locked or kc.ground_body_id != deck.id:
                raise Error(label + " lost the pivot support at tick " + String(c.tick))
            var foot = deck_inverse.rotate(right - deck.pos)
            if pivot_ticks == 0:
                pivot_anchor = foot
            pivot_drift = max(pivot_drift, (foot - pivot_anchor).length())
            pivot_ticks += 1
        if i >= 12 and response_ticks < 0 and place.position.z - start.z > 0.01:
            response_ticks = i - 11
        if i == 113:
            stop_origin = deck_inverse.rotate(place.position - deck.pos)
        if i == 143:
            stop_distance = (deck_inverse.rotate(place.position - deck.pos) - stop_origin).length()
        if i >= 148 and i < 228:
            var soles = [left, right]
            if ac.foot_plant.legs[0].locked and ac.foot_plant.legs[1].locked and kc.ground_body_id == deck.id:
                supported_ticks += 1
            for j in range(2):
                var sole = deck_inverse.rotate(soles[j] - deck.pos)
                if i == 148:
                    deck_anchors[j] = sole
                drift = max(drift, (sole - deck_anchors[j]).length())
        if rc.recovery_count > 0 and rc.getup_time > 0.5 and rc.state != RAGDOLL_SM_ANIMATED:
            if min(left.y, right.y) < deck.pos.y + deck.half_extents.y - 0.03:
                raise Error("A recovery foot went below the supporting deck at tick " + String(c.tick))
        trace += String(c.tick) + "," + clip_label(c) + "," + String(source_frame(c))
        trace += "," + String(place.position.x) + "," + String(place.position.y) + "," + String(place.position.z) + "," + String(c.intent.traj_pos_800.z)
        trace += "," + String(Int(ac.enable_foot_plant and ac.foot_plant.legs[0].locked)) + "," + String(Int(ac.enable_foot_plant and ac.foot_plant.legs[1].locked))
        for p in [left, right]:
            trace += "," + String(p.x) + "," + String(p.y) + "," + String(p.z)
        trace += "," + String(kc.ground_body_id) + "," + String(deck.pos.x) + "," + String(deck.pos.z) + "," + String(deck.rot.y) + "," + String(deck.rot.w) + "," + String(place.yaw) + "," + String(c.intent.traj_facing_800) + "\n"
        # Pick a mid-recovery frame from the actual classified get-up.
        if sample == 3 and rc.recovery_count > 0 and rc.getup_time >= 0.6:
            checkpoint_ticks[3] = c.tick
        var snapshot = sample < 4 and (sample != 3 or rc.getup_time >= 0.6) and checkpoint_ticks[sample] == c.tick
        if snapshot or every_frame:
            for view in range(2):
                draw(c, tile, view == 0)
                if snapshot:
                    for y in range(tile.height):
                        for x in range(tile.width):
                            sheet.pixels[(view*tile.height+y)*sheet.width + sample*tile.width+x] = tile.pixels[y*tile.width+x]
                if every_frame:
                    var path = folder + "/" + label + "-" + String(view) + "-" + String(c.tick) + ".png"
                    if not write_png(path, tile.pixels, tile.width, tile.height):
                        raise Error("Could not write " + path)
            if snapshot:
                sample += 1
    var recovered = c.app.world.get_ptr[RagdollComponent](c.actor).value()[].state_machine.state == RAGDOLL_SM_ANIMATED
    c.app.shutdown()
    var path = folder + "/" + label
    var bytes = List[UInt8]()
    for byte in trace.as_bytes():
        bytes.append(byte)
    if not write_bytes(path + ".csv", bytes) or not write_png(path + ".png", sheet.pixels, sheet.width, sheet.height):
        raise Error("Failed to write capture to " + path)
    if walk_end.z - start.z < 0.6 or not saw_fall or not saw_getup or not recovered or sample != 4:
        raise Error("Incomplete " + label + " course: travel=" + String(walk_end.z-start.z) + " captures=" + String(sample) + " recovered=" + String(recovered))
    if supported_ticks != 80 or drift > 0.02:
        raise Error(label + " deck stance failed: supported ticks=" + String(supported_ticks) + " drift_m=" + String(drift))
    if abs(pivot_net_yaw - 90) > 0.25 or abs(pivot_first_time) > 1e-5 or abs(pivot_last_time - pivot_duration) > 1e-5 or pivot_rewinds > 0:
        raise Error(label + " pivot failed: yaw_deg=" + String(pivot_net_yaw) + " first_s=" + String(pivot_first_time) + " last_s=" + String(pivot_last_time) + " rewinds=" + String(pivot_rewinds))
    if pivot_drift > 0.02:
        raise Error(label + " planted pivot ankle drifted " + String(pivot_drift) + " m")
    print(label, "pivot_yaw_deg =", pivot_net_yaw, "pivot_first_s =", pivot_first_time, "pivot_last_s =", pivot_last_time, "pivot_rewinds =", pivot_rewinds, "pivot_ankle_drift_m =", pivot_drift)
    print(label, "response_ticks =", response_ticks, "stop_distance_m =", stop_distance, "planted_drift_m =", drift)
    print("Completed", label, "course. Forward travel", walk_end.z-start.z, "m. Capture", path + ".png")


def play() raises:
    var c = Course()
    var win = WaylandWindow()
    win.set_title("Flint | Character course")
    if not win.initialize(900, 560):
        raise Error("A display is required for interactive play. Use --capture for headless replay.")
    var canvas = SimdCanvas(900, 560)
    var input = InputState()
    var side = False
    var paused = False
    var last = get_time()
    var accumulator = Float64(0)
    var pivot = False
    var fall = False
    var rise = False
    while not win.close_requested:
        input.begin_frame()
        win.poll(input)
        if input.key_pressed(256):
            break
        if input.key_pressed(77):
            c.matching = not c.matching
            c.previous_action = -1
        if input.key_pressed(67):
            side = not side
        if input.key_pressed(32):
            paused = not paused
        pivot = pivot or input.key_pressed(80)
        fall = fall or input.key_pressed(70)
        rise = rise or input.key_pressed(71)
        var now = get_time()
        accumulator += min(0.1, now-last) if not paused else 0.0
        last = now
        if paused and input.key_pressed(46):
            accumulator = Float64(DT)
        while accumulator >= Float64(DT):
            var steer = Float32(Int(input.key_down(65)) - Int(input.key_down(68)))
            c.step(input.key_down(87), steer, pivot, fall, rise)
            pivot = False
            fall = False
            rise = False
            accumulator -= Float64(DT)
        draw(c, canvas, side)
        canvas.draw_text(12, canvas.height-20, "W walk   A/D steer   P pivot   F fall   G rise   M graph/match   C camera   Space pause   . step", 0xFFFFFF, 1)
        win.present(canvas.pixels, canvas.width, canvas.height)
        _ = external_call["usleep", Int32](UInt32(4000))
    win.close()
    c.app.shutdown()


def main() raises:
    var args = argv()
    if len(args) == 1:
        play()
    elif len(args) == 2 and (args[1] == "--capture" or args[1] == "--frames"):
        for matching in [False, True]:
            capture("build/character-course", matching, args[1] == "--frames")
    elif len(args) == 3 and args[1] == "--match-tick":
        inspect_match(Int(String(args[2])))
    else:
        raise Error("Usage: character-course [--capture|--frames|--match-tick TICK]")
