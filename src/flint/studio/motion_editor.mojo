"""Flint Studio motion editor. Open a versioned move preset from the project."""
from flint.studio.source_editor import EmbeddedIde
from flint.studio.motion_tools import (
    MotionHistory,
    MotionTools,
    QuitGuard,
    TRANSITION,
    CONTACT,
    TIMING,
    effect_sound,
    last_frame,
    nearest_frame,
    frame_joint,
    shake_offset,
    draw_dust,
)
from flint.anim.choreography import TRACK_FPS, cue_kind
from std.ffi import external_call
from std.math import sin, cos
from std.sys import argv
from flint.anim.choreography import MotionPlayer, MotionSequence
from flint.anim.pose import write_pose
from flint.anim.procedural import Placement
from flint.anim.skinning import SKIN_DQS, skin_mesh
from flint.assets.glb_native import load_native_glb
from flint.assets.image import write_png
from flint.assets.model import NativeModel
from flint.core.math import Vec3
from flint.platform.audio import NativeAudio, WavClip, load_wav
from flint.platform.input import InputState
from flint.platform.os import get_time, parent_directory
from flint.platform.wayland import WaylandWindow
from flint.render.simd_canvas import Camera3D, SimdCanvas

comptime W = 960
comptime H = 640
comptime BAR = 482


def relative_to(preset: String, asset: String) -> String:
    return asset if asset.startswith("/") else parent_directory(preset) + "/" + asset


def point(canvas: SimdCanvas, cam: Camera3D, position: Vec3) -> Tuple[Int, Int, Bool]:
    var p = cam.project(position)
    return (Int(p[0]), Int(p[1]), p[3] and p[0] >= 0 and p[0] < Float32(canvas.width) and p[1] >= 0 and p[1] < BAR)


def draw_mesh(mut canvas: SimdCanvas, cam: Camera3D, mut model: NativeModel, player: MotionPlayer, actor: Placement):
    write_pose(model, player.bones.pose)
    for i in range(len(model.meshes)):
        if model.meshes[i].skin < 0:
            continue
        var vertices = skin_mesh(model, i, SKIN_DQS)
        ref mesh = model.meshes[i]
        var positions = List[Vec3](capacity=len(vertices) // 8)
        for v in range(0, len(vertices), 8):
            positions.append(actor.to_world(Vec3(vertices[v], vertices[v + 1], vertices[v + 2])))
        for t in range(0, len(mesh.indices) - 2, 3):
            canvas.draw_tri_3d_lit(cam, positions[mesh.indices[t]], positions[mesh.indices[t + 1]],
                                   positions[mesh.indices[t + 2]], 0x9AB5CA)


trait MotionCamera:
    @staticmethod
    def view(actor: Placement, seconds: Float32) -> Camera3D:
        ...


struct FollowCamera(MotionCamera):
    @staticmethod
    def view(actor: Placement, seconds: Float32) -> Camera3D:
        var facing = Placement(actor.position, Vec3(1, 1, 1), actor.yaw)
        return Camera3D(W, H, facing.to_world(Vec3(0.55, 1.9, -4.3)), facing.to_world(Vec3(0.4, 1.1, 0.6)), 55)


def frame[C: MotionCamera](mut canvas: SimdCanvas, mut model: NativeModel, seq: MotionSequence, mut player: MotionPlayer,
          phase: Float32, mesh: Bool, status: String, follow: Bool):
    var tools = MotionTools()
    render_motion[C](canvas, model, seq, player, phase, mesh, status, follow, tools, False)


def render_motion[C: MotionCamera](mut canvas: SimdCanvas, mut model: NativeModel, seq: MotionSequence, mut player: MotionPlayer,
          phase: Float32, mesh: Bool, status: String, follow: Bool, mut tools: MotionTools, interactive: Bool,
          dirty: Bool = False):
    var wall = Placement(seq.ledge, Vec3(1, 1, 1), seq.yaw)
    var seconds = phase * player.duration(model, seq)
    var dust: List[Tuple[Vec3,Float32]] = []
    for event in seq.events:
        var age = seconds - Float32(event.frame)/TRACK_FPS
        if event.kind == 2 and age >= 0 and age < 0.32:
            var origin = player.sample(model,seq,wall,Float32(event.frame)/TRACK_FPS).position
            dust.append((origin,age))
    var actor = player.sample(model, seq, wall, seconds)
    var shot = seq.camera(phase, wall)
    var cam = Camera3D(W, H, shot[0], shot[1], 50)
    if follow:
        cam = C.view(actor, phase * player.duration(model, seq))
    var shake = shake_offset(seq, seconds)
    cam.eye = cam.eye + shake
    cam.target = cam.target + shake
    tools.view = cam
    canvas.clear()
    for i in range(-5, 6):
        var a = Float32(i)
        canvas.draw_line_3d(cam, Vec3(a, 0, -5), Vec3(a, 0, 5), 0x394854)
        canvas.draw_line_3d(cam, Vec3(-5, 0, a), Vec3(5, 0, a), 0x394854)
    var lip_a = wall.to_world(Vec3(-0.8, 0, 0))
    var lip_b = wall.to_world(Vec3(0.8, 0, 0))
    canvas.draw_line_3d(cam, lip_a, lip_b, 0xE0A457, 3)
    canvas.draw_line_3d(cam, lip_a, lip_a - Vec3(0, 1, 0), 0x82694D)
    canvas.draw_line_3d(cam, lip_b, lip_b - Vec3(0, 1, 0), 0x82694D)
    if mesh:
        draw_mesh(canvas, cam, model, player, actor)
    else:
        for i in range(len(model.nodes)):
            var parent = player.bones.parents[i]
            if parent >= 0:
                canvas.draw_line_3d(cam, actor.to_world(player.bones.global_position(parent)),
                                    actor.to_world(player.bones.global_position(i)), 0xABC5D6, 2)
    for joint in player.markers:
        var p = point(canvas, cam, actor.to_world(player.bones.global_position(joint)))
        if p[2]:
            canvas.draw_rect(p[0] - 3, p[1] - 3, 7, 7, 0xFF69E5CE)
    if tools.tab == CONTACT and player.markers:
        tools.shown_joint = actor.to_world(player.bones.global_position(player.markers[tools.marker]))
        var marker = point(canvas, cam, tools.shown_joint)
        if marker[2]:
            canvas.draw_rect(marker[0]-4,marker[1]-4,9,9,0xFFFFAE51)
        if tools.pinned:
            canvas.draw_line_3d(cam, tools.anchor, tools.shown_joint, 0xFFED8269, 2)
            var pin = point(canvas, cam, tools.anchor)
            if pin[2]:
                canvas.draw_rect(pin[0]-7,pin[1]-1,15,3,0xFFFFE18A)
                canvas.draw_rect(pin[0]-1,pin[1]-7,3,15,0xFFFFE18A)
    for particle in dust:
        draw_dust(canvas,cam,particle[0],particle[1])
    var target = point(canvas, cam, wall.position)
    if target[2]:
        canvas.draw_rect(target[0] - 7, target[1], 15, 2, 0xFFFFC455)
        canvas.draw_rect(target[0], target[1] - 7, 2, 15, 0xFFFFC455)
    canvas.draw_rect(0, BAR - 8, W, H - BAR + 8, 0xFF15202E)
    canvas.draw_rect(16, BAR + 8, W - 32, 6, 0xFF546472)
    var span = Float32(W - 32)
    var duration = player.duration(model, seq)
    var a = model.animations[player.first].duration
    var b = model.animations[player.second].duration
    var begin = Int(span * (duration - b) / duration)
    canvas.draw_rect(16 + begin, BAR + 8, max(1, Int(span * a / duration) - begin), 6, 0xFF699A9C)
    canvas.draw_rect(16 + Int(span * seq.cue), BAR, 3, 20, 0xFFFFBC64)
    canvas.draw_rect(16 + Int(span * phase), BAR + 3, 3, 16, 0xFFFFFFFF)
    canvas.draw_text(16, BAR + 26, ("* " if dirty else "") + seq.first + " > " + seq.second + "    " + String(Int(1000 * phase * duration)) + " MS   BLEND " + String(Int(seq.overlap * 100)) + "%", 0xFFFFFFFF, 2)
    canvas.draw_text(16, BAR + 51, "SPACE PLAY   ARROWS STEP   DRAG TO SEEK   M MESH/RIG   G GAME CAMERA", 0xFFBDCFDC, 2)
    canvas.draw_text(16, BAR + 72, "1/2 CLIPS   -/= BLEND   T CUE   UP/DOWN LEDGE   R TURN   S SAVE", 0xFFBDCFDC, 2)
    canvas.draw_text(16, BAR + 93, "TAB CAMERA KEY   A/D ORBIT   W/X HEIGHT   CTRL+Z/Y UNDO/REDO", 0xFFBDCFDC, 2)
    canvas.draw_text(16, BAR + 120, status, 0xFFFFC455, 2)
    if interactive:
        tools.hud(canvas, model, seq, player, nearest_frame(seconds), BAR, status)


def next_clip(model: NativeModel, index: Int) -> Int:
    for step in range(1, len(model.animations) + 1):
        var i = (index + step) % len(model.animations)
        if model.animations[i].duration > 0:
            return i
    return index


def timing_click() -> WavClip:
    var clip = WavClip()
    clip.channels = 1
    clip.sample_rate = 24000
    clip.valid = True
    for i in range(1200):
        clip.samples.append(Int16(6000 * (1 - Float32(i) / 1200) * sin(Float32(i) * 0.22)))
    return clip^


def clip_index(model: NativeModel, name: String, fallback: Int) -> Int:
    for i in range(len(model.animations)):
        if model.animations[i].name == name and model.animations[i].duration > 0:
            return i
    return fallback


def contact_self_check(model: NativeModel, seq: MotionSequence, mut player: MotionPlayer) raises -> Int:
    """Pin the first marked joint at frame 0 and check the Contact Inspector's first slip
    against a brute-force scan of every track frame: the inspector must report the first
    frame drifting more than 20 mm, and `-1` exactly when no frame drifts. Returns that
    frame (or -1); also -1 when the preset marks no joints."""
    if not player.markers:
        return -1
    var inspector = MotionTools()
    var anchor = frame_joint(player, model, seq, 0, 0)
    inspector.pin(model, seq, player, 0, anchor)
    if not inspector.pinned:
        raise Error("Contact Inspector did not pin the marked joint")
    var expected = -1
    for i in range(last_frame(player.duration(model, seq)) + 1):
        if frame_joint(player, model, seq, i, 0).distance_to(anchor) > 0.02:
            expected = i
            break
    if inspector.first_slip != expected:
        raise Error(
            "Contact Inspector reported first slip F" + String(inspector.first_slip)
            + " but the joint first drifts past 20 mm at F" + String(expected)
        )
    return expected


def verify(model: NativeModel, seq: MotionSequence, mut player: MotionPlayer) raises:
    var total = player.duration(model, seq)
    for wall in [Placement(seq.ledge), Placement(seq.ledge + Vec3(3, 2, -1), Vec3(1, 1, 1), 80)]:
        var end = player.sample(model, seq, wall, total)
        if player.contact >= 0 and end.to_world(player.bones.global_position(player.contact)).distance_to(wall.position) > 0.0001:
            raise Error("Contact missed the placed ledge")
        if player.wrist >= 0:
            var local = wall.to_model(end.to_world(player.bones.global_position(player.wrist)))
            var miss = Vec3(0, local.y, local.z).length()
            if miss > 0.02:
                raise Error("Supporting wrist misses ledge by " + String(miss) + " metres")
        if player.contact >= 0:
            var near = wall.to_world(Vec3(0, 0, -0.1))
            var toward = wall.vector_to_world(Vec3(0, -1, 1))
            if not player.can_catch(wall, near, toward, 0.2) or player.can_catch(wall, near, -toward, 0.2) or player.can_catch(wall, near, toward, 0.01):
                raise Error("Catch entry failed approach or reach test")
        var entry = wall.position + Vec3(1, -2, 0)
        if player.sample(model, seq, wall, 0, entry).position.distance_to(entry) > 0.0001:
            raise Error("Move did not start at the actor's position")
        var first = player.sample(model, seq, wall, total * 0.6)
        var p = player.bones.global_position(max(0, player.contact))
        _ = player.sample(model, seq, wall, 0)
        var again = player.sample(model, seq, wall, total * 0.6)
        if again.position.distance_to(first.position) > 0.0001 or p.distance_to(player.bones.global_position(max(0, player.contact))) > 0.0001:
            raise Error("Seeking changed the sampled pose")
    if not seq.cue_crossed(-1, seq.cue) or seq.cue_crossed(seq.cue, 1):
        raise Error("Cue must fire once at the crossing")
    var edited = seq.copy()
    edited.events.clear()
    edited.set_event(2, 0)
    edited.set_event(0, 2)
    edited.set_event(2, 0)
    if len(edited.events) != 2 or not edited.events[0].crossed(-1, 0):
        raise Error("Event order, duplicate insertion, or frame zero failed")
    var seconds = Float32(2)/TRACK_FPS
    var ev = edited.events[1]
    if not ev.crossed(0,seconds) or ev.crossed(seconds,seconds+1) or ev.crossed(seconds,0):
        raise Error("A timing event must fire once on its forward crossing")
    _ = contact_self_check(model, seq, player)
    print("Verified wall placements, reverse seek, contact drift and timing crossings.")


def run_motion_editor[C: MotionCamera]() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("studio --motion PRESET.json [--check | --shot FRAME.png PHASE | --game-shot FRAME.png PHASE]")
    var path = String(args[1])
    var seq = MotionSequence(path)
    var original = seq.copy()
    var tools = MotionTools()
    var model = load_native_glb(relative_to(path, seq.model_path))
    var player = MotionPlayer(model, seq)
    var baseline = MotionPlayer(model, original)
    var canvas = SimdCanvas(W, H)
    if len(args) == 3 and args[2] == "--check":
        if seq.sound and not load_wav(relative_to(path, seq.sound)).valid:
            raise Error("Could not load WAV: " + seq.sound)
        verify(model, seq, player)
        return
    if len(args) == 5 and (args[2] == "--shot" or args[2] == "--game-shot"):
        var t = Float32(Float64(String(args[4])))
        if not (t >= 0 and t <= 1):
            raise Error("Shot phase must be in [0, 1]")
        frame[C](canvas, model, seq, player, t, True, "CONTACT PREVIEW", args[2] == "--game-shot")
        if not write_png(String(args[3]), canvas.pixels, W, H):
            raise Error("Could not write shot")
        return
    if len(args) == 3 and args[2] == "--transition":
        tools.tab = TRANSITION
    elif len(args) == 3 and args[2] == "--contact":
        tools.tab = CONTACT
    elif len(args) == 3 and args[2] == "--timing":
        tools.tab = TIMING
    elif len(args) != 2:
        raise Error("Expected --check or --shot FRAME.png PHASE")
    var win = WaylandWindow()
    win.set_title("Flint Studio - Motion")
    if not win.initialize(W, H):
        raise Error("Motion editor needs a display. Use --shot for headless preview.")
    var audio = NativeAudio()
    var sound = audio.initialize()
    var clip = timing_click()
    if seq.sound:
        clip = load_wav(relative_to(path, seq.sound))
        if not clip.valid:
            raise Error("Could not load WAV: " + seq.sound)
    var sound_id = len(audio.clips)
    audio.clips.append(clip^)
    var effects = len(audio.clips)
    audio.clips.append(effect_sound(0))
    audio.clips.append(effect_sound(1))
    audio.music_loaded = False
    var inp = InputState()
    var phase = Float32(0)
    var playing = False
    var mesh = True
    var follow = False
    var end_key = False
    var redraw = True
    var last = get_time()
    var status = String("CAMERA START KEY" if tools.tab == 0 else "")
    var ide = EmbeddedIde()
    var history = MotionHistory(seq)
    var quit = QuitGuard()
    comptime UNSAVED = "UNSAVED CHANGES: ESC OR CLOSE AGAIN TO DISCARD, S TO SAVE"
    while True:
        var now = get_time()
        var dt = Float32(now - last)
        last = now
        inp.begin_frame()
        win.poll(inp)
        var asked = False
        if win.close_requested:
            win.close_requested = False
            if quit.request(history.dirty(seq)):
                break
            asked = True
            status = UNSAVED
            redraw = True
        if ide.present(win, inp):
            redraw = True
            last = get_time()
            continue
        if inp.key_pressed(256):
            if quit.request(history.dirty(seq)):
                break
            status = UNSAVED
            redraw = True
        elif quit.armed and not asked and (inp.any_key_pressed() or inp.mouse_pressed(0)):
            quit.cancel()
            status = ""
            redraw = True
        var x = Int(inp.last_mouse_x)
        var y = Int(inp.last_mouse_y)
        var click = inp.mouse_pressed(0)
        var prior_duration = baseline.duration(model,original) if tools.original else player.duration(model,seq)
        if click and y >= 0 and y < 31:
            tools.tab = max(0,min(3,x*4//W))
            tools.original = False
            tools.pinned = False
            playing = False
            status = ""
            redraw = True
        if tools.tab == TRANSITION:
            var controls = click and BAR+40 <= y < BAR+73
            if inp.key_pressed(ord("A")) or (controls and 525 <= x < 655):
                tools.original = not tools.original
            if inp.key_pressed(ord("Q")) or (controls and 655 <= x < 750):
                tools.quarter = not tools.quarter
            if inp.key_pressed(ord("L")) or (controls and 750 <= x < 935):
                tools.loop_join = not tools.loop_join
            if click and 205 <= x <= 525 and BAR+40 <= y <= BAR+74:
                tools.drag_blend = True
            if not inp.mouse_down(0):
                tools.drag_blend = False
            if tools.drag_blend:
                seq.overlap = max(Float32(0.05),min(Float32(1),Float32(x-215)/294))
                tools.original = False
                redraw = True
        if tools.tab == TRANSITION or tools.tab == 0:
            if inp.key_pressed(ord("1")):
                player.first = next_clip(model, player.first)
                seq.first = model.animations[player.first].name
                tools.original = False
            if inp.key_pressed(ord("2")):
                player.second = next_clip(model, player.second)
                seq.second = model.animations[player.second].name
                tools.original = False
            if inp.key_pressed(ord("-")) or inp.key_pressed(ord("=")):
                seq.overlap = max(Float32(0.05), min(Float32(1), seq.overlap + Float32(0.05 if inp.key_pressed(ord("=")) else -0.05)))
                tools.original = False
        var duration = baseline.duration(model,original) if tools.original else player.duration(model,seq)
        if duration != prior_duration:
            phase = min(Float32(1),phase*prior_duration/duration)
        var previous = phase
        var from_seconds = phase*duration
        var low = Float32(0)
        var high = Float32(1)
        if tools.tab == TRANSITION and tools.loop_join:
            var first = baseline.first if tools.original else player.first
            var second = baseline.second if tools.original else player.second
            var overlap = original.overlap if tools.original else seq.overlap
            var a = model.animations[first].duration
            low = max(Float32(0),(a-min(a,model.animations[second].duration)*overlap-0.12)/duration)
            high = min(Float32(1),(a+0.12)/duration)
        if inp.key_pressed(32):
            playing = not playing
            if playing and (phase < low or phase >= high):
                phase = low
            from_seconds = phase*duration
            if playing and phase == 0:
                from_seconds = -1
        if inp.mouse_down(0) and BAR-8 <= y <= BAR+22:
            phase = Float32(x-16)/Float32(W-32)
            playing = False
        if inp.key_pressed(263) or inp.key_pressed(262):
            var frame_index = nearest_frame(phase*duration) + (1 if inp.key_pressed(262) else -1)
            phase = Float32(frame_index)/(TRACK_FPS*duration)
            playing = False
        if playing:
            var delta = min(Float32(0.12), dt) * (Float32(0.25) if tools.tab == TRANSITION and tools.quarter else Float32(1))
            phase = max(low, phase) + delta/duration
            if sound:
                ref viewed = original if tools.original else seq
                if viewed.cue_crossed(from_seconds/duration,min(phase,high)):
                    audio.play_sound(sound_id)
                for ev in viewed.events:
                    if ev.kind < 2 and ev.crossed(from_seconds,min(phase,high)*duration):
                        audio.play_sound(effects + ev.kind)
            if phase >= high:
                if tools.tab == TRANSITION and tools.loop_join:
                    phase = low
                else:
                    phase = high
                    playing = False
        phase = max(Float32(0), min(Float32(1), phase))
        var index = nearest_frame(phase*duration)
        if tools.tab == CONTACT and player.markers:
            if inp.key_pressed(ord("J")):
                tools.marker = (tools.marker+1)%len(player.markers)
                tools.pinned = False
            if inp.key_pressed(ord("P")) or inp.key_pressed(ord("L")) or (click and 31 <= y < BAR-8):
                var target = frame_joint(player,model,seq,index,tools.marker)
                if inp.key_pressed(ord("L")):
                    target = seq.ledge
                elif click:
                    target = tools.clicked_point(x,y)
                tools.pin(model,seq,player,index,target)
                phase = min(Float32(1),Float32(index)/(duration*TRACK_FPS))
                playing = False
                redraw = True
        if tools.tab == TIMING:
            var final_frame = last_frame(duration)
            index = min(index,final_frame)
            if click and 140 <= x < W-16 and BAR+28 <= y < BAR+100:
                tools.kind = (y-BAR-28)//18
                index = ((x-140)*final_frame+(W-157)//2)//(W-157)
                phase = Float32(index)/(TRACK_FPS*duration)
                seq.set_event(index,tools.kind)
                status = "Added "+cue_kind(tools.kind)+" at F"+String(index)
                playing = False
                redraw = True
            for k in range(4):
                if inp.key_pressed(ord("1")+k):
                    tools.kind = k
            if inp.key_pressed(257):
                seq.set_event(index,tools.kind)
                status = "Added "+cue_kind(tools.kind)+" at F"+String(index)
            if inp.key_pressed(261):
                for i in range(len(seq.events)):
                    if seq.events[i].frame == index and seq.events[i].kind == tools.kind:
                        _ = seq.events.pop(i)
                        status = "Removed event"
                        break
            if inp.key_pressed(ord("[")) or inp.key_pressed(ord("]")):
                var forward = inp.key_pressed(ord("]"))
                for j in range(len(seq.events)):
                    var ev = seq.events[j if forward else len(seq.events)-j-1]
                    if (ev.frame>index if forward else ev.frame<index) and ev.frame<=final_frame:
                        phase = min(Float32(1), Float32(ev.frame)/(duration*TRACK_FPS))
                        tools.kind = ev.kind
                        playing = False
                        break
        if inp.key_pressed(ord("G")):
            follow = not follow
            status = "GAME CAMERA" if follow else "AUTHORED CAMERA"
        if inp.key_pressed(ord("M")):
            mesh = not mesh
        if tools.tab == 0:
            if inp.key_pressed(ord("T")):
                seq.cue = phase
            if inp.key_pressed(265) or inp.key_pressed(264):
                seq.ledge.y += Float32(0.25 if inp.key_pressed(265) else -0.25)
            if inp.key_pressed(ord("R")):
                seq.yaw = (seq.yaw + 15) % 360
            if inp.key_pressed(258):
                follow = False
                end_key = not end_key
                phase = 1 if end_key else 0
                playing = False
                status = "CAMERA END KEY" if end_key else "CAMERA START KEY"
            var eye = seq.eye_b if end_key else seq.eye_a
            var offset = eye - seq.aim
            var turn = Float32(0.12) * Float32(Int(inp.key_pressed(ord("D"))) - Int(inp.key_pressed(ord("A"))))
            if turn != 0:
                eye.x = seq.aim.x + offset.x * cos(turn) + offset.z * sin(turn)
                eye.z = seq.aim.z + offset.z * cos(turn) - offset.x * sin(turn)
            eye.y += Float32(0.1) * Float32(Int(inp.key_pressed(ord("W"))) - Int(inp.key_pressed(ord("X"))))
            if end_key:
                seq.eye_b = eye
            else:
                seq.eye_a = eye
        var ctrl = inp.key_down(341) or inp.key_down(345)
        var shift = inp.key_down(340) or inp.key_down(344)
        if ctrl and (inp.key_pressed(ord("Z")) or inp.key_pressed(ord("Y"))):
            var forward = inp.key_pressed(ord("Y")) or shift
            if history.redo(seq) if forward else history.undo(seq):
                player.first = clip_index(model, seq.first, player.first)
                player.second = clip_index(model, seq.second, player.second)
                tools.original = False
                tools.pinned = False
                tools.drag_blend = False
                status = "Redo" if forward else "Undo"
            else:
                status = "Nothing to redo" if forward else "Nothing to undo"
            playing = False
            redraw = True
        else:
            _ = history.track(seq, tools.drag_blend)
        if inp.key_pressed(ord("S")):
            try:
                if tools.original:
                    var chosen = seq.copy()
                    chosen.first = original.first
                    chosen.second = original.second
                    chosen.overlap = original.overlap
                    chosen.save(path)
                    seq = chosen^
                    player.first = baseline.first
                    player.second = baseline.second
                    tools.original = False
                else:
                    seq.save(path)
                history.mark_saved(seq)
                quit.cancel()
                status = "Saved motion preset"
            except err:
                status = String(err)
        if redraw or click or inp.any_key_pressed() or phase != previous:
            if tools.original:
                render_motion[C](canvas, model, original, baseline, phase, mesh, status, follow, tools, True, history.dirty(seq))
            else:
                render_motion[C](canvas, model, seq, player, phase, mesh, status, follow, tools, True, history.dirty(seq))
            redraw = False
        win.present(canvas.pixels, W, H)
        if sound:
            audio.update(0)
        var spare = 1.0 / 60.0 - (get_time() - now)
        if spare > 0:
            _ = external_call["usleep", Int32](UInt32(spare * 1000000))
    audio.close()
    ide.close()
    win.close()
