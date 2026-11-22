"""Studio movement prompt, preview and BVH export."""
from std.ffi import external_call
from std.math import sin, cos
from flint.core.math import Vec3
from flint.assets.image import write_png
from flint.ml.motionbricks import MbClip, MbXform, MbVec3, load_bvh, world_pose
from flint.platform.input import InputState
from flint.platform.os import file_size, get_time, make_dirs
from flint.platform.wayland import WaylandWindow
from flint.render.simd_canvas import Camera3D, SimdCanvas


def draw_pose(mut canvas: SimdCanvas, clip: MbClip, f: Int):
    var pose = world_pose(clip.skeleton, clip.frames[f], MbXform.identity())
    var p = pose[0].p
    var center = Vec3(p.x, 0.86, p.z)
    var cam = Camera3D(
        canvas.width, canvas.height, center + Vec3(2.55, 0.53, 3.0), center, 39
    )
    for n in range(-12, 13):
        var v = Float32(n) * 0.5
        canvas.draw_line_3d(cam, Vec3(-6, 0, v), Vec3(6, 0, v), 0x344652)
        canvas.draw_line_3d(cam, Vec3(v, 0, -6), Vec3(v, 0, 6), 0x344652)
    for j in range(1, len(pose)):
        if clip.skeleton.names[j] in ("LeftEye", "RightEye", "Jaw"):
            continue
        var parent = clip.skeleton.parents[j]
        var a = pose[parent].p
        var b = pose[j].p
        var color = 0x68CCC0 if clip.skeleton.names[j].startswith(
            "Left"
        ) else 0xE8C689
        canvas.draw_line_3d(
            cam, Vec3(a.x, a.y, a.z), Vec3(b.x, b.y, b.z), UInt32(color), 3
        )

    var head = clip.skeleton.joint_index("Head")
    if head >= 0:
        for plane in range(2):
            for i in range(24):
                var a = Float32(i) * 0.2617994
                var b = Float32(i + 1) * 0.2617994
                var u = 0.083 * cos(a)
                var v = 0.083 * cos(b)
                var start = MbVec3(
                    u if plane == 0 else 0,
                    0.07 + 0.115 * sin(a),
                    u if plane == 1 else 0,
                )
                var end = MbVec3(
                    v if plane == 0 else 0,
                    0.07 + 0.115 * sin(b),
                    v if plane == 1 else 0,
                )
                start = pose[head].p + pose[head].q.rotate(start)
                end = pose[head].p + pose[head].q.rotate(end)
                canvas.draw_line_3d(
                    cam,
                    Vec3(start.x, start.y, start.z),
                    Vec3(end.x, end.y, end.z),
                    0xDFDAD0,
                    1,
                )


def save_sheet(path: String, output: String) raises:
    var clip = load_bvh(path)
    if clip.num_frames() < 2 or clip.skeleton.num_joints() != 30:
        raise Error("Expected a SOMA30 take")
    var tile = SimdCanvas(400, 320)
    var sheet = SimdCanvas(1200, 320)
    for k in range(3):
        var f = (clip.num_frames() - 1) * k // 2
        tile.clear()
        draw_pose(tile, clip, f)
        tile.draw_text(12, 12, "Frame " + String(f + 1), 0xE8ECEE, 2)
        for y in range(320):
            for x in range(400):
                sheet.pixels[y * 1200 + k * 400 + x] = tile.pixels[y * 400 + x]
    if not write_png(output, sheet.pixels, 1200, 320):
        raise Error("Could not write preview image")


def run_editor(var binary: String, dir: String) raises:
    var win = WaylandWindow()
    win.set_title("Flint Studio - Kimodo")
    if not win.initialize(900, 560):
        raise Error(
            "Kimodo editor needs a display. Pass a prompt and an output BVH for"
            " headless generation."
        )
    var input = InputState()
    var canvas = SimdCanvas(900, 560)
    var prompt = String("A person walks forward.")
    var select_all = True
    var status = String("Enter a movement prompt, then generate a BVH take.")
    var has_weights = (
        file_size(dir + "/denoiser.onnx") > 0
        and file_size(dir + "/model_int4_b128_fp32act_s64.onnx_data") > 0
    )
    if not has_weights:
        status = (
            "Install the weights with bash tools/kimodo-models.sh, then reopen."
        )
    var clip = MbClip()
    var output = String()
    var worker = Int32(-1)
    var take = 0
    var started = get_time()
    var pid = external_call["getpid", Int32]()
    while not win.close_requested:
        input.begin_frame()
        win.poll(input)
        if input.key_pressed(256):
            break
        var click = input.mouse_pressed(0)
        var generate = (
            click
            and input.last_mouse_x >= 722
            and input.last_mouse_y >= 458
            and input.last_mouse_y < 501
        )
        if worker <= 0:
            if input.key_pressed(65) and (
                input.key_down(341) or input.key_down(345)
            ):
                select_all = True
            elif input.key_pressed(259):
                if select_all:
                    prompt = ""
                elif prompt.byte_length():
                    var shortened = String(
                        prompt[byte = 0 : prompt.byte_length() - 1]
                    )
                    prompt = shortened^
                select_all = False
            elif input.text_input_utf8 and not input.key_pressed(257):
                var chars = List[UInt8]()
                for b in input.text_input_utf8.as_bytes():
                    if b >= 32 and b < 127:
                        chars.append(b)
                if len(chars):
                    if select_all:
                        prompt = ""
                    if prompt.byte_length() + len(chars) <= 100:
                        prompt += String(unsafe_from_utf8=chars)
                    select_all = False
            if (generate or input.key_pressed(257)) and has_weights and prompt:
                if not make_dirs("build/kimodo-takes"):
                    status = "Could not create build/kimodo-takes"
                else:
                    take += 1
                    output = (
                        "build/kimodo-takes/take-"
                        + String(pid)
                        + "-"
                        + String(take)
                        + ".bvh"
                    )
                    worker = external_call["fork", Int32]()
                    if worker == 0:
                        _ = external_call["close_range", Int32](
                            UInt32(3), UInt32(0xFFFFFFFF), UInt32(0)
                        )
                        _ = external_call["execl", Int32](
                            binary.as_c_string_span(),
                            binary.as_c_string_span(),
                            prompt.as_c_string_span(),
                            output.as_c_string_span(),
                            Int(0),
                        )
                        external_call["_exit", NoneType](Int32(127))
                    status = (
                        "Generating..." if worker
                        > 0 else "Could not launch generator"
                    )
                    select_all = False
        else:
            var code = Int32(0)
            if (
                external_call["waitpid", Int32](
                    worker, Pointer(to=code), Int32(1)
                )
                == worker
            ):
                worker = -1
                if code == 0:
                    clip = load_bvh(output)
                    started = get_time()
                    status = "Saved " + output
                else:
                    status = "Generation failed. See the Studio console."
        canvas.clear()
        if clip.num_frames():
            draw_pose(
                canvas,
                clip,
                Int((get_time() - started) * 30) % clip.num_frames(),
            )
        canvas.draw_text(22, 20, "Kimodo", 0xF2E7D2, 3)
        canvas.draw_rect(0, 418, 900, 142, 0xFF16232F)
        canvas.draw_text(22, 430, "Movement prompt", 0xCADDDC, 2)
        canvas.draw_rect(
            22, 458, 685, 43, UInt32(0xFF35506A if select_all else 0xFF213342)
        )
        var visible = max(0, prompt.byte_length() - 54)
        canvas.draw_text(32, 472, String(prompt[byte=visible:]), 0xEDF5F6, 2)
        canvas.draw_rect(
            722,
            458,
            155,
            43,
            UInt32(0xFF285E60 if has_weights and worker <= 0 else 0xFF424C53),
        )
        canvas.draw_text(744, 471, "Generate", 0xFFFFFF, 2)
        canvas.draw_text(22, 520, status, 0xD0DCDD, 1)
        win.present(canvas.pixels, 900, 560)
        _ = external_call["usleep", Int32](UInt32(16000))
    if worker > 0:
        _ = external_call["kill", Int32](worker, Int32(15))
        var code = Int32(0)
        _ = external_call["waitpid", Int32](worker, Pointer(to=code), Int32(0))
    win.close()
