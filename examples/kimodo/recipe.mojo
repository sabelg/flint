"""Compose, capture, constrain and export native Studio motion recipes."""
from flint.assets.json import JsonDocument, parse_json
from flint.anim.soma_retarget import mapped_bone
from flint.assets.glb_native import load_native_glb
from flint.assets.glb_take import append_take_glb
from flint.assets.image import write_png
from flint.anim.take_optimize import optimize_bake
from flint.anim.take_constraints import (
    check_interval,
    clean_stances,
    position,
    pin_joint,
    pin_frame,
    warp_contact,
    follow_path,
    mb,
)
from flint.core.math import Vec3
from flint.ml.motionbricks import MbClip, MbXform, world_pose
from flint.platform.os import read_bytes, env_now, file_size, remove_file
from flint.render.simd_canvas import SimdCanvas, Camera3D
from beats import timeline, prompt_take
from capture import capture_take
from character import CharacterPreview, character_sheet
from editor import save_sheet
from recipe_io import string, num, integer, vector, field, relative
from take_files import load_take, save_new


def document(path: String) raises -> JsonDocument:
    var d = parse_json(read_bytes(path))
    if not d.valid or d.root < 0 or d.nodes[d.root].kind != 5:
        raise Error("Invalid motion recipe JSON")
    if integer(d, d.root, "version", 0) != 1:
        raise Error("Expected motion recipe version 1")
    return d^


def character_path(d: JsonDocument, file: String) raises -> String:
    var path = string(d, d.root, "character")
    if path:
        return relative(file, path)
    var env = env_now("FLINT_TAKE_CHARACTER")
    return env if env else String("assets/mozu.glb")


def bone_map(
    d: JsonDocument, node_index: Int = -1
) raises -> Dict[String, String]:
    var out = Dict[String, String]()
    var node = field(d, d.root if node_index < 0 else node_index, "bones")
    if node >= 0:
        if d.nodes[node].kind != 5:
            raise Error("bones must map SOMA names to skin bone names")
        for key in d.nodes[node].keys:
            var name = string(d, node, key)
            if not name:
                raise Error("Empty target bone for " + key)
            out[key] = name.copy()
    return out^


def apply_edits(mut clip: MbClip, d: JsonDocument, arr: Int) raises:
    for i in range(d.array_size(arr)):
        var n = d.at(arr, i)
        var op = string(d, n, "op")
        var a = integer(d, n, "start", 0)
        var b = integer(d, n, "end", clip.num_frames() - 1)
        check_interval(clip, a, b)
        var joint = string(d, n, "joint")
        if op == "pin" or op == "plant":
            var j = clip.skeleton.joint_index(joint)
            if j < 0:
                raise Error("Unknown pin joint " + joint)
            var target = position(clip, a, j) if op == "plant" else vector(
                d, n, "target"
            )
            if op == "plant":
                target.y = num(d, n, "height", Float64(target.y))
            var peak = pin_joint(
                clip,
                joint,
                a,
                b,
                target,
                integer(d, n, "fade", 2),
                num(d, n, "weight", 1),
            )
            print(joint, "peak pin error (m)", peak)
        elif op == "cleanup":
            var count = clean_stances(
                clip,
                num(d, n, "speed", 0.32),
                num(d, n, "height", 0.08),
                integer(d, n, "minimum", 3),
            )
            print("Planted", count, "detected stances")
        elif op == "warp":
            warp_contact(
                clip,
                joint,
                a,
                b,
                vector(d, n, "target"),
                num(d, n, "reach", 0.8),
            )
        elif op == "path":
            var pts = field(d, n, "markers")
            var frames = List[Int]()
            var points = List[Vec3]()
            for m in range(d.array_size(pts)):
                var mark = d.at(pts, m)
                frames.append(integer(d, mark, "frame", -1))
                points.append(vector(d, mark, "position"))
            follow_path(clip, frames, points)
        else:
            raise Error("Unknown take operation " + op)


def build_recipe(
    d: JsonDocument, file: String, mut partner: MbClip
) raises -> MbClip:
    var root = d.root
    var clip: MbClip
    var cap = field(d, root, "capture")
    var beats = field(d, root, "beats")
    if cap >= 0:
        var rig = load_take(relative(file, string(d, root, "source")))
        clip = capture_take(rig, d, cap, file)
    elif beats >= 0:
        clip = timeline(d, beats, file)
    elif field(d, root, "prompt") >= 0 or field(d, root, "cue") >= 0:
        clip = prompt_take(d, root, file)
    else:
        clip = load_take(relative(file, string(d, root, "source")))
    apply_edits(clip, d, field(d, root, "edits"))
    var actor = field(d, root, "partner")
    if actor >= 0:
        var other = load_take(relative(file, string(d, actor, "source")))
        var shift = vector(d, actor, "offset")
        var delay = num(d, actor, "delay", 0)
        partner = MbClip()
        partner.skeleton = other.skeleton.copy()
        partner.sample_rate = clip.sample_rate
        for f in range(clip.num_frames()):
            var pose = other.sample(Float32(f) / clip.sample_rate - delay)
            pose[0].p = pose[0].p + mb(shift)
            partner.frames.append(pose^)
        apply_edits(partner, d, field(d, actor, "edits"))
        var pins = field(d, actor, "contacts")
        for i in range(d.array_size(pins)):
            var n = d.at(pins, i)
            var follow = string(d, n, "joint")
            var tip = partner.skeleton.joint_index(follow)
            var ref_joint = clip.skeleton.joint_index(
                string(d, n, "primaryJoint")
            )
            if (
                tip < 0
                or follow
                not in ("LeftHand", "RightHand", "LeftFoot", "RightFoot")
                or ref_joint < 0
            ):
                raise Error("Unknown actor contact joints")
            var a = integer(d, n, "start", 0)
            var b = integer(d, n, "end", clip.num_frames() - 1)
            check_interval(clip, a, b)
            var peak: Float32 = 0
            for f in range(a, b + 1):
                var target = position(clip, f, ref_joint)
                pin_frame(partner, f, tip, target)
                peak = max(peak, (position(partner, f, tip) - target).length())
            print(follow, "peak partner contact error (m)", peak)
    return clip^


def draw_pair(mut canvas: SimdCanvas, a: MbClip, b: MbClip, f: Int):
    var p = a.frames[f][0].p
    var q = b.frames[f][0].p
    var target = Vec3((p.x + q.x) * 0.5, 0.9, (p.z + q.z) * 0.5)
    var dist = 1 + (p - q).length() * 0.18
    var cam = Camera3D(
        canvas.width,
        canvas.height,
        target + Vec3(1.1, 0.42, 3.55) * dist,
        target,
        40,
    )
    for i in range(-8, 9):
        var v = Float32(i) / 2
        canvas.draw_line_3d(cam, Vec3(-4, 0, v), Vec3(4, 0, v), 0x344652)
        canvas.draw_line_3d(cam, Vec3(v, 0, -4), Vec3(v, 0, 4), 0x344652)
    for actor in range(2):
        var w = world_pose(
            a.skeleton, a.frames[f], MbXform.identity()
        ) if actor == 0 else world_pose(
            b.skeleton, b.frames[f], MbXform.identity()
        )
        for j in range(1, len(w)):
            var k = (
                a.skeleton.parents[j] if actor == 0 else b.skeleton.parents[j]
            )
            var u = w[j].p
            var v = w[k].p
            canvas.draw_line_3d(
                cam,
                Vec3(u.x, u.y, u.z),
                Vec3(v.x, v.y, v.z),
                UInt32(0x75DBD0 if actor == 0 else 0xEDAF74),
                3,
            )


def partner_path(path: String) -> String:
    var i = path.rfind(".")
    return String(path[byte=0:i]) + ".partner" + String(path[byte=i:])


def export_character(
    clip: MbClip,
    rig: String,
    aliases: Dict[String, String],
    tolerance: Float32,
    path: String,
    shot: String = "",
    stage_origin: Bool = True,
) raises:
    if tolerance < 0:
        raise Error("Optimization tolerance must be nonnegative")
    var c = CharacterPreview()
    c.load(rig)
    c.bake(clip, aliases, stage_origin)
    if tolerance > 0:
        _ = optimize_bake(c.model, c.clip, tolerance)
    if shot:
        character_sheet(c, clip, shot)
    append_take_glb(rig, path, c.model.animations[c.clip])


comptime Contact = Tuple[String, String, Int, Int, Int]


def contacts(d: JsonDocument, frames: Int) raises -> List[Contact]:
    var actor = field(d, d.root, "partner")
    var pins = field(d, actor, "contacts")
    var result = List[Contact]()
    for i in range(d.array_size(pins)):
        var p = d.at(pins, i)
        var fade = integer(d, p, "fade", 2)
        if fade < 0:
            raise Error("Negative contact fade")
        result.append(
            (
                string(d, p, "primaryJoint"),
                string(d, p, "joint"),
                integer(d, p, "start", 0),
                integer(d, p, "end", frames - 1),
                fade,
            )
        )
    return result^


def side(name: String, mirror: Bool) -> String:
    if mirror and name.startswith("Left"):
        return "Right" + String(name[byte=4:])
    if mirror and name.startswith("Right"):
        return "Left" + String(name[byte=5:])
    return name.copy()


def export_partner(
    take: MbClip,
    primary_path: String,
    rig: String,
    aliases: Dict[String, String],
    primary_aliases: Dict[String, String],
    pins: List[Contact],
    tol: Float32,
    first: Int = 0,
    mirror: Bool = False,
    reverse: Bool = False,
) raises:
    var primary = load_native_glb(primary_path)
    var c = CharacterPreview()
    c.load(rig)
    c.bake(take, aliases, True)
    for pin in pins:
        var source = mapped_bone(primary, side(pin[0], mirror), primary_aliases)
        var tip = mapped_bone(c.model, side(pin[1], mirror), aliases)
        var mid = c.model.parents[tip]
        var upper = c.model.parents[mid]
        var peak: Float32 = 0
        for f in range(take.num_frames()):
            var src = first + (take.num_frames() - 1 - f if reverse else f)
            var outside = max(pin[2] - src, max(0, src - pin[3]))
            if outside > pin[4]:
                continue
            var time = Float32(f) / take.sample_rate
            primary.sample_clip(len(primary.animations) - 1, time)
            c.model.sample_clip(c.clip, time)
            var target = primary.global_position(source)
            var pole = c.model.global_position(mid)
            var w = 1 - Float32(outside) / Float32(pin[4] + 1)
            c.model.solve_two_bone(
                upper, mid, tip, target, pole, w * w * (3 - 2 * w)
            )
            for i in range(len(c.model.animations[c.clip].channels)):
                ref ch = c.model.animations[c.clip].channels[i]
                if ch.path == 1 and (ch.node == upper or ch.node == mid):
                    var rot = c.model.nodes[ch.node].rotation
                    var sign: Float32 = 1
                    if (
                        rot.x * ch.values[4 * f]
                        + rot.y * ch.values[4 * f + 1]
                        + rot.z * ch.values[4 * f + 2]
                        + rot.w * ch.values[4 * f + 3]
                        < 0
                    ):
                        sign = -1
                    ch.values[4 * f] = rot.x * sign
                    ch.values[4 * f + 1] = rot.y * sign
                    ch.values[4 * f + 2] = rot.z * sign
                    ch.values[4 * f + 3] = rot.w * sign
            if outside == 0:
                peak = max(
                    peak, (c.model.global_position(tip) - target).length()
                )
        print(pin[1], "peak skinned contact error (m)", peak)
    if tol > 0:
        _ = optimize_bake(c.model, c.clip, tol)
    append_take_glb(rig, partner_path(primary_path), c.model.animations[c.clip])


def write_asset(
    clip: MbClip,
    d: JsonDocument,
    file: String,
    path: String,
    shot: String,
) raises:
    if path.lower().endswith(".bvh"):
        save_new(clip, path)
        if shot:
            try:
                save_sheet(path, shot)
            except err:
                _ = remove_file(path)
                raise Error(String(err))
    elif path.lower().endswith(".glb"):
        var rig = character_path(d, file)
        var aliases = bone_map(d)
        export_character(
            clip, rig, aliases, num(d, d.root, "tolerance", 0), path, shot
        )
    else:
        raise Error("Choose a new .bvh or .glb output")


def run_recipe(file: String, path: String, var shot: String) raises:
    var d = document(file)
    var partner = MbClip()
    if shot and (not shot.lower().endswith(".png") or shot == path):
        raise Error("Use a separate .png filename for the preview")
    if file_size(path) >= 0 or (shot and file_size(shot) >= 0):
        raise Error("Choose new output filenames")
    var clip = build_recipe(d, file, partner)
    from std.ffi import external_call

    var staging = (
        shot
        + "."
        + String(external_call["getpid", Int32]())
        + ".stage" if shot else String()
    )
    var paired = partner.num_frames() > 0
    var second = partner_path(path) if paired else String()
    var saved_partner = False
    var saved_primary = False
    try:
        if paired:
            if path.lower().endswith(".bvh"):
                write_asset(partner, d, file, second, "")
                saved_partner = True
            if shot:
                var canvas = SimdCanvas(760, 420)
                canvas.clear()
                draw_pair(canvas, clip, partner, clip.num_frames() // 2)
                if not write_png(staging, canvas.pixels, 760, 420):
                    raise Error("Could not write paired preview")
        write_asset(clip, d, file, path, "" if paired else staging)
        saved_primary = True
        if paired and path.lower().endswith(".glb"):
            var actor = field(d, d.root, "partner")
            var other = string(d, actor, "character")
            var target = relative(file, other) if other else character_path(
                d, file
            )
            var aliases = bone_map(d, actor) if other or field(
                d, actor, "bones"
            ) >= 0 else bone_map(d)
            export_partner(
                partner,
                path,
                target,
                aliases,
                bone_map(d),
                contacts(d, clip.num_frames()),
                num(d, d.root, "tolerance", 0),
            )
            saved_partner = True
        if (
            shot
            and external_call["link", Int32](
                staging.as_c_string_span(), shot.as_c_string_span()
            )
            != 0
        ):
            raise Error("Could not publish preview")
    except err:
        if saved_primary:
            _ = remove_file(path)
        if saved_partner:
            _ = remove_file(second)
        if shot:
            _ = remove_file(staging)
        raise Error(String(err))
    if shot:
        _ = remove_file(staging)
    print("Saved", clip.num_frames(), "frames to", path)
