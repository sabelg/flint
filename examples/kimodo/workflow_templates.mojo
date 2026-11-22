"""Create editable Studio recipes from the loaded take."""
from std.ffi import external_call
from flint.ml.motionbricks import MbClip, MbXform, world_pose
from take_files import load_take
from flint.anim.take_constraints import position, vec
from flint.core.math import Vec3
from flint.platform.os import (
    file_size,
    make_dirs,
    env_now,
    write_file_durable,
    remove_file,
)


def labels() -> List[String]:
    return [
        "Bone mapping",
        "Foot cleanup",
        "Ledge warp",
        "Optimize GLB",
        "Path markers",
        "Joint pin",
        "Beat timeline",
        "Prompt providers",
        "Two actors",
        "Single camera",
        "Stereo capture",
        "Capture to GLB",
    ]


def absolute(path: String) raises -> String:
    if path.startswith("/"):
        return path.copy()
    var buf = List[UInt8](length=4096, fill=0)
    if external_call["getcwd", Int](buf.unsafe_ptr(), len(buf)) == 0:
        raise Error("Unable to resolve the project directory")
    var n = 0
    while n < len(buf) and buf[n] != 0:
        n += 1
    return String(unsafe_from_utf8=buf[:n]) + "/" + path


def quoted(s: String) raises -> String:
    for b in s.as_bytes():
        if b < 32:
            raise Error("Control character in asset path")
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def xyz(p: Vec3) -> String:
    return "[" + String(p.x) + ", " + String(p.y) + ", " + String(p.z) + "]"


def new_template(
    kind: Int, clip: MbClip, source: String, second: String, rig: String
) raises -> String:
    var dir = String("build/motion-recipes")
    if not make_dirs(dir):
        raise Error("Cannot create build/motion-recipes")
    var s = quoted(absolute(source))
    var r = quoted(absolute(rig))
    var other = quoted(absolute(second)) if second else s.copy()
    var last = clip.num_frames() - 1
    var mid = last // 2
    var j = clip.skeleton.joint_index("LeftHand")
    var pos = position(clip, mid, j)
    var h0 = position(clip, 0, 0)
    var h1 = position(clip, last, 0)
    h0.y = 0
    h1.y = 0
    var body: String
    if kind == 0:
        body = '"bones": {"Hips": "Hips", "Chest": "Spine2"}'
    elif kind == 1:
        body = (
            '"edits": [{"op":"cleanup","speed":0.32,"height":0.08,"minimum":3}]'
        )
    elif kind == 2:
        body = (
            '"edits": [{"op":"warp","joint":"LeftHand","start":0,"end":'
            + String(max(1, mid))
            + ',"target":'
            + xyz(position(clip, max(1, mid), j) + Vec3(0.1, 0.05, 0))
            + ',"reach":0.5}]'
        )
    elif kind == 3:
        body = '"tolerance": 0.004'
    elif kind == 4:
        body = (
            '"edits": [{"op":"path","markers":[{"frame":0,"position":'
            + xyz(h0)
            + '},{"frame":'
            + String(last)
            + ',"position":'
            + xyz(h1 + Vec3(0.5, 0, 0.25))
            + "}]}]"
        )
    elif kind == 5:
        body = (
            '"edits": [{"op":"pin","joint":"LeftHand","start":'
            + String(mid)
            + ',"end":'
            + String(min(last, mid + 5))
            + ',"target":'
            + xyz(pos)
            + ',"weight":1,"fade":2}]'
        )
    elif kind == 6:
        body = '"beats": [{"source":' + s + '},{"source":' + other + "}]"
    elif kind == 7:
        if (
            file_size(env_now("HOME") + "/.cache/flint/hymotion-lite/qwen.bf16")
            > 0
        ):
            body = (
                '"provider": "hymotion-lite",\n  "prompt": "A person takes a'
                ' deliberate step sideways to the left.",\n  "frames": 45,\n '
                ' "steps": 20,\n  "skeleton": '
                + s
            )
        else:
            body = (
                '"provider": "library",\n  "prompt": "A person walks'
                ' forward.",\n  "frames": 60,\n  "library": [{"caption":"A'
                ' person walks forward.","source":'
                + s
                + '},{"caption":"A person crouches.","source":'
                + other
                + "}]"
            )
    elif kind == 8:
        var secondary = load_take(second) if second else clip.copy()
        var pose = secondary.sample(Float32(mid) / clip.sample_rate)
        var world = world_pose(secondary.skeleton, pose, MbXform.identity())
        var hand = secondary.skeleton.joint_index("LeftHand")
        var shift = position(
            clip, mid, clip.skeleton.joint_index("RightHand")
        ) - vec(world[hand].p)
        body = (
            '"partner": {"source":'
            + other
            + ',"offset":'
            + xyz(shift)
            + ',"contacts":[{"joint":"LeftHand","primaryJoint":"RightHand","start":'
            + String(mid)
            + ',"end":'
            + String(min(last, mid + 2))
            + "}]}"
        )
    elif kind == 10:
        body = (
            '"capture": {"fps":30,"views":[\n   '
            ' {"frames":"camera-left","eye":[-0.5,1,3],"lookAt":[0,1,0],"up":[0,1,0],"fx":800,"fy":800},\n'
            '    {"frames":"camera-right","eye":[0.5,1,3],"lookAt":[0,1,0],"up":[0,1,0],"fx":800,"fy":800}]}'
        )
    else:
        body = '"capture": {"views":[{"frames":"performance.avi"}]}'
        if kind == 11:
            body += ',\n  "edits": [{"op":"cleanup"}],\n  "tolerance": 0.004'
    var json = (
        '{\n  "version": 1,\n  "source": '
        + s
        + ',\n  "character": '
        + r
        + ",\n  "
        + body
        + "\n}\n"
    )
    var stem = (
        dir
        + "/"
        + labels()[kind].lower().replace(" ", "-")
        + "-"
        + String(external_call["getpid", Int32]())
        + "-"
    )
    var idx = 1
    while file_size(stem + String(idx) + ".motionrecipe.json") >= 0:
        idx += 1
    var path = stem + String(idx) + ".motionrecipe.json"
    var data = List[UInt8](json.as_bytes())
    if not write_file_durable(path, data):
        raise Error("Could not write recipe")
    return path^
