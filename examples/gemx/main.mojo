"""GEM-X video-to-motion CLI: PNG frames of one person in, a SOMA BVH clip out.

usage: mojo run -I src examples/gemx/main.mojo MODEL_DIR FRAMES_DIR out.bvh
           (--boxes BOXES.txt | --init x0,y0,x1,y1) [--skeleton SOMA.bvh] [--frames N] [--incam] [--cpu]

  MODEL_DIR     gemx_vitpose.onnx, gemx_pose.onnx, gemx_motion.onnx from `tools/gemx/export.py`
  FRAMES_DIR    00000.png, 00001.png, ... at 30 fps: `ffmpeg -i clip.mp4 -vf fps=30 FRAMES_DIR/%05d.png`
  --boxes       one person box per frame, "x0 y0 x1 y1" in pixels, from any detector
  --init        or a first-frame box: the person is then tracked with the 77 keypoints
  --skeleton    a SOMA-skeleton BVH (e.g. a MotionBricks clip) whose bones carry the motion;
                the neutral SOMA-X skeleton by default
  --incam       camera-space root motion instead of the gravity-aligned world rollout
"""
from std.sys import argv
from std.time import perf_counter_ns
from flint.platform.os import read_bytes
from flint.ml.gemx import Gemx, load_frames, boxes_to_crops, gemx_clip, write_bvh, soma77_skeleton
from flint.ml.motionbricks import load_bvh


def _floats(text: String) raises -> List[Float32]:
    var out = List[Float32]()
    var cur = String()
    for b in text.as_bytes():
        var c = Int(b)
        if c == 32 or c == 9 or c == 10 or c == 13 or c == 44:
            if cur.byte_length() > 0:
                out.append(Float32(atof(cur)))
                cur = String()
        else:
            cur += chr(c)
    if cur.byte_length() > 0:
        out.append(Float32(atof(cur)))
    return out^


def _secs(t0: Int) -> Float64:
    return Float64(Int(perf_counter_ns()) - t0) / 1e9


def main() raises:
    var a = argv()
    if len(a) < 4:
        print("usage: gemx MODEL_DIR FRAMES_DIR out.bvh (--boxes BOXES.txt | --init x0,y0,x1,y1)"
              + " [--skeleton SOMA.bvh] [--frames N] [--incam] [--cpu]")
        return
    var gpu = True
    var world = True
    var limit = -1
    var boxes_path = String()
    var init = List[Float32]()
    var skeleton_path = String()
    var i = 4
    while i < len(a):
        var s = String(a[i])
        if s == "--cpu":
            gpu = False
        elif s == "--incam":
            world = False
        elif s == "--frames" and i + 1 < len(a):
            limit = Int(atol(String(a[i + 1])))
            i += 1
        elif s == "--boxes" and i + 1 < len(a):
            boxes_path = String(a[i + 1])
            i += 1
        elif s == "--skeleton" and i + 1 < len(a):
            skeleton_path = String(a[i + 1])
            i += 1
        elif s == "--init" and i + 1 < len(a):
            init = _floats(String(a[i + 1]))
            i += 1
        i += 1
    if boxes_path.byte_length() == 0 and len(init) != 4:
        raise Error("give --boxes BOXES.txt or --init x0,y0,x1,y1")
    var t0 = Int(perf_counter_ns())
    var frames = load_frames(String(a[2]), limit)
    var n = len(frames)
    print("[gemx]", n, "frames", frames[0].width, "x", frames[0].height, "decoded in", _secs(t0), "s")
    t0 = Int(perf_counter_ns())
    var g = Gemx.load(String(a[1]), gpu)
    print("[gemx] models loaded (GPU:", g.gpu, ") in", _secs(t0), "s")
    var xyxy: List[Float32]
    if boxes_path.byte_length() > 0:
        xyxy = _floats(String(unsafe_from_utf8=read_bytes(boxes_path)))
        if len(xyxy) < 4 * n:
            raise Error(boxes_path + " has fewer than one box per frame")
    else:
        t0 = Int(perf_counter_ns())
        xyxy = g.track(frames, init[0], init[1], init[2], init[3])
        print("[gemx] tracked the person in", _secs(t0), "s")
    var xys = boxes_to_crops(xyxy, frames[0].width, frames[0].height)
    t0 = Int(perf_counter_ns())
    var kp = g.keypoints(frames, xys)
    print("[gemx] native77 keypoints in", _secs(t0), "s")
    t0 = Int(perf_counter_ns())
    var tok = g.pose_tokens(frames, xys)
    print("[gemx] Body Pose tokens in", _secs(t0), "s")
    t0 = Int(perf_counter_ns())
    var m = g.motion(kp, xys, tok, n, frames[0].width, frames[0].height)
    print("[gemx] Motion transformer + decoding in", _secs(t0), "s")
    var skel = soma77_skeleton() if skeleton_path.byte_length() == 0 else load_bvh(skeleton_path).skeleton.copy()
    var clip = gemx_clip(m, skel, world)
    write_bvh(clip, String(a[3]))
    print("[gemx] wrote", String(a[3]), "(", clip.num_frames(), "frames,", clip.skeleton.num_joints(), "joints )")
