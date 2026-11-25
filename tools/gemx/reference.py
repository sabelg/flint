"""Run the PyTorch GEM-X reference on a directory of PNG frames and dump every stage for
`tests/test_flint_v5_gemx.mojo` / `examples/gemx` parity checks.

usage: python tools/gemx/reference.py GEMX_CHECKOUT WEIGHTS_DIR FRAMES_DIR OUT_DIR
           [--boxes BOXES.txt | --init x0,y0,x1,y1] [--frames N] [--dump-images K]

  FRAMES_DIR  00000.png, 00001.png, ... (30 fps RGB frames, e.g. `ffmpeg -i clip.mp4 -vf fps=30 %05d.png`)
  --boxes     one person box per frame, "x0 y0 x1 y1" in pixels (any detector)
  --init      no box file: track the person from this first-frame box with the native77 keypoints
              (the same tracker as `flint.ml.gemx.Gemx.track`); the boxes are written to
              OUT_DIR/boxes.txt so Flint can be fed the identical boxes
  --frames    use only the first N frames
  --dump-images  also dump the network input crops / heatmaps of the first K frames (default 2)

OUT_DIR receives `<name>.f32` (raw little-endian float32) plus `manifest.txt` ("name dim0 dim1 ...").
The camera is static (GEM-X `--camera-mode static`) with the default diagonal-focal intrinsics, and
Contact/IK postprocessing is off: Flint reproduces the raw Motion predictions.
"""
import math
import os
import sys

args = sys.argv[1:]
if len(args) < 4:
    print(__doc__)
    sys.exit(1)
GEMX, WEIGHTS, FRAMES, OUT = (os.path.abspath(a) for a in args[:4])
opts = {}
i = 4
while i < len(args):
    opts[args[i]] = args[i + 1]
    i += 2
sys.path.insert(0, GEMX)
os.makedirs(OUT, exist_ok=True)

import cv2
import numpy as np
import torch
import yaml

torch.set_grad_enabled(False)
from gem.common.detectors.tracks import smooth_boxes, square_crops
from gem.common.pose2d.native77 import crop_native77_inputs, load_native77_detector
from gem.common.geometry.rotations import rotation_6d_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle
from gem.inference.body_video import extract_body_features, motion_intrinsics
from gem.inference.camera_motion import camera_motion
from gem.inference.motion import load_predictor, run_motion
from gem.pose.api import body_config, load_pose_checkpoint
from gem.pose.data.crop import prepare_pose_batch

manifest = []


def dump(name, a):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    a.tofile(os.path.join(OUT, name + ".f32"))
    manifest.append(name + " " + " ".join(str(d) for d in a.shape))


files = sorted(f for f in os.listdir(FRAMES) if f.endswith(".png"))
if "--frames" in opts:
    files = files[: int(opts["--frames"])]
frames = [cv2.cvtColor(cv2.imread(os.path.join(FRAMES, f), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for f in files]
T = len(frames)
H, W = frames[0].shape[:2]
print(f"{T} frames {W}x{H}")
release = "vitpose77-dinov3-huge-metrosim-256x192-v1"
detector = load_native77_detector(os.path.join(WEIGHTS, "vitpose.pth"), release_id=release, batch_size=4)


def keypoint_box(kp, width, height):
    """The tracker's person box from native77 keypoints: the confident joints' extent, padded by
    15% of its size on every side (a detector box covers the head top and feet) and clamped."""
    good = kp[kp[:, 2] > 0.3]
    if len(good) < 4:
        return None
    x0, y0 = good[:, 0].min(), good[:, 1].min()
    x1, y1 = good[:, 0].max(), good[:, 1].max()
    pw, ph = (x1 - x0) * 0.15, (y1 - y0) * 0.15
    return np.array([max(0.0, x0 - pw), max(0.0, y0 - ph), min(width - 1.0, x1 + pw), min(height - 1.0, y1 + ph)],
                    np.float32)


if "--boxes" in opts:
    xyxy = np.loadtxt(opts["--boxes"], dtype=np.float32).reshape(-1, 4)[:T]
else:
    box = np.array([float(v) for v in opts["--init"].split(",")], np.float32)
    xyxy = np.zeros((T, 4), np.float32)
    for t in range(T):
        for _ in range(3 if t == 0 else 1):
            xys = square_crops(box[None], 1.2)
            kp = detector.predict([frames[t]], torch.from_numpy(xys)).numpy()[0]
            nb = keypoint_box(kp, W, H)
            if nb is not None:
                box = nb
        xyxy[t] = box
    np.savetxt(os.path.join(OUT, "boxes.txt"), xyxy, fmt="%.4f")
# GEM-X track post-processing (clamp, 5-frame moving average, 1.2x square crop)
xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, W - 1)
xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, H - 1)
boxes_xys = square_crops(smooth_boxes(xyxy), 1.2).astype(np.float32)
dump("boxes_xyxy", xyxy)
dump("bbx_xys", boxes_xys)

K = int(opts.get("--dump-images", "2"))
if K > 0:
    crops = crop_native77_inputs(frames[:K], torch.from_numpy(boxes_xys[:K]))
    dump("vitpose_crops", crops)
    m = detector.model
    o, fl = m(torch.cat((crops, crops.flip(-1)))).chunk(2)
    dump("vitpose_heatmaps", o)
    dump("vitpose_heatmaps_flipped", fl)

print("2D keypoints")
kp2d = detector.predict(frames, torch.from_numpy(boxes_xys))
dump("kp2d", kp2d)
del detector

intrinsics = motion_intrinsics(W, H)
camera = camera_motion(frames, intrinsics.numpy(), xyxy, mode="static")

print("Body Pose features")
pose = load_pose_checkpoint(os.path.join(WEIGHTS, "body_s240000.ckpt"), body_config())
if K > 0:
    batch = prepare_pose_batch(frames[:K], boxes_xys[:K], intrinsics, crop_convention="motion_training")
    dump("pose_crops", batch["img"])
    dump("pose_cliff", pose.conditioning(batch))
features, version = extract_body_features(frames, boxes_xys, intrinsics, pose, batch_size=4, device="cpu")
dump("pose_tokens", features)
del pose

with open(os.path.join(WEIGHTS, "body_video_native77.yaml")) as f:
    cfg = yaml.safe_load(f)
sequence = {
    "condition_spec": {"pose_release_id": version, "observation_source": release,
                       "observation_joint_set": "soma77_native_v1"},
    "length": torch.tensor([T]),
    "f_imgseq": features.unsqueeze(0),
    "kp2d_native77": kp2d.unsqueeze(0),
    "bbx_xys": torch.from_numpy(boxes_xys).unsqueeze(0),
    "K_fullimg": intrinsics.expand(T, 3, 3).unsqueeze(0).clone(),
    "cam_angvel": camera["cam_angvel"].unsqueeze(0),
    "frame_times": (torch.arange(T, dtype=torch.float64) / 30).unsqueeze(0),
    "sampling_fps": 30.0,
}
print("Body Motion")
predictor = load_predictor(os.path.join(WEIGHTS, "motion_native77_e774_s292175.ckpt"), cfg["motion"],
                           stats_path=os.path.join(WEIGHTS, "motion_stats_raw.npz"))
# capture the raw network outputs too
captured = {}
net = predictor.pipeline.pipeline.denoiser3d
hook = net.register_forward_hook(lambda m, i, o: captured.update(o))
result = run_motion(predictor, sequence)
hook.remove()
dump("cam_angvel", camera["cam_angvel"])
dump("motion_x", captured["pred_x"][0])
dump("motion_cam", captured["pred_cam"][0])
dump("motion_conf", captured["static_conf_logits"][0])
incam = result["pred_soma_params_incam"]
world = result["pred_soma_params_global"]
dump("body_pose_R", axis_angle_to_matrix(incam["body_pose"][0].reshape(T, 76, 3)))
dump("incam_orient_R", axis_angle_to_matrix(incam["global_orient"][0]))
dump("incam_transl", incam["transl"][0])
dump("world_orient_R", axis_angle_to_matrix(world["global_orient"][0]))
dump("world_transl", world["transl"][0])
# the same world rollout in float64: GEM-X's float32 axis-angle round trips near 180 degrees cost
# ~1e-2 in the world orientation, while Flint keeps rotation matrices
from gem.motion.codec.world import rollout_world
codec = predictor.pipeline.pipeline.endecoder
m64 = captured["pred_x"].double() * codec.std.double() + codec.mean.double()
roots = rollout_world(matrix_to_axis_angle(rotation_6d_to_matrix(m64[..., 646:652])), m64[..., 652:655],
                      matrix_to_axis_angle(rotation_6d_to_matrix(m64[..., 640:646])),
                      camera["cam_angvel"].unsqueeze(0).double())
dump("world_orient_R_f64", axis_angle_to_matrix(roots["global_orient"][0]))
dump("world_transl_f64", roots["transl"][0])
dump("identity", incam["identity_coeffs"][0][0])
dump("scale", incam["scale_params"][0][0])
with open(os.path.join(OUT, "manifest.txt"), "w") as f:
    f.write("\n".join(manifest) + "\n")
print("wrote", OUT)
