"""PyTorch reference dumps for the Make-It-Animatable (v1) parity test.

usage: python tools/make_it_animatable/reference.py MIA_CHECKOUT WEIGHTS_DIR MESH.glb OUT_DIR [--seed S]

Runs upstream app.py's default pipeline (prepare_input -> preprocess -> infer -> vis; mesh input,
input_normal=False, bw_fix=True, hands re-sampling 0.5) with the upstream PCAE modules and the
released v1 weights, and writes every stage as raw little-endian float32 (`.f32`) / int32 (`.i32`)
files for tests/test_flint_v5_make_it_animatable.mojo:

  verts.f32        [V, 3]   input vertices (trimesh's merged mesh, original frame)
  pts1.f32         [N, 3]   stage-1 surface samples after the bounding-box normalisation
  fps1.i32         [512]    farthest-point picks on pts1 (one group)
  coarse.f32       [52, 6]  coarse joints (head xyz, tail xyz) in the stage-1 frame
  hips.f32         [4, 4]   row-major Hips transform (stage-1 frame -> facing-forward frame)
  pts2_raw.f32     [N, 3]   stage-2 samples (uniform + around both hands) in the Hips frame
  pts2.f32         [N, 3]   the same after the |max| normalisation (network input)
  scale2.f32       [1]      that normalisation's scale
  fps2.i32         [512]    farthest-point picks on pts2 (two groups of 256)
  qverts.f32       [V, 3]   the vertices in the network frame (skin-weight queries)
  bw_raw.f32       [V, 52]  skin weights straight from the network
  bw.f32           [V, 52]  after upstream bw_post_process (4 influences per vertex)
  joints.f32       [52, 6]  joints (head, tail) in the network frame
  pose6.f32        [52, 6]  pose network output (global ortho6d rotations, pose -> rest)
  pose.f32         [52, 4, 4] row-major global rest transforms (pose_rot_to_global, root = I)
  rest_verts.f32   [V, 3]   vertices skinned into the rest (T) pose (rows renormalised), network frame
  rest_joints.f32  [52, 3]  rest-pose joint heads, network frame
  global.f32       [4, 4]   row-major input frame -> network frame (similarity transform)

Point sampling is random (numpy, seeded); the Mojo test starts every stage from these dumps so the
comparison does not depend on the sampler. Farthest point sampling uses the deterministic shim in
shims/torch_cluster (first point first), which Flint implements bit for bit.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("mia_src")
ap.add_argument("weights")
ap.add_argument("mesh")
ap.add_argument("out")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

import mia_torch  # noqa: E402

mia_model, _, U = mia_torch.import_upstream(args.mia_src)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import trimesh  # noqa: E402
from torch_cluster import fps  # noqa: E402  (the shim)

N = mia_torch.N_POINTS
IDX = {n: i for i, n in enumerate(mia_torch.MIXAMO_NAMES)}
PARENTS = torch.tensor(mia_torch.MIXAMO_PARENTS)


def dump(name, a, dt=np.float32):
    a = np.ascontiguousarray(np.asarray(a, dtype=dt))
    a.tofile(os.path.join(args.out, name))
    print(f"  {name:16s} {tuple(a.shape)}")


def normalize_transform(points, recenter):
    """upstream utils.get_normalize_transform(keep_ratio=True) as a 4x4 matrix (points: [N, 3])."""
    mn = points.min(0).values
    mx = points.max(0).values
    if not recenter:
        mx = torch.maximum(mx.abs(), mn.abs())
        mn = -mx
    s = 2.0 / (mx - mn).max()
    m = torch.eye(4)
    m[:3, :3] *= s
    if recenter:
        m[:3, 3] = -(mx + mn) / 2.0 * s
    return m


def xf(m, p):
    return p @ m[:3, :3].T + m[:3, 3]


# upstream util/dataset_mixamo.py get_plane_transform / get_hips_transform (MIT, (c) 2025 Zhiyang
# Guo), restated because that module needs Blender at import time.
def get_plane_transform(origin, normal, tangent_ends):
    B = normal.shape[0]
    normal = F.normalize(normal, dim=-1)
    target_normal = F.normalize(torch.tensor([0, 0, 1]).to(normal), dim=-1).unsqueeze(0).repeat(B, 1)
    cross = torch.cross(normal, target_normal, dim=-1)
    dot = torch.sum(normal * target_normal, dim=-1, keepdim=True)
    cross_norm = cross.norm(dim=-1, keepdim=True)
    sk = torch.zeros((B, 3, 3)).to(normal)
    sk[:, 0, 1] = -cross[:, 2]
    sk[:, 0, 2] = cross[:, 1]
    sk[:, 1, 0] = cross[:, 2]
    sk[:, 1, 2] = -cross[:, 0]
    sk[:, 2, 0] = -cross[:, 1]
    sk[:, 2, 1] = cross[:, 0]
    rotation = torch.eye(3).to(normal).unsqueeze(0).repeat(B, 1, 1) + sk + sk @ sk * ((1 - dot) / (cross_norm**2)).unsqueeze(-1)
    rm = torch.eye(4).to(normal).unsqueeze(0).repeat(B, 1, 1)
    rm[:, :3, :3] = rotation
    tm = torch.eye(4).to(normal).unsqueeze(0).repeat(B, 1, 1)
    tm[:, :3, 3] = -origin
    matrix = rm @ tm
    tangent_ends = U.apply_transform(tangent_ends, matrix.unsqueeze(1).expand(-1, 2, -1, -1))
    tangent = tangent_ends[:, 1] - tangent_ends[:, 0]
    angles = torch.atan2(tangent[:, 1], tangent[:, 0])
    c, s = torch.cos(-angles), torch.sin(-angles)
    r2 = torch.zeros((B, 4, 4)).to(tangent)
    r2[:, 0, 0] = c
    r2[:, 0, 1] = -s
    r2[:, 1, 0] = s
    r2[:, 1, 1] = c
    r2[:, 2, 2] = 1
    r2[:, 3, 3] = 1
    return r2 @ matrix


def get_hips_transform(hips, rightupleg, leftupleg):
    normal = torch.cross(rightupleg - hips, leftupleg - hips, dim=-1)
    return get_plane_transform(hips, normal, torch.stack((rightupleg, leftupleg), dim=-2))


def fps_groups(pts, ratio):
    """PCAE.fps index selection (hierarchical split of the points and of the 512 latents)."""
    n_h = int(N * ratio)
    l_h = int(512 * ratio)
    groups = [(0, N - n_h, 512 - l_h), (N - n_h, n_h, l_h)]
    out = []
    for start, n, m in groups:
        if n == 0:
            continue
        idx = fps(pts[start:start + n], torch.zeros(n, dtype=torch.long), ratio=1.0 * m / n)[:m]
        out.append(idx + start)
    return torch.cat(out)


np.random.seed(args.seed)
torch.manual_seed(args.seed)
mesh = trimesh.load(args.mesh, force="mesh")
verts = torch.from_numpy(np.array(mesh.vertices, dtype=np.float32))
dump("verts.f32", verts)

# prepare_input
pts = U.sample_mesh(mesh, N, get_normals=True).astype(np.float32)
pts = torch.from_numpy(pts[:, :3].copy())

# preprocess: bbox normalisation, coarse joints, Hips frame, hands re-sampling
norm1 = normalize_transform(pts, recenter=True)
pts1 = xf(norm1, pts)
dump("pts1.f32", pts1)
dump("fps1.i32", fps_groups(pts1, 0.0), np.int32)
t0 = time.time()
with torch.no_grad():
    coarse = mia_torch.build_model(args.mia_src, args.weights, "coarse")(pts1[None]).joints[0]
print(f"coarse joints {time.time() - t0:.2f}s")
dump("coarse.f32", coarse)
H = get_hips_transform(coarse[None, IDX["Hips"], :3], coarse[None, IDX["RightUpLeg"], :3],
                       coarse[None, IDX["LeftUpLeg"], :3])[0]
dump("hips.f32", H)
tails = xf(H, coarse[:, 3:])
mesh.vertices = xf(H, xf(norm1, verts)).numpy()
pts2 = U.sample_mesh(mesh, N, get_normals=True, attn_ratio=mia_torch.HANDS_RATIO,
                     attn_centers=[tails[IDX["LeftHand"]].numpy(), tails[IDX["RightHand"]].numpy()],
                     attn_geo_ratio=0.0).astype(np.float32)
pts2 = torch.from_numpy(pts2[:, :3].copy())
dump("pts2_raw.f32", pts2)

# infer
norm2 = normalize_transform(pts2, recenter=False)
pts2 = xf(norm2, pts2)
dump("pts2.f32", pts2)
dump("scale2.f32", norm2[0, 0].reshape(1))
dump("fps2.i32", fps_groups(pts2, mia_torch.HANDS_RATIO), np.int32)
qverts = xf(norm2, torch.from_numpy(mesh.vertices.astype(np.float32)))
dump("qverts.f32", qverts)
dump("global.f32", norm2 @ H @ norm1)
with torch.no_grad():
    t0 = time.time()
    m = mia_torch.build_model(args.mia_src, args.weights, "bw")
    bw = torch.cat([m(pts2[None], q).bw[0] for q in torch.split(qverts[None], 100000, dim=1)], 0)
    del m
    m = mia_torch.build_model(args.mia_src, args.weights, "joints")
    joints = m(pts2[None]).joints
    del m
    m = mia_torch.build_model(args.mia_src, args.weights, "pose")
    pose6 = m(pts2[None], joints=joints.clone()).pose_trans
    del m
    print(f"bw + joints + pose {time.time() - t0:.2f}s")
joints = joints[0]
pose6 = pose6[0]
dump("bw_raw.f32", bw)
dump("joints.f32", joints)
dump("pose6.f32", pose6)

# vis: upstream bw_post_process (needs the app's globals; restated from app.py, MIT) and rest pose
sys.argv = [sys.argv[0]]

src = open(os.path.join(args.mia_src, "app.py")).read()
a = src.index("def get_conflict_mask")
b = src.index("def reorganize_bone_data_")
ns = {"torch": torch, "MIXAMO_PREFIX": "mixamorig:"}
exec(src[a:b], ns)
bones = {"mixamorig:" + n: i for i, n in enumerate(mia_torch.MIXAMO_NAMES)}
head_tail_y = joints[IDX["Head"], 4]
bw_post = ns["bw_post_process"](bw[None].clone(), bones_idx_dict=bones,
                               above_head_mask=(qverts[:, 1] >= head_tail_y)[None], no_fingers=False)[0]
dump("bw.f32", bw_post)
pose, _ = U.pose_rot_to_global(pose6[None], joints[None, :, :3], PARENTS)
pose = pose[0]
pose[0] = torch.eye(4)
dump("pose.f32", pose)
# rows renormalised as Blender's armature deform does (upstream leaves the top-4 rows unnormalised)
lbs = torch.einsum("kij,nk->nij", pose, bw_post / bw_post.sum(-1, keepdim=True).clamp(min=1e-10))
dump("rest_verts.f32", U.apply_transform(qverts, lbs))
dump("rest_joints.f32", U.apply_transform(joints[:, :3], pose))
print("done:", args.out)
