"""Shared PyTorch side of the Make-It-Animatable (v1) tools: imports the upstream code with the
shims in `shims/`, builds the five v1 networks exactly as upstream `app.py:init_models` does and
loads the released checkpoints.

Only v1 is supported (`output/best/new/*.pth` in the Hugging Face repo jasongzy/Make-It-Animatable,
Apache-2.0). v2 is not: it depends on the Hunyuan3D 2.1 ShapeVAE (Tencent's territory-restricted
licence).

The upstream code (MIT) is imported from a user-supplied checkout, never copied into Flint. Its
`util/dataset_mixamo.py` builds the kinematic tree by opening `bones.fbx` in Blender at import time;
here the same 52-joint Mixamo tree is written out by hand (`MIXAMO_PARENTS`) and passed in as a
small `Joint`-compatible object.
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))

# The 52 joints in upstream MIXAMO_JOINTS order (the "mixamorig:" prefix dropped) and each one's
# parent (index into this list, -1 for the root). Same table as flint.ml.make_it_animatable.
MIXAMO_NAMES = [
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3",
    "RightHandRing1", "RightHandRing2", "RightHandRing3",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
]


def _parents():
    idx = {n: i for i, n in enumerate(MIXAMO_NAMES)}
    par = {"Hips": None, "Spine": "Hips", "Spine1": "Spine", "Spine2": "Spine1", "Neck": "Spine2", "Head": "Neck"}
    for side in ("Left", "Right"):
        par[side + "Shoulder"] = "Spine2"
        par[side + "Arm"] = side + "Shoulder"
        par[side + "ForeArm"] = side + "Arm"
        par[side + "Hand"] = side + "ForeArm"
        for f in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            par[f"{side}Hand{f}1"] = side + "Hand"
            par[f"{side}Hand{f}2"] = f"{side}Hand{f}1"
            par[f"{side}Hand{f}3"] = f"{side}Hand{f}2"
        par[side + "UpLeg"] = "Hips"
        par[side + "Leg"] = side + "UpLeg"
        par[side + "Foot"] = side + "Leg"
        par[side + "ToeBase"] = side + "Foot"
    return [-1 if par[n] is None else idx[par[n]] for n in MIXAMO_NAMES]


MIXAMO_PARENTS = _parents()


class Joint:
    """The parts of upstream util.dataset_mixamo.Joint that model.py uses."""

    def __init__(self, index, parents, nodes):
        self.index = index
        self.name = "mixamorig:" + MIXAMO_NAMES[index]
        self._parents = parents
        self._nodes = nodes

    @property
    def parent(self):
        p = self._parents[self.index]
        return None if p < 0 else self._nodes[p]

    @property
    def parent_recursive(self):
        out, p = [], self.parent
        while p is not None:
            out.append(p)
            p = p.parent
        return out

    def __iter__(self):  # the whole tree (only called on the root)
        return iter(self._nodes)

    def __len__(self):
        return len(self._nodes)

    @property
    def tree_levels_mask(self):
        depth = [len(n.parent_recursive) for n in self._nodes]
        return [[d == l for d in depth] for l in range(max(depth) + 1)]

    @property
    def parent_indices(self):
        return list(self._parents)


def kinematic_tree():
    nodes = []
    for i in range(len(MIXAMO_NAMES)):
        nodes.append(Joint(i, MIXAMO_PARENTS, nodes))
    return nodes[0]


def import_upstream(mia_src):
    """Puts the shims and the upstream checkout on sys.path and returns (model, models_ae, utils)."""
    sys.path.insert(0, os.path.join(HERE, "shims"))
    sys.path.insert(1, os.path.abspath(mia_src))
    stub = types.ModuleType("util.dataset_mixamo")
    stub.Joint = Joint
    sys.modules["util.dataset_mixamo"] = stub
    import model as mia_model
    import models_ae
    from util import utils as mia_utils
    return mia_model, models_ae, mia_utils


N_POINTS = 32768
HANDS_RATIO = 0.5


MODEL_FILES = {"bw": "bw", "joints": "joints", "coarse": "joints_coarse", "pose": "pose"}


def build_model(mia_src, weights_dir, name):
    """One of the four v1 networks of upstream app.py:init_models (ADDITIONAL_BONES = False), in
    eval mode with its released weights: "coarse" (joints_coarse.pth), "joints", "pose" or "bw".
    `bw_normal` is not built: the default pipeline runs with input_normal=False."""
    import torch
    mia_model, _, _ = import_upstream(mia_src)
    PCAE = mia_model.PCAE
    tree = kinematic_tree()
    common = dict(N=N_POINTS, input_normal=False, deterministic=True, output_dim=52)
    if name == "bw":
        m = PCAE(hierarchical_ratio=HANDS_RATIO, **common)
    elif name == "joints":
        m = PCAE(hierarchical_ratio=HANDS_RATIO, kinematic_tree=tree, predict_bw=False, predict_joints=True,
                 predict_joints_tail=True, joints_attn_causal=True, **common)
    elif name == "coarse":
        m = PCAE(predict_bw=False, predict_joints=True, predict_joints_tail=True, **common)
    elif name == "pose":
        m = PCAE(hierarchical_ratio=HANDS_RATIO, kinematic_tree=tree, predict_bw=False, predict_pose_trans=True,
                 pose_mode="ortho6d", pose_input_joints=True, pose_attn_causal=True, **common)
    else:
        raise ValueError(name)
    sd = torch.load(os.path.join(weights_dir, MODEL_FILES[name] + ".pth"), map_location="cpu",
                    weights_only=False)["model"]
    m.load_state_dict(m.adapt_ckpt(sd), strict=True)
    return m.eval()
