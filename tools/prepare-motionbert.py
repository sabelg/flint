"""Export MotionBERT's released H36M DSTformer to fixed 9-frame windows for Flint."""

import argparse
import sys
from functools import partial
from pathlib import Path
import numpy as np
import torch
import onnx
from onnx import helper, numpy_helper


def convert(checkout, checkpoint, dest):
    if dest.exists():
        raise FileExistsError(dest)
    sys.path.insert(0, str(checkout.resolve()))
    from lib.model.DSTformer import DSTformer

    torch.set_num_threads(4)
    model = DSTformer(
        dim_in=3,
        dim_out=3,
        dim_feat=512,
        dim_rep=512,
        depth=5,
        num_heads=8,
        mlp_ratio=2,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        maxlen=243,
        num_joints=17,
    )
    params = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_pos"]
    model.load_state_dict({k.removeprefix("module."): v for k, v in params.items()})
    model.eval()
    import io

    buf = io.BytesIO()
    torch.onnx.export(
        model,
        torch.zeros(1, 9, 17, 3),
        buf,
        opset_version=17,
        dynamo=False,
        input_names=["keypoints"],
        output_names=["pose3d"],
    )
    graph = onnx.load_model_from_string(buf.getvalue())
    fixed = []
    for n in graph.graph.node:
        if n.op_type == "Identity":
            fixed.append(helper.make_node("Add", [n.input[0], "flint_zero"], n.output))
        elif n.op_type == "Tanh":
            key = n.output[0]
            fixed.extend(
                [
                    helper.make_node("Mul", [n.input[0], "flint_two"], [key + "_x2"]),
                    helper.make_node("Sigmoid", [key + "_x2"], [key + "_sig"]),
                    helper.make_node("Mul", [key + "_sig", "flint_two"], [key + "_s2"]),
                    helper.make_node("Sub", [key + "_s2", "flint_one"], [key]),
                ]
            )
        else:
            fixed.append(n)
    for name, value in [("flint_one", 1), ("flint_two", 2), ("flint_zero", 0)]:
        graph.graph.initializer.append(numpy_helper.from_array(np.float32(value), name))
    del graph.graph.node[:]
    graph.graph.node.extend(fixed)
    onnx.checker.check_model(graph)
    onnx.save(graph, dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["checkout", "checkpoint", "output"]:
        p.add_argument(name, type=Path)
    a = p.parse_args()
    convert(a.checkout, a.checkpoint, a.output)
