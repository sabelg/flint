"""Offline conversion of the RTMPose-m COCO 256x192 ONNX graph for Flint's Mojo interpreter."""

import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as H, numpy_helper as NH, TensorProto as T


def convert(source, dest):
    model = onnx.load(source)
    graph = model.graph
    graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
    model = onnx.shape_inference.infer_shapes(model)
    graph = model.graph
    shapes = {
        v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
        for v in [*graph.input, *graph.value_info]
    }
    weights = {v.name: NH.to_array(v) for v in graph.initializer}
    nodes, constants = [], []

    def const(data):
        name = "flint_const_" + str(len(constants))
        constants.append(NH.from_array(np.asarray(data), name))
        return name

    def op(kind, inputs, output=None, **attrs):
        name = output or "flint_node_" + str(len(nodes))
        nodes.append(H.make_node(kind, inputs, [name], **attrs))
        return name

    def reshape(name, dims):
        return op("Reshape", [name, const(np.asarray(dims, np.int64))])

    for node in graph.node:
        x, *args = node.input
        y = node.output[0]
        a = {v.name: H.get_attribute_value(v) for v in node.attribute}
        if node.op_type in ("Conv", "MaxPool"):
            _, ci, ih, iw = shapes[x]
            kh, kw = a["kernel_shape"]
            sy, sx = a.get("strides", [1, 1])
            pt, pl, pb, pr = a.get("pads", [0] * 4)
            assert a.get("dilations", [1, 1]) == [1, 1] and not a.get("ceil_mode", 0)
            oh, ow = (ih + pt + pb - kh) // sy + 1, (iw + pl + pr - kw) // sx + 1
            iy, ix = np.indices((oh, ow))
            flat = reshape(x, [ci, ih * iw])
            if kh * kw != 1 or sy * sx != 1 or any([pt, pl, pb, pr]):
                fill = 0 if node.op_type == "Conv" else -np.finfo(np.float32).max
                flat = op(
                    "Concat", [flat, const(np.full((ci, 1), fill, np.float32))], axis=1
                )
                indices = []
                for dy in range(kh):
                    for dx in range(kw):
                        ry, rx = iy * sy + dy - pt, ix * sx + dx - pl
                        indices.append(
                            np.where(
                                (ry >= 0) & (ry < ih) & (rx >= 0) & (rx < iw),
                                ry * iw + rx,
                                ih * iw,
                            ).ravel()
                        )
                if node.op_type == "MaxPool":
                    part = [op("Gather", [flat, const(i)], axis=1) for i in indices]
                    val = part[0]
                    for p in part[1:]:
                        val = op("Max", [val, p])
                    op("Reshape", [val, const([1, ci, oh, ow])], y)
                    continue
                flat = op("Gather", [flat, const(np.stack(indices))], axis=1)
            w = weights[args[0]]
            co = w.shape[0]
            groups = a.get("group", 1)
            width = ci // groups * kh * kw
            val = op(
                "MatMul",
                [
                    const(w.reshape(groups, co // groups, width)),
                    reshape(flat, [groups, width, oh * ow]),
                ],
            )
            val = reshape(val, [1, co, oh, ow])
            if len(args) == 2:
                op("Add", [val, const(weights[args[1]].reshape(1, co, 1, 1))], y)
            else:
                nodes[-1].output[0] = y
        elif node.op_type == "GlobalAveragePool":
            h, w = shapes[x][-2:]
            val = op("ReduceSum", [x], axes=[2, 3], keepdims=1)
            op("Div", [val, const(np.float32(h * w))], y)
        elif node.op_type == "HardSigmoid":
            v = op("Mul", [x, const(np.float32(a.get("alpha", 0.2)))])
            v = op("Add", [v, const(np.float32(a.get("beta", 0.5)))])
            op("Clip", [v, const(np.float32(0)), const(np.float32(1))], y)
        else:
            nodes.append(node)
    used = {name for node in nodes for name in node.input}
    kept = [v for v in graph.initializer if v.name in used]
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.initializer[:]
    graph.initializer.extend(kept + constants)
    del graph.value_info[:]
    onnx.checker.check_model(model)
    if dest.exists():
        raise FileExistsError(dest)
    onnx.save(model, dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    convert(args.source, args.output)
