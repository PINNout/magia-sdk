# Generator for a PINN-style MLP test (forward pass only).
#
# Builds an fp16 ONNX graph:
#   X[P, in] -> Gemm -> Tanh -> (Gemm -> Tanh) x (layers-1) -> Gemm -> Y[P, out]
# and writes network.onnx, inputs.npz and outputs.npz next to this file,
# in the same format as the SDK's other deployment tests.
#
# Also writes outputs_fp32.npz (an fp32 reference, not used by the deployment
# flow) so the fp16 error of the whole network can be measured.

import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def parse_args():
    p = argparse.ArgumentParser(description="PINN-style tanh MLP test generator")
    p.add_argument("--points", type=int, default=256, help="collocation points (batch size)")
    p.add_argument("--in-dim", type=int, default=2, help="input dimension, e.g. 2 for (x, t)")
    p.add_argument("--width", type=int, default=64, help="hidden layer width")
    p.add_argument("--layers", type=int, default=3, help="number of hidden layers")
    p.add_argument("--out-dim", type=int, default=1, help="output dimension")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def flush_subnormals(arr):
    # Same treatment as the SDK's resnet-reduced generator: fp16 subnormals -> 0.
    arr = arr.copy()
    arr[np.abs(arr) < np.float16(2.0**-14)] = 0
    return arr


def build_model(args, rng, dtype):
    """Returns an ONNX model with all weights as initializers, in the given dtype."""
    onnx_type = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
    dims = [args.in_dim] + [args.width] * args.layers + [args.out_dim]

    nodes, inits = [], []
    prev = "X"
    for i in range(len(dims) - 1):
        k, n = dims[i], dims[i + 1]
        # Xavier/Glorot normal init (suits tanh)
        W = rng.normal(0.0, np.sqrt(2.0 / (k + n)), size=(k, n))
        b = rng.normal(0.0, 0.1, size=(n,))
        # The MAGIA gemm kernel copies C as a full [M, N] matrix (no broadcast),
        # so the bias is materialised per row.
        B_full = np.broadcast_to(b, (args.points, n))

        W_name, C_name, gemm_out = f"W{i}", f"C{i}", f"gemm{i}"
        inits.append(numpy_helper.from_array(flush_subnormals(W.astype(dtype)), W_name))
        inits.append(numpy_helper.from_array(flush_subnormals(B_full.astype(dtype)), C_name))
        nodes.append(helper.make_node("Gemm", [prev, W_name, C_name], [gemm_out],
                                      name=f"Gemm_{i}", alpha=1.0, beta=1.0, transA=0, transB=0))
        prev = gemm_out

        if i < len(dims) - 2:  # tanh after every hidden layer, not after the output layer
            act_out = f"tanh{i}"
            nodes.append(helper.make_node("Tanh", [prev], [act_out], name=f"Tanh_{i}"))
            prev = act_out

    nodes[-1].output[0] = "Y"  # rename the last output

    X_info = helper.make_tensor_value_info("X", onnx_type, (args.points, args.in_dim))
    Y_info = helper.make_tensor_value_info("Y", onnx_type, (args.points, args.out_dim))
    graph = helper.make_graph(nodes, "pinn-mlp", [X_info], [Y_info], initializer=inits)
    # Pin the IR version: a newer onnx package otherwise writes an IR version
    # that an older onnxruntime refuses to load. IR 8 is valid for opset 13.
    model = helper.make_model(graph, producer_name="pinn-mlp-generator",
                              opset_imports=[helper.make_operatorsetid("", 13)],
                              ir_version=8)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


def main():
    args = parse_args()
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Collocation points in a normalised domain [-1, 1]^in_dim.
    X = np.random.default_rng(args.seed + 1).uniform(-1.0, 1.0, size=(args.points, args.in_dim))

    # Same seed for both models -> identical weights before the dtype cast.
    model16 = build_model(args, np.random.default_rng(args.seed), np.float16)
    model32 = build_model(args, np.random.default_rng(args.seed), np.float32)

    X16 = X.astype(np.float16)
    Y16 = ort.InferenceSession(model16.SerializeToString()).run(None, {"X": X16})[0].astype(np.float16)
    Y32 = ort.InferenceSession(model32.SerializeToString()).run(None, {"X": X.astype(np.float32)})[0]

    onnx.save(model16, os.path.join(out_dir, "network.onnx"))
    np.savez(os.path.join(out_dir, "inputs.npz"), X=X16)
    np.savez(os.path.join(out_dir, "outputs.npz"), Y=Y16)
    np.savez(os.path.join(out_dir, "outputs_fp32.npz"), Y=Y32)

    rel = np.linalg.norm(Y16.astype(np.float32) - Y32) / np.linalg.norm(Y32)
    ops = [n.op_type for n in model16.graph.node]
    print(f"Generated PINN MLP: points={args.points} in={args.in_dim} width={args.width} "
          f"layers={args.layers} out={args.out_dim}")
    print(f"Nodes ({len(ops)}): {' -> '.join(ops)}")
    print(f"Reference fp16 vs fp32 (onnxruntime), normwise relative error: {rel:.2e}")


if __name__ == "__main__":
    main()
