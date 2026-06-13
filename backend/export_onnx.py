"""Export the trained AZNet checkpoint to ONNX for in-browser inference.

Produces an ONNX graph that takes a flat (N, 486) float32 input and returns
two outputs: `policy` (N, 81) raw logits and `value` (N,) tanh scalar.
The frontend (onnxruntime-web) masks/softmaxes the policy over legal moves,
mirroring net.predict() exactly.

Run from the backend/ directory:
    python export_onnx.py
"""
import os

import torch

from net import AZNet, INPUT_DIM, INPUT_DIM_V2

# UTTT_PLANES=11 exports the v2 (tactical-plane) net; UTTT_EXPORT_SRC overrides
# which checkpoint to export (default az_net.pt).
PLANES = int(os.environ.get("UTTT_PLANES", 6))
INPUT = INPUT_DIM_V2 if PLANES == 11 else INPUT_DIM
CKPT_PATH = os.environ.get(
    "UTTT_EXPORT_SRC", os.path.join(os.path.dirname(__file__), "az_net.pt")
)
OUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "frontend", "public", "az_net.onnx"
)


def main():
    if not os.path.exists(CKPT_PATH):
        raise SystemExit(f"Checkpoint not found: {CKPT_PATH}")

    net = AZNet(input_dim=INPUT)
    net.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    net.eval()

    dummy = torch.zeros(1, INPUT, dtype=torch.float32)
    out_path = os.path.abspath(OUT_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    torch.onnx.export(
        net,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes={
            "input": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    # Sanity-check the exported graph against the PyTorch model.
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    probe = torch.randn(1, INPUT, dtype=torch.float32)
    with torch.no_grad():
        ref_policy, ref_value = net(probe)
    ort_policy, ort_value = sess.run(None, {"input": probe.numpy()})

    p_err = np.abs(ref_policy.numpy() - ort_policy).max()
    v_err = np.abs(ref_value.numpy() - ort_value).max()
    print(f"Exported -> {out_path}")
    print(f"max policy abs error: {p_err:.2e}")
    print(f"max value  abs error: {v_err:.2e}")
    if p_err > 1e-4 or v_err > 1e-4:
        raise SystemExit("ONNX output diverges from PyTorch; aborting.")
    print("ONNX export verified OK.")


if __name__ == "__main__":
    main()
