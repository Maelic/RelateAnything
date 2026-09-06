"""Export the re-parameterized relation head to ONNX.

Design decision that matters: the graph outputs RAW per-pair, per-predicate
scores. It does NOT threshold, rank or select triplets. Thresholding happens
on the host (see ``deploy/postprocess.py``), which is what keeps the score
cutoff *dynamic* — a slider in the demo, not a re-export.

Putting a threshold inside the graph would make the output shape
data-dependent (a variable number of surviving triplets). ONNX expresses that
only via NonZero/compress, whose output shape is unknown at graph-build time;
that disables shape inference downstream, blocks TensorRT/most NPU/CoreML
backends, and forces a dynamic allocation per frame. The cost of keeping it
outside is nil: the postprocess is a [K,V] elementwise compare on 128x30
floats, microseconds against the ViT's tens of milliseconds.

Two vocabulary modes:

  --vocab-mode baked    W/alpha become graph constants. Smallest, fastest,
                        fully foldable. Predicate set is frozen at export.
  --vocab-mode input    W [V,text_dim] and alpha [V] become graph INPUTS, so
                        you can swap the predicate vocabulary at runtime
                        without re-exporting (encode new predicate names with
                        dino.txt offline, feed the matrix in). V stays a
                        dynamic axis.

Usage:
    python deploy/export_onnx.py \
        --checkpoint runs/train/relsgg-vits16plus/model.pth \
        --vocab-mode input --check --out relateanything.onnx
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("HF_HOME", os.path.join(REPO, ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from relsgg.api import RelateAnything  # noqa: E402
from deploy.vocab import PREDICATE_VOCAB  # noqa: E402


class RelSGGExport(nn.Module):
    """Inference-only wrapper with a flat tensor signature.

    Returns raw scores; no thresholding, no ranking, no Python-side control
    flow. ``pred_logits`` and ``pair_logits`` are kept SEPARATE so the host can
    apply per-predicate thresholds and re-weight the pair-existence term
    without re-exporting.
    """

    def __init__(self, model: nn.Module, vocab_as_input: bool = False):
        super().__init__()
        self.model = model
        self.vocab_as_input = vocab_as_input

    def forward(self, image, boxes, box_counts, W=None, alpha=None):
        if self.vocab_as_input:
            # Swap the baked vocabulary for the runtime-supplied one. Plain
            # attribute assignment on the module: traced as data flow, so the
            # matmul reads the graph input instead of a constant.
            self.model.vocab_head.W = W
            self.model.vocab_head.alpha = alpha

        out = self.model(image, boxes, box_counts=box_counts, targets=None)

        # Raw logits, not probabilities: the score contract is
        # sigmoid(a * (pred + w * pair) + b) (relsgg/scoring.py), which cannot
        # be recovered from two separate sigmoids, and emitting logits keeps
        # the calibration on the host so (a, b) can be refitted without a
        # re-export.
        pred_logits = out["logits"]                        # [B, K, V]
        pair_logits = out.get("pair_logits")
        if pair_logits is None:
            # 0.0 is the identity for an ADDITIVE fusion (1.0 was the identity
            # for the old multiplicative one — a silent sign error if copied).
            pair_logits = torch.zeros_like(pred_logits[..., 0])

        return (
            pred_logits,                                   # [B, K, V]
            pair_logits,                                   # [B, K]
            out["sub_idx"].to(torch.int64),                # [B, K]
            out["obj_idx"].to(torch.int64),                # [B, K]
            out["valid_mask"].to(torch.bool),              # [B, K]
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--text-student", default=None,
                    help="text student checkpoint; default: the one the checkpoint names")
    ap.add_argument("--out", default="relateanything.onnx")
    ap.add_argument("--predicates", nargs="*", default=None)
    ap.add_argument("--vocab-npz", default="",
                    help="a release bundle's predicate_bank.npz (names + W). Exports "
                         "against the ALREADY-ENCODED deployment vocabulary, so no text "
                         "encoder is needed at export time -- which is both the natural "
                         "path when re-exporting a shipped bundle and the only path when "
                         "the checkpoint's text student is not on the machine.")
    ap.add_argument("--vocab-mode", choices=["baked", "input"], default="baked")
    ap.add_argument("--max-boxes", type=int, default=32,
                    help="N the graph is built for (boxes are zero-padded to this)")
    ap.add_argument("--img-size", type=int, default=448)
    ap.add_argument("--weights", default="ema", choices=["ema", "raw"])
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamo", action="store_true", help="use the TorchDynamo exporter")
    ap.add_argument("--check", action="store_true", help="validate against torch with onnxruntime")
    args = ap.parse_args()

    os.chdir(REPO)
    # nn.TransformerEncoderLayer takes a fused "BetterTransformer" fastpath in
    # eval+no_grad, which traces to aten::_transformer_encoder_layer_fwd — an
    # op with no ONNX symbolic. Disabling the fastpath makes it trace as the
    # ordinary MHA/LayerNorm/Linear sequence, which exports cleanly. Numerics
    # are equivalent (same math, unfused); this only affects export.
    torch.backends.mha.set_fastpath_enabled(False)

    bank_E = None
    if args.vocab_npz:
        import numpy as _np
        z = _np.load(args.vocab_npz, allow_pickle=True)
        preds = [str(q) for q in z["names"]]
        bank_E = z["W"]
        print(f"[export] vocabulary from {args.vocab_npz}: {len(preds)} predicates, "
              f"W {bank_E.shape}")
    else:
        preds = args.predicates if args.predicates else PREDICATE_VOCAB
    print(f"[export] re-parameterizing to {len(preds)} predicates")

    # Release exports must load every trained tensor or fail — a silently
    # dropped module here ships a wrong model with no error anywhere.
    ra = RelateAnything.from_checkpoint(
        args.checkpoint, preds, text_student=args.text_student,
        device="cpu", weights=args.weights, img_size=args.img_size,
        strict=True, embeddings=bank_E)
    bt = "dinov3"
    print("[export] text encoder: " + (f"none (vocabulary from {args.vocab_npz})"
                                       if bank_E is not None else str(ra.text_student)))
    model = ra.model.eval()

    V = model.vocab_head.W.shape[0]
    N, S = args.max_boxes, args.img_size
    print(f"[export] V={V}  N={N}  img={S}  final_budget(K)={model.config.final_budget}")

    wrapper = RelSGGExport(model, vocab_as_input=(args.vocab_mode == "input")).eval()

    image = torch.rand(1, 3, S, S)
    boxes = torch.rand(1, N, 4) * 0.5 + 0.25          # cxcywh in [0,1]
    box_counts = torch.tensor([N], dtype=torch.long)

    input_names = ["image", "boxes", "box_counts"]
    inputs = [image, boxes, box_counts]
    dynamic_axes = {
        "image": {0: "batch"},
        "boxes": {0: "batch", 1: "num_boxes"},
        "box_counts": {0: "batch"},
        "pred_logits": {0: "batch", 1: "num_pairs", 2: "num_predicates"},
        "pair_logits": {0: "batch", 1: "num_pairs"},
        "sub_idx": {0: "batch", 1: "num_pairs"},
        "obj_idx": {0: "batch", 1: "num_pairs"},
        "valid_mask": {0: "batch", 1: "num_pairs"},
    }
    if args.vocab_mode == "input":
        W = model.vocab_head.W.detach().clone()
        alpha = model.vocab_head.alpha.detach().clone()
        inputs += [W, alpha]
        input_names += ["W", "alpha"]
        dynamic_axes["W"] = {0: "num_predicates"}
        dynamic_axes["alpha"] = {0: "num_predicates"}

    # deploy/runtime.py keys off these names.
    output_names = ["pred_logits", "pair_logits", "sub_idx", "obj_idx",
                    "valid_mask"]

    with torch.no_grad():
        ref = wrapper(*inputs)
    print(f"[export] torch reference OK — pred_logits {tuple(ref[0].shape)} "
          f"range [{ref[0].min():.2f}, {ref[0].max():.2f}], "
          f"pair_logits {tuple(ref[1].shape)}")

    print(f"[export] exporting (opset {args.opset}, dynamo={args.dynamo}) -> {args.out}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, tuple(inputs), args.out,
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes, opset_version=args.opset,
            do_constant_folding=True, dynamo=args.dynamo,
)
    sz = os.path.getsize(args.out) / 1e6
    print(f"[export] wrote {args.out} ({sz:.0f} MB)")

    import subprocess
    import transformers as _tf
    ck_args = torch.load(args.checkpoint, map_location="cpu",
                         weights_only=False).get("args") or {}
    ck_args = dict(ck_args if isinstance(ck_args, dict) else vars(ck_args))
    _student = ra.text_student or ""

    def _sha256(path):
        if not path or not os.path.exists(path):
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    # The score contract travels WITH the artifact. Without it the host has no
    # way to know whether the graph emits logits or sigmoids, nor which (a, b)
    # the numbers were calibrated against.
    from relsgg.scoring import CONTRACT, ScoreContract
    contract = ScoreContract.for_checkpoint(args.checkpoint)
    meta = {
        "predicates": preds, "img_size": S, "max_boxes": N,
        "final_budget": model.config.final_budget, "vocab_mode": args.vocab_mode,
        "text_dim": int(model.vocab_head.W.shape[1]),
        "outputs": output_names,
        "output_kind": "logits",
        "score_contract": CONTRACT,
        "calibration": {"a": contract.calib_a, "b": contract.calib_b},
        "calibrated": contract.is_calibrated,
        # Provenance: enough to reproduce or audit the artifact. The old
        # basename-only field could not even say which RUN it came from.
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "run_name": os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint))),
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip(),
        "backbone_type": bt,
        "backbone_model": ck_args.get("backbone_model"),
        "text_student": _student,
        "text_student_sha256": _sha256(_student),
        "pred_embeds": ck_args.get("pred_embeds"),
        "epoch": ck_args.get("epochs"),
        "torch": torch.__version__,
        "transformers": _tf.__version__,
        "opset": args.opset,
        "exported": __import__("datetime").date.today().isoformat(),
    }
    meta_path = os.path.splitext(args.out)[0] + ".json"
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"[export] wrote {meta_path}")

    if args.check:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        feed = {n: t.numpy() for n, t in zip(input_names, inputs)}
        got = sess.run(output_names, feed)
        worst = 0.0
        for name, a, b in zip(output_names, ref, got):
            a = a.numpy()
            if a.dtype == bool or a.dtype == np.int64:
                agree = float((a == b).mean())
                print(f"[check] {name:12s} exact-match {agree:.4f}")
            else:
                d = np.abs(a - b).max()
                worst = max(worst, float(d))
                print(f"[check] {name:12s} max|Δ| {d:.3e}")
        # The check result is release evidence — persist it with the artifact.
        meta["check_max_abs_delta"] = worst
        json.dump(meta, open(meta_path, "w"), indent=2)
        print(f"[check] recorded max|Δ| {worst:.3e} into {meta_path}")


if __name__ == "__main__":
    main()
