"""Convert the deploy ONNX graphs to OpenVINO IR, optionally quantized.

Produces, inside --dist (self-contained laptop bundle):
    relateanything_fp16.xml/.bin     fp16 IR (weights compressed; math is fp32
                                     on CPU, fp16 on the iGPU)
    detector_fp16.xml/.bin
    relateanything_int8.xml/.bin     --int8: NNCF post-training quantization,
    detector_int8.xml/.bin           calibrated on REAL images through the
                                     exact inference preprocessing
    relateanything_w4.xml/.bin       --weights4: int4 weight-only compression
                                     (download-size lever, not a speed lever)
Each.xml gets a.json sidecar (the ONNX sidecar + conversion provenance), so
deploy/ov_runtime.py reads metadata the same way the ONNX classes do.

Calibration feeds are built by running the fp16 DETECTOR on the calibration
images and pushing its boxes through OnnxRelationHead.make_feed — i.e. the
relation head is calibrated on the distribution it will actually see (detector
boxes, zero-padding, the bank's W/alpha), not on synthetic tensors.

The W/alpha inputs and everything downstream of them (the cosine scoring
against unit-norm text embeddings + the alpha gate) are EXCLUDED from int8:
that subgraph is microseconds of compute, and quantizing unit-norm embedding
activations is precision spent exactly where the model keeps its meaning.

    python deploy/export_openvino.py --dist deploy/dist/relsgg-vits16plus \
        --int8 --calib-dir../DATASETS/PSG_coco_format/val
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import openvino as ov  # noqa: E402


def _sidecar(src_json: str, dst_xml: str, extra: dict) -> None:
    meta = json.load(open(src_json)) if os.path.exists(src_json) else {}
    meta.update(extra)
    meta["openvino"] = ov.__version__
    dst = os.path.splitext(dst_xml)[0] + ".json"
    json.dump(meta, open(dst, "w"), indent=2)


def _save(model, path: str, fp16: bool) -> None:
    ov.save_model(model, path, compress_to_fp16=fp16)
    mb = (os.path.getsize(path)
          + os.path.getsize(path.replace(".xml", ".bin"))) / 1e6
    print(f"[ov] wrote {path} ({mb:.0f} MB)")


def _downstream_of(model, param_names) -> list:
    """Friendly names of every op reachable from the given graph inputs.
    Used to keep the W/alpha scoring subgraph out of quantization."""
    from collections import deque
    q = deque(p for p in model.get_parameters()
              if p.get_friendly_name() in param_names)
    seen, names = set(), []
    while q:
        node = q.popleft()
        for out in node.outputs():
            for tgt in out.get_target_inputs():
                nxt = tgt.get_node()
                key = nxt.get_friendly_name()
                if key in seen:
                    continue
                seen.add(key)
                if nxt.get_type_name() not in ("Result", "Parameter"):
                    names.append(key)
                q.append(nxt)
    return names


def _calib_images(calib_dir: str, n: int, seed: int = 0) -> list:
    paths = sorted(glob.glob(os.path.join(calib_dir, "*.jpg"))
                   + glob.glob(os.path.join(calib_dir, "*.png")))
    if not paths:
        raise SystemExit(f"[ov] no images in {calib_dir}")
    random.Random(seed).shuffle(paths)
    return paths[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", required=True,
                    help="bundle dir holding relateanything.onnx (+.json, bank)")
    ap.add_argument("--detector", default="deploy/dist/detector.onnx",
                    help="detector ONNX; copied into --dist if not already there")
    ap.add_argument("--int8", action="store_true",
                    help="NNCF post-training int8 quantization (needs --calib-dir)")
    ap.add_argument("--det-int8", action="store_true",
                    help="also quantize the detector. OFF by default — "
                         "MEASURED on PSG val: naive int8 drops box IoU to "
                         "0.80 and label agreement to 0.62 vs fp32; the fp16 "
                         "detector is 0.996/0.996. Needs a head-aware ignored "
                         "scope before it is shippable.")
    ap.add_argument("--int8-scope", choices=["backbone", "full"],
                    default="backbone",
                    help="backbone (default): quantize only the ViT, keep the "
                         "head/sampler/geometry fp — full-graph int8 is faster "
                         "but measurably wrong (top-1 agreement 0.50)")
    ap.add_argument("--calib-dir", default="",
                    help="directory of raw.jpg calibration images (e.g. PSG val)")
    ap.add_argument("--calib-n", type=int, default=192,
                    help="calibration images (those yielding <2 boxes are skipped "
                         "for the relation head, so it sees slightly fewer)")
    ap.add_argument("--weights4", action="store_true",
                    help="also emit int4 weight-only relation IR (size, not speed)")
    ap.add_argument("--static-vocab", type=int, default=0,
                    help="freeze all shapes incl. V predicates (NPU needs this); "
                         "0 = keep the dynamic vocabulary")
    ap.add_argument("--check", action="store_true",
                    help="numeric parity fp16-IR vs onnxruntime on a real feed")
    args = ap.parse_args()
    os.chdir(REPO)

    rel_onnx = os.path.join(args.dist, "relateanything.onnx")
    if not os.path.exists(rel_onnx):
        raise SystemExit(f"[ov] {rel_onnx} missing — run deploy/export_onnx.py first")

    # a self-contained bundle: the detector ONNX (+sidecar) lives beside the IRs
    det_onnx = os.path.join(args.dist, "detector.onnx")
    if not os.path.exists(det_onnx):
        import shutil
        shutil.copy2(args.detector, det_onnx)
        shutil.copy2(os.path.splitext(args.detector)[0] + ".json",
                     os.path.splitext(det_onnx)[0] + ".json")
        print(f"[ov] copied detector into {args.dist}")

    print(f"[ov] converting {rel_onnx}")
    rel_model = ov.convert_model(rel_onnx)
    if args.static_vocab:
        V = args.static_vocab
        rel_model.reshape({"image": [1, 3, 448, 448], "boxes": [1, 32, 4],
                           "box_counts": [1], "W": [V, -1], "alpha": [V]})
        print(f"[ov] static shapes, V={V}")
    det_model = ov.convert_model(det_onnx)

    rel_fp16 = os.path.join(args.dist, "relateanything_fp16.xml")
    det_fp16 = os.path.join(args.dist, "detector_fp16.xml")
    _save(rel_model, rel_fp16, fp16=True)
    _save(det_model, det_fp16, fp16=True)
    rel_json = os.path.splitext(rel_onnx)[0] + ".json"
    det_json = os.path.splitext(det_onnx)[0] + ".json"
    _sidecar(rel_json, rel_fp16, {"ov_variant": "fp16", "source_onnx": rel_onnx,
                                  "static_vocab": args.static_vocab or None})
    _sidecar(det_json, det_fp16, {"ov_variant": "fp16", "source_onnx": det_onnx})

    if args.check:
        _parity_check(args.dist, rel_onnx)

    if args.weights4:
        import nncf
        # group_size=64 divides the transformer dims; the odd-sized layers
        # (64-ch head projections, the 19-ch geometry MLPs) fall back to an
        # adjusted per-layer group instead of erroring out.
        from nncf.quantization.advanced_parameters import (
            AdvancedCompressionParameters)
        w4 = nncf.compress_weights(
            ov.convert_model(rel_onnx),      # fresh copy; compression mutates
            mode=nncf.CompressWeightsMode.INT4_SYM, group_size=64, ratio=1.0,
            advanced_parameters=AdvancedCompressionParameters(
                group_size_fallback_mode=nncf.GroupSizeFallbackMode.ADJUST))
        p = os.path.join(args.dist, "relateanything_w4.xml")
        _save(w4, p, fp16=True)
        _sidecar(rel_json, p, {"ov_variant": "w4_int4_sym_g64_adjust",
                               "source_onnx": rel_onnx})

    if not args.int8:
        return
    if not args.calib_dir:
        raise SystemExit("[ov] --int8 needs --calib-dir")

    import cv2
    import nncf
    from deploy.runtime import DetectorConfig
    from deploy.ov_runtime import OVDetector, OVRelationHead

    det = OVDetector(det_fp16, device="CPU")
    rel = OVRelationHead(rel_fp16,
                         bank_path=os.path.join(args.dist, "predicate_bank.npz"),
                         device="CPU")
    det_cfg = DetectorConfig()

    det_feeds, rel_feeds = [], []
    paths = _calib_images(args.calib_dir, args.calib_n)
    print(f"[ov] building calibration feeds from {len(paths)} images "
          f"({args.calib_dir})")
    for i, p in enumerate(paths):
        frame = cv2.imread(p)
        if frame is None:
            continue
        x, _, _, _ = det.make_input(frame)
        det_feeds.append({det.iname: x})
        boxes, _, _ = det(frame, det_cfg)
        if len(boxes) >= 2:
            rel_feeds.append(rel.make_feed(frame, boxes[:rel.max_boxes]))
        if (i + 1) % 50 == 0:
            print(f"[ov]   {i + 1}/{len(paths)}  (relation feeds: {len(rel_feeds)})")
    print(f"[ov] calibration: {len(det_feeds)} detector / {len(rel_feeds)} "
          "relation feeds")

    rel32 = ov.convert_model(rel_onnx)       # quantize from fp32, not fp16
    if args.int8_scope == "backbone":
        # Quantize ONLY the ViT backbone (~82% of relation compute, and the
        # part PTQ is known-good on). MEASURED reason: full-graph int8 halves
        # latency but scrambles the output (top-1 agreement 0.50 on fixed
        # boxes) — the sampler's ranking logic, the geometry PE and the pair
        # scoring do fine-grained comparisons that don't survive 8-bit
        # activations. rope_embeddings stays fp too: quantized rotary phase
        # angles corrupt every attention head downstream.
        ignored = [
            n for op in rel32.get_ops()
            if op.get_type_name() not in ("Constant", "Parameter", "Result")
            and not ((n:= op.get_friendly_name()).startswith("/model/model/")
                     and "/rope_embeddings/" not in n)]
    else:
        ignored = _downstream_of(rel32, {"W", "alpha"})
    print(f"[ov] int8 scope={args.int8_scope}: keeping {len(ignored)} ops in fp")
    rel_int8 = nncf.quantize(
        rel32,
        nncf.Dataset(rel_feeds),
        model_type=nncf.ModelType.TRANSFORMER,
        ignored_scope=nncf.IgnoredScope(names=ignored, validate=False),
        subset_size=len(rel_feeds))
    p = os.path.join(args.dist, "relateanything_int8.xml")
    _save(rel_int8, p, fp16=False)
    _sidecar(rel_json, p, {
        "ov_variant": "int8_ptq", "source_onnx": rel_onnx,
        "int8_scope": args.int8_scope,
        "calib_dir": os.path.abspath(args.calib_dir),
        "calib_feeds": len(rel_feeds), "ignored_scope_ops": len(ignored)})

    if args.det_int8:
        det_int8 = nncf.quantize(
            ov.convert_model(det_onnx), nncf.Dataset(det_feeds),
            subset_size=len(det_feeds))
        p = os.path.join(args.dist, "detector_int8.xml")
        _save(det_int8, p, fp16=False)
        _sidecar(det_json, p, {"ov_variant": "int8_ptq", "source_onnx": det_onnx,
                               "calib_dir": os.path.abspath(args.calib_dir),
                               "calib_feeds": len(det_feeds)})


def _parity_check(dist: str, rel_onnx: str) -> None:
    """fp16 IR vs onnxruntime fp32 on one real feed; loud numbers, no verdict.
    Full triplet-level comparison across variants lives in bench_openvino.py
    --compare (this here is just conversion sanity)."""
    import onnxruntime as ort
    from deploy.ov_runtime import OVRelationHead
    rel = OVRelationHead(os.path.join(dist, "relateanything_fp16.xml"),
                         bank_path=os.path.join(dist, "predicate_bank.npz"),
                         device="CPU")
    rng = np.random.default_rng(0)
    frame = (rng.uniform(0, 255, (720, 1280, 3))).astype(np.uint8)
    n = 12
    cx, cy = rng.uniform(0.2, 0.8, n) * 1280, rng.uniform(0.2, 0.8, n) * 720
    w, h = rng.uniform(0.05, 0.3, n) * 1280, rng.uniform(0.05, 0.3, n) * 720
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    feed = rel.make_feed(frame, boxes.astype(np.float32))
    got_ov = rel._run_engine(feed)
    sess = ort.InferenceSession(rel_onnx, providers=["CPUExecutionProvider"])
    got_ort = sess.run(None, feed)
    for name, a, b in zip(("pred_logits", "pair_logits", "sub_idx", "obj_idx",
                           "valid_mask"), got_ort, got_ov):
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype in (np.int64, np.bool_):
            print(f"[check] {name:12s} exact-match {(a == b).mean():.4f}")
        else:
            print(f"[check] {name:12s} max|Δ| {np.abs(a - b).max():.3e}")


if __name__ == "__main__":
    main()
