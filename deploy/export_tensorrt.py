"""Build a local TensorRT 10 engine from a released ONNX graph.

Batch, image size and padded box count are fixed; W/alpha retain a dynamic
predicate axis. FP32 with TF32 disabled is intentional: reduced precision in
the sampler can change which pairs are scored. No detector weights are fetched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy.trt_runtime import TensorRTEngine, load_tensorrt


def input_profiles(meta, bank_path=None, max_vocab=None):
    """Return min/opt/max shapes for the deployment contract (batch one)."""
    if "img_size" not in meta:  # locally exported raw YOLO graph
        size = int(meta["imgsz"])
        return {"images": ((1, 3, size, size),) * 3}
    size, boxes = int(meta["img_size"]), int(meta["max_boxes"])
    shapes = {"image": (1, 3, size, size), "boxes": (1, boxes, 4), "box_counts": (1,)}
    profiles = {n: (s, s, s) for n, s in shapes.items()}
    if meta.get("vocab_mode", "baked") == "input":
        if not bank_path or not Path(bank_path).is_file():
            raise ValueError(
                "A predicate_bank.npz is required for a dynamic vocabulary"
            )
        with np.load(bank_path, allow_pickle=True) as bank:
            count, dim = bank["W"].shape
            optimum = len(bank["default"])
        maximum = count if max_vocab is None else max_vocab
        if maximum < 1:
            raise ValueError("--max-vocab must be positive")
        if dim != int(meta["text_dim"]):
            raise ValueError(
                "Predicate bank text dimension does not match the ONNX sidecar"
            )
        optimum = max(1, min(optimum, maximum))
        profiles["W"] = ((1, dim), (optimum, dim), (maximum, dim))
        profiles["alpha"] = ((1,), (optimum,), (maximum,))
    return profiles


def build_engine(
    onnx_path,
    out_path=None,
    bank_path=None,
    max_vocab=None,
    workspace_gib=4.0,
    device="cuda",
):
    import torch

    source = Path(onnx_path)
    out = Path(out_path) if out_path else source.with_name(source.stem + "_trt.engine")
    if (
        out.suffix != ".engine"
        or out.with_suffix(".json").resolve() == source.with_suffix(".json").resolve()
    ):
        raise ValueError(
            "Use a distinct engine stem, e.g. relateanything_trt.engine, to preserve the ONNX sidecar"
        )
    if workspace_gib <= 0:
        raise ValueError("--workspace-gib must be positive")
    meta = json.loads(source.with_suffix(".json").read_text())
    bank_path = bank_path or source.parent / "predicate_bank.npz"
    profiles = input_profiles(meta, bank_path, max_vocab)
    trt = load_tensorrt()
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise ValueError(
            "Building TensorRT engines requires an NVIDIA GPU and CUDA-enabled PyTorch"
        )

    with torch.cuda.device(target):
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(source.resolve())):
            errors = "\n".join(
                str(parser.get_error(i)) for i in range(parser.num_errors)
            )
            raise RuntimeError(f"TensorRT could not parse {source}:\n{errors}")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, int(workspace_gib * (1 << 30))
        )
        config.clear_flag(trt.BuilderFlag.TF32)
        profile = builder.create_optimization_profile()
        actual_profiles = {}
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            name = tensor.name
            if "img_size" not in meta and network.num_inputs == 1:
                shapes = profiles["images"]
            elif name in profiles:
                shapes = profiles[name]
            else:
                raise ValueError(f"Unexpected graph input {name!r}")
            expected = tuple(tensor.shape)
            if any(
                len(s) != len(expected)
                or any(d >= 0 and d != v for d, v in zip(expected, s))
                for s in shapes
            ):
                raise ValueError(
                    f"{name}: graph shape {expected} disagrees with sidecar/profile {shapes}"
                )
            # Specialize batch / padded boxes before building so shape-driven
            # TopK and reshapes are constants; only vocabulary remains dynamic.
            if shapes[0] == shapes[2]:
                tensor.shape = shapes[0]
            profile.set_shape(name, *shapes)  # raises ValueError on invalid shapes
            actual_profiles[name] = shapes
        config.add_optimization_profile(profile)
        print(
            f"[trt] building FP32 on {torch.cuda.get_device_name(target)}; this can take several minutes",
            flush=True,
        )
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError(
                "TensorRT engine build failed; see the TensorRT diagnostics above"
            )
        with source.open("rb") as source_file:
            source_hash = hashlib.file_digest(source_file, "sha256").hexdigest()
        metadata = dict(meta)
        metadata["tensorrt"] = {
            "version": trt.__version__,
            "precision": "fp32",
            "tf32": False,
            "gpu": torch.cuda.get_device_name(target),
            "compute_capability": list(torch.cuda.get_device_capability(target)),
            "source_onnx": source.name,
            "source_onnx_sha256": source_hash,
            "profiles": actual_profiles,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(bytes(plan))
        out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[trt] wrote {out}", flush=True)
    return out


def compare_relation_outputs(reference, actual, atol=1e-3, rtol=1e-4):
    """Compare by ordered region pair: TopK may emit ties in another order.

    Invalid padded slots are not predictions and have no ordering contract.
    The set of valid pairs must still agree exactly; no intersection-only test.
    """
    if len(reference) != 5 or len(actual) != 5:
        raise AssertionError("Expected five relation outputs")
    for ref_value, got_value in zip(reference, actual):
        if ref_value.shape != got_value.shape:
            raise AssertionError(
                f"Output shape mismatch: {ref_value.shape} != {got_value.shape}"
            )

    def rows(values):
        pred, pair, sub, obj, valid = values
        indices = np.flatnonzero(valid[0])
        keys = [(int(sub[0, i]), int(obj[0, i])) for i in indices]
        if len(set(keys)) != len(keys):
            raise AssertionError("Duplicate valid pairs")
        return {key: (pred[0, i], pair[0, i]) for key, i in zip(keys, indices)}

    ref, got = rows(reference), rows(actual)
    if set(ref) != set(got):
        raise AssertionError("TensorRT and ONNX selected different valid region pairs")
    worst = 0.0
    for key in ref:
        for a, b in zip(ref[key], got[key]):
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise AssertionError(f"Non-finite logits for pair {key}")
            np.testing.assert_allclose(b, a, atol=atol, rtol=rtol)
            worst = max(worst, float(np.max(np.abs(a - b))))
    return worst


def check_engine(onnx_path, engine_path, bank_path=None, device="cuda", image_paths=()):
    import cv2
    import onnxruntime as ort
    from deploy.runtime import OnnxRelationHead

    meta = json.loads(Path(onnx_path).with_suffix(".json").read_text())
    engine_meta = json.loads(Path(engine_path).with_suffix(".json").read_text())
    engine = TensorRTEngine(str(engine_path), device)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    sess = ort.InferenceSession(
        str(onnx_path), options, providers=["CPUExecutionProvider"]
    )
    names = [o.name for o in sess.get_outputs()]
    rng = np.random.default_rng(0)
    worst, cases = 0.0, 0
    if "img_size" in meta:
        bank = bank_path or Path(onnx_path).parent / "predicate_bank.npz"
        head = OnnxRelationHead(str(onnx_path), str(bank), threads=4)
        frames = [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)]
        for path in image_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f"Cannot read image {path}")
            frames.append(frame)
        vocabularies = [None]
        if head.bank is not None:
            maximum = engine_meta["tensorrt"]["profiles"]["W"][2][0]
            all_names = head.available_predicates()
            vocabularies = [
                all_names[:1],
                head.predicates[:maximum],
                all_names[:maximum],
            ]
        order = ["pred_logits", "pair_logits", "sub_idx", "obj_idx", "valid_mask"]
        for predicates in vocabularies:
            if predicates is not None:
                head.set_predicates(predicates)
            for frame in frames:
                h, w = frame.shape[:2]
                xy = rng.uniform(0.0, 0.65, (head.max_boxes, 2))
                wh = rng.uniform(0.05, 0.35, (head.max_boxes, 2))
                boxes = (
                    np.concatenate([xy, np.minimum(xy + wh, 1)], axis=1) * [w, h, w, h]
                ).astype(np.float32)
                for count in sorted({0, 1, 2, min(10, head.max_boxes), head.max_boxes}):
                    feed = head.make_feed(frame, boxes[:count])
                    ref = sess.run(order, feed)
                    got = engine.run(feed, order)
                    worst = max(worst, compare_relation_outputs(ref, got))
                    cases += 1
    else:
        shape = tuple(engine_meta["tensorrt"]["profiles"][engine.input_names[0]][0])
        feed = {engine.input_names[0]: rng.random(shape, dtype=np.float32)}
        for ref, got in zip(sess.run(names, feed), engine.run(feed, names)):
            if not np.isfinite(ref).all() or not np.isfinite(got).all():
                raise AssertionError("Non-finite detector outputs")
            np.testing.assert_allclose(got, ref, atol=1e-3, rtol=1e-4)
            worst = max(worst, float(np.max(np.abs(ref - got))))
        cases = 1
    report = {"cases": cases, "max_abs_logit_delta": worst, "atol": 1e-3, "rtol": 1e-4}
    engine_meta["tensorrt"]["onnx_parity"] = report
    Path(engine_path).with_suffix(".json").write_text(
        json.dumps(engine_meta, indent=2) + "\n"
    )
    print(f"[trt] parity passed: {cases} cases, max absolute delta {worst:.6g}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--onnx",
        required=True,
        help="relation or locally rebuilt detector ONNX, with JSON sidecar",
    )
    ap.add_argument("--out", help="default: <onnx_stem>_trt.engine")
    ap.add_argument("--bank", help="default: predicate_bank.npz beside the ONNX graph")
    ap.add_argument(
        "--max-vocab", type=int, help="largest vocabulary, default: all bank rows"
    )
    ap.add_argument("--workspace-gib", type=float, default=4.0)
    ap.add_argument("--device", default="cuda", help="cuda or cuda:N")
    ap.add_argument(
        "--check",
        action="store_true",
        help="compare with ONNX Runtime, including empty/singleton regions and vocabulary swaps",
    )
    ap.add_argument(
        "--check-images",
        nargs="*",
        default=[],
        help="also check real images with generated boxes (implies --check)",
    )
    args = ap.parse_args()
    out = build_engine(
        args.onnx, args.out, args.bank, args.max_vocab, args.workspace_gib, args.device
    )
    if args.check or args.check_images:
        check_engine(args.onnx, out, args.bank, args.device, args.check_images)


if __name__ == "__main__":
    main()
