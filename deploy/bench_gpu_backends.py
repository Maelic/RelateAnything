"""Compare warm GPU latency using identical inputs, vocabulary and decoding.

Requires CUDA PyTorch, TensorRT 10 and onnxruntime-gpu. Build and check the
TensorRT engine first. See docs/benchmarks/README.md for complete commands.
Each invocation measures one checkpoint; run checkpoints sequentially.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARMS = ["PyTorch eager FP32", "PyTorch eager BF16", "ONNX CUDA FP32", "TensorRT FP32"]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gpu_status():
    """Record observed clocks/power without changing device settings."""
    query = (
        "driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,"
        "clocks_event_reasons_counters.sw_thermal_slowdown,clocks_event_reasons_counters.hw_thermal_slowdown,"
        "clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,enforced.power.limit"
    )
    try:
        lines = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    "0",
                    "--query-gpu=" + query,
                    "--format=csv,noheader",
                ],
                text=True,
            )
            .strip()
            .split(", ")
        )
        return dict(zip(query.split(","), lines))
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--boxes", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--cooldown-temperature",
        type=int,
        help="wait for GPU to reach this temperature before each block (requires nvidia-smi)",
    )
    args = parser.parse_args()
    if (
        args.cooldown_temperature is not None
        and not 1 <= args.cooldown_temperature <= 100
    ):
        parser.error("cooldown temperature must be between 1 and 100 degrees C")
    if min(args.rounds, args.warmup, args.iterations, args.boxes, args.threads) < 1:
        parser.error("rounds, warmup, iterations, boxes and threads must be positive")

    import cv2
    import numpy as np
    import torch
    import onnxruntime as ort
    import tensorrt as trt

    from deploy.export_tensorrt import compare_relation_outputs
    from deploy.postprocess import ThresholdConfig, decode
    from deploy.runtime import OnnxRelationHead
    from deploy.trt_runtime import TensorRTRelationHead
    from relsgg.api import RelateAnything

    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("All four arms require a CUDA GPU with BF16 support")
    ort.preload_dlls()
    bundle = args.bundle
    bank = bundle / "predicate_bank.npz"
    engine_path = bundle / "relateanything_trt.engine"
    onnx_path = bundle / "relateanything.onnx"
    engine_meta = json.loads(engine_path.with_suffix(".json").read_text())
    onnx_meta = json.loads(onnx_path.with_suffix(".json").read_text())
    trt_meta = engine_meta["tensorrt"]
    if not trt_meta.get("onnx_parity"):
        raise RuntimeError("Run export_tensorrt.py --check before benchmarking")
    if trt_meta["source_onnx_sha256"] != sha256(onnx_path):
        raise RuntimeError("Engine does not correspond to this ONNX graph")
    if trt_meta["precision"] != "fp32" or trt_meta["tf32"]:
        raise RuntimeError("This comparison requires TensorRT FP32 with TF32 disabled")
    t_head = TensorRTRelationHead(str(engine_path), str(bank))
    cuda_options = {"use_tf32": 0, "device_id": 0}
    o_head = OnnxRelationHead(
        str(onnx_path),
        str(bank),
        threads=args.threads,
        providers=[("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"],
    )
    if "CUDAExecutionProvider" not in o_head.sess.get_providers():
        raise RuntimeError("ORT CUDA unavailable; refusing a CPU comparison")
    ra = RelateAnything.from_checkpoint(
        str(args.checkpoint),
        predicates=t_head.predicates,
        embeddings=t_head._W,
        device="cuda",
        strict=True,
    )
    model = ra.model.eval()
    if (
        ra.img_size != t_head.img_size
        or model.config.final_budget != engine_meta["final_budget"]
        or args.boxes > t_head.max_boxes
    ):
        raise ValueError("PyTorch and exported deployment dimensions must agree")

    paths = sorted((ROOT / "assets/reel/images").glob("*.jpg"))
    frames = [cv2.imread(str(path)) for path in paths]
    if not frames or any(frame is None for frame in frames):
        raise ValueError("Missing or unreadable benchmark images")
    rng = np.random.default_rng(20260922)
    boxsets = []
    for frame in frames:
        h, w = frame.shape[:2]
        xy = rng.uniform(0.02, 0.65, (args.boxes, 2))
        wh = rng.uniform(0.08, 0.30, (args.boxes, 2))
        boxsets.append(
            (
                np.concatenate([xy, np.minimum(1, xy + wh)], axis=1) * [w, h, w, h]
            ).astype(np.float32)
        )
    cfg = ThresholdConfig(
        topk=20,
        threshold=0.5,
        calib_a=t_head.contract.calib_a,
        calib_b=t_head.contract.calib_b,
    )
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    results = {
        "complete": False,
        "date": datetime.now(timezone.utc).isoformat(),
        "checkpoint": "maelic/" + args.checkpoint.parent.name,
        "gpu": torch.cuda.get_device_name(0),
        "cpu": cpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tensorrt": trt.__version__,
        "onnxruntime": ort.__version__,
        "providers": o_head.sess.get_providers(),
        "ort_cuda_options": cuda_options,
        "scope": "relation backbone + head, CPU preprocessing, H2D/D2H transfers, CPU decode; no detector",
        "batch": 1,
        "image_size": t_head.img_size,
        "valid_boxes": args.boxes,
        "padded_boxes": t_head.max_boxes,
        "pair_budget": model.config.final_budget,
        "images": len(frames),
        "threads": args.threads,
        "opencv_threads": 1,
        "tf32": False,
        "torch_compile": False,
        "rounds": args.rounds,
        "warmup_per_round": args.warmup,
        "timed_iterations_per_round": args.iterations,
        "timing": "synchronized wall time; backend/vocabulary configurations shuffled within each round",
        "cooldown_temperature_c": args.cooldown_temperature,
        "box_seed": 20260922,
        "arm_shuffle_seeds": list(range(42, 42 + args.rounds)),
        "decode": {
            "topk": cfg.topk,
            "threshold": cfg.threshold,
            "calib_a": cfg.calib_a,
            "calib_b": cfg.calib_b,
        },
        "input_images": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in paths
        ],
        "artifact_sha256": {
            "checkpoint": sha256(args.checkpoint),
            "onnx": sha256(onnx_path),
            "engine": sha256(engine_path),
            "bank": sha256(bank),
            "benchmark_script": sha256(__file__),
        },
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "engine_validation": trt_meta["onnx_parity"],
        "onnx_export": {
            key: onnx_meta.get(key)
            for key in (
                "torch",
                "transformers",
                "opset",
                "exported",
                "git_sha",
                "check_max_abs_delta",
            )
        },
        "gpu_status_before_timing": gpu_status(),
        "rows": [],
        "measurement_blocks": [],
        "fp32_cross_backend_checks": [],
    }

    def torch_raw(feed, bf16=False):
        inputs = {
            key: torch.from_numpy(feed[key]).to("cuda")
            for key in ("image", "boxes", "box_counts")
        }
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            out = model(
                inputs["image"],
                inputs["boxes"],
                box_counts=inputs["box_counts"],
                targets=None,
            )
        return [
            out["logits"].float().cpu().numpy(),
            out["pair_logits"].float().cpu().numpy(),
            out["sub_idx"].cpu().numpy(),
            out["obj_idx"].cpu().numpy(),
            out["valid_mask"].cpu().numpy(),
        ]

    def select_vocab(vocab):
        t_head.set_predicates(vocab)
        o_head.set_predicates(vocab)
        # Feed identical released embeddings and gate weights to every arm.
        model.vocab_head.W = torch.from_numpy(t_head._W).cuda()
        model.vocab_head.alpha = torch.from_numpy(t_head._alpha).cuda()
        model.vocab_head.is_reparameterized = True

    def call(arm, index):
        f, b = frames[index % len(frames)], boxsets[index % len(frames)]
        feed = t_head.make_feed(f, b)
        if arm == "TensorRT FP32":
            raw = t_head._run_engine(feed)
        elif arm == "ONNX CUDA FP32":
            raw = o_head._run_engine(feed)
        else:
            raw = torch_raw(feed, bf16=arm == "PyTorch eager BF16")
        return decode(*(x[0] for x in raw), t_head.predicates, cfg, boxes_xyxy=b)

    def cool_down():
        start = time.monotonic()
        status = gpu_status()
        if args.cooldown_temperature is None:
            return status, 0.0
        while True:
            if status is None or not status.get("temperature.gpu", "").isdigit():
                raise RuntimeError(
                    "GPU temperature unavailable; cannot enforce cooling interval"
                )
            thermal_clear = all(
                status.get("clocks_event_reasons." + kind) == "Not Active"
                for kind in ("sw_thermal_slowdown", "hw_thermal_slowdown")
            )
            if (
                int(status["temperature.gpu"]) <= args.cooldown_temperature
                and thermal_clear
            ):
                return status, time.monotonic() - start
            if time.monotonic() - start > 180:
                raise RuntimeError(
                    "GPU did not reach the requested temperature within 180 seconds"
                )
            time.sleep(2)
            status = gpu_status()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    vocabularies = [list(t_head.predicates), t_head.available_predicates()]
    configurations = [(arm, i) for i in range(len(vocabularies)) for arm in ARMS]
    times = {key: [] for key in configurations}
    per_round = {key: [] for key in configurations}
    with torch.inference_mode():
        for vocab in vocabularies:
            select_vocab(vocab)
            for index, (frame, boxes) in enumerate(zip(frames, boxsets)):
                feed = t_head.make_feed(frame, boxes)
                reference = t_head._run_engine(feed)
                for arm, raw in [
                    ("PyTorch eager FP32", torch_raw(feed)),
                    ("ONNX CUDA FP32", o_head._run_engine(feed)),
                ]:
                    delta = compare_relation_outputs(
                        reference, raw, atol=1e-2, rtol=1e-4
                    )
                    results["fp32_cross_backend_checks"].append(
                        {
                            "backend": arm,
                            "predicates": len(vocab),
                            "image": paths[index].name,
                            "same_valid_pairs": True,
                            "max_logit_delta": delta,
                            "atol": 0.01,
                            "rtol": 0.0001,
                        }
                    )
            print(
                f"Parity passed: {len(vocab)} predicates, {len(frames)} images",
                flush=True,
            )
        for round_no in range(args.rounds):
            order = list(configurations)
            random.Random(42 + round_no).shuffle(order)
            for arm, vocab_index in order:
                select_vocab(vocabularies[vocab_index])
                before, cooling = cool_down()
                for i in range(args.warmup):
                    call(arm, i)
                torch.cuda.synchronize()
                samples = []
                for i in range(args.iterations):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    call(arm, i)
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - start) * 1000)
                after = gpu_status()
                times[arm, vocab_index].extend(samples)
                per_round[arm, vocab_index].append(float(np.median(samples)))
                results["measurement_blocks"].append(
                    {
                        "backend": arm,
                        "predicates": len(t_head.predicates),
                        "round": round_no + 1,
                        "cooling_seconds": cooling,
                        "gpu_before_warmup": before,
                        "gpu_after_timing": after,
                    }
                )
                print(
                    f"{len(t_head.predicates)} predicates, round {round_no + 1}, {arm}: "
                    f"{np.median(samples):.3f} ms (cooled {cooling:.1f}s)",
                    flush=True,
                )
            # Save partial evidence; incomplete records cannot generate the tables.
            results["rows"] = []
            for (arm, vocab_index), values in times.items():
                results["rows"].append(
                    {
                        "backend": arm,
                        "predicates": len(vocabularies[vocab_index]),
                        "n": len(values),
                        "median_ms": float(np.median(values)),
                        "p95_ms": float(np.percentile(values, 95)),
                        "mean_ms": float(np.mean(values)),
                        "round_medians_ms": per_round[arm, vocab_index],
                        "fps_at_median": 1000 / float(np.median(values)),
                        "samples_ms": values,
                    }
                )
            args.out.write_text(json.dumps(results, indent=2) + "\n")
    results["complete"] = True
    results["gpu_status_after_timing"] = gpu_status()
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Results: {args.out}", flush=True)


if __name__ == "__main__":
    main()
