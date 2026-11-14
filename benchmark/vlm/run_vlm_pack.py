"""Generate the Table 5 Qwen baseline on numbered GT regions using vLLM.

Run in a separate vLLM environment; scoring needs only NumPy. Defaults are the
effective original job settings: bbox_no_label, describe_then_relate, 2048 output
tokens, bf16, greedy decoding, seed 0, at most 60 regions and 16384 context tokens.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from relsgg.paths import resolve_img_dir  # noqa: E402

if __package__:
    from .common import cxcywh_to_xyxy_abs, read_meta, sha256
    from .protocol import _objects_of, annotate_image, build_prompt, parse_relations
else:
    from common import cxcywh_to_xyxy_abs, read_meta, sha256
    from protocol import _objects_of, annotate_image, build_prompt, parse_relations


def select_indices(n_images: int, subsample: int, seed: int, shard: int,
                   num_shards: int, limit: int) -> np.ndarray:
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError("Require 0 <= shard < num_shards")
    if min(subsample, limit) < 0:
        raise ValueError("subsample and limit must be nonnegative")
    indices = np.arange(n_images)
    if subsample and subsample < len(indices):
        indices = np.sort(np.random.default_rng(seed).choice(indices, subsample, replace=False))
    indices = indices[shard::num_shards]
    return indices[:limit] if limit else indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--image_root", help="Override the machine-specific img_dir in meta.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--revision", help="Model commit or tag; resolved commit is saved when available")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--subsample", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_regions", type=int, default=60)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu_mem", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=16384)
    args = parser.parse_args()
    if min(args.max_regions, args.max_new_tokens, args.chunk, args.tp, args.max_model_len) < 1:
        parser.error("region, token, chunk, tensor-parallel and context limits must be positive")
    pack, out = Path(args.pack), Path(args.out)
    sidecar = Path(str(out) + ".meta.json")
    if out.exists() or sidecar.exists():
        raise FileExistsError(f"Use a new --out path; {out} or its metadata already exists")
    meta = read_meta(pack)
    img_meta = np.load(pack / "img_meta.npy", allow_pickle=False)
    boxes = np.load(pack / "boxes.npy", mmap_mode="r", allow_pickle=False)
    files = json.loads((pack / "file_names.json").read_text())
    if len(files) != len(img_meta):
        raise ValueError("file_names.json and img_meta.npy have different lengths")
    indices = select_indices(len(img_meta), args.subsample, args.seed, args.shard,
                             args.num_shards, args.limit)
    if not len(indices):
        raise ValueError("No images selected")
    image_root = Path(args.image_root) if args.image_root else resolve_img_dir(meta)
    # Fail before loading a large model when images have not been relocated.
    for i in indices:
        if not (image_root / files[i]).is_file():
            raise FileNotFoundError(f"{image_root / files[i]} missing; set --image_root")

    from transformers import AutoConfig, AutoProcessor
    from vllm import LLM, SamplingParams

    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    revision = getattr(config, "_commit_hash", None) or args.revision
    processor = AutoProcessor.from_pretrained(args.model, revision=revision)
    llm = LLM(model=args.model, revision=revision, dtype="bfloat16", tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              limit_mm_per_prompt={"image": 1}, seed=args.seed, disable_log_stats=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                              skip_special_tokens=True)

    def template(prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image"},
                                                  {"type": "text", "text": prompt}]}]
        try:
            return processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    versions = {}
    for name in ("vllm", "transformers", "torch", "numpy", "Pillow"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    manifest = {
        "args": vars(args), "resolved_revision": revision, "python": platform.python_version(),
        "versions": versions, "mode": "bbox_no_label",
        "prompt_style": "describe_then_relate",
        "temperature": 0.0, "pack_rows": indices.tolist(), "complete": False,
        "font_sha256": sha256(font) if font.exists() else None,
        "pack_sha256": {n: sha256(pack / n) for n in
                        ("meta.json", "img_meta.npy", "boxes.npy", "file_names.json")},
        "source_sha256": {n: sha256(Path(__file__).with_name(n)) for n in
                          ("run_vlm_pack.py", "protocol.py", "common.py")},
        "prompt": build_prompt(3),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(manifest, indent=2) + "\n")
    started, n_done = time.time(), 0
    with out.open("x") as stream:
        for start in range(0, len(indices), args.chunk):
            requests, keep = [], []
            for i in indices[start:start + args.chunk]:
                image_id, _, _, boff, nb, _, _ = (int(v) for v in img_meta[i])
                n = min(nb, args.max_regions)
                with Image.open(image_root / files[i]) as source:
                    image = source.convert("RGB")
                width, height = image.size
                xyxy = cxcywh_to_xyxy_abs(boxes[boff:boff + n], width, height)
                objects = [{"bbox": [int(v) for v in b], "label": ""} for b in xyxy]
                shown = annotate_image(image, objects)
                prompt = build_prompt(n)
                record = {"pack_row": int(i), "image_id": image_id, "file": files[i],
                          "n_regions": n, "n_boxes_pack": nb, "width": width, "height": height}
                if n < 2:
                    record.update(raw="", relations=[], objects=[], n_tokens=0, truncated=False,
                                  skipped="fewer_than_two_regions")
                    stream.write(json.dumps(record) + "\n")
                    n_done += 1
                    continue
                requests.append({"prompt": template(prompt), "multi_modal_data": {"image": shown}})
                keep.append(record)
            if requests:
                outputs = llm.generate(requests, sampling)
                if len(outputs) != len(keep):
                    raise RuntimeError("vLLM returned a different number of outputs than requests")
                for record, output in zip(keep, outputs):
                    generation = output.outputs[0]
                    record.update(
                        raw=generation.text, relations=parse_relations(generation.text, record["n_regions"]),
                        objects=_objects_of(generation.text), n_tokens=len(generation.token_ids),
                        truncated=len(generation.token_ids) >= args.max_new_tokens,
                    )
                    stream.write(json.dumps(record) + "\n")
                    n_done += 1
            stream.flush()
            print(f"{n_done}/{len(indices)} images, {time.time() - started:.1f}s", flush=True)
    manifest.update(complete=True, n_images=n_done, output_sha256=sha256(out))
    sidecar.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
