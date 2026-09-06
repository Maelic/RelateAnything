"""Build every model file the browser demo (github.com/Maelic/RelateAnything_demo) loads.

One script, seven steps, each with a numerical check against the graph it was made from:

  1. relation head   deploy/dist/<dist>/relateanything.onnx  -> fp16-STORED graph (weights fp16 + Cast, compute fp32)
  2. predicate text  runs/packed/text_student_v2_512/student.pt + the checkpoint's gate MLP -> predicate_encoder.onnx
                     (ids [templates, len] -> W [512], alpha []); parity vs the shipped predicate bank
  3. detectors       ultralytics weights -> ONNX with a browser-friendly head:
                       * yolo26 n/s: ultralytics export, [1,300,6] xyxy/conf/cls (NMS-free)
                       * yoloe-11s prompt-free: ultralytics export + in-graph max/argmax over 4,585 classes, masks dropped
                       * yoloe-11s text-prompt: OWN export with the class vocabulary as a graph INPUT
                                                 (txt_feats [1,C,512] = raw MobileCLIP text features; reprta runs inside)
                       * yolov8s-worldv2 MEGASG-497: deploy/dist detector.onnx + in-graph max/argmax
                       * FastSAM-s: ultralytics export + surgery over its ONE class; segments
                                                 everything and names nothing, so the page labels each instance
                                                 by its own colour (labels="color" in the manifest)
  4. MobileCLIP-BLT text tower (TorchScript -> ONNX), so the page can encode class names that are not in the bank
  5. class bank      MobileCLIP features for COCO + Objects365 + LVIS + OpenImages + MEGASG names (1,760), fp16
  6. banks -> JSON   predicate_bank.npz, corpus_type_map.json, CLIP vocab/merges for the JS tokenizer
  7. package         fp16-store everything, split any Concat wider than WebGPU's storage-buffer budget,
                     chunk files > 90 MB into ONNX external data (GitHub's 100 MB limit),
                     write models/manifest.json with sizes + sha256

Why fp16-STORED rather than an fp16 graph: onnxruntime-web's WASM (CPU) provider has no fp16 kernels for most ops,
so an fp16 graph only runs on WebGPU. Storing weights as fp16 initializers followed by Cast(to=float) halves the
download and runs on both providers; ORT constant-folds the Casts at session creation. Measured: relation head
top-20 Jaccard 1.000 vs fp32, detectors 100% matched detections (IoU>=0.5, same label) on real photos.

Why the class vocabulary is a graph input: ultralytics' exporter fuses the text embeddings into the class-head conv
(static classes). Keeping `txt_feats` as an input makes "re-parameterise to my classes" a matter of feeding different
rows — no re-export — exactly like the relation head's W/alpha inputs (deploy/export_onnx.py --vocab-mode input).

    python deploy/web/export_web_models.py --site../RelateAnything_site --scratch /tmp/ra_web_export

Needs the training venv (torch, ultralytics 8.4.x, onnx, onnxruntime, clip) and internet for the ultralytics assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
CHUNK = 90 * 1024 * 1024
DETECTOR_WEIGHTS = ["yolo26n.pt", "yolo26s.pt", "yoloe-11s-seg.pt", "yoloe-11s-seg-pf.pt",
                    "yolo12s.pt", "yolo26s-seg.pt", "FastSAM-s.pt"]
DEFAULT_CKPT = "runs/train/relsgg-vits16plus/model.pth"


# --------------------------------------------------------------------------- onnx helpers
def store_fp16(m, min_size: int = 1024):
    """fp32 initializers (>= min_size elements) -> fp16 initializer + Cast(to=float). Compute stays fp32."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    g, casts = m.graph, []
    for t in list(g.initializer):
        if t.data_type == TensorProto.FLOAT and int(np.prod(t.dims)) >= min_size:
            a = numpy_helper.to_array(t)
            a16 = a.astype(np.float16)
            if not np.isfinite(a16).all():
                print(f"    [fp16] keeping {t.name} in fp32 (overflow)")
                continue
            g.initializer.remove(t)
            t16 = numpy_helper.from_array(a16, t.name + "_f16")
            g.initializer.append(t16)
            casts.append(helper.make_node("Cast", [t16.name], [t.name], name="w16cast_" + t.name, to=TensorProto.FLOAT))
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    return m


def head_surgery(src: str, dst: str, n_classes: int, raw_out: str = "output0", input_name: str = "images",
                 masks: bool = False, proto_out: str = "output1", n_mask: int = 32):
    """Raw ultralytics head [1, 4+C(+M), A] -> boxes [1,4,A] cxcywh px, conf [1,A], cls [1,A].

    With masks=True the segmentation branch is kept as well: mask_coef [1,M,A] (the trailing M channels of the
    raw head) and proto [1,M,mh,mw]. The host multiplies the two for the kept detections, which is how
    ultralytics' ops.process_mask does it — doing it in-graph would need the post-NMS indices the graph does
    not have. Without masks=True the whole prototype branch is pruned away instead.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    m = onnx.load(src)
    g = m.graph
    for name, arr in (("_s0", [0]), ("_s4", [4]), ("_sC", [4 + n_classes]),
                      ("_sCM", [4 + n_classes + n_mask]), ("_ax1", [1])):
        g.initializer.append(numpy_helper.from_array(np.array(arr, np.int64), name))
    g.node.extend([
        helper.make_node("Slice", [raw_out, "_s0", "_s4", "_ax1"], ["boxes"], name="ra_boxes"),
        helper.make_node("Slice", [raw_out, "_s4", "_sC", "_ax1"], ["_cls_scores"], name="ra_cls_slice"),
        helper.make_node("ReduceMax", ["_cls_scores"], ["conf"], name="ra_conf", axes=[1], keepdims=0),
        helper.make_node("ArgMax", ["_cls_scores"], ["cls"], name="ra_cls", axis=1, keepdims=0),
    ])
    outs = ["boxes", "conf", "cls"]
    vis = [helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 4, None]),
           helper.make_tensor_value_info("conf", TensorProto.FLOAT, [1, None]),
           helper.make_tensor_value_info("cls", TensorProto.INT64, [1, None])]
    if masks:
        g.node.append(helper.make_node("Slice", [raw_out, "_sC", "_sCM", "_ax1"], ["mask_coef"], name="ra_coef"))
        g.node.append(helper.make_node("Identity", [proto_out], ["proto"], name="ra_proto"))
        outs += ["mask_coef", "proto"]
        vis += [helper.make_tensor_value_info("mask_coef", TensorProto.FLOAT, [1, n_mask, None]),
                helper.make_tensor_value_info("proto", TensorProto.FLOAT, [1, n_mask, None, None])]
    del g.output[:]
    g.output.extend(vis)
    tmp = dst + ".tmp.onnx"
    onnx.save(m, tmp)
    m2 = onnx.utils.Extractor(onnx.load(tmp)).extract_model([input_name], outs)
    os.remove(tmp)
    onnx.checker.check_model(m2)
    onnx.save(m2, dst)
    import onnxruntime as ort
    x = np.random.RandomState(0).rand(1, 3, 640, 640).astype(np.float32)
    ref = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(None, {input_name: x})
    raw = ref[0]
    got = ort.InferenceSession(dst, providers=["CPUExecutionProvider"]).run(None, {input_name: x})
    assert np.abs(got[0] - raw[:,:4]).max() == 0 and np.abs(got[1] - raw[:, 4:4 + n_classes].max(1)).max() == 0, "surgery parity"
    if masks:
        assert np.abs(got[3] - raw[:, 4 + n_classes:4 + n_classes + n_mask]).max() == 0, "mask coef parity"
        assert np.abs(got[4] - ref[1]).max() == 0, "proto parity"
    print(f"    surgery ok: {os.path.basename(dst)} ({os.path.getsize(dst)/1e6:.1f} MB)"
          f"{' + masks ' + str(got[4].shape) if masks else ''}")


def split_wide_concats(m, max_inputs: int = 6):
    """Concat(N inputs) -> a tree of Concats with at most `max_inputs` each. Exact, no numerics change.

    onnxruntime-web's WebGPU Concat kernel binds one storage buffer per input plus one for the output, and
    WebGPU only guarantees 8 storage buffers per compute stage (Intel iGPUs commonly report 8-10). The
    relation graph has two 19-input Concats in the geometry/pair features, which ask for 20 and so cannot be
    compiled there -- and because WebGPU shaders are compiled lazily, that surfaces at the FIRST inference,
    not at session creation. max_inputs=6 keeps every Concat at 7 buffers, one under the guaranteed floor.
    """
    from onnx import helper
    g, split, uid = m.graph, [], 0
    while True:
        wide = [n for n in g.node if n.op_type == "Concat" and len(n.input) > max_inputs]
        if not wide:
            break
        for nd in wide:
            axis = next(a.i for a in nd.attribute if a.name == "axis")
            at = next(k for k, x in enumerate(g.node) if x is nd)
            ins, parts, new_nodes = list(nd.input), [], []
            for j in range(0, len(ins), max_inputs):
                grp = ins[j:j + max_inputs]
                if len(grp) == 1:
                    parts.append(grp[0])
                    continue
                uid += 1
                out = f"{nd.output[0]}_cat{uid}"
                new_nodes.append(helper.make_node("Concat", grp, [out], name=f"{nd.name}_cat{uid}", axis=axis))
                parts.append(out)
            new_nodes.append(helper.make_node("Concat", parts, [nd.output[0]], name=nd.name, axis=axis))
            split.append((nd.name, len(ins)))
            g.node.remove(nd)
            for k, node in enumerate(new_nodes):
                g.node.insert(at + k, node)
    for name, n in split:
        print(f"    [webgpu] split Concat {name} ({n} inputs) into a tree of <= {max_inputs}")
    return m


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def emit(m, name: str, models_dir: str):
    """Save `name.onnx` (+ `name.data0..N` external chunks < CHUNK when the weights exceed CHUNK)."""
    import onnx
    from onnx import numpy_helper
    from onnx.external_data_helper import set_external_data, write_external_data_tensors
    split_wide_concats(m)                 # keep every kernel inside WebGPU's storage-buffer budget
    # ultralytics stamps an export timestamp into metadata_props, which makes an otherwise byte-identical
    # re-export a fresh 20 MB in the Pages repo. Drop it so a rebuild only changes what actually changed.
    keep = [q for q in m.metadata_props if q.key != "date"]
    del m.metadata_props[:]; m.metadata_props.extend(keep)
    for f in os.listdir(models_dir):      # onnx APPENDS to an existing chunk, so a rebuild must clear them first
        if f.startswith(name + ".data"):
            os.remove(os.path.join(models_dir, f))
    nbytes = lambda t: len(t.raw_data) if t.raw_data else numpy_helper.to_array(t).nbytes
    total = sum(nbytes(t) for t in m.graph.initializer)
    if total > CHUNK:
        idx, used = 0, 0
        for t in m.graph.initializer:
            nb = nbytes(t)
            if nb < 1024:
                continue
            if used + nb > CHUNK:
                idx, used = idx + 1, 0
            set_external_data(t, location=f"{name}.data{idx}")
            used += nb
        write_external_data_tensors(m, models_dir)
    onnx.save(m, os.path.join(models_dir, name + ".onnx"))
    files = [os.path.join(models_dir, name + ".onnx")]
    files += sorted(os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.startswith(name + ".data"))
    for f in files:
        os.chmod(f, 0o644)
    return [{"path": "models/" + os.path.basename(f), "size": os.path.getsize(f), "sha256": sha256(f)} for f in files]


# --------------------------------------------------------------------------- steps
def step_relation(args, scratch):
    import onnx
    src = os.path.join(args.dist, "relateanything.onnx")
    dst = os.path.join(scratch, "relateanything_w16.onnx")
    onnx.save(store_fp16(onnx.load(src)), dst)
    print(f"[1] relation fp16-stored: {os.path.getsize(src)/1e6:.0f} -> {os.path.getsize(dst)/1e6:.0f} MB")
    return src, dst


def step_predicate_encoder(args, scratch):
    import onnx
    import onnxruntime as ort
    import torch
    from relsgg.vocabulary import TRAIN_TEMPLATES
    from relsgg.text.student import PredicateTextStudent
    student = PredicateTextStudent.from_checkpoint(args.student)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    sd = sd.get("ema_model") or sd["model"]
    g = {k: sd[k].float().clone() for k in sd if k.startswith("vocab_head.gate_mlp")}

    class PredicateEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.s = student
            self.g0, self.g2 = torch.nn.Linear(512, 128), torch.nn.Linear(128, 1)
            with torch.no_grad():
                self.g0.weight.copy_(g["vocab_head.gate_mlp.0.weight"]); self.g0.bias.copy_(g["vocab_head.gate_mlp.0.bias"])
                self.g2.weight.copy_(g["vocab_head.gate_mlp.2.weight"]); self.g2.bias.copy_(g["vocab_head.gate_mlp.2.bias"])

        def forward(self, ids):                       # ids [T, L]: the T template renderings of ONE predicate
            e = self.s(ids, ids == 0)
            w = torch.nn.functional.normalize(e.sum(0), dim=-1)
            return w, torch.sigmoid(self.g2(torch.nn.functional.gelu(self.g0(w)))).reshape(())

    torch.backends.mha.set_fastpath_enabled(False)
    enc = PredicateEncoder().eval()
    tok = lambda p: student.tokenize([t.format(p=p) for t in TRAIN_TEMPLATES])[0]
    dst = os.path.join(scratch, "predicate_encoder.onnx")
    torch.onnx.export(enc, (tok("holding"),), dst, input_names=["ids"], output_names=["W", "alpha"],
                      dynamic_axes={"ids": {0: "templates", 1: "length"}}, opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(dst))
    z = np.load(os.path.join(args.dist, "predicate_bank.npz"), allow_pickle=True)
    s = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    worst, da = 1.0, 0.0
    for i, n in enumerate(z["names"]):
        W, a = s.run(None, {"ids": tok(str(n)).numpy()})
        worst = min(worst, float(W @ z["W"][i])); da = max(da, abs(float(a) - float(z["alpha"][i])))
    assert worst > 0.9999, f"predicate encoder does not reproduce the bank (min cos {worst})"
    print(f"[2] predicate encoder: {os.path.getsize(dst)/1e6:.1f} MB, bank parity min cos {worst:.6f} max|dalpha| {da:.1e}")
    return dst, list(TRAIN_TEMPLATES), student.max_len


def step_detectors(args, scratch):
    import onnx
    import onnxruntime as ort
    import torch
    import torch.nn as nn
    from ultralytics import FastSAM, YOLO, YOLOE
    from ultralytics.nn.modules import Detect
    from ultralytics.utils.downloads import attempt_download_asset
    wdir = os.path.join(scratch, "weights"); os.makedirs(wdir, exist_ok=True)
    cwd = os.getcwd(); os.chdir(wdir)
    try:
        for w in DETECTOR_WEIGHTS:
            attempt_download_asset(w)
        out = {}
        # yolo26 n/s: end-to-end, [1,300,6]
        for k in ("n", "s"):
            # ultralytics returns a path relative to the cwd it exported in; make it absolute before the chdir back
            p = os.path.join(wdir, os.path.basename(str(YOLO(f"yolo26{k}.pt").export(
                format="onnx", imgsz=640, opset=17, simplify=False, dynamic=False, nms=False))))
            out[f"yolo26{k}"] = (p, "e2e_xyxy", None)
        # yolo12s: classic head, [1, 4+80, A] -> surgery -> boxes/conf/cls (host-side NMS)
        p = os.path.join(wdir, os.path.basename(str(YOLO("yolo12s.pt").export(
            format="onnx", imgsz=640, opset=17, simplify=False, dynamic=False, nms=False))))
        head_surgery(p, os.path.join(wdir, "yolo12s_web.onnx"), 80, masks=False)
        out["yolo12s"] = (os.path.join(wdir, "yolo12s_web.onnx"), "boxes_conf_cls", None)
        # the -seg end-to-end heads need NO surgery: [1, 300, 4+1+1+32] rows + [1, 32, 160, 160] prototypes,
        # i.e. NMS-free detection AND masks, with the coefficients already reduced to the kept 300 rows.
        # NOTE yoloe-26s-seg-pf is deliberately NOT here: its end-to-end export does not reduce the 4,585
        # class columns the way predict() does — every one of the 300 rows comes back with conf 0.82-0.92 and
        # 298 of them labelled "technician", against 8 correct detections from the.pt path. The 11s variant
        # works because head_surgery does that reduction explicitly instead of trusting the exporter.
        for wname, mid, cls in (("yolo26s-seg.pt", "yolo26s-seg", YOLO),):
            net = cls(wname)
            names = [net.names[i] for i in range(len(net.names))]
            p = os.path.join(wdir, os.path.basename(str(net.export(
                format="onnx", imgsz=640, opset=17, simplify=False, dynamic=False, nms=False))))
            out[mid] = (p, "e2e_xyxy_seg", names if len(names) > 80 else None)
        # FastSAM-s: a YOLOv8-seg trained on SA-1B with a single class, so the surgery is the ordinary raw-head
        # one with n_classes=1. It is the SAM-family model that fits a browser: SAM/SAM2/MobileSAM are
        # prompt-driven, and "segment everything" there means running the mask decoder over a point grid.
        p = os.path.join(wdir, os.path.basename(str(FastSAM("FastSAM-s.pt").export(
            format="onnx", imgsz=640, opset=17, simplify=False, dynamic=False, nms=False))))
        head_surgery(p, os.path.join(wdir, "fastsam-s_web.onnx"), 1, masks=True)
        out["fastsam-s"] = (os.path.join(wdir, "fastsam-s_web.onnx"), "boxes_conf_cls", ["object"])
        # yoloe prompt-free: raw head + surgery
        pf = YOLOE("yoloe-11s-seg-pf.pt")
        p = os.path.join(wdir, os.path.basename(str(pf.export(
            format="onnx", imgsz=640, opset=17, simplify=False, dynamic=False, nms=False))))
        names = [pf.names[i] for i in range(len(pf.names))]
        head_surgery(p, os.path.join(wdir, "yoloe-11s-pf_web.onnx"), len(names), masks=True)
        out["yoloe-11s-pf"] = (os.path.join(wdir, "yoloe-11s-pf_web.onnx"), "boxes_conf_cls", names)
        # yoloe text-prompt: own export, vocabulary as input
        def prep(model, imgsz=640):
            for mod in model.modules():
                if isinstance(mod, Detect):
                    mod.dynamic = False; mod.export = True; mod.format = "onnx"; mod.xyxy = False; mod.shape = None
                    mod.max_det = min(300, sum(int(imgsz / s) ** 2 for s in model.stride.tolist())); mod.agnostic_nms = False
            return model

        class DynVocabYOLOE(nn.Module):
            """YOLOE with the class vocabulary as an INPUT. A YOLOESegModel returns (detect_out, proto) in export
            mode, so the segmentation branch comes along for free — the trailing 32 channels of the detect output
            are the per-anchor mask coefficients."""

            def __init__(self, net):
                super().__init__(); self.net = net

            def forward(self, images, txt_feats):
                out = self.net.predict(images, tpe=txt_feats)
                y, proto = (out[0], out[1]) if isinstance(out, (tuple, list)) else (out, None)
                if isinstance(y, (tuple, list)):
                    y = y[0]
                C = txt_feats.shape[1]
                conf, cls = y[:, 4:4 + C].max(1)
                if proto is None:
                    return y[:,:4], conf, cls
                return y[:,:4], conf, cls, y[:, 4 + C:4 + C + proto.shape[1]], proto

        m = YOLOE("yoloe-11s-seg.pt"); net = prep(m.model.eval().float())
        probe = ["person", "chair", "cup", "laptop", "dog", "bicycle", "umbrella", "hat"]
        pre = net.get_text_pe(probe, cache_clip_model=True, without_reprta=True).detach().clone().float()
        wrapper = DynVocabYOLOE(net).eval()
        im = torch.rand(1, 3, 640, 640)
        with torch.no_grad():
            b, c, k, coef, proto = wrapper(im, pre)
        print(f"    dyn-vocab outputs: boxes {tuple(b.shape)} coef {tuple(coef.shape)} proto {tuple(proto.shape)}")
        m2 = YOLOE("yoloe-11s-seg.pt"); m2.set_classes(probe, m2.get_text_pe(probe)); net2 = prep(m2.model.eval().float()); net2.model[-1].fuse(net2.pe)
        with torch.no_grad():
            ref = net2(im); ref = ref[0] if isinstance(ref, (tuple, list)) else ref
        assert (b - ref[:,:4]).abs().max() < 1e-3 and (c - ref[:, 4:4 + len(probe)].max(1)[0]).abs().max() < 1e-4, "dyn-vocab path != fused path"
        dst = os.path.join(wdir, "yoloe-11s_dynvocab_web.onnx")
        with torch.no_grad():
            torch.onnx.export(wrapper, (im, pre), dst, input_names=["images", "txt_feats"],
                              output_names=["boxes", "conf", "cls", "mask_coef", "proto"],
                              dynamic_axes={"txt_feats": {1: "num_classes"}}, opset_version=17, dynamo=False, do_constant_folding=True)
        onnx.checker.check_model(onnx.load(dst))
        oo = ort.InferenceSession(dst, providers=["CPUExecutionProvider"]).run(None, {"images": im.numpy(), "txt_feats": pre.numpy()})
        assert np.abs(oo[1] - c.numpy()).max() < 1e-3, "onnx dyn-vocab parity"
        assert np.abs(oo[4] - proto.numpy()).max() < 1e-2, "onnx proto parity"
        print(f"    dyn-vocab yoloe-11s: parity vs fused ultralytics path ok ({os.path.getsize(dst)/1e6:.1f} MB)")
        out["yoloe-11s-text"] = (dst, "boxes_conf_cls", None)
    finally:
        os.chdir(cwd)
    # YOLO-World MEGASG-497 from the laptop bundle
    src = os.path.join(args.dist, "detector.onnx")
    classes = json.load(open(os.path.join(args.dist, "detector.json")))["classes"]
    dst = os.path.join(scratch, "yoloworld-s-megasg497_web.onnx")
    head_surgery(src, dst, len(classes))
    out["yoloworld-s-megasg497"] = (dst, "boxes_conf_cls", classes)
    print(f"[3] detectors: {list(out)}")
    return out


def step_mobileclip(args, scratch):
    import onnx
    import onnxruntime as ort
    import torch
    import clip
    enc = torch.jit.load(args.mobileclip, map_location="cpu")
    tok = clip.clip.tokenize(["a photo of a cat"])
    dst = os.path.join(scratch, "mobileclip_text.onnx")
    torch.onnx.export(enc, (tok,), dst, input_names=["tokens"], output_names=["feat"],
                      dynamic_axes={"tokens": {0: "batch"}, "feat": {0: "batch"}}, opset_version=17, dynamo=False)
    with torch.no_grad():
        ref = enc(tok).numpy()
    z = ort.InferenceSession(dst, providers=["CPUExecutionProvider"]).run(None, {"tokens": tok.numpy()})[0]
    assert np.abs(z - ref).max() < 1e-4
    print(f"[4] mobileclip text tower: {os.path.getsize(dst)/1e6:.0f} MB, parity {np.abs(z - ref).max():.1e}")
    return dst


def step_class_bank(args, scratch):
    import torch
    import yaml
    import ultralytics
    from ultralytics import YOLOE
    UL = os.path.dirname(ultralytics.__file__)

    def ynames(f):
        d = yaml.safe_load(open(f"{UL}/cfg/datasets/{f}"))["names"]
        return [d[k] for k in sorted(d)] if isinstance(d, dict) else list(d)
    sources = {"coco": ynames("coco.yaml"), "objects365": ynames("Objects365.yaml"), "lvis": ynames("lvis.yaml"),
               "openimages": ynames("open-images-v7.yaml"), "megasg": json.load(open(os.path.join(REPO, "deploy/megasg_categories.json")))}
    norm = lambda s: re.sub(r"\s+", " ", s.replace("_", " ").strip().lower())
    names, origin = [], {}
    for src, lst in sources.items():
        for n in lst:
            k = norm(n)
            if k not in origin:
                origin[k] = []; names.append(k)
            origin[k].append(src)
    net = YOLOE(os.path.join(scratch, "weights", "yoloe-11s-seg.pt")).model.eval()
    with torch.no_grad():
        pre = net.get_text_pe(names, batch=256, cache_clip_model=True, without_reprta=True)[0].cpu().numpy().astype(np.float16)
    print(f"[5] class bank: {len(names)} names from {list(sources)}")
    return names, [",".join(origin[n]) for n in names], pre


def step_banks(args, site):
    """predicate bank -> JSON, corpus type map, CLIP tokenizer files."""
    z = np.load(os.path.join(args.dist, "predicate_bank.npz"), allow_pickle=True)
    fin = lambda v, nd: None if not np.isfinite(v) else round(float(v), nd)
    bank = {"names": [str(x) for x in z["names"]], "dim": int(z["W"].shape[1]), "W": np.round(z["W"].astype(np.float32), 6).tolist(),
            "alpha": np.round(z["alpha"], 6).tolist(), "thr": [fin(v, 5) for v in z["thr"]], "is_spatial": z["is_spatial"].astype(int).tolist(),
            "type_source": [str(x) for x in z["type_source"]], "recall": [fin(v, 4) for v in z["recall"]], "gt": z["gt"].tolist(),
            "default": [str(x) for x in z["default"]], "checkpoint": str(z["checkpoint"]), "student": str(z["student"])}
    bd = os.path.join(site, "assets", "banks"); os.makedirs(bd, exist_ok=True)
    json.dump(bank, open(os.path.join(bd, "predicate_bank.json"), "w"), separators=(",", ":"))
    shutil.copy(os.path.join(REPO, "deploy", "corpus_type_map.json"), os.path.join(bd, "corpus_type_map.json"))
    td = os.path.join(site, "assets", "tokenizer"); os.makedirs(td, exist_ok=True)
    from transformers import CLIPTokenizer
    tk = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    tk.save_pretrained(td)                                  # writes vocab.json + merges.txt (+ configs) in this transformers version
    for f in os.listdir(td):
        if f not in ("vocab.json", "merges.txt"):
            os.remove(os.path.join(td, f))
    assert os.path.exists(os.path.join(td, "vocab.json")) and os.path.exists(os.path.join(td, "merges.txt"))
    print("[6] banks + tokenizer files written")
    return bank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, help="checkout of Maelic/RelateAnything (the Pages repo)")
    ap.add_argument("--scratch", default="/tmp/ra_web_export")
    ap.add_argument("--dist", default="deploy/dist/vits16plus", help="laptop bundle: relateanything.onnx/.json, predicate_bank.npz, detector.onnx/.json")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT, help="relation checkpoint (gate MLP for the predicate encoder)")
    ap.add_argument("--student", default="runs/packed/text_student_v2_512/student.pt")
    ap.add_argument("--mobileclip", default="checkpoints/detectors/mobileclip_blt.ts")
    ap.add_argument("--skip", nargs="*", default=[], help="steps to skip: relation predicate detectors mobileclip classbank")
    args = ap.parse_args()
    os.chdir(REPO)
    import onnx
    import torch
    torch.set_num_threads(4)
    scratch, site = os.path.abspath(args.scratch), os.path.abspath(args.site)
    os.makedirs(scratch, exist_ok=True)
    models = os.path.join(site, "models"); os.makedirs(models, exist_ok=True)
    # Merge into whatever is already published so a partial rebuild (--skip) does not drop entries.
    mpath = os.path.join(models, "manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    manifest.update(version=2, generated=time.strftime("%Y-%m-%d"))
    manifest.setdefault("relation", None); manifest.setdefault("detectors", []); manifest.setdefault("text_encoders", {})

    bank = step_banks(args, site)
    rj = json.load(open(os.path.join(args.dist, "relateanything.json")))
    if "relation" not in args.skip:
        _, dst = step_relation(args, scratch)
        files = emit(onnx.load(dst), "relateanything_vits16plus_w16", models)
        manifest["relation"] = {"id": "relsgg-vits16plus", "name": "RelateAnything ViT-S+ (DINOv3, full recipe)", "precision": "fp16 weights, fp32 compute", "files": files,
            "meta": {"img_size": rj["img_size"], "max_boxes": rj["max_boxes"], "final_budget": rj["final_budget"], "text_dim": rj["text_dim"], "calibration": rj["calibration"],
                     "score_contract": rj["score_contract"], "default_predicates": bank["default"], "run_name": rj["run_name"], "git_sha": rj["git_sha"], "epoch": rj["epoch"],
                     "backbone": "DINOv3 ViT-S+/16 (LVD-1689M)", "text_student_sha256": rj["text_student_sha256"]},
            "bank": "assets/banks/predicate_bank.json", "license": "DINOv3 License (backbone derivative)", "license_url": "https://ai.meta.com/resources/models-and-libraries/dinov3-license/"}
    if "predicate" not in args.skip:
        dst, templates, max_len = step_predicate_encoder(args, scratch)
        files = emit(store_fp16(onnx.load(dst)), "predicate_encoder_w16", models)
        manifest["text_encoders"]["predicate"] = {"id": "predicate-student-v2-512", "name": "Predicate text encoder (distilled dino.txt student)", "files": files,
            "input": "ids [templates, length] int64 (CLIP BPE, max_len 32, pad 0)", "outputs": "W [512] unit-norm, alpha [] gate", "templates": templates, "max_len": max_len,
            "license": "DINOv3 License (dino.txt distillation derivative)"}
    if "detectors" not in args.skip:
        manifest["detectors"] = []
        dets = step_detectors(args, scratch)
        import yaml, ultralytics
        coco = yaml.safe_load(open(os.path.join(os.path.dirname(ultralytics.__file__), "cfg/datasets/coco.yaml")))["names"]; coco = [coco[i] for i in range(80)]
        # default_conf: measured on the sample photos so every detector opens with a usable number of objects.
        # The MEGASG-497 head in particular spreads its scores over 497 classes and yields ~1 object at 0.25.
        meta = {"yolo12s": ("YOLO12s \u00b7 COCO-80", "Attention-centric YOLO12. Closed-set COCO, host-side NMS.", "fixed", "yolo12s.pt", 0.25, False),
                "yolo26s-seg": ("YOLO26s-seg \u00b7 COCO-80", "NMS-free end-to-end head that also returns instance masks \u2014 the cheapest way to get masks.", "fixed", "yolo26s-seg.pt", 0.25, False),
                "yoloe-11s-pf": ("YOLOE-11s prompt-free", "4,585 built-in classes (LVIS+Objects365 vocabulary). Detects 'anything' with no class list.", "fixed", "yoloe-11s-seg-pf.pt", 0.25, True),
                "yoloe-11s-text": ("YOLOE-11s text-prompt", "Open-vocabulary: the class list is a graph input, re-parameterised live from the 1,760-name bank or the MobileCLIP text encoder.", "dynamic", "yoloe-11s-seg.pt", 0.15, False),
                "yoloworld-s-megasg497": ("YOLO-World v2-S · MEGASG-497", "Re-parameterised to the 497 categories the relation model was trained with (the laptop-bundle detector).", "fixed", "yolov8s-worldv2.pt", 0.15, False),
                "yolo26n": ("YOLO26n · COCO-80", "Closed-set COCO detector, NMS-free end-to-end head. Smallest and fastest.", "fixed", "yolo26n.pt", 0.20, False),
                "yolo26s": ("YOLO26s · COCO-80", "Closed-set COCO detector, NMS-free end-to-end head.", "fixed", "yolo26s.pt", 0.20, False),
                "fastsam-s": ("FastSAM-s · class-agnostic", "Segments everything the way SAM does, with no class list at all — the page names each instance by its colour. Relations do not read object names, so the graph is unaffected.", "agnostic", "FastSAM-s.pt", 0.40, False)}
        for mid in ("yoloe-11s-pf", "yoloe-11s-text", "yoloworld-s-megasg497",
                    "yolo26s-seg", "fastsam-s", "yolo12s", "yolo26n", "yolo26s"):
            src, kind, classes = dets[mid]; name, desc, vocab, upstream, dconf, reco = meta[mid]
            files = emit(store_fp16(onnx.load(src)), mid, models)
            if classes is None and vocab == "fixed":       # every fixed-vocabulary head here is COCO-80
                classes = coco
            entry = {"id": mid, "name": name, "description": desc, "kind": kind, "vocab": vocab, "imgsz": 640, "files": files, "precision": "fp16 weights, fp32 compute",
                     "upstream": f"ultralytics/assets {upstream} (ultralytics {ultralytics.__version__})", "license": "AGPL-3.0", "license_url": "https://github.com/ultralytics/ultralytics/blob/main/LICENSE",
                     "default_conf": dconf, **({"recommended": True} if reco else {})}
            if mid.startswith("yoloe") or mid.startswith("fastsam") or kind == "e2e_xyxy_seg":   # graphs that also emit mask coefficients + prototypes
                entry.update(masks=True, mask_dim=32, mask_stride=4)
            if vocab == "agnostic":     # one nameless class: the page reads a colour word off each instance mask
                entry["labels"] = "color"
            if classes is not None:
                json.dump(classes, open(os.path.join(models, f"{mid}.classes.json"), "w"), separators=(",", ":"))
                entry["classes"], entry["num_classes"] = f"models/{mid}.classes.json", len(classes)
            if vocab == "dynamic":
                entry.update(bank="assets/banks/yoloe-11s_classbank.json", text_encoder="mobileclip-blt-text", txt_feats_dim=512)
            manifest["detectors"].append(entry)
    if "mobileclip" not in args.skip:
        dst = step_mobileclip(args, scratch)
        files = emit(store_fp16(onnx.load(dst)), "mobileclip_blt_text_w16", models)
        manifest["text_encoders"]["mobileclip-blt-text"] = {"id": "mobileclip-blt-text", "name": "MobileCLIP-BLT text encoder (YOLOE's)", "files": files,
            "input": "tokens [batch, 77] int32 (CLIP BPE, zero-padded)", "outputs": "feat [batch, 512] unit-norm", "ctx": 77, "license": "Apple ML MobileCLIP license",
            "license_url": "https://github.com/apple/ml-mobileclip/blob/main/LICENSE_weights_data", "upstream": "ultralytics/assets mobileclip_blt.ts (text tower only)"}
    if "classbank" not in args.skip:
        names, origin, pre = step_class_bank(args, scratch)
        bd = os.path.join(site, "assets", "banks")
        pre.tofile(os.path.join(bd, "yoloe-11s_classbank.f16.bin"))
        json.dump({"names": names, "origin": origin, "dim": 512, "dtype": "float16", "bin": "yoloe-11s_classbank.f16.bin",
                   "sha256": sha256(os.path.join(bd, "yoloe-11s_classbank.f16.bin")),
                   "space": "MobileCLIP-BLT text features, pre-reprta, unit-norm"}, open(os.path.join(bd, "yoloe-11s_classbank.json"), "w"), separators=(",", ":"))
    json.dump(manifest, open(os.path.join(models, "manifest.json"), "w"), indent=1)
    total = sum(os.path.getsize(os.path.join(models, f)) for f in os.listdir(models))
    print(f"[7] wrote {models}/manifest.json — {total/1e6:.0f} MB of model files")


if __name__ == "__main__":
    main()
