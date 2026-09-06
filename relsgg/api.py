"""RelateAnything — plug-and-play inference API.

The product surface: give it an image and a set of boxes (from ANY detector or
ground truth) and it returns open-vocabulary relations. No object labels are
needed, and the predicate vocabulary is swappable at will.

Two ways to load:

  * ``RelateAnything.from_deploy("relateanything_deploy.pt")``
        A self-contained bundle produced by ``deploy/prepare_deploy_ckpt.py``:
        the predicate vocabulary is already baked into the head, so no text
        encoder (dino.txt) is needed and inference is pure vision. This is what
        you ship to a laptop / edge device.

  * ``RelateAnything.from_checkpoint(ckpt, predicates, dinotxt_weights=...)``
        A training checkpoint + a predicate list, re-parameterized on the fly
        with the dino.txt text tower. Needs the dino.txt weights present (so
        run it where they are — e.g. the training box).

Then, per frame::

    triplets = ra.predict(image, boxes_xyxy, box_scores=confs, topk=20)
    # -> [Triplet(subject_idx, subject_box, predicate, score, object_idx, object_box), ...]

Re-parameterizing to a new vocabulary at runtime (needs dino.txt):

    ra.set_vocabulary(["holding", "riding", "next to", ...])
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from relsgg.model import RelSGG, RelSGGConfig

try:                                            # optional, only for image decode
    from PIL import Image
except Exception:                               # pragma: no cover
    Image = None

# The template ensemble the training W was built with (must match training).
TRAIN_TEMPLATES = ["{p}", "one object is {p} another object",
                   "a photo of something {p} something"]


@dataclass
class Triplet:
    subject_idx: int
    subject_box: np.ndarray      # xyxy pixels in the ORIGINAL image frame
    predicate: str
    score: float
    object_idx: int
    object_box: np.ndarray       # xyxy pixels
    subject_label: Optional[str] = None
    object_label: Optional[str] = None

    def __repr__(self) -> str:
        s = self.subject_label or f"obj{self.subject_idx}"
        o = self.object_label or f"obj{self.object_idx}"
        return f"({s}) --{self.predicate} [{self.score:.2f}]--> ({o})"


# Config fields that must NEVER be taken from the checkpoint's args by the
# drift guard below. `backbone_pretrained` is a LOAD-TIME choice (the deploy
# path loads trained weights and must not re-download the pretrained tower);
# the other two are resolved explicitly above because their arg spelling or
# their derivation differs.
_CFG_NEVER_FROM_ARGS = frozenset({
    "backbone_pretrained", "backbone_model", "compose_query",
})


def _cfg_from_args(a: dict, backbone_pretrained: bool) -> RelSGGConfig:
    if not isinstance(a, dict):
        a = vars(a)
    cfg = RelSGGConfig(
        backbone_type=a["backbone_type"],
        **({"backbone_model": a["backbone_model"]} if a.get("backbone_model") else {}),
        backbone_pretrained=backbone_pretrained,
        # Fields that CREATE PARAMETERS must round-trip from the checkpoint's
        # own args, or the rebuilt model silently diverges from the trained
        # one: patch_size sets the feature-grid geometry, beta_relatedness
        # allocates beta_mlp, lambda_sigmoid>0 marks the sigmoid-aux head as
        # live. They were previously dropped here — correct only by luck, for
        # checkpoints that happened to use the defaults.
        patch_size=a.get("patch_size", 16),
        beta_relatedness=a.get("beta_relatedness") or False,
        lambda_sigmoid=a.get("lambda_sigmoid") or 0.0,
        n_heads=a.get("n_heads", 8),
        ffn_ratio=a.get("ffn_ratio", 2.0),
        lora_rank=a["lora_rank"],
        lora_layers=a.get("lora_layers"),
        d_model=a["d_model"],
        text_dim=a["text_dim"],
        geo_budget=a["geo_budget"],
        final_budget=a["final_budget"],
        n_self_layers=a["n_self_layers"],
        n_cross_layers=a["n_cross_layers"],
        logit_scale_init=a["logit_scale_init"],
        logit_bias_init=a.get("logit_bias_init") or 0.0,
        sampler_type=a.get("sampler_type", "cascade"),
        rel_neg_weight=a.get("rel_neg_weight", 0.3),
        swap_include=a.get("swap_include", True),
        proj_layers=a.get("proj_layers", 1),
        compose_query=(a.get("loss_type") == "batch_infonce"),
        dual_spatial_head=a.get("dual_spatial_head", False),
        fast_bilinear_head=a.get("fast_bilinear", False),
        lambda_fast=a.get("lambda_fast", 0.5),
        use_rel_interaction=a.get("use_rel_interaction", True),
        n_dep_layers=a.get("n_dep_layers", 2),
        n_gnd_layers=a.get("n_gnd_layers", 1),
        # Same "creates parameters" rule as beta_relatedness above: without
        # these the deformable read is never built and its trained weights land
        # in the ignored-keys bucket, silently deploying a DIFFERENT model than
        # the one that was evaluated (caught on the full-recipe checkpoint,
        # which carries deformable_read.* — [[relsgg-deformable-read-verdict]]).
        deformable_points=a.get("deformable_points") or 0,
        deformable_heads=a.get("deformable_heads", 1),
        deformable_v2=a.get("deformable_v2", False),
    )
    # DRIFT GUARD. Everything above is hand-enumerated, and train.py's
    # build_model() enumerates the SAME config independently — so every new
    # training flag has to be added in two places, and when it is missed here
    # the deploy path rebuilds a differently-shaped model. That is not
    # hypothetical: this function silently lacked `pe_num_freqs` (making every
    # box positional encoding 4x too wide: proj 4*64 instead of 4*16) and the
    # whole `deformable_nulls/clamp/ring/gain/border` group, so the current
    # recipe could not be loaded by the product at all.
    #
    # So: after the explicit block, copy through ANY remaining RelSGGConfig
    # field that the checkpoint's own args carry under the SAME NAME. Explicit
    # entries above always win (they cover the cases where the arg name differs
    # or a value must be derived), and fields absent from args keep the
    # dataclass default, which is what an older checkpoint should get. New
    # same-named flags now round-trip with no edit here.
    for f in dataclasses.fields(RelSGGConfig):
        if f.name in _CFG_NEVER_FROM_ARGS or f.name not in a:
            continue
        v = a[f.name]
        if v is None or f.default is dataclasses.MISSING:
            continue
        cur = getattr(cfg, f.name)
        # Only fill fields the explicit block left AT the dataclass default;
        # anything it deliberately set (including deliberately set TO the
        # default) is already the authority for that field.
        if cur != f.default or cur == v:
            continue
        setattr(cfg, f.name, v)
    return cfg


def _load_state(model: RelSGG, sd: dict, strict_release: bool = False) -> None:
    """Resize freshly-built buffers (W/alpha/...) to match the checkpoint, then
    load. Mirrors eval_zeroshot.build_model_from_ckpt.

    strict_release: release/export paths set this. A print-and-continue on
    unexpected keys is fine for experimentation, but for a shipped artifact a
    dropped tensor IS the bug (a config field not round-tripped through
    _cfg_from_args shows up exactly here, as unexpected checkpoint keys with
    no module to receive them) — fail so it cannot ship.
    """
    for name, t in sd.items():
        mod = model
        *path, leaf = name.split(".")
        for p in path:
            mod = getattr(mod, p, None)
            if mod is None:
                break
        if mod is None:
            continue
        cur = getattr(mod, leaf, None)
        if (leaf in dict(mod.named_buffers(recurse=False))
                and isinstance(cur, torch.Tensor) and cur.shape != t.shape):
            setattr(mod, leaf, torch.empty_like(t))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    unexpected = [k for k in unexpected if "ontology" not in k]
    # vocab_head.beta is a buffer newer code creates with a safe default;
    # a checkpoint that predates it simply never used it. Its absence is
    # benign — its PRESENCE (a trained beta) still loads and still counts.
    strict_missing = [k for k in missing if k != "vocab_head.beta"]
    if strict_release and (strict_missing or unexpected):
        raise RuntimeError(
            "[RelateAnything] strict release load failed — "
            f"missing={strict_missing[:6]} unexpected={unexpected[:6]}. "
            "A release artifact must load every trained tensor; if a key is "
            "listed here, _cfg_from_args is not round-tripping the config "
            "field that creates it.")
    if unexpected:
        print(f"[RelateAnything] unexpected keys ignored: {unexpected[:4]}"
              f"{'...' if len(unexpected) > 4 else ''}")


class RelateAnything:
    def __init__(self, model: RelSGG, predicates: List[str], img_size: int = 448,
                 device: Union[str, torch.device] = "cpu",
                 dinotxt_weights: Optional[str] = None,
                 score_mode: str = "sigmoid",
                 text_student: Optional[str] = None):
        self.model = model.to(device).eval()
        self.predicates = list(predicates)
        self.img_size = img_size
        self.device = torch.device(device)
        self.dinotxt_weights = dinotxt_weights
        # v34+ encodes the vocabulary with the DISTILLED student (768-d), not
        # raw dino.txt (2048-d). Whichever the checkpoint was trained with must
        # be used here — the head's W lives in that space and mixing them
        # silently produces garbage cosines.
        self.text_student = text_student
        self.score_mode = score_mode
        self._type_vec = None      # per-vocabulary two-graph split, lazy
        # Deployment calibration: sigmoid(calib_a * (pred + rel) + calib_b).
        # Identity by default so nothing changes until a fit is installed.
        # The head's own logit_scale/logit_bias are trained by a mass-balanced
        # BCE (equal positive and negative mass per slot), which calibrates it
        # to a 50/50 prior; a deployed frame is 0.2-4% positive, so raw scores
        # saturate into [0.9, 1.0) and the threshold does nothing. Fitting
        # these two on a val split is the missing step, not a hack: monotone,
        # so R@K / mR@K / AP / AUC are bit-identical, while ECE goes
        # 0.92 -> 0.009 and a threshold starts meaning precision.
        # Fit with benchmark/eval_deploy_metrics.py --fit_platt.
        from relsgg.scoring import ScoreContract
        self._set_contract(ScoreContract())

    def _set_contract(self, contract) -> None:
        self.contract = contract
        # keep model.predict on the same contract; older pickled models that
        # predate the attribute simply keep their own default.
        if hasattr(self.model, "set_score_contract"):
            self.model.set_score_contract(contract)

    def set_calibration(self, a: float, b: float) -> None:
        """Install a Platt fit. Ranking-invariant for a > 0 (enforced)."""
        from relsgg.scoring import ScoreContract
        self._set_contract(ScoreContract(calib_a=float(a), calib_b=float(b)))

    # kept so callers and the pipeline can read the two scalars directly
    @property
    def calib_a(self) -> float:
        return self.contract.calib_a

    @property
    def calib_b(self) -> float:
        return self.contract.calib_b

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_deploy(cls, path: str, device: Union[str, torch.device] = "cpu",
                    score_mode: str = "sigmoid",
                    strict_release: bool = False) -> "RelateAnything":
        """Load a self-contained deploy bundle (vocab already baked, no text
        encoder needed)."""
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        a = bundle["args"]; a = dict(a if isinstance(a, dict) else vars(a))
        # Rebuild the backbone architecture from the bundled config (no HF
        # download, no network): write config.json to a temp dir and point the
        # backbone loader at it. Weights come from bundle["model"].
        if bundle.get("backbone_config"):
            import json, tempfile
            d = tempfile.mkdtemp(prefix="ra_backbone_")
            json.dump(bundle["backbone_config"], open(os.path.join(d, "config.json"), "w"))
            a["backbone_model"] = d
        cfg = _cfg_from_args(a, backbone_pretrained=False)
        model = RelSGG(cfg)
        # Mirror the training-time gate so its weights load rather than being
        # reported as unexpected. alpha is already baked (is_reparameterized
        # below), so the MLP is inert at inference — this just keeps the load
        # clean and the bundle round-trippable.
        if any(k.startswith("vocab_head.gate_mlp") for k in bundle["model"]):
            model.vocab_head.build_gate_mlp()
        _load_state(model, bundle["model"], strict_release=strict_release)
        # A --fp16 bundle halves the file; compute stays fp32 (runtime autocasts
        # to bf16 anyway). Parameters are upcast by load_state_dict's copy_, but
        # the dynamically-sized buffers (vocab_head.W/alpha, W_obj) are ASSIGNED
        # and would stay fp16 -> dtype mismatch at the first matmul. Upcast the
        # whole module, not just the params.
        if bundle.get("storage_dtype") == "fp16":
            model.float()
        # W/alpha/gates were baked at prepare time and travel in bundle["model"].
        model.vocab_head.is_reparameterized = True
        # A --embed-text-encoder bundle carries the student AND its CLIP-BPE
        # tokenizer: unpack both into ONE temp dir so the student finds the
        # tokenizer co-located (no Hub, no HF cache) and set_vocabulary() can
        # re-parameterize to arbitrary strings at runtime.
        student_path = None
        if bundle.get("text_encoder"):
            import tempfile
            te = bundle["text_encoder"]
            d = tempfile.mkdtemp(prefix="ra_text_")
            for fname, blob in te.get("tokenizer_files", {}).items():
                with open(os.path.join(d, fname), "wb") as fh:
                    fh.write(blob)
            student_path = os.path.join(d, "student.pt")
            torch.save({"cfg": te["cfg"], "clip_ids": te["clip_ids"],
                        "state_dict": te["state_dict"]}, student_path)
        ra = cls(model, bundle["predicates"], img_size=bundle.get("img_size", 448),
                 device=device, score_mode=bundle.get("score_mode", score_mode),
                 text_student=student_path)
        # A bundle may carry its deployment calibration; without one, a
        # score threshold is not interpretable (see set_calibration).
        cal = bundle.get("calibration")
        if cal:
            ra.set_calibration(float(cal["a"]), float(cal["b"]))
        return ra

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, predicates: Sequence[str],
                        dinotxt_weights: str = "",
                        device: Union[str, torch.device] = "cpu",
                        templates: Optional[List[str]] = None,
                        weights: str = "ema", score_mode: str = "sigmoid",
                        img_size: int = 448,
                        text_student: Optional[str] = None,
                        strict_release: bool = False,
                        embeddings=None) -> "RelateAnything":
        """Load a training checkpoint and re-parameterize to ``predicates``.

        The text encoder must be the one the checkpoint was TRAINED with:
        ``text_student`` (distilled student, v34+) or ``dinotxt_weights``
        (raw dino.txt, v33 and earlier). When ``text_student`` is None it is
        read from the checkpoint's own args, so callers normally need not
        pass it.
        """
        from relsgg.checkpoint import (hub_id_for_backbone,
                                       materialize_backbone_config)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ck_args = ckpt["args"]
        ck_args = dict(ck_args if isinstance(ck_args, dict) else vars(ck_args))
        # Released checkpoints embed the backbone config: build the tower from
        # it and take every weight from the checkpoint (no download, no gated
        # login). Training checkpoints name the pretrained tower instead.
        offline = materialize_backbone_config(ckpt, ck_args)
        if not offline:
            ck_args["backbone_model"] = hub_id_for_backbone(ck_args.get("backbone_model"))
        cfg = _cfg_from_args(ck_args, backbone_pretrained=not offline)
        model = RelSGG(cfg)
        # The trainable spatialness gate is built post-hoc at train time
        # (train.py --gate_mlp), so it does not exist on a freshly constructed
        # model. Build it BEFORE loading, or its weights land in the
        # "unexpected keys" bucket and alpha silently falls back to the frozen
        # logistic probe — a quiet accuracy regression on v34+ checkpoints.
        if ck_args.get("gate_mlp"):
            model.vocab_head.build_gate_mlp()
        sd = ckpt["ema_model"] if (weights == "ema" and "ema_model" in ckpt) else ckpt["model"]
        # beta_mlp mirrors the gate_mlp situation: built post-hoc when
        # --beta_relatedness trained one, so construct it before loading.
        if ck_args.get("beta_relatedness") and any(
                k.startswith("vocab_head.beta_mlp") for k in sd):
            model.vocab_head.build_beta_mlp()
        _load_state(model, sd, strict_release=strict_release)
        if text_student is None:
            text_student = ck_args.get("text_student") or None
        if text_student:
            from relsgg.text_student import resolve_student_path
            text_student = resolve_student_path(text_student, near=ckpt_path)
        ra = cls(model, list(predicates), img_size=img_size, device=device,
                 dinotxt_weights=dinotxt_weights, score_mode=score_mode,
                 text_student=text_student)
        ra.set_vocabulary(predicates, templates=templates, embeddings=embeddings)
        return ra

    # -- vocabulary ---------------------------------------------------------

    def set_vocabulary(self, predicates: Sequence[str],
                       templates: Optional[List[str]] = None,
                       embeddings=None) -> "RelateAnything":
        """Re-parameterize the relation head to a new predicate vocabulary.
        Needs the text encoder the checkpoint was trained with (only at set-up
        time; inference stays pure vision afterwards).

        ``embeddings`` supplies the [V, text_dim] matrix directly and skips the
        encoder. That is the path for a vocabulary that is already encoded — a
        release bundle's ``predicate_bank.npz``, or a stand-in used to measure
        cost — and it is the only path available when the text encoder the
        checkpoint names is not on the machine."""
        if embeddings is None and not self.text_student and not self.dinotxt_weights:
            raise RuntimeError(
                "set_vocabulary needs a text encoder — construct with "
                "from_checkpoint(text_student=...) for v34+ checkpoints or "
                "from_checkpoint(dinotxt_weights=...) for older ones, pass "
                "embeddings=<[V, text_dim]>, or ship a deploy bundle whose "
                "vocabulary is already baked (from_deploy).")
        self.predicates = list(predicates)
        if embeddings is not None:
            W = torch.as_tensor(embeddings, dtype=torch.float32, device=self.device)
            if W.shape[0] != len(self.predicates):
                raise ValueError(f"embeddings has {W.shape[0]} rows but "
                                 f"{len(self.predicates)} predicates were given")
            self.model.vocab_head.set_vocabulary_matrix(self.predicates, W)
        elif self.text_student:
            from relsgg.text_student import encode_texts_student
            W = encode_texts_student(
                self.predicates, self.text_student,
                templates=templates or TRAIN_TEMPLATES, device=self.device)
            self.model.vocab_head.set_vocabulary_matrix(self.predicates, W)
        else:
            self.model.vocab_head.encode_vocabulary_dinotxt(
                self.predicates, dinotxt_weights=self.dinotxt_weights,
                templates=templates or TRAIN_TEMPLATES)
        self.model.reparameterize()
        self._type_vec = None      # vocabulary changed -> recompute lazily
        return self

    def _type_vector(self) -> np.ndarray:
        """[V] bool spatial/semantic split for the CURRENT vocabulary.

        Hybrid rule (measured trade-offs in relsgg/decompose.py): corpus flag
        when the string is known — read from the cached
        relsgg/corpus_type_map.json so no pack access is needed at runtime —
        and the checkpoint's own gate alpha >= 0.5 for novel strings (the
        gate under-routes unseen spatial predicates, so it is the fallback,
        not the default).
        """
        if self._type_vec is not None:
            return self._type_vec
        from relsgg.decompose import type_vector
        cmap = None
        cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "corpus_type_map.json")
        if os.path.exists(cache):
            import json as _json
            cmap = _json.load(open(cache))
        alpha = getattr(self.model.vocab_head, "alpha", None)
        alpha_np = (alpha.detach().cpu().numpy()
                    if isinstance(alpha, torch.Tensor) and
                    alpha.numel() == len(self.predicates) else None)
        is_sp, src = type_vector(self.predicates, corpus_map=cmap,
                                 alpha=alpha_np)
        n = {k: int((src == k).sum()) for k in ("corpus", "gate", "default")}
        print(f"[RelateAnything] two-graph split: {int(is_sp.sum())} spatial "
              f"/ {len(is_sp)} (sources {n})")
        if n["default"] > 0.5 * len(is_sp):
            print("[RelateAnything] WARNING: most predicates untyped (no "
                  "corpus map, no aligned alpha) — the spatial stream will "
                  "be near-empty.")
        self._type_vec = is_sp
        return self._type_vec

    # -- inference ----------------------------------------------------------

    @staticmethod
    def _to_chw(image, size) -> "tuple[torch.Tensor, int, int]":
        """Return (image tensor [1,3,size,size] in [0,1], orig_W, orig_H)."""
        if isinstance(image, np.ndarray):          # HWC uint8 (BGR or RGB)
            arr = image
            H, W = arr.shape[:2]
            if Image is not None:
                pil = Image.fromarray(arr[..., ::-1] if arr.shape[2] == 3 else arr)
                pil = pil.resize((size, size), Image.BILINEAR)
                t = torch.from_numpy(np.asarray(pil, np.float32).transpose(2, 0, 1) / 255.0)
            else:                                   # no PIL: crude resize via torch
                t = torch.from_numpy(arr[..., ::-1].copy().astype(np.float32).transpose(2, 0, 1) / 255.0)
                t = torch.nn.functional.interpolate(t[None], (size, size), mode="bilinear",
                                                    align_corners=False)[0]
            return t.unsqueeze(0), W, H
        # PIL image
        W, H = image.size
        pil = image.convert("RGB").resize((size, size), Image.BILINEAR)
        t = torch.from_numpy(np.asarray(pil, np.float32).transpose(2, 0, 1) / 255.0)
        return t.unsqueeze(0), W, H

    @torch.no_grad()
    def predict(self, image, boxes_xyxy: np.ndarray,
                box_labels: Optional[Sequence[str]] = None,
                box_scores: Optional[np.ndarray] = None,
                topk: int = 20, max_boxes: int = 60,
                decompose: bool = False):
        """Predict open-vocabulary relations.

        Args:
            image:       PIL.Image, or HWC numpy array (BGR from OpenCV is fine).
            boxes_xyxy:  [N, 4] float pixels in the original image frame.
            box_labels:  optional [N] detector class names (for display only).
            box_scores:  optional [N] detector confidences; when given, triplets
                         are ranked by conf(sub)*conf(obj)*pred_score (the SGDet
                         convention — suppresses low-confidence-box pairs).
            topk:        number of triplets to return, ranked by score.
            max_boxes:   cap on boxes fed to the head (top by score if scores given).
            decompose:   False -> one ranked List[Triplet] (a pair emits its
                         single argmax predicate). True -> TWO graphs from the
                         SAME forward pass, {"spatial": [...], "semantic":
                         [...]}, each an independently ranked List[Triplet] of
                         up to `topk`: within each stream the other type's
                         predicate columns are masked out, one argmax edge per
                         pair (the measured type-stratified protocol —
                         benchmark/eval_decomposed.py). A pair may appear in
                         both graphs, holding a layout relation AND an
                         interaction simultaneously; that coexistence is the
                         point of the feature.
        """
        boxes_xyxy = np.asarray(boxes_xyxy, np.float32).reshape(-1, 4)
        N = len(boxes_xyxy)
        if N < 2:
            return []
        if box_scores is not None:
            box_scores = np.asarray(box_scores, np.float32).reshape(-1)
        if N > max_boxes:                          # keep the most confident boxes
            order = (np.argsort(-box_scores) if box_scores is not None
                     else np.arange(N))[:max_boxes]
            boxes_xyxy = boxes_xyxy[order]
            box_scores = box_scores[order] if box_scores is not None else None
            box_labels = [box_labels[i] for i in order] if box_labels is not None else None
            N = max_boxes

        img_t, W, H = self._to_chw(image, self.img_size)
        img_t = img_t.to(self.device)

        # xyxy px -> normalized cxcywh (features are scale-invariant to resize)
        b = boxes_xyxy.copy()
        b[:, [0, 2]] /= max(W, 1); b[:, [1, 3]] /= max(H, 1)
        cx = (b[:, 0] + b[:, 2]) / 2; cy = (b[:, 1] + b[:, 3]) / 2
        bw = (b[:, 2] - b[:, 0]);     bh = (b[:, 3] - b[:, 1])
        boxes_t = torch.from_numpy(np.stack([cx, cy, bw, bh], -1).astype(np.float32))
        boxes_t = boxes_t.unsqueeze(0).to(self.device)                 # [1,N,4]
        box_counts = torch.tensor([N], device=self.device)

        out = self.model(img_t, boxes_t, box_counts=box_counts, targets=None)
        logits = out["logits"][0].float()          # [K, V]
        sub_idx = out["sub_idx"][0].cpu().numpy()   # [K]
        obj_idx = out["obj_idx"][0].cpu().numpy()
        valid = out["valid_mask"][0].cpu().numpy().astype(bool)

        if decompose:
            # Log-space fusion (logits + pair_logits) — bit-identical to the
            # measured protocol in benchmark/eval_decomposed.py, so the api
            # path reproduces the evaluated semantics exactly.
            from relsgg.decompose import split_ranked
            fused = logits.clone()
            if out.get("pair_logits") is not None:
                fused = fused + out["pair_logits"][0].float().unsqueeze(-1)
            keep = valid.copy()
            si_np, oi_np = sub_idx, obj_idx
            keep &= (si_np < N) & (oi_np < N) & (si_np != oi_np)
            streams = split_ranked(fused.cpu().numpy(), si_np, oi_np, keep,
                                   self._type_vector(), topk=topk)
            result = {}
            for tag, edges in streams.items():
                ts = []
                for si, oi, pi, sc in edges:
                    s_disp = float(torch.sigmoid(torch.tensor(sc)))
                    if box_scores is not None:
                        s_disp *= float(box_scores[si]) * float(box_scores[oi])
                    ts.append(Triplet(
                        subject_idx=si, subject_box=boxes_xyxy[si],
                        predicate=self.predicates[pi], score=s_disp,
                        object_idx=oi, object_box=boxes_xyxy[oi],
                        subject_label=(box_labels[si] if box_labels is not None else None),
                        object_label=(box_labels[oi] if box_labels is not None else None)))
                result[tag] = ts
            return result

        # Sigmoid mode fuses ADDITIVELY — sigmoid(pred + rel) — which is the
        # v34 score contract and what relsgg/evaluator.py scores. This path
        # used to multiply the two sigmoids in BOTH modes, so every reported
        # number came from a formula the product did not use. Multiplicative
        # looks better on PSG/VG150 (VG150 AUC .832 vs .763) but that is the
        # incomplete-GT trap: it up-weights the relatedness term, which is an
        # annotation-propensity prior. On Haystack's explicit negatives the
        # ordering INVERTS — additive .9108, multiplicative .9038, relatedness
        # alone worst at .7475. See [[relsgg-confidence-knob]]. Softmax mode
        # keeps the multiplicative form: that IS its evaluator contract.
        if self.score_mode == "sigmoid":
            scores = self.contract.scores(
                logits.float(),
                None if out.get("pair_logits") is None
                else out["pair_logits"][0].float())
        else:
            scores = torch.softmax(logits, dim=-1)
            if out.get("pair_logits") is not None:
                scores = scores * torch.sigmoid(
                    out["pair_logits"][0].float()).unsqueeze(-1)
        # best predicate per pair
        best_s, best_p = scores.max(dim=-1)         # [K], [K]
        best_s = best_s.cpu().numpy(); best_p = best_p.cpu().numpy()

        cand = []
        for k in range(len(sub_idx)):
            if not valid[k]:
                continue
            si, oi = int(sub_idx[k]), int(obj_idx[k])
            if si >= N or oi >= N or si == oi:
                continue
            s = float(best_s[k])
            if box_scores is not None:              # SGDet triplet weighting
                s *= float(box_scores[si]) * float(box_scores[oi])
            cand.append((s, si, oi, int(best_p[k])))

        cand.sort(key=lambda x: -x[0])
        out_triplets = []
        for s, si, oi, p in cand[:topk]:
            out_triplets.append(Triplet(
                subject_idx=si, subject_box=boxes_xyxy[si],
                predicate=self.predicates[p], score=s,
                object_idx=oi, object_box=boxes_xyxy[oi],
                subject_label=(box_labels[si] if box_labels is not None else None),
                object_label=(box_labels[oi] if box_labels is not None else None)))
        return out_triplets
