"""RelateAnything: image + boxes (or masks) -> ranked relations.

    from relsgg import RelateAnything

    ra = RelateAnything.from_pretrained("maelic/relsgg-vits16plus")
    triplets = ra.predict(image, boxes_xyxy, topk=20)
    ra.set_vocabulary(["holding", "riding", "tethered to"])   # any strings
    graphs = ra.predict(image, boxes_xyxy, decompose=True)     # spatial + semantic

The predicate vocabulary is encoded once by the text student that ships with
every model; inference afterwards is vision only. Boxes come from any
detector or from ground truth; object class labels are never needed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from.checkpoint import (build_model_from_ckpt, hub_id_for_backbone, load_checkpoint,
                         load_state, materialize_backbone_config)
from.config import config_from_args
from.model import RelSGG
from.scoring import ScoreContract
from.vocabulary import DEFAULT_PREDICATES, TRAIN_TEMPLATES

try:
    from PIL import Image
except Exception:                               # pragma: no cover
    Image = None

RASTER_RES = 32   # mask rasters are g x g coverage grids over the image


@dataclass
class Triplet:
    subject_idx: int
    subject_box: np.ndarray      # xyxy pixels in the original image frame
    predicate: str
    score: float
    object_idx: int
    object_box: np.ndarray
    subject_label: Optional[str] = None
    object_label: Optional[str] = None

    def __repr__(self) -> str:
        s = self.subject_label or f"obj{self.subject_idx}"
        o = self.object_label or f"obj{self.object_idx}"
        return f"({s}) --{self.predicate} [{self.score:.2f}]--> ({o})"


class RelateAnything:
    def __init__(self, model: RelSGG, predicates: Sequence[str], text_student: str,
                 img_size: int = 448, device: Union[str, torch.device] = "cpu"):
        self.model = model.to(device).eval()
        self.predicates = list(predicates)
        self.text_student = text_student
        self.img_size = img_size
        self.device = torch.device(device)
        self._type_vec = None
        self._set_contract(ScoreContract())

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_pretrained(cls, repo_id: str = "maelic/relsgg-vits16plus",
                        predicates: Optional[Sequence[str]] = None,
                        device: Union[str, torch.device] = "cpu",
                        revision: Optional[str] = None, **kw) -> "RelateAnything":
        """Download a released model from the Hugging Face Hub and load it."""
        from huggingface_hub import snapshot_download
        local = snapshot_download(
            repo_id, revision=revision,
            allow_patterns=["model.pth", "text_student.pt", "tokenizer*", "vocab.json",
                            "merges.txt", "special_tokens_map.json", "calibration.json"])
        return cls.from_checkpoint(os.path.join(local, "model.pth"), predicates,
                                   device=device, **kw)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, predicates: Optional[Sequence[str]] = None,
                        device: Union[str, torch.device] = "cpu", weights: str = "ema",
                        img_size: int = 448, text_student: Optional[str] = None,
                        strict: bool = True, embeddings=None,
                        calibration: bool = True) -> "RelateAnything":
        """Load ``model.pth`` and encode ``predicates`` (default: the
        vocabulary in ``relsgg.vocabulary``).

        ``text_student`` defaults to the ``text_student.pt`` next to the
        checkpoint (the layout of every released model). ``embeddings``
        supplies a ready ``[V, text_dim]`` matrix instead of encoding.
        A ``calibration.json`` next to the checkpoint is applied when present.
        """
        from.text.student import resolve_student_path
        ckpt = load_checkpoint(ckpt_path)
        model = build_model_from_ckpt(ckpt, weights=weights, strict=strict)
        if text_student is None:
            text_student = ckpt["args"].get("text_student") or "text_student.pt"
        text_student = resolve_student_path(text_student, near=ckpt_path)
        ra = cls(model, list(predicates or DEFAULT_PREDICATES), text_student,
                 img_size=img_size, device=device)
        ra.set_vocabulary(ra.predicates, embeddings=embeddings)
        if calibration:
            contract = ScoreContract.for_checkpoint(ckpt_path)
            if contract.is_calibrated:
                ra._set_contract(contract)
        return ra

    # -- calibration ----------------------------------------------------------

    def _set_contract(self, contract: ScoreContract) -> None:
        self.contract = contract
        self.model.set_score_contract(contract)

    def set_calibration(self, a: float, b: float) -> None:
        """Install a fitted (a, b); ranking is unchanged, thresholds gain meaning."""
        self._set_contract(ScoreContract(calib_a=float(a), calib_b=float(b)))

    @property
    def calib_a(self) -> float:
        return self.contract.calib_a

    @property
    def calib_b(self) -> float:
        return self.contract.calib_b

    # -- vocabulary -----------------------------------------------------------

    def set_vocabulary(self, predicates: Sequence[str],
                       templates: Optional[List[str]] = None,
                       embeddings=None) -> "RelateAnything":
        """Swap the predicate vocabulary. Strings are encoded by the text
        student (once); ``embeddings`` skips the encoder."""
        self.predicates = list(predicates)
        if embeddings is not None:
            W = torch.as_tensor(embeddings, dtype=torch.float32, device=self.device)
            if W.shape[0] != len(self.predicates):
                raise ValueError(f"embeddings has {W.shape[0]} rows for "
                                 f"{len(self.predicates)} predicates")
        else:
            if not self.text_student or not os.path.exists(self.text_student):
                raise FileNotFoundError(
                    f"text student not found ({self.text_student}); pass "
                    "text_student=... or embeddings=...")
            from.text.student import encode_texts_student
            W = encode_texts_student(self.predicates, self.text_student,
                                     templates=templates or TRAIN_TEMPLATES,
                                     device=self.device)
        self.model.vocab_head.set_vocabulary_matrix(self.predicates, W)
        self.model.reparameterize()
        self._type_vec = None
        return self

    def _type_vector(self) -> np.ndarray:
        """Spatial / semantic type of each predicate: the corpus flag when the
        string is known, the head's routing weight (alpha >= 0.5) otherwise."""
        if self._type_vec is not None:
            return self._type_vec
        import json
        from.decompose import type_vector
        cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus_type_map.json")
        cmap = json.load(open(cache)) if os.path.exists(cache) else None
        alpha = self.model.vocab_head.alpha
        alpha_np = alpha.detach().cpu().numpy() if alpha.numel() == len(self.predicates) else None
        self._type_vec, _ = type_vector(self.predicates, corpus_map=cmap, alpha=alpha_np)
        return self._type_vec

    # -- inference ------------------------------------------------------------

    @staticmethod
    def _to_chw(image, size):
        """(tensor [1, 3, size, size] in [0, 1], width, height)."""
        if isinstance(image, np.ndarray):            # HWC uint8, BGR (OpenCV)
            arr = image
            H, W = arr.shape[:2]
            if Image is not None:
                pil = Image.fromarray(arr[...,::-1] if arr.shape[2] == 3 else arr)
                pil = pil.resize((size, size), Image.BILINEAR)
                t = torch.from_numpy(np.asarray(pil, np.float32).transpose(2, 0, 1) / 255.0)
            else:
                t = torch.from_numpy(arr[...,::-1].copy().astype(np.float32).transpose(2, 0, 1) / 255.0)
                t = torch.nn.functional.interpolate(t[None], (size, size), mode="bilinear",
                                                    align_corners=False)[0]
            return t.unsqueeze(0), W, H
        W, H = image.size                             # PIL
        pil = image.convert("RGB").resize((size, size), Image.BILINEAR)
        t = torch.from_numpy(np.asarray(pil, np.float32).transpose(2, 0, 1) / 255.0)
        return t.unsqueeze(0), W, H

    @staticmethod
    def _rasterize(masks, boxes_xyxy: np.ndarray, W: int, H: int):
        """Masks ``[N, H, W]`` -> (cov [1, N, g, g] float, fill [1, N])."""
        from.data.rasters import mask_raster
        g = RASTER_RES
        cov = np.zeros((len(masks), g, g), dtype=np.float32)
        fill = np.ones(len(masks), dtype=np.float32)
        for i, m in enumerate(masks):
            m = np.asarray(m)
            if m.shape != (H, W):
                raise ValueError(f"mask {i} has shape {m.shape}, image is {(H, W)}")
            m = m.astype(np.float64)
            cov[i] = mask_raster(m, g)
            x1, y1, x2, y2 = boxes_xyxy[i]
            box_area = max((x2 - x1) * (y2 - y1), 1.0)
            fill[i] = float(np.clip(m.sum() / box_area, 0.0, 1.0))
        return (torch.from_numpy(cov).unsqueeze(0), torch.from_numpy(fill).unsqueeze(0))

    @torch.no_grad()
    def predict(self, image, boxes_xyxy: np.ndarray, masks=None,
                box_labels: Optional[Sequence[str]] = None,
                box_scores: Optional[np.ndarray] = None,
                topk: int = 20, max_boxes: int = 60, decompose: bool = False):
        """Rank relations between the given regions.

        Args:
            image:       PIL image or HWC uint8 array (BGR, as OpenCV reads).
            boxes_xyxy:  ``[N, 4]`` pixel boxes.
            masks:       optional ``[N, H, W]`` binary masks of the same regions.
            box_labels:  optional names, for display only.
            box_scores:  optional detector confidences; triplets are then
                         ranked by ``conf(sub) * conf(obj) * score``.
            topk:        triplets to return.
            max_boxes:   keep at most this many boxes (highest score first).
            decompose:   return two graphs from one pass, ``{"spatial": [...],
                         "semantic": [...]}``, each ranked on its own.
        Returns:
            ``List[Triplet]``, or a dict of two lists with ``decompose``.
        """
        boxes_xyxy = np.asarray(boxes_xyxy, np.float32).reshape(-1, 4)
        N = len(boxes_xyxy)
        if N < 2:
            return {"spatial": [], "semantic": []} if decompose else []
        if box_scores is not None:
            box_scores = np.asarray(box_scores, np.float32).reshape(-1)
        if N > max_boxes:
            order = (np.argsort(-box_scores) if box_scores is not None else np.arange(N))[:max_boxes]
            boxes_xyxy = boxes_xyxy[order]
            box_scores = box_scores[order] if box_scores is not None else None
            box_labels = [box_labels[i] for i in order] if box_labels is not None else None
            masks = [masks[i] for i in order] if masks is not None else None
            N = max_boxes

        img_t, W, H = self._to_chw(image, self.img_size)
        img_t = img_t.to(self.device)
        b = boxes_xyxy.copy()
        b[:, [0, 2]] /= max(W, 1)
        b[:, [1, 3]] /= max(H, 1)
        cx = (b[:, 0] + b[:, 2]) / 2
        cy = (b[:, 1] + b[:, 3]) / 2
        boxes_t = torch.from_numpy(np.stack([cx, cy, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], -1)
.astype(np.float32)).unsqueeze(0).to(self.device)
        region = {}
        if masks is not None:
            cov, fill = self._rasterize(masks, boxes_xyxy, W, H)
            region = {"cov": cov.to(self.device), "fill": fill.to(self.device)}

        out = self.model(img_t, boxes_t, box_counts=torch.tensor([N], device=self.device),
                         targets=None, **region)
        logits = out["logits"][0].float()
        pair = out["pair_logits"][0].float()
        sub_idx = out["sub_idx"][0].cpu().numpy()
        obj_idx = out["obj_idx"][0].cpu().numpy()
        valid = out["valid_mask"][0].cpu().numpy().astype(bool)
        keep = valid & (sub_idx < N) & (obj_idx < N) & (sub_idx != obj_idx)

        def _triplet(si, oi, pi, score):
            if box_scores is not None:
                score *= float(box_scores[si]) * float(box_scores[oi])
            return Triplet(subject_idx=si, subject_box=boxes_xyxy[si],
                           predicate=self.predicates[pi], score=score,
                           object_idx=oi, object_box=boxes_xyxy[oi],
                           subject_label=(box_labels[si] if box_labels is not None else None),
                           object_label=(box_labels[oi] if box_labels is not None else None))

        if decompose:
            from.decompose import split_ranked
            fused = (logits + pair.unsqueeze(-1)).cpu().numpy()
            streams = split_ranked(fused, sub_idx, obj_idx, keep, self._type_vector(), topk=topk)
            return {tag: [_triplet(si, oi, pi, float(torch.sigmoid(torch.tensor(sc))))
                          for si, oi, pi, sc in edges]
                    for tag, edges in streams.items()}

        scores = self.contract.scores(logits, pair)
        best_s, best_p = scores.max(dim=-1)
        best_s, best_p = best_s.cpu().numpy(), best_p.cpu().numpy()
        cand = [(float(best_s[k]), int(sub_idx[k]), int(obj_idx[k]), int(best_p[k]))
                for k in range(len(sub_idx)) if keep[k]]
        cand = [(s * (float(box_scores[si]) * float(box_scores[oi]) if box_scores is not None else 1.0),
                 si, oi, p) for s, si, oi, p in cand]
        cand.sort(key=lambda x: -x[0])
        return [Triplet(subject_idx=si, subject_box=boxes_xyxy[si], predicate=self.predicates[p],
                        score=s, object_idx=oi, object_box=boxes_xyxy[oi],
                        subject_label=(box_labels[si] if box_labels is not None else None),
                        object_label=(box_labels[oi] if box_labels is not None else None))
                for s, si, oi, p in cand[:topk]]
