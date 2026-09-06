"""RelSGG: the relation head.

    Backbone                 dense patch features F of the image
    SoftSpatialPool          one feature per box (and per union / contact zone)
    RelatednessPairSampler   K ordered pairs worth scoring
    RelGeomEncoder           geometry of each pair
    pair_proj                fused pair representation
    RelationTransformer      pairs attend to each other and to the scene
    DeformableRelRead        pairs read the scene at learned points
    RelationInteractionBlock refinement with the scene
    VocabHead                cosine against the predicate vocabulary

Inputs are an image and boxes (masks optional). Object class labels are
never an input.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from..config import RelSGGConfig
from..scoring import ScoreContract
from.backbone import Backbone, RelationInteractionBlock
from.deformable import DeformableRelRead
from.geometry import BoxPromptEncoder, RelGeomEncoder
from.pooling import SoftSpatialPool, union_box
from.sampler import RelatednessPairSampler
from.transformer import RelationTransformer
from.vocab_head import VocabHead


def _cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], -1).clamp(0, 1)


class RelSGG(nn.Module):
    def __init__(self, config: Optional[RelSGGConfig] = None):
        super().__init__()
        config = config if config is not None else RelSGGConfig()
        self.config = config
        c = config

        self.backbone = Backbone(model_name=c.backbone_model, patch_size=c.patch_size,
                                 pretrained=c.backbone_pretrained)
        backbone_dim = self.backbone.d_model

        self.box_prompt_encoder = BoxPromptEncoder(
            d_model=c.d_model, num_freqs=c.pe_num_freqs, max_octave=c.pe_max_octave)
        self.geo_encoder = RelGeomEncoder(d_model=c.d_model)
        self.spatial_pool = SoftSpatialPool(
            d_model=backbone_dim, num_freqs=c.pe_num_freqs, max_octave=c.pe_max_octave)
        self.sampler = RelatednessPairSampler(
            geo_budget=c.geo_budget, final_budget=c.final_budget, feat_dim=backbone_dim,
            neg_weight=c.rel_neg_weight)

        # [v_sub; v_obj; v_union; v_contact; geo] -> d_model. The contact
        # zone is the intersection of the two boxes, or the gap between them
        # when they do not overlap.
        pair_in = backbone_dim * 4 + c.d_model
        self.pair_proj = nn.Linear(pair_in, c.d_model)
        nn.init.xavier_uniform_(self.pair_proj.weight)
        nn.init.zeros_(self.pair_proj.bias)

        self.rel_transformer = RelationTransformer(
            d_model=c.d_model, backbone_dim=backbone_dim, n_self=c.n_self_layers,
            n_cross=c.n_cross_layers, n_heads=c.n_heads, ffn_ratio=c.ffn_ratio,
            dropout=c.dropout)
        self.deformable_read = DeformableRelRead(
            d_model=c.d_model, n_points=c.deformable_points, heads=c.deformable_heads,
            null_slots=c.deformable_nulls)
        self.rel_interaction = RelationInteractionBlock(
            d_model=c.d_model, scene_dim=backbone_dim, n_dep=c.n_dep_layers,
            n_gnd=c.n_gnd_layers, n_heads=c.n_heads, ffn_ratio=c.ffn_ratio, dropout=c.dropout)

        self.vocab_head = VocabHead(d_model=c.d_model, text_dim=c.text_dim,
                                    logit_scale_init=c.logit_scale_init,
                                    proj_layers=c.proj_layers)

        # Semantic query: pair context plus the subject and object features
        # projected into text space, gated (gates start small so training
        # begins from the pure pair-context query).
        self.sub_text_proj = nn.Linear(backbone_dim, c.text_dim, bias=False)
        self.obj_text_proj = nn.Linear(backbone_dim, c.text_dim, bias=False)
        nn.init.xavier_uniform_(self.sub_text_proj.weight)
        nn.init.xavier_uniform_(self.obj_text_proj.weight)
        self.compose_norm = nn.LayerNorm(c.text_dim)
        self.compose_gate = nn.Parameter(torch.tensor([0.1, 0.1]))
        # Category-name embeddings for the training-time object alignment.
        self.register_buffer("W_obj", torch.empty(0))

        # Spatial query: pair context and geometry through their own MLP, so
        # direction (above / below) does not compete with semantics for one
        # projection.
        hidden = max(c.d_model * 2, c.text_dim // 2)
        self.spa_proj = nn.Sequential(
            nn.Linear(c.d_model * 2, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, c.text_dim, bias=False))
        for m in self.spa_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # How predicate and pair-existence logits become one score. Fitted
        # after training (calibration.json), not part of the state dict.
        self.score_contract = ScoreContract()
        # Set by the trainer: [n_sources, V] bool, which vocabulary columns an
        # anchor from each source may be contrasted against.
        self.source_col_allow: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # vocabulary and training installs
    # ------------------------------------------------------------------

    def reparameterize(self) -> None:
        self.vocab_head.reparameterize()

    def set_score_contract(self, contract: ScoreContract) -> None:
        self.score_contract = contract

    def install_losses(self, ontology, n_neg: int = 512, hard_frac: float = 0.5) -> None:
        """Attach the training objective (call before DDP / optimiser)."""
        from..training.losses import BatchLocalInfoNCE
        self.ontology = ontology
        self.batch_infonce = BatchLocalInfoNCE(
            ontology, temp=self.config.infonce_temp, n_neg=n_neg, hard_frac=hard_frac)
        # The contrastive term is scale-free, so the head's affine
        # (logit_scale, logit_bias) is trained only by the sigmoid terms.
        if self.config.lambda_sigmoid <= 0.0 and self.config.lambda_bg <= 0.0:
            self.vocab_head.logit_scale.requires_grad_(False)
            self.vocab_head.logit_bias.requires_grad_(False)
            print("[vocab_head] logit_scale/logit_bias frozen: no sigmoid term is active")

    @torch.no_grad()
    def set_object_vocabulary(self, names: List[str], W_obj: torch.Tensor) -> None:
        """Category-name embeddings for the object alignment term (training only)."""
        self.obj_names = list(names)
        self.W_obj = F.normalize(W_obj.float(), dim=-1).to(self.vocab_head.logit_scale.device).clone()

    # ------------------------------------------------------------------
    # training-time helpers
    # ------------------------------------------------------------------

    def _object_text_loss(self, v_obj: torch.Tensor, box_counts: torch.Tensor,
                          targets: List[dict]) -> torch.Tensor:
        """InfoNCE aligning the projected box features to their category
        names, for both the subject and the object projection."""
        if self.W_obj.numel() == 0:
            return v_obj.new_zeros(())
        feats, labels = [], []
        for b, t in enumerate(targets):
            el = t.get("entity_labels")
            if el is None:
                continue
            n = min(int(box_counts[b]), el.shape[0])
            keep = el[:n] >= 0
            if keep.any():
                feats.append(v_obj[b,:n][keep])
                labels.append(el[:n][keep])
        if not feats:
            return v_obj.new_zeros(())
        feats = torch.cat(feats)
        labels = torch.cat(labels).to(feats.device)
        loss = feats.new_zeros(())
        for proj in (self.sub_text_proj, self.obj_text_proj):
            q = F.normalize(proj(feats), dim=-1)
            loss = loss + F.cross_entropy(q @ self.W_obj.T / self.config.infonce_temp, labels)
        return loss * 0.5

    def _mix_partners(self, labels: torch.Tensor, ok: torch.Tensor):
        """For every labelled slot, another slot of the batch with the same
        predicate. Returns ``(src, dst)`` flat indices or ``None``."""
        idx = ok.reshape(-1).nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            return None
        g = labels.reshape(-1)[idx]
        order = torch.argsort(g)
        _, counts = torch.unique_consecutive(g[order], return_counts=True)
        starts = torch.cumsum(counts, 0) - counts
        start_per = torch.repeat_interleave(starts, counts)
        count_per = torch.repeat_interleave(counts, counts)
        pos = torch.arange(g.shape[0], device=g.device) - start_per
        r = (torch.rand(g.shape[0], device=g.device) * (count_per - 1).clamp(min=1)).long()
        r = torch.minimum(r, (count_per - 2).clamp(min=0))
        r = r + (r >= pos).long()
        keep = count_per > 1
        if not bool(keep.any()):
            return None
        src = idx[order[keep]]
        dst = idx[order[(start_per + r)[keep]]]
        return (src, dst) if src.numel() else None

    def _mix_entities(self, v_sub, v_obj_k, labels, ok):
        """Same-predicate feature mixing: blend the subject and object features
        of a slot with those of another slot carrying the same predicate.
        Label-preserving, so no loss change is needed."""
        pairs = self._mix_partners(labels, ok)
        if pairs is None:
            return v_sub, v_obj_k
        src, dst = pairs
        n, dev = src.shape[0], src.device
        conc = torch.full((1,), float(self.config.cfa_alpha), device=dev)
        lam = torch.distributions.Beta(conc, conc).sample((n,)).reshape(n)
        lam = torch.where(torch.rand(n, device=dev) < self.config.cfa_prob,
                          lam, torch.ones_like(lam)).unsqueeze(-1)
        out = []
        for x in (v_sub, v_obj_k):
            flat = x.reshape(-1, x.shape[-1])
            mixed = flat.clone()
            mixed[src] = lam * flat[src] + (1.0 - lam) * flat[dst]
            out.append(mixed.reshape(x.shape))
        return out[0], out[1]

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor, boxes: torch.Tensor,
                box_counts: Optional[torch.Tensor] = None,
                targets: Optional[List[dict]] = None,
                cov: Optional[torch.Tensor] = None,
                fill: Optional[torch.Tensor] = None,
                precomputed_features: Optional[torch.Tensor] = None) -> Dict:
        """
        Args:
            images:     ``[B, 3, H, W]`` float in [0, 1] (normalised here).
            boxes:      ``[B, N, 4]`` normalised cxcywh, zero-padded.
            box_counts: ``[B]`` valid boxes per image (default: all N).
            targets:    per-image dicts with ``relations`` ``[R, 3]`` of
                        (sub_idx, obj_idx, predicate) at training time.
            cov:        ``[B, N, g, g]`` region coverage rasters from masks
                        (uint8 or float in [0, 1]); a box is its rectangle.
            fill:       ``[B, N]`` region area over box area.
            precomputed_features: fused backbone map to skip the backbone.
        Returns:
            ``logits [B, K, V]``, ``pair_logits [B, K]``, ``sub_idx``,
            ``obj_idx``, ``valid_mask``, ``pred_labels`` (all ``[B, K]``),
            ``pair_features [B, K, d_model]``; with targets also ``loss`` and
            ``loss_dict``.
        """
        B, max_N, _ = boxes.shape
        if cov is None:
            cov = getattr(targets, "cov", None)
        if fill is None:
            fill = getattr(targets, "fill", None)

        r_iou = r_contact = None
        if cov is not None:
            if cov.dtype == torch.uint8:
                cov = cov.float() / 255.0
            cov = cov.to(boxes.device, dtype=boxes.dtype)
            cov_flat = cov.flatten(2)
            area = cov_flat.sum(-1)
            inter = torch.einsum("bnc,bmc->bnm", cov_flat, cov_flat)
            a_s, a_o = area.unsqueeze(2), area.unsqueeze(1)
            r_iou = inter / (a_s + a_o - inter).clamp(min=1e-6)
            r_contact = inter / torch.minimum(a_s, a_o).clamp(min=1e-6)
            if fill is not None:
                fill = fill.to(boxes.device, dtype=boxes.dtype)

        # 1. scene features
        if precomputed_features is not None:
            F_map = precomputed_features
        else:
            F_map = self.backbone.extract(self.backbone.preprocess(images))   # [B,h,w,d]
        h_f, w_f = F_map.shape[1], F_map.shape[2]
        cov_grid = None
        if cov is not None:
            cov_grid = F.adaptive_avg_pool2d(
                cov.reshape(B * cov.shape[1], 1, cov.shape[2], cov.shape[3]),
                (h_f, w_f)).reshape(B, cov.shape[1], h_f * w_f)

        # 2. per-box features
        v_obj = self.spatial_pool(F_map, boxes, cov=cov_grid)

        # 3. pairs. Entity labels only weight the training loss.
        el_sampler = None
        if targets is not None and self.sampler.neg_rate is not None:
            el_sampler = boxes.new_full((B, max_N), -1, dtype=torch.long)
            for b, t in enumerate(targets):
                el = t.get("entity_labels")
                if el is not None and el.numel():
                    n = min(el.shape[0], max_N)
                    el_sampler[b,:n] = el[:n].to(el_sampler.device)
        (sub_idx, obj_idx, valid_mask, pred_labels,
         geo_loss, rel_loss, rel_logits) = self.sampler(
            boxes=boxes, obj_feats=v_obj, box_counts=box_counts, targets=targets,
            entity_labels=el_sampler)
        K = sub_idx.shape[1]
        B_idx = torch.arange(B, device=boxes.device).unsqueeze(1).expand(B, K)

        # 4. pair features
        sub_boxes_k = boxes[B_idx, sub_idx]
        obj_boxes_k = boxes[B_idx, obj_idx]
        v_sub = v_obj[B_idx, sub_idx]
        v_obj_k = v_obj[B_idx, obj_idx]
        union_boxes_k = union_box(sub_boxes_k, obj_boxes_k)

        union_cov = region_k = None
        if cov_grid is not None:
            union_cov = torch.maximum(cov_grid[B_idx, sub_idx], cov_grid[B_idx, obj_idx])
            f = fill if fill is not None else torch.ones_like(boxes[..., 0])
            region_k = (f[B_idx, sub_idx], f[B_idx, obj_idx],
                        r_iou[B_idx, sub_idx, obj_idx], r_contact[B_idx, sub_idx, obj_idx])
        geo_feat = self.geo_encoder(sub_boxes_k, obj_boxes_k, region_k)          # [B,K,d]

        pair_xyxy = _cxcywh_to_xyxy(torch.stack([sub_boxes_k, obj_boxes_k], 0))
        sub_xyxy, obj_xyxy = pair_xyxy[0], pair_xyxy[1]
        box_tokens = self.box_prompt_encoder.encode_pairs(sub_xyxy, obj_xyxy)   # [B,K,4,d]

        # Contact zone: the intersection when the boxes overlap, the gap
        # between their facing edges when they do not.
        inner_x1 = torch.maximum(sub_xyxy[..., 0], obj_xyxy[..., 0])
        inner_y1 = torch.maximum(sub_xyxy[..., 1], obj_xyxy[..., 1])
        inner_x2 = torch.minimum(sub_xyxy[..., 2], obj_xyxy[..., 2])
        inner_y2 = torch.minimum(sub_xyxy[..., 3], obj_xyxy[..., 3])
        contact_boxes_k = torch.stack(
            [(inner_x1 + inner_x2) * 0.5, (inner_y1 + inner_y2) * 0.5,
             (inner_x2 - inner_x1).abs(), (inner_y2 - inner_y1).abs()], -1)
        # Union and contact zones are pooled in one call (same keys).
        pool_boxes = torch.cat([union_boxes_k, contact_boxes_k], dim=1)
        pool_cov = None
        if union_cov is not None:
            # The contact half keeps a flat raster: log(1) = 0, no bias.
            pool_cov = torch.cat([union_cov, union_cov.new_full(union_cov.shape, 1.0 - 1e-4)], dim=1)
        v_pool = self.spatial_pool(F_map, pool_boxes, cov=pool_cov)
        v_union, v_contact = v_pool[:,:K], v_pool[:, K:]

        if self.training and targets is not None and self.config.cfa_prob > 0.0:
            v_sub, v_obj_k = self._mix_entities(
                v_sub, v_obj_k, pred_labels.clamp(min=0), (pred_labels >= 0) & valid_mask)

        pair_input = torch.cat([v_sub, v_obj_k, v_union, v_contact, geo_feat], dim=-1)
        pair_feat = self.pair_proj(pair_input)

        # 5. context: relation transformer, deformable read, interaction block
        padding_mask = ~valid_mask
        bt_drop = None
        if self.training and self.config.box_token_dropout > 0.0:
            bt_drop = torch.rand(B, device=boxes.device) < self.config.box_token_dropout
        r = self.rel_transformer(pair_feat, F_map, box_tokens=box_tokens,
                                 pair_padding_mask=padding_mask, box_token_drop=bt_drop)
        scene_d = self.rel_transformer.scene_proj(F_map).permute(0, 3, 1, 2).contiguous()
        anchors = torch.stack([sub_boxes_k, obj_boxes_k, union_boxes_k, contact_boxes_k], dim=2)
        r = self.deformable_read(r, scene_d, anchors)
        scene_flat = F_map.reshape(B, h_f * w_f, F_map.shape[3])
        r = self.rel_interaction(r, scene_flat, query_padding_mask=padding_mask,
                                 grid_hw=(h_f, w_f))

        # 6. queries and scores
        g = self.compose_gate
        q = self.compose_norm(self.vocab_head.proj(r)
                              + g[0] * self.sub_text_proj(v_sub)
                              + g[1] * self.obj_text_proj(v_obj_k))          # [B,K,text_dim]
        q_spa = self.spa_proj(torch.cat([r, geo_feat], dim=-1))
        logits = self.vocab_head.score_query_dual(q, q_spa)                    # [B,K,V]

        out = {"logits": logits, "pair_logits": rel_logits, "sub_idx": sub_idx,
               "obj_idx": obj_idx, "valid_mask": valid_mask, "pred_labels": pred_labels,
               "pair_features": r}
        if targets is None:
            return out

        # 7. training objective
        from..training.losses import build_slot_targets, swap_direction_hinge
        W = self.vocab_head.W
        V = W.shape[0]
        gt_hot, slot_w = build_slot_targets(sub_idx, obj_idx, valid_mask, targets, V)
        has_gt = gt_hot.any(-1) & valid_mask
        hot = gt_hot[has_gt]
        w_gt = slot_w[has_gt]
        alpha = self.vocab_head.current_alpha()

        col_allow = src_img = None
        allow = self.source_col_allow
        if allow is not None and all("src" in t for t in targets):
            allow = allow.to(boxes.device)
            src_img = torch.as_tensor([int(t["src"]) for t in targets], device=boxes.device)
            col_allow = allow[src_img[B_idx][has_gt]]

        nce_loss = self.batch_infonce(q[has_gt], hot, W, feats_spa=q_spa[has_gt],
                                      alpha=alpha, weights=w_gt, col_allow=col_allow)
        obj_loss = self._object_text_loss(v_obj, box_counts, targets)
        swap_loss = nce_loss.new_zeros(())
        if self.config.lambda_swap > 0:
            swap_loss = swap_direction_hinge(
                q, q_spa, alpha, W, sub_idx, obj_idx, valid_mask, targets,
                sym=self.ontology.sym.to(W.device), margin=self.config.swap_margin)

        # Per-cell sigmoid term: within a slot, the negative columns share the
        # mass of the positives, so the term needs no negative weight.
        sig_loss = nce_loss.new_zeros(())
        if self.config.lambda_sigmoid > 0.0:
            lg_gt = logits[has_gt]
            tgt = hot.to(lg_gt.dtype)
            pos_mass = tgt.sum(-1, keepdim=True)
            allow_f = col_allow.to(tgt.dtype) if col_allow is not None else torch.ones_like(tgt)
            n_neg = ((tgt <= 0) & (allow_f > 0)).sum(-1, keepdim=True).clamp(min=1)
            w = torch.where(tgt > 0, torch.ones_like(tgt), pos_mass / n_neg * allow_f)
            bce = F.binary_cross_entropy_with_logits(lg_gt, tgt.clamp(0, 1), weight=w,
                                                     reduction="none")
            sig_loss = bce.sum(-1).mean()

        # Background suppression on valid slots without a label: push down the
        # top-k columns of the fused score, weighted by how likely the pair is
        # a genuine negative.
        bg_loss = nce_loss.new_zeros(())
        if self.config.lambda_bg > 0.0:
            neg_slots = valid_mask & ~has_gt
            lg_neg = logits[neg_slots]
            if src_img is not None:
                lg_neg = lg_neg.masked_fill(~allow[src_img[B_idx][neg_slots]],
                                            torch.finfo(lg_neg.dtype).min)
            kk = min(self.config.bg_topk, lg_neg.shape[-1])
            per_slot = F.softplus(lg_neg.topk(kk, dim=-1).values).mean(-1)
            if el_sampler is not None and self.sampler.neg_rate is not None:
                w_slot = self.sampler._pu_neg_weight(
                    el_sampler[B_idx, sub_idx][neg_slots],
                    el_sampler[B_idx, obj_idx][neg_slots], per_slot)
            else:
                w_slot = torch.full_like(per_slot, self.sampler.neg_weight)
            bg_loss = (per_slot * w_slot).sum() / w_slot.sum().clamp(min=1e-6)

        c = self.config
        total = (nce_loss + c.lambda_obj * obj_loss + c.lambda_swap * swap_loss
                 + c.lambda_sigmoid * sig_loss + c.lambda_bg * bg_loss
                 + c.lambda_geo * geo_loss + c.lambda_rel * rel_loss)
        out["loss"] = total
        out["loss_dict"] = {
            "loss_nce": nce_loss.detach(), "loss_obj": obj_loss.detach(),
            "loss_swap": swap_loss.detach(), "loss_sig": sig_loss.detach(),
            "loss_bg": bg_loss.detach(), "loss_geo": geo_loss.detach(),
            "loss_rel": rel_loss.detach(), "loss_total": total.detach(),
        }
        return out

    # ------------------------------------------------------------------
    # inference helper
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict(self, images: torch.Tensor, boxes: torch.Tensor,
                box_counts: Optional[torch.Tensor] = None, threshold: float = 0.3,
                topk_per_pair: int = 1, **region_kwargs) -> List[List[dict]]:
        """Triplets per image as dicts with ``subject``, ``object``,
        ``predicate``, ``predicate_id`` and ``score`` (the score contract
        applied to the fused logits)."""
        out = self.forward(images, boxes, box_counts=box_counts, targets=None, **region_kwargs)
        logits, pair_logits = out["logits"], out["pair_logits"]
        sub_idx, obj_idx, valid_mask = out["sub_idx"], out["obj_idx"], out["valid_mask"]
        pred_names = self.vocab_head.pred_names
        results: List[List[dict]] = []
        for b in range(images.shape[0]):
            valid_k = valid_mask[b]
            scores = self.score_contract.scores(logits[b, valid_k].float(),
                                                pair_logits[b, valid_k].float())
            sub_b, obj_b = sub_idx[b, valid_k], obj_idx[b, valid_k]
            kk = min(topk_per_pair, scores.shape[-1])
            top_scores, top_preds = scores.topk(kk, dim=-1)
            keep = top_scores >= threshold
            pair_of_hit = keep.nonzero(as_tuple=True)[0]
            sc, order = top_scores[keep].sort(descending=True, stable=True)
            sc_l = sc.tolist()
            pred_l = top_preds[keep][order].tolist()
            sub_l = sub_b[pair_of_hit[order]].tolist()
            obj_l = obj_b[pair_of_hit[order]].tolist()
            results.append([
                {"subject": s, "object": o,
                 "predicate": pred_names[p] if pred_names else str(p),
                 "predicate_id": p, "score": sc_l[i]}
                for i, (s, o, p) in enumerate(zip(sub_l, obj_l, pred_l))])
        return results
