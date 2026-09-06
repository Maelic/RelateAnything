"""Embedding-space quality analysis for relation pair features.

``EmbeddingAnalyzer`` accumulates GT-labelled pair embeddings from
``model.forward()``'s ``pair_features`` output and computes:

    - Intra / inter-class cosine similarity & gap
    - Silhouette score (sklearn)
    - Linear probe accuracy (sklearn LogisticRegression)
    - Centroid nearest-neighbour recall@1
    - t-SNE visualization saved as a PNG

    Latent Space health tiers (required for EXP-000 and all subsequent entries):

    Tier 1 — always required:
        T1-A  isotropy         — fraction of variance explained by top SVD component;
                                 target < 0.2, flag if > 0.5
        T1-B  entity_bias_acc  — linear-probe accuracy predicting subject entity class
                                 from pair features; target < 30 %, flag if > 60 %

    Tier 2 — required for vocab.py / backbone.py changes; also run for EXP-000:
        T2-A  vl_r@{1,5,10}   — VL text-to-pair retrieval R@K
        T2-B  cka              — linear CKA between pair features and predicate text embeddings
        T2-C  nmi / ari        — k-means clustering NMI and ARI vs ground-truth predicate labels

Usage::

    analyzer = EmbeddingAnalyzer(pred_names)
    analyzer.set_text_embeddings(model.vocab_head.W.cpu().numpy())  # for T2
    collect_embeddings(raw_model, val_loader, device, args, analyzer)
    metrics = analyzer.compute(args.output_dir, epoch=0)
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch


class EmbeddingAnalyzer:
    """Accumulate GT-labelled pair embeddings and measure latent-space quality.

    Args:
        pred_names:    List of predicate class names — used for t-SNE legend.
        max_per_class: Cap on embeddings collected per predicate class
                       (prevents OOM on large validation sets).
    """

    def __init__(self, pred_names: List[str], max_per_class: int = 500):
        self.pred_names    = pred_names
        self.max_per_class = max_per_class
        self._feats:  List[np.ndarray] = []
        self._labels: List[int]        = []
        self._class_count: Dict[int, int] = defaultdict(int)
        # Entity labels for T1-B entity bias metric
        self._sub_entity_labels: List[int] = []
        self._obj_entity_labels: List[int] = []
        # Text embeddings for T2-A/B (set externally via set_text_embeddings)
        self._text_embs: Optional[np.ndarray] = None  # [V, d_model]

    def set_text_embeddings(self, W: np.ndarray) -> None:
        """Supply projected text embedding matrix for T2-A/B metrics.

        Args:
            W: Projected text embeddings [V, d_model] from
               ``model.vocab_head.W.detach().cpu().float().numpy()``.
               Must be called AFTER ``reparameterize()`` so that W is in
               the same d_model space as pair features.
        """
        self._text_embs = W.astype(np.float32)

    def reset(self) -> None:
        self._feats.clear()
        self._labels.clear()
        self._class_count.clear()
        self._sub_entity_labels.clear()
        self._obj_entity_labels.clear()
        self._text_embs = None

    @torch.no_grad()
    def update(
        self,
        out: dict,
        entity_labels: Optional[List[torch.Tensor]] = None,
) -> None:
        """Collect GT-labelled embeddings from one batch.

        Args:
            out: Model output dict with keys ``pair_features`` [B,K,d],
                 ``pred_labels`` [B,K] (−1 = no GT), ``valid_mask`` [B,K],
                 ``sub_idx`` [B,K], ``obj_idx`` [B,K].
            entity_labels: Optional list of B tensors each [max_objects],
                           entity class index per box (-1 = padding).
                           Required for T1-B entity bias metric.
        """
        pair_features = out["pair_features"]  # [B, K, d]
        pred_labels   = out["pred_labels"]    # [B, K]
        valid_mask    = out["valid_mask"]     # [B, K]
        sub_idx       = out.get("sub_idx")    # [B, K] or None
        obj_idx       = out.get("obj_idx")    # [B, K] or None

        for b in range(pair_features.shape[0]):
            ent_b = entity_labels[b].cpu() if entity_labels is not None else None
            for k in range(pair_features.shape[1]):
                if not valid_mask[b, k]:
                    continue
                lbl = int(pred_labels[b, k])
                if lbl < 0:
                    continue
                if self._class_count[lbl] >= self.max_per_class:
                    continue
                self._feats.append(pair_features[b, k].float().cpu().numpy())
                self._labels.append(lbl)
                self._class_count[lbl] += 1

                # Entity labels for T1-B
                if ent_b is not None and sub_idx is not None and obj_idx is not None:
                    s_i = int(sub_idx[b, k].item())
                    o_i = int(obj_idx[b, k].item())
                    self._sub_entity_labels.append(int(ent_b[s_i].item()))
                    self._obj_entity_labels.append(int(ent_b[o_i].item()))
                else:
                    self._sub_entity_labels.append(-1)
                    self._obj_entity_labels.append(-1)

    def compute(self, output_dir: str, epoch: int) -> Dict[str, float]:
        """Compute all metrics and save the t-SNE plot.

        Returns an empty dict if fewer than 50 samples have been collected.
        """
        if len(self._feats) < 50:
            return {}

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import silhouette_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import normalize

        X = np.array(self._feats, dtype=np.float32)
        y = np.array(self._labels, dtype=np.int64)
        X_norm = normalize(X, norm="l2")

        results: Dict[str, float] = {}

        # ---- 1. Intra / inter-class cosine similarity ----
        classes        = np.unique(y)
        centroids      = np.zeros((len(classes), X_norm.shape[1]), dtype=np.float32)
        for i, c in enumerate(classes):
            centroids[i] = X_norm[y == c].mean(0)

        c_to_i = {c: i for i, c in enumerate(classes)}
        intra_sims, inter_sims = [], []
        for xi, yi in zip(X_norm, y):
            ci = c_to_i[yi]
            intra_sims.append(float(xi @ centroids[ci]))
            other = [centroids[j] for j in range(len(classes)) if j != ci]
            if other:
                inter_sims.append(float(np.mean([xi @ oc for oc in other])))

        results["cosine_intra"] = float(np.mean(intra_sims))
        results["cosine_inter"] = float(np.mean(inter_sims)) if inter_sims else 0.0
        results["cosine_gap"]   = results["cosine_intra"] - results["cosine_inter"]

        # ---- 2. Silhouette score (subsampled) ----
        rng   = np.random.default_rng(42)
        sub_n = min(len(X_norm), 3000)
        idx   = rng.choice(len(X_norm), sub_n, replace=False)
        X_sub, y_sub = X_norm[idx], y[idx]
        if len(np.unique(y_sub)) >= 2:
            results["silhouette"] = float(
                silhouette_score(X_sub, y_sub, metric="cosine")
)

        # ---- 3. Linear probe accuracy ----
        if len(np.unique(y)) >= 2:
            try:
                # stratify only when every class has ≥5 samples; otherwise
                # train_test_split raises ValueError on rare predicates
                _, counts = np.unique(y, return_counts=True)
                stratify = y if int(counts.min()) >= 5 else None
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_norm, y, test_size=0.2, stratify=stratify, random_state=42
)
                clf = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs")
                clf.fit(X_tr, y_tr)
                results["linear_probe_acc"] = float(clf.score(X_te, y_te))
            except Exception as exc:
                print(f"[EmbeddingAnalyzer] Linear probe failed: {exc}")

        # ---- 4. Centroid nearest-neighbour recall@1 ----
        nn_hits = sum(
            int(classes[np.argmax(centroids @ xi)] == yi)
            for xi, yi in zip(X_norm, y)
)
        results["nn_recall@1"] = nn_hits / len(y)

        # ---- T1-A: Isotropy — fraction of variance in top singular value ----
        # target < 0.2; flag if > 0.5 (space collapsing into one direction)
        try:
            X_c = X - X.mean(0)
            _, s_vals, _ = np.linalg.svd(X_c, full_matrices=False)
            results["isotropy"] = float((s_vals[0] ** 2) / max((s_vals ** 2).sum(), 1e-12))
        except Exception as exc:
            print(f"[EmbeddingAnalyzer] Isotropy computation failed: {exc}")

        # ---- T1-B: Entity bias accuracy ----
        # Linear probe: predict subject entity class from pair features.
        # target < 0.30; flag if > 0.60.
        y_ent = np.array(self._sub_entity_labels, dtype=np.int64)
        valid_ent = y_ent >= 0
        if valid_ent.sum() > 50:
            X_ent = X_norm[valid_ent]
            y_ent_f = y_ent[valid_ent]
            unique_ent = np.unique(y_ent_f)
            if len(unique_ent) >= 2:
                try:
                    strat = y_ent_f if (y_ent_f.shape[0] / len(unique_ent)) >= 2 else None
                    X_etr, X_ete, y_etr, y_ete = train_test_split(
                        X_ent, y_ent_f, test_size=0.2, stratify=strat, random_state=42
)
                    clf_ent = LogisticRegression(max_iter=100, C=1.0, solver="lbfgs")
                    clf_ent.fit(X_etr, y_etr)
                    results["entity_bias_acc"] = float(clf_ent.score(X_ete, y_ete))
                except Exception as exc:
                    print(f"[EmbeddingAnalyzer] Entity bias probe failed: {exc}")

        # ---- T2-A/B: VL-retrieval R@K and CKA (requires text embeddings in
        # the SAME space as pair_features — d_model, not raw text_dim; a
        # dimension mismatch here must not take down every other metric
        # already computed above) ----
        if (self._text_embs is not None and len(self._text_embs) > 0
                and self._text_embs.shape[-1] != X_norm.shape[-1]):
            print(f"[EmbeddingAnalyzer] text embeddings dim "
                 f"{self._text_embs.shape[-1]} != pair-feature dim "
                 f"{X_norm.shape[-1]} — skipping vl_r@K/cka (need d_model-"
                 f"space text embeddings, not raw text_dim W)")
        if (self._text_embs is not None and len(self._text_embs) > 0
                and self._text_embs.shape[-1] == X_norm.shape[-1]):
            from sklearn.preprocessing import normalize as sk_normalize

            W_norm = sk_normalize(self._text_embs, norm="l2")  # [V, d]
            sims = X_norm @ W_norm.T                           # [N, V]

            # T2-A: VL-retrieval R@K
            for k_ret in [1, 5, 10]:
                if k_ret <= W_norm.shape[0]:
                    topk_cols = np.argsort(-sims, axis=1)[:,:k_ret]
                    hits = np.array([y[i] in topk_cols[i] for i in range(len(y))])
                    results[f"vl_r@{k_ret}"] = float(hits.mean())

            # T2-B: Linear CKA between pair features and per-pair text embeddings
            try:
                valid_txt = y < W_norm.shape[0]
                if valid_txt.sum() >= 10:
                    X_v = X_norm[valid_txt]
                    y_v = y[valid_txt]
                    sub_n_cka = min(len(X_v), 1000)
                    rng_cka = np.random.default_rng(42)
                    cka_idx = rng_cka.choice(len(X_v), sub_n_cka, replace=False)
                    X_v = X_v[cka_idx]
                    y_v = y_v[cka_idx]
                    T_v = W_norm[y_v]          # [n, d]
                    X_vc = X_v - X_v.mean(0)
                    T_vc = T_v - T_v.mean(0)
                    K_xt = X_vc @ T_vc.T
                    K_xx = X_vc @ X_vc.T
                    K_tt = T_vc @ T_vc.T
                    num = float((K_xt * K_xt).sum())
                    den = float(np.sqrt((K_xx * K_xx).sum() * (K_tt * K_tt).sum()))
                    results["cka"] = num / max(den, 1e-12)
            except Exception as exc:
                print(f"[EmbeddingAnalyzer] CKA computation failed: {exc}")

        # ---- T2-C: Clustering NMI / ARI ----
        try:
            from sklearn.cluster import MiniBatchKMeans
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

            n_clusters = min(len(classes), max(2, len(classes)))
            sub_n_cl = min(len(X_norm), 5000)
            rng_cl = np.random.default_rng(42)
            cl_idx = rng_cl.choice(len(X_norm), sub_n_cl, replace=False)
            X_cl, y_cl = X_norm[cl_idx], y[cl_idx]
            km = MiniBatchKMeans(
                n_clusters=n_clusters, random_state=42, n_init=3, max_iter=100
)
            cluster_lbl = km.fit_predict(X_cl)
            results["nmi"] = float(normalized_mutual_info_score(y_cl, cluster_lbl))
            results["ari"] = float(adjusted_rand_score(y_cl, cluster_lbl))
        except Exception as exc:
            print(f"[EmbeddingAnalyzer] Clustering metrics failed: {exc}")

        # ---- 5. t-SNE visualization ----
        try:
            self._save_tsne(X_norm, y, output_dir, epoch)
        except Exception as exc:
            print(f"[EmbeddingAnalyzer] t-SNE failed: {exc}")

        return results

    def _save_tsne(
        self,
        X: np.ndarray,
        y: np.ndarray,
        output_dir: str,
        epoch: int,
) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.pyplot as _plt
        from sklearn.manifold import TSNE

        max_tsne = 5000
        if len(X) > max_tsne:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(X), max_tsne, replace=False)
            X, y = X[idx], y[idx]

        coords = TSNE(
            n_components=2, perplexity=30, max_iter=1000, random_state=42
).fit_transform(X)

        fig, ax = plt.subplots(figsize=(14, 10))
        cmap = _plt.get_cmap("tab20", len(self.pred_names))
        for lbl in np.unique(y):
            mask = y == lbl
            name = self.pred_names[lbl] if lbl < len(self.pred_names) else str(lbl)
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                s=6, alpha=0.5, color=cmap(lbl % 20), label=name,
)

        ax.legend(loc="upper right", fontsize=5, ncol=3, markerscale=1.5, framealpha=0.5)
        ax.set_title(f"t-SNE of pair embeddings — epoch {epoch}")
        ax.axis("off")
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"tsne_epoch{epoch:03d}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[EmbeddingAnalyzer] t-SNE saved → {out_path}")
