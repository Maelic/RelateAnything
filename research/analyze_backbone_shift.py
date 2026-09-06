"""What did relation fine-tuning do to the DINOv3 backbone?

Compares the PRETRAINED DINOv3 backbone against the backbone inside one or
more fine-tuned RelSGG checkpoints, on the same images:

  numerical (over --n_stats images)
    - per-layer linear CKA between pretrained and fine-tuned patch tokens
      (population-level: is the representation GEOMETRY still the same?)
    - per-layer mean per-token cosine (token-level: did individual patch
      features move?), plus CLS-token cosine and patch-norm ratio
    - pairwise between the fine-tuned arms too, so "did both arms drift to
      the same place" is answered, not assumed
    - weight-space drift straight off the state dicts: relative Frobenius
      change per transformer block, split by parameter type (Q/K/V,
      attn-out, MLP, LayerNorm, LayerScale, embeddings)
    - the learned multi-layer combiner (backbone.layer_weights softmax) per
      arm — which taps the relation head actually reads after training

  visual (--n_viz images, at each --viz_layers hidden state)
    - joint-PCA RGB maps: PCA is fit on the CONCATENATED tokens of all
      models for the same image, so the SAME projection colors every panel
      and color differences are feature differences, not per-panel PCA
      arbitrariness (per-panel PCA — like visualize_features.pca_rgb — can
      make identical features look different)
    - a 1-cos(pretrained, fine-tuned) heatmap: WHERE in the image the
      features moved

Usage:
    python training/analyze_backbone_shift.py \
        --checkpoints runs/train/armA/checkpoint_best.pth \
                      runs/train/armB/checkpoint_best.pth \
        --labels armA armB \
        --data_root runs/packed/psg --split val \
        --n_stats 200 --n_viz 6 --out_dir runs/analysis/backbone_shift
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from relsgg.data.dataset import RelationDataset      # noqa: E402
from relsgg.model.backbone import Backbone        # noqa: E402


# ---------------------------------------------------------------- weights ---

_TYPE_RULES = [
    ("q_proj", "attn.q"), ("k_proj", "attn.k"), ("v_proj", "attn.v"),
    ("o_proj", "attn.out"), ("mlp.", "mlp"),
    ("layer_scale", "layerscale"), ("norm", "layernorm"),
    ("embeddings", "embed"), ("layer_weights", "combiner"),
]


def _param_type(key: str) -> str:
    for pat, name in _TYPE_RULES:
        if pat in key:
            return name
    return "other"


def _block_index(key: str):
    m = re.search(r"layer\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def weight_drift(init_sd: dict, ft_sd: dict) -> dict:
    """Relative Frobenius drift sqrt(sum||dW||^2 / sum||W_init||^2), grouped
    per (block, type) and per type overall."""
    num = {}   # (block, type) -> sum ||dW||^2
    den = {}
    for k, w0 in init_sd.items():
        if k == "layer_weights":
            # zero-init combiner => relative drift is meaningless; the learned
            # softmax is reported separately as combiner_softmax.
            continue
        if k not in ft_sd:
            print(f"[shift] WARNING key missing in ft state dict: {k}")
            continue
        d = (ft_sd[k].float() - w0.float()).pow(2).sum().item()
        n = w0.float().pow(2).sum().item()
        g = (_block_index(k), _param_type(k))
        num[g] = num.get(g, 0.0) + d
        den[g] = den.get(g, 0.0) + n
    out = {}
    for g in num:
        blk = "global" if g[0] is None else g[0]
        out.setdefault(str(blk), {})[g[1]] = (
            float(np.sqrt(num[g] / max(den[g], 1e-30))))
    # per-type across all blocks
    tnum, tden = {}, {}
    for (blk, t), v in num.items():
        tnum[t] = tnum.get(t, 0.0) + v
        tden[t] = tden.get(t, 0.0) + den[(blk, t)]
    out["by_type"] = {t: float(np.sqrt(tnum[t] / max(tden[t], 1e-30)))
                      for t in tnum}
    return out


def drift_heatmap(drifts: dict, labels: list[str], out_path: str) -> None:
    types = ["attn.q", "attn.k", "attn.v", "attn.out", "mlp",
             "layernorm", "layerscale"]
    n_blocks = max(int(b) for b in drifts[labels[0]] if b.isdigit()) + 1
    fig, axes = plt.subplots(1, len(labels), figsize=(7.2 * len(labels), 4.6),
                             squeeze=False)
    for ai, lab in enumerate(labels):
        M = np.zeros((len(types), n_blocks))
        for b in range(n_blocks):
            row = drifts[lab].get(str(b), {})
            for ti, t in enumerate(types):
                M[ti, b] = row.get(t, np.nan)
        ax = axes[0][ai]
        im = ax.imshow(M, aspect="auto", cmap="magma")
        ax.set_xticks(range(n_blocks))
        ax.set_yticks(range(len(types)))
        ax.set_yticklabels(types, fontsize=8)
        ax.set_xlabel("transformer block")
        ax.set_title(f"{lab}: relative weight drift ||dW||/||W||", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------- features ---

class CKAAccumulator:
    """Streaming linear CKA + mean cosine between two token streams.

    Accumulates X^T Y / X^T X / Y^T Y and the column means, then computes
    centered linear CKA exactly:
        CKA = ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F)
    """

    def __init__(self, d: int, device):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
        self.sxy, self.sxx, self.syy = z(d, d), z(d, d), z(d, d)
        self.mx, self.my = z(d), z(d)
        self.n = 0
        self.cos_sum = 0.0

    @torch.no_grad()
    def update(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        Xd, Yd = X.double(), Y.double()
        self.sxy += Xd.T @ Yd
        self.sxx += Xd.T @ Xd
        self.syy += Yd.T @ Yd
        self.mx += Xd.sum(0)
        self.my += Yd.sum(0)
        self.n += X.shape[0]
        self.cos_sum += float(F.cosine_similarity(X, Y, dim=-1).double().sum())

    def result(self) -> dict:
        n = self.n
        mx, my = self.mx / n, self.my / n
        cxy = self.sxy - n * torch.outer(mx, my)
        cxx = self.sxx - n * torch.outer(mx, mx)
        cyy = self.syy - n * torch.outer(my, my)
        cka = (cxy.pow(2).sum()
               / (cxx.norm() * cyy.norm()).clamp_min(1e-30)).item()
        return {"cka": float(cka), "mean_cos": self.cos_sum / n}


@torch.no_grad()
def hidden_stack(bb: Backbone, images: torch.Tensor, n_patch: int):
    """All hidden states + fused map + the tap decomposition.

    Returns (hs, cls, fused, taps_used, tap_idx, w) where ``taps_used`` are the
    tensors ACTUALLY summed (post-LayerNorm when norm_taps is on), so
    ``w[i] * ||taps_used[i]||`` is the tap's real contribution to the fused map
    rather than its nominal softmax weight.
    """
    x = bb.preprocess(images)
    out = bb.model(pixel_values=x, output_hidden_states=True)
    hs = [h[:, -n_patch:,:] for h in out.hidden_states]
    cls = [h[:, 0,:] for h in out.hidden_states]
    idx = bb._resolve_layer_indices(len(out.hidden_states))
    w = F.softmax(bb.layer_weights, dim=0)
    taps = [out.hidden_states[i][:, -n_patch:,:] for i in idx]
    if bb.norm_taps:
        taps = [F.layer_norm(t, t.shape[-1:]) for t in taps]
    fused = sum(w[i] * t for i, t in enumerate(taps))
    return hs, cls, fused, taps, idx, w


class MapStats:
    """Streaming spatial statistics of a [B, n_patch, d] feature map.

    These target the actual hypothesis behind norm_taps — that giving mid-depth
    taps real weight puts more LOCALLY DISCRIMINATIVE signal into the map the
    relation head reads:

      hifreq   energy left after subtracting a 3x3 spatial average, over total
               energy. Higher = more detail that survives local averaging.
      nbr_cos  mean cosine between horizontally adjacent patch tokens. LOWER
               means neighbouring patches are more distinguishable, which is
               what a fine-grained read needs.
      pr       participation ratio of the token covariance spectrum
               ((sum l)^2 / sum l^2), i.e. effective rank. A map dominated by
               one tap inherits that tap's spectrum; a balanced one need not.
               Accumulated over ALL images and diagonalized once at the end, so
               it is the global spectrum, not a per-batch average.
      norm     mean per-token L2 norm — the magnitude that decides a tap's real
               share of the fused sum regardless of its softmax weight.
    """

    def __init__(self, d: int, device):
        self.s2 = torch.zeros(d, d, dtype=torch.float64, device=device)
        self.mu = torch.zeros(d, dtype=torch.float64, device=device)
        self.n = 0
        self.hi_num = self.hi_den = self.nbr = self.norm = 0.0
        self.n_tok = 0

    @torch.no_grad()
    def update(self, m: torch.Tensor, hp: int) -> None:
        B, n, d = m.shape
        g = m.reshape(B, hp, hp, d).permute(0, 3, 1, 2).float()
        blur = F.avg_pool2d(F.pad(g, (1, 1, 1, 1), mode="replicate"),
                            3, stride=1)
        self.hi_num += (g - blur).pow(2).sum().item()
        self.hi_den += g.pow(2).sum().item()
        a = F.normalize(g[:,:,:,:-1], dim=1)
        b = F.normalize(g[:,:,:, 1:], dim=1)
        self.nbr += float((a * b).sum(1).sum())
        self.n_tok += a.shape[0] * a.shape[2] * a.shape[3]
        x = m.reshape(-1, d).double()
        self.s2 += x.T @ x
        self.mu += x.sum(0)
        self.n += x.shape[0]
        self.norm += float(m.norm(dim=-1).float().sum())

    def result(self) -> dict:
        mu = self.mu / self.n
        cov = self.s2 / self.n - torch.outer(mu, mu)
        ev = torch.linalg.eigvalsh(cov).clamp_min(0)
        pr = float(ev.sum().pow(2) / ev.pow(2).sum().clamp_min(1e-30))
        return {"hifreq": round(self.hi_num / max(self.hi_den, 1e-12), 5),
                "nbr_cos": round(self.nbr / max(self.n_tok, 1), 4),
                "pr": round(pr, 2),
                "norm": round(self.norm / max(self.n, 1), 2)}


def joint_pca_rgb(token_sets: list[np.ndarray], h: int, w: int):
    """One PCA fit on the union of all sets -> comparable RGB maps."""
    from sklearn.decomposition import PCA
    allt = np.concatenate(token_sets, 0)
    pca = PCA(n_components=3).fit(allt)
    projs = [pca.transform(t) for t in token_sets]
    cat = np.concatenate(projs, 0)
    lo = np.percentile(cat, 1, axis=0)
    hi = np.percentile(cat, 99, axis=0)
    return [np.clip((p - lo) / (hi - lo + 1e-8), 0, 1)
.reshape(h, w, 3).astype(np.float32) for p in projs]


def upsample(m: np.ndarray, H: int, W: int) -> np.ndarray:
    t = torch.from_numpy(m).permute(2, 0, 1)[None] if m.ndim == 3 else \
        torch.from_numpy(m)[None, None]
    t = F.interpolate(t.float(), size=(H, W), mode="nearest")
    return (t[0].permute(1, 2, 0).numpy() if m.ndim == 3 else t[0, 0].numpy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--weights", default="ema", choices=["ema", "raw"])
    ap.add_argument("--data_root", default="runs/packed/psg")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n_stats", type=int, default=200)
    ap.add_argument("--n_viz", type=int, default=6)
    ap.add_argument("--viz_layers", default="6,12",
                    help="comma list of hidden-state indices (0=embeddings, "
                         "12=last block for ViT-B)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    labels = args.labels or [Path(c).parent.name for c in args.checkpoints]
    assert len(labels) == len(args.checkpoints)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- build the pretrained backbone + one fine-tuned clone per checkpoint
    ck0 = torch.load(args.checkpoints[0], map_location="cpu",
                     weights_only=False)
    a0 = ck0["args"] if isinstance(ck0["args"], dict) else vars(ck0["args"])
    bb_kwargs = dict(backbone_type=a0.get("backbone_type", "dinov3"),
                     model_name=a0.get("backbone_model") or None,
                     pretrained=True)
    models: dict[str, Backbone] = {}
    # The pretrained reference keeps the ARCHITECTURAL default (norm_taps off).
    # Its fused map is only a formality anyway: layer_weights is zero-init, so
    # "pretrained fused" is an unweighted mean, not something the head ever saw.
    models["pretrained"] = Backbone(
        **bb_kwargs, norm_taps=False).to(device).eval()
    drifts = {}
    init_sd = {k: v.cpu() for k, v in models["pretrained"].state_dict().items()}
    for ckpt_path, lab in zip(args.checkpoints, labels):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        key = ("ema_model" if args.weights == "ema" and "ema_model" in ck
               else "model")
        sd = {k[len("backbone."):]: v for k, v in ck[key].items()
              if k.startswith("backbone.")}
        # norm_taps is PER CHECKPOINT, read from that run's own args. Taking it
        # from checkpoints[0] (the old behaviour) silently gave every arm the
        # first arm's fusion rule — which is exactly the variable under test
        # when a norm_taps arm is compared against a non-norm_taps control.
        a = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
        nt = bool(a.get("norm_taps", False))
        bb = Backbone(**bb_kwargs, norm_taps=nt).to(device).eval()
        missing, unexpected = bb.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected backbone keys: {unexpected[:5]}"
        if missing:
            print(f"[shift] {lab}: {len(missing)} keys kept at pretrained "
                  f"values: {missing[:5]}")
        models[lab] = bb
        drifts[lab] = weight_drift(init_sd, sd)
        w = F.softmax(bb.layer_weights.detach().cpu(), 0).tolist()
        drifts[lab]["combiner_softmax"] = [round(v, 4) for v in w]
        drifts[lab]["norm_taps"] = nt
        # Trajectory proxy: EMA lags the raw weights, so raw-minus-EMA gives the
        # direction the combiner was still travelling at the end of training.
        if "ema_model" in ck and "model" in ck:
            other = "model" if key == "ema_model" else "ema_model"
            lw = ck[other].get("backbone.layer_weights")
            if lw is not None:
                drifts[lab][f"combiner_softmax_{other}"] = [
                    round(v, 4) for v in F.softmax(lw.float(), 0).tolist()]
        print(f"[shift] {lab}: weights={key}  norm_taps={nt}  by_type="
              f"{json.dumps(drifts[lab]['by_type'])}  combiner={w}")
    drift_heatmap(drifts, labels, os.path.join(args.out_dir,
                                               "weight_drift_heatmap.png"))

    ds = RelationDataset(root=args.data_root, split=args.split,
                         resolution=args.img_size)
    rng = random.Random(args.seed)
    stat_idx = rng.sample(range(len(ds)), min(args.n_stats, len(ds)))
    viz_idx = rng.sample(range(len(ds)), min(args.n_viz, len(ds)))
    n_patch = (args.img_size // models["pretrained"].patch_size) ** 2
    hp = args.img_size // models["pretrained"].patch_size

    # --- streaming per-layer stats over all model pairs
    probe = ds[stat_idx[0]][0][None].to(device)
    n_layers = len(models["pretrained"].model(
        pixel_values=models["pretrained"].preprocess(probe),
        output_hidden_states=True).hidden_states)
    d = models["pretrained"].d_model
    names = ["pretrained"] + labels
    pairs = list(itertools.combinations(names, 2))
    layer_keys = [str(i) for i in range(n_layers)] + ["fused"]
    acc = {p: {lk: CKAAccumulator(d, device) for lk in layer_keys}
           for p in pairs}
    cls_cos = {p: np.zeros(n_layers) for p in pairs}
    norm_sum = {m: np.zeros(n_layers) for m in names}
    # tap/fused spatial statistics, per model. "tap{i}" is the tensor actually
    # summed (post-LayerNorm under norm_taps); "tap{i}_raw" is the hidden state
    # as the backbone produced it, so the two arms' backbones stay comparable.
    ms_keys = ["fused"] + [f"tap{i}" for i in range(3)] + \
              [f"tap{i}_raw" for i in range(3)]
    mstat = {m: {} for m in names}
    eff_share = {m: np.zeros(3) for m in names}
    tap_idx_seen = {}
    n_seen = 0

    for start in range(0, len(stat_idx), args.batch_size):
        chunk = stat_idx[start:start + args.batch_size]
        images = torch.stack([ds[i][0] for i in chunk]).to(device)
        feats, clss, fused = {}, {}, {}
        for m in names:
            (feats[m], clss[m], fused[m],
             taps_m, tidx_m, w_m) = hidden_stack(models[m], images, n_patch)
            tap_idx_seen[m] = tidx_m
            for k, t in [("fused", fused[m])] + \
                    [(f"tap{i}", taps_m[i]) for i in range(len(taps_m))] + \
                    [(f"tap{i}_raw", feats[m][tidx_m[i]])
                     for i in range(len(taps_m))]:
                if k not in mstat[m]:
                    mstat[m][k] = MapStats(d, device)
                mstat[m][k].update(t, hp)
            # effective share: nominal softmax weight x the tap's actual scale.
            for i, t in enumerate(taps_m):
                eff_share[m][i] += float(w_m[i]) * float(
                    t.norm(dim=-1).float().mean())
        for li in range(n_layers):
            for m in names:
                norm_sum[m][li] += float(
                    feats[m][li].norm(dim=-1).float().sum())
            for p in pairs:
                x = feats[p[0]][li].reshape(-1, d).float()
                y = feats[p[1]][li].reshape(-1, d).float()
                acc[p][str(li)].update(x, y)
                cls_cos[p][li] += float(F.cosine_similarity(
                    clss[p[0]][li], clss[p[1]][li], dim=-1).sum())
        for p in pairs:
            acc[p]["fused"].update(fused[p[0]].reshape(-1, d).float(),
                                   fused[p[1]].reshape(-1, d).float())
        n_seen += len(chunk)
        if (start // args.batch_size) % 5 == 0:
            print(f"[shift] stats {n_seen}/{len(stat_idx)} images")

    report = {"n_images": n_seen, "weights": args.weights,
              "weight_drift": drifts, "layers": {},
              "tap_indices": {m: tap_idx_seen[m] for m in names},
              "map_stats": {m: {k: mstat[m][k].result() for k in ms_keys}
                            for m in names},
              "effective_share": {
                  m: [round(v, 4) for v in
                      (eff_share[m] / max(eff_share[m].sum(), 1e-12)).tolist()]
                  for m in names}}
    for m in names:
        print(f"[shift] {m}: effective_share={report['effective_share'][m]}  "
              f"fused={json.dumps(report['map_stats'][m]['fused'])}")
    for lk in layer_keys:
        row = {}
        for p in pairs:
            r = acc[p][lk].result()
            if lk != "fused":
                li = int(lk)
                r["cls_cos"] = cls_cos[p][li] / n_seen
            row[f"{p[0]}__vs__{p[1]}"] = {k: round(v, 4)
                                          for k, v in r.items()}
        if lk != "fused":
            li = int(lk)
            row["mean_patch_norm"] = {
                m: round(norm_sum[m][li] / (n_seen * n_patch), 2)
                for m in names}
        report["layers"][lk] = row
    json.dump(report, open(os.path.join(args.out_dir,
                                        "backbone_shift.json"), "w"),
              indent=2)

    # --- summary line chart: CKA + cosine per layer for each pair
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    xs = np.arange(n_layers)
    for p in pairs:
        k = f"{p[0]}__vs__{p[1]}"
        cka = [report["layers"][str(i)][k]["cka"] for i in range(n_layers)]
        cos = [report["layers"][str(i)][k]["mean_cos"]
               for i in range(n_layers)]
        axes[0].plot(xs, cka, marker="o", label=k)
        axes[1].plot(xs, cos, marker="o", label=k)
    axes[0].set_title("linear CKA per hidden state")
    axes[1].set_title("mean per-token cosine per hidden state")
    for ax in axes:
        ax.set_xlabel("hidden state (0 = embeddings)")
        ax.set_ylim(None, 1.005)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "backbone_shift_curves.png"),
                dpi=140)
    plt.close(fig)

    # --- joint-PCA panels
    viz_layers = [int(v) for v in args.viz_layers.split(",")]
    for vl in viz_layers:
        ncols = 1 + len(names) + 1
        fig, axes = plt.subplots(len(viz_idx), ncols,
                                 figsize=(3.3 * ncols, 3.45 * len(viz_idx)),
                                 squeeze=False)
        for r, idx in enumerate(viz_idx):
            image = ds[idx][0]
            img_np = image.permute(1, 2, 0).numpy()
            H, W = img_np.shape[:2]
            toks, last_ft = {}, None
            for m in names:
                hs = hidden_stack(models[m], image[None].to(device),
                                  n_patch)[0]
                toks[m] = hs[vl][0].float().cpu().numpy()
            rgbs = joint_pca_rgb([toks[m] for m in names], hp, hp)
            axes[r][0].imshow(img_np)
            axes[r][0].set_title(f"ds{idx}", fontsize=9)
            for c, m in enumerate(names):
                axes[r][1 + c].imshow(upsample(rgbs[c], H, W))
                axes[r][1 + c].set_title(m, fontsize=9)
            # where features moved: 1 - cos(pretrained, LAST arm) at this layer
            x0 = torch.from_numpy(toks["pretrained"])
            x1 = torch.from_numpy(toks[labels[-1]])
            dmap = (1 - F.cosine_similarity(x0, x1, dim=-1)).reshape(hp, hp)
            im = axes[r][ncols - 1].imshow(upsample(dmap.numpy(), H, W),
                                           cmap="inferno")
            axes[r][ncols - 1].set_title(f"1-cos(pre, {labels[-1]})",
                                         fontsize=9)
            fig.colorbar(im, ax=axes[r][ncols - 1], shrink=0.8)
            for ax in axes[r]:
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"hidden state {vl} — joint-PCA RGB (one shared "
                     f"projection per row)", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"pca_layer{vl}.png"), dpi=130)
        plt.close(fig)
        print(f"[shift] wrote pca_layer{vl}.png")

    print(f"[shift] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
