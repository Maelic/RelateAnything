"""Training monitor: cluster-friendly metric logging + live plots.

W&B cannot phone home from offline compute nodes, so this module is the
primary record and W&B (offline mode) is a mirror you can `wandb sync`
later from a login node. Everything lands under ``<output_dir>/metrics/``:

    metrics/iters.jsonl     one row every ``log_every`` optimizer steps:
                            {step, epoch, lr, loss_*...}
    metrics/epochs.jsonl    one row per epoch: {"epoch", "train": {...},
                            "eval": {...}}
    metrics/summary.json    best value + best epoch per eval metric,
                            refreshed every epoch
    metrics/plots/*.png     re-rendered every epoch:
                            losses_iter.png   per-component iteration losses
                            lr.png            schedule actually applied
                            eval_recall.png   SoftR@50 splits + fast head
                            eval_tail.png     SoftmR/mR + GT_MRR
                            throughput.png    img/s + GPU peak mem

Usage:
    mon = TrainMonitor(output_dir, log_every=50)   # rank-0 only
    mon.log_iter(step, epoch, lr, loss_dict)       # from train_one_epoch
    mon.log_epoch(epoch, train_metrics, eval_metrics)
    mon.render()                                   # cheap; called per epoch

Offline W&B: pass ``wandb_run`` (an initialized run object) and every
log_iter/log_epoch is mirrored with ``step``-keyed logging.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, Optional


class TrainMonitor:
    def __init__(self, output_dir: str, log_every: int = 50,
                 wandb_run=None) -> None:
        self.dir = os.path.join(output_dir, "metrics")
        self.plot_dir = os.path.join(self.dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        self.log_every = max(1, log_every)
        self.wandb = wandb_run
        self._iter_fh = open(os.path.join(self.dir, "iters.jsonl"), "a")
        self._epoch_path = os.path.join(self.dir, "epochs.jsonl")
        # reload history on resume so plots stay complete
        self._iters: list[dict] = []
        ip = os.path.join(self.dir, "iters.jsonl")
        if os.path.getsize(ip) if os.path.exists(ip) else 0:
            self._iters = [json.loads(l) for l in open(ip)]
        self._epochs: list[dict] = []
        if os.path.exists(self._epoch_path):
            self._epochs = [json.loads(l) for l in open(self._epoch_path)]

    # ------------------------------------------------------------------

    def log_iter(self, step: int, epoch: int, lr: float,
                 losses: Dict[str, float]) -> None:
        if step % self.log_every:
            return
        row = {"step": step, "epoch": epoch, "lr": round(lr, 8),
               **{k: round(float(v), 5) for k, v in losses.items()}}
        self._iters.append(row)
        self._iter_fh.write(json.dumps(row) + "\n")
        self._iter_fh.flush()
        if self.wandb is not None:
            self.wandb.log({f"iter/{k}": v for k, v in row.items()
                            if k not in ("step", "epoch")}, step=step)

    def log_epoch(self, epoch: int, train: Dict[str, float],
                  eval_: Dict[str, float]) -> None:
        row = {"epoch": epoch,
               "train": {k: round(float(v), 5) for k, v in train.items()},
               "eval": {k: round(float(v), 5) for k, v in eval_.items()}}
        self._epochs.append(row)
        with open(self._epoch_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        self._write_summary()
        if self.wandb is not None:
            self.wandb.log({**{f"train/{k}": v for k, v in train.items()},
                            **{f"eval/{k}": v for k, v in eval_.items()},
                            "epoch": epoch})

    def _write_summary(self) -> None:
        best: Dict[str, dict] = {}
        for row in self._epochs:
            for k, v in row["eval"].items():
                lower_better = "MedRank" in k or k.startswith("loss")
                cur = best.get(k)
                better = (cur is None
                          or (v < cur["value"] if lower_better
                              else v > cur["value"]))
                if better:
                    best[k] = {"value": v, "epoch": row["epoch"]}
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(best, f, indent=1, sort_keys=True)

    # ------------------------------------------------------------------

    def render(self) -> None:
        """Re-draw all plots from the accumulated history (cheap, ~100ms)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def _save(fig, name):
            fig.tight_layout()
            fig.savefig(os.path.join(self.plot_dir, name), dpi=110,
                        bbox_inches="tight")
            plt.close(fig)

        # -- iteration losses + lr ------------------------------------
        if self._iters:
            steps = [r["step"] for r in self._iters]
            loss_keys = sorted({k for r in self._iters for k in r
                                if k.startswith("loss_")})
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for k in loss_keys:
                ax.plot(steps, [r.get(k) for r in self._iters],
                        label=k, lw=1.0, alpha=0.9)
            ax.set_xlabel("optimizer step"); ax.set_ylabel("loss")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            ax.set_title("iteration losses")
            _save(fig, "losses_iter.png")

            fig, ax = plt.subplots(figsize=(9, 2.8))
            ax.plot(steps, [r["lr"] for r in self._iters], lw=1.2)
            ax.set_xlabel("optimizer step"); ax.set_ylabel("lr")
            ax.grid(alpha=0.3); ax.set_title("learning rate")
            _save(fig, "lr.png")

        if not self._epochs:
            return
        ep = [r["epoch"] for r in self._epochs]

        def series(scope, key):
            return [r[scope].get(key) for r in self._epochs]

        def plot_group(keys, title, name, scope="eval"):
            keys = [k for k in keys
                    if any(r[scope].get(k) is not None for r in self._epochs)]
            if not keys:
                return
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for k in keys:
                ax.plot(ep, series(scope, k), marker="o", ms=3, label=k)
            ax.set_xlabel("epoch"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
            ax.set_title(title)
            _save(fig, name)

        plot_group(["SoftR@50", "SoftR@50_semantic", "SoftR@50_spatial",
                    "fast_SoftR@50", "R@50"],
                   "head recall (val)", "eval_recall.png")
        plot_group(["SoftmR@50", "SoftmR@100", "mR@50", "GT_MRR",
                    "fast_SoftmR@50", "fast_GT_MRR"],
                   "tail / ranking (val)", "eval_tail.png")
        plot_group(["img_per_s", "gpu_mem_peak_gb"],
                   "throughput", "throughput.png", scope="train")

    def close(self) -> None:
        self._iter_fh.close()
