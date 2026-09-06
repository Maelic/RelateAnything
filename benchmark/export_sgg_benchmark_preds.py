"""Export our detector-box predictions in sgg_benchmark's ``predictions.pth`` format so that
REACT's own evaluator (sgg_benchmark/data/datasets/evaluation/sgg_eval.py) scores us and REACT
identically: same GT (their RelationDataset, duplicate predicates sampled), same per-image
averaging, same non-greedy IoU>=0.5 + label-triplet matching, same 100-pair budget, same ranking
rule (triplet score = rel_prob * conf(sub) * conf(obj), graph-constrained argmax per pair).

Output: list (their dataset order, keyed by COCO image id) of dicts with the fields their
evaluator reads: boxes (xyxy, ORIGINAL pixel frame — that is what their GT is in), mode,
image_size, pred_labels/labels (their contiguous class ids, 0 = background), pred_scores,
rel_pair_idxs [N,2], pred_rel_scores [N, 1+P] (col 0 = background), pred_rel_labels.

Usage: python training/export_sgg_benchmark_preds.py --checkpoint... --dataset_root runs/packed/vg150
       --dataset_name vg150 --det runs/detect/yolov8m_vg150_test.npz --det_weights <yolo.pt>
       --their runs/benchmark/react_compare/their_test_vg150.json --out <predictions.pth> [--keep_W]
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import benchmark.eval_zeroshot_detbox as D                       # noqa: E402  (model loading + DetBoxDataset)
from relsgg.training.engine import region_kwargs          # noqa: E402


class IndexedDetBox(D.DetBoxDataset):
    def __getitem__(self, i):
        image, boxes, target, n = super().__getitem__(i)
        target["idx"] = i
        return image, boxes, target, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True); p.add_argument("--keep_W", action="store_true")
    p.add_argument("--dataset_root", required=True); p.add_argument("--dataset_name", required=True)
    p.add_argument("--split", default="test"); p.add_argument("--det", required=True)
    p.add_argument("--det_weights", required=True); p.add_argument("--their", required=True)
    p.add_argument("--out", required=True); p.add_argument("--weights", default="ema")
    p.add_argument("--text_student", default=None)
    p.add_argument("--det_conf", type=float, default=0.10); p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--max_objects", type=int, default=60); p.add_argument("--eval_budget", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32); p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--topk", type=int, default=100, help="pairs kept per image (REACT: rel_topk=100)")
    p.add_argument("--limit", type=int, default=0, help="smoke: only the first N pack images")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = D.build_model_from_ckpt(ckpt, args.weights).to(device).eval()
    meta = json.load(open(os.path.join(args.dataset_root, args.split, "meta.json")))
    pred_names, cat_names = meta["predicates"], meta["categories"]
    class_remap = {n: i for i, n in enumerate(cat_names)}
    ck_args = ckpt.get("args") or {}; ck_args = ck_args if isinstance(ck_args, dict) else vars(ck_args)
    text_student = args.text_student if args.text_student is not None else (ck_args.get("text_student") or "")
    templates = getattr(D, "TRAIN_TEMPLATES")
    if args.keep_W and list(ckpt.get("pred_names") or []) == list(pred_names):
        print(f"--keep_W: keeping the checkpoint's W ({len(pred_names)} predicates)")
        model.vocab_head.pred_names = list(pred_names)
    elif text_student:
        from relsgg.text.student import encode_texts_student
        E = encode_texts_student(pred_names, text_student, templates=templates, device=device)
        model.vocab_head.set_vocabulary_matrix(pred_names, E)
    else:
        raise SystemExit(
            "this checkpoint names no text student. The vocabulary has to be "
            "encoded by the encoder the head was trained against; pass "
            "--text_student, or use a released model, which ships its own.")
    model.reparameterize()

    their = json.load(open(args.their))
    t_cls, t_pred, t_ids = their["ind_to_classes"], their["ind_to_predicates"], their["image_ids"]
    pred_col = np.array([t_pred.index(n) for n in pred_names], dtype=np.int64)       # ours -> their column
    cls_map = np.array([t_cls.index(n) if n in t_cls else 0 for n in cat_names] + [0], dtype=np.int64)  # [-1] -> bg
    assert (pred_col > 0).all(), "predicate name missing on their side"
    print(f"vocab map: {len(pred_names)} predicates -> their {len(t_pred)-1}; {sum(cls_map[:-1]==0)} categories unmapped")
    pos_of_id = {iid: k for k, iid in enumerate(t_ids)}

    d_names = D.det_class_names(args.det_weights)
    ds = IndexedDetBox(args.dataset_root, args.det, d_names, 0.5, args.det_conf, args.max_objects,
                       args.img_size, False, class_remap, weight_by_conf=True, split=args.split)
    if args.limit: ds.file_names = ds.file_names[:args.limit]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=D.collate,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    raw = model.module if hasattr(model, "module") else model
    raw.sampler.final_budget = min(args.eval_budget, raw.sampler.geo_budget)
    Vt = len(t_pred)

    def empty(iid):
        return {"boxes": torch.zeros(0, 4), "mode": "xyxy", "image_size": (0, 0), "pred_labels": torch.zeros(0, dtype=torch.long),
                "labels": torch.zeros(0, dtype=torch.long), "pred_scores": torch.zeros(0), "rel_pair_idxs": torch.zeros(0, 2, dtype=torch.long),
                "pred_rel_scores": torch.zeros(0, Vt), "pred_rel_labels": torch.zeros(0, dtype=torch.long), "image_id": iid}
    out_list = [empty(iid) for iid in t_ids]
    n_written = 0
    with torch.no_grad():
        for images, boxes, box_counts, targets in tqdm(loader, desc="export"):
            images, boxes, box_counts = images.to(device), boxes.to(device), box_counts.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                out = model(images, boxes, box_counts, targets=None, **region_kwargs(targets, device))
            logits, sub_idx, obj_idx, valid = out["logits"], out["sub_idx"], out["obj_idx"], out["valid_mask"]
            scores = torch.softmax(logits.float(), -1)
            if out.get("pair_logits") is not None:
                scores = scores * torch.sigmoid(out["pair_logits"].float()).unsqueeze(-1)
            for b, tgt in enumerate(targets):
                i = tgt["idx"]; iid, W, H = int(ds.img_meta[i][0]), float(ds.img_meta[i][1]), float(ds.img_meta[i][2])
                if iid not in pos_of_id: continue
                n = int(box_counts[b]); bsc = tgt["box_scores"][:n].float()
                n_det = int((bsc > 0).sum())            # DetBoxDataset pads an empty image to 1 zero box
                cx = boxes[b,:n].float().cpu()
                xyxy = torch.stack([cx[:, 0] - cx[:, 2] / 2, cx[:, 1] - cx[:, 3] / 2, cx[:, 0] + cx[:, 2] / 2, cx[:, 1] + cx[:, 3] / 2], -1) * torch.tensor([W, H, W, H])
                # labels: pack category idx per det box is not stored in the target -> recover from the dataset
                s0, s1 = ds.starts[i], ds.starts[i + 1]
                conf = ds.d_conf[s0:s1]; top = np.argsort(-conf)[:min(args.max_objects, len(conf))]
                det_cls = ds.d_cls_remap[s0:s1][top]
                lab = torch.from_numpy(cls_map[np.where(det_cls < 0, len(cat_names), det_cls)])
                rec = {"boxes": xyxy[:n_det], "mode": "xyxy", "image_size": (int(W), int(H)), "pred_labels": lab[:n_det],
                       "labels": lab[:n_det], "pred_scores": bsc[:n_det], "image_id": iid}
                m = valid[b]
                if n_det == 0 or int(m.sum()) == 0:
                    rec.update({"rel_pair_idxs": torch.zeros(0, 2, dtype=torch.long), "pred_rel_scores": torch.zeros(0, Vt), "pred_rel_labels": torch.zeros(0, dtype=torch.long)})
                    out_list[pos_of_id[iid]] = rec; n_written += 1; continue
                pr = scores[b][m]; si = sub_idx[b][m].long().clamp(max=n - 1); oi = obj_idx[b][m].long().clamp(max=n - 1)
                bd = bsc.to(pr.device)
                pr = pr * (bd[si] * bd[oi]).unsqueeze(-1)                      # triplet score, REACT convention
                best, bp = pr.max(-1); k = min(args.topk, best.numel()); _, top_pairs = best.topk(k)
                rows = torch.zeros(k, Vt); rows[:, torch.from_numpy(pred_col)] = pr[top_pairs].cpu()
                rec.update({"rel_pair_idxs": torch.stack([si[top_pairs], oi[top_pairs]], -1).cpu(),
                            "pred_rel_scores": rows, "pred_rel_labels": torch.from_numpy(pred_col)[bp[top_pairs].cpu()]})
                out_list[pos_of_id[iid]] = rec; n_written += 1
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out_list, args.out)
    print(f"wrote {args.out}: {len(out_list)} entries in their order, {n_written} with predictions, "
          f"{len(out_list) - n_written} empty (not in pack / limit)")


if __name__ == "__main__":
    main()
