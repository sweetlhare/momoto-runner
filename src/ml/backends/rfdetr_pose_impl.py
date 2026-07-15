"""RF-DETR keypoint/pose (Roboflow, Apache-2.0) train/predict helper — runs in the model venv,
which must have an `rfdetr` new enough to expose `RFDETRKeypointPreview`. Speaks the same
PROGRESS/RESULT/OBJ line protocol as rfdetr_impl.py.

RF-DETR keypoint support is a Roboflow *preview* (class `RFDETRKeypointPreview`), pretrained on COCO
person keypoints (K=17) but NOT locked to it: verified on a 4090 that `.train(dataset_dir=…)` reads a
project's own K straight from the COCO metadata and fine-tunes to it (bird K=5 confirmed end to end —
train → checkpoint → predict, no person-17 lock). Checkpoints: RF-DETR's BestModelCheckpoint writes
stripped `checkpoint_best_*.pth` only once `val/keypoint_map` improves, while the standard Lightning
ModelCheckpoint always writes `last.ckpt` / `checkpoint_{epoch}.ckpt` — so `_find_checkpoint` prefers
the best `.pth` and falls back to the `.ckpt` (rfdetr's `load_pretrain_weights` normalizes PTL .ckpt).
`.predict()` returns a supervision 0.29 `KeyPoints` (xy / keypoint_confidence / visible / class_id /
data["xyxy"]) — parsed below; the parser was confirmed on real pretrained person-17 output (21
detections, K=17, boolean `visible`, xyxy boxes, and a 1-indexed `class_id` correctly resolved via
data["class_name"]). NB: the fine-tune GPU run exercised the code path, not accuracy (tiny synthetic
set → map 0).
Dataset layout: same Roboflow-COCO as detection (<data>/{train,valid}/_annotations.coco.json), but
the annotations carry `keypoints`/`num_keypoints` (db_to_coco task='keypose' already emits them).
"""
import argparse
import glob
import json
import os
import sys


def _ctor():
    import rfdetr
    if hasattr(rfdetr, "RFDETRKeypointPreview"):
        return rfdetr.RFDETRKeypointPreview
    raise RuntimeError("this rfdetr build has no RFDETRKeypointPreview — upgrade the rfdetr package")


def _find_checkpoint(out_dir):
    # Prefer the stripped "best" .pth files RF-DETR's BestModelCheckpoint writes (checkpoint_best_total
    # is the regular-vs-EMA winner) — but those only appear once val/keypoint_map improves.
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth",
                 "checkpoint_best.pth", "checkpoint.pth"):
        p = os.path.join(out_dir, name)
        if os.path.isfile(p):
            return p
    # Fall back to any .pth, then the PTL last.ckpt / checkpoint_{epoch}.ckpt that the standard
    # Lightning ModelCheckpoint always writes — a short or degenerate run may never improve the metric
    # and so leave only a .ckpt. rfdetr's load_pretrain_weights normalizes PTL .ckpt (state_dict→model).
    cands = sorted(glob.glob(os.path.join(out_dir, "**", "*.pth"), recursive=True) +
                   glob.glob(os.path.join(out_dir, "**", "*.ckpt"), recursive=True),
                   key=os.path.getmtime, reverse=True)
    return cands[0] if cands else None


def train(a):
    net = _ctor()()                                  # COCO person-keypoints pretrained by default
    os.makedirs(a.out, exist_ok=True)
    state = {"epoch": 0, "map": 0.0}

    def _emit(epoch=None, total=None, mp=None):
        if epoch is not None:
            state["epoch"] = epoch
        if mp is not None:
            state["map"] = float(mp)
        print("PROGRESS " + json.dumps({"epoch": state["epoch"], "epochs": total or a.epochs,
                                        "map": state["map"]}), flush=True)

    # Same callback hookup as detection — the keypoint eval reports OKS AP; scrape it defensively.
    try:
        def on_epoch(payload=None, **kw):
            d = payload if isinstance(payload, dict) else kw
            ep = d.get("epoch")
            mp = (d.get("test_coco_eval_keypoints") or d.get("test_coco_eval_bbox") or [None])
            mp = mp[0] if isinstance(mp, list) else (d.get("map") or d.get("mAP"))
            _emit(epoch=(ep + 1) if isinstance(ep, int) else state["epoch"] + 1, mp=mp)
        cbs = getattr(net, "callbacks", None)
        if isinstance(cbs, dict) and "on_fit_epoch_end" in cbs:
            cbs["on_fit_epoch_end"].append(on_epoch)
    except Exception as e:
        print(f"[rfdetr-pose] callback hookup skipped: {e}", flush=True)

    kw = dict(dataset_dir=a.data, epochs=a.epochs, batch_size=a.bs, output_dir=a.out)
    if a.cpu:
        kw["device"] = "cpu"
    net.train(**kw)

    best = _find_checkpoint(a.out)
    if not best:
        print("RESULT " + json.dumps({"error": "no checkpoint produced"}), flush=True)
        sys.exit(1)
    print("RESULT " + json.dumps({"weights": best, "map": state["map"],
                                  "classes": list(a.classes or [])}), flush=True)


def predict(a):
    from PIL import Image
    net = _ctor()(pretrain_weights=a.weights)
    img = Image.open(a.image).convert("RGB")
    W, H = img.size
    kp = net.predict(img, threshold=a.conf)
    if isinstance(kp, list):                         # predict may return a list for batched input
        kp = kp[0] if kp else None
    if kp is None:
        return
    names = list(a.classes or [])
    # supervision 0.29 KeyPoints fields: xy (N, K, 2) abs px; keypoint_confidence (N, K) per-point
    # findability; visible (N, K) bool; class_id (N,) 0-based into class_names; data["xyxy"] (N, 4)
    # detection boxes, data["class_name"] (N,). (.confidence is a deprecated alias for keypoint_confidence.)
    xy = getattr(kp, "xy", None)
    kconf = getattr(kp, "keypoint_confidence", None)
    vis = getattr(kp, "visible", None)
    cids = getattr(kp, "class_id", None)
    data = getattr(kp, "data", None) or {}
    xyxy = data.get("xyxy")
    cnames = data.get("class_name")
    n = len(xy) if xy is not None else 0
    for i in range(n):
        pts = xy[i]                                  # (K, 2) absolute px
        flat, xs, ys = [], [], []
        for k in range(len(pts)):
            x, y = float(pts[k][0]), float(pts[k][1])
            if vis is not None:                      # model-emitted visibility wins
                v = 2 if bool(vis[i][k]) else 0
            elif kconf is not None:
                v = 2 if float(kconf[i][k]) >= a.conf else 0
            else:
                v = 2 if (x > 0 or y > 0) else 0
            flat += [round(x / W, 6), round(y / H, 6), v]
            if v:
                xs.append(x); ys.append(y)
        # bbox: prefer the model's own detection box, else derive a loose one from visible keypoints.
        if xyxy is not None and i < len(xyxy):
            x0, y0, x1, y1 = (float(c) for c in xyxy[i])
            bbox = {"x_center": (x0 + x1) / 2 / W, "y_center": (y0 + y1) / 2 / H,
                    "width": abs(x1 - x0) / W, "height": abs(y1 - y0) / H}
        elif xs and ys:
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            pad = 0.05
            bbox = {"x_center": (x0 + x1) / 2 / W, "y_center": (y0 + y1) / 2 / H,
                    "width": min(1.0, (x1 - x0) / W + pad), "height": min(1.0, (y1 - y0) / H + pad)}
        else:
            bbox = {"x_center": 0.5, "y_center": 0.5, "width": 0.0, "height": 0.0}
        cid = int(cids[i]) if cids is not None else 0
        if cnames is not None and i < len(cnames):
            cat = str(cnames[i])
        elif 0 <= cid < len(names):
            cat = names[cid]
        else:
            cat = names[0] if names else str(cid)
        print("OBJ " + json.dumps({"category": cat, "keypoints": flat, "bbox": bbox}), flush=True)


def export(a):
    net = _ctor()(pretrain_weights=a.weights)
    out_dir = os.path.dirname(os.path.abspath(a.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    try:
        net.export(output_dir=out_dir)
    except TypeError:
        net.export()
    for c in sorted(glob.glob(os.path.join(out_dir, "**", "*.onnx"), recursive=True),
                    key=os.path.getmtime, reverse=True):
        if os.path.abspath(c) != os.path.abspath(a.out):
            os.replace(c, a.out)
        return
    print("no onnx produced", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True); t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=50); t.add_argument("--bs", type=int, default=4)
    t.add_argument("--cpu", action="store_true"); t.add_argument("--classes", nargs="*", default=[])
    p = sub.add_parser("predict")
    p.add_argument("--weights", required=True); p.add_argument("--image", required=True)
    p.add_argument("--conf", type=float, default=0.3); p.add_argument("--classes", nargs="*", default=[])
    e = sub.add_parser("export")
    e.add_argument("--weights", required=True); e.add_argument("--out", required=True)
    args = ap.parse_args()
    {"train": train, "predict": predict, "export": export}[args.mode](args)
