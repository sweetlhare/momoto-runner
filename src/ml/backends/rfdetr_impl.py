"""RF-DETR (Roboflow, Apache-2.0) train/predict helper — runs in the model venv, which must
have the `rfdetr` package. Speaks the same PROGRESS/RESULT/OBJ line protocol as the other impls
so the backend orchestrator can stream progress and pick up the checkpoint.

Modes:
  train   --data <coco_roboflow_dir> --out <dir> --model RFDETRBase --epochs N --bs B [--cpu]
  predict --weights <ckpt> --image <img> --conf C --model RFDETRBase

Dataset layout RF-DETR expects (Roboflow COCO): <data>/{train,valid}/_annotations.coco.json
plus the images in the same folder. The backend's prepare_dataset() lays it out that way.
"""
import argparse
import glob
import json
import os
import sys


def _model_ctor(name):
    import rfdetr
    if hasattr(rfdetr, name):
        return getattr(rfdetr, name)
    # Older rfdetr only ships Base/Large — fall back so a Nano/Small/Medium request still trains.
    return getattr(rfdetr, "RFDETRBase")


def _find_checkpoint(out_dir):
    for name in ("checkpoint_best_ema.pth", "checkpoint_best_regular.pth", "checkpoint_best.pth",
                 "checkpoint.pth"):
        p = os.path.join(out_dir, name)
        if os.path.isfile(p):
            return p
    cands = sorted(glob.glob(os.path.join(out_dir, "**", "*.pth"), recursive=True),
                   key=os.path.getmtime, reverse=True)
    return cands[0] if cands else None


def train(a):
    Model = _model_ctor(a.model)
    net = Model()                                  # COCO-pretrained weights by default
    os.makedirs(a.out, exist_ok=True)

    state = {"epoch": 0, "map": 0.0}

    def _emit(epoch=None, total=None, mp=None):
        if epoch is not None:
            state["epoch"] = epoch
        if mp is not None:
            state["map"] = float(mp)
        print("PROGRESS " + json.dumps({"epoch": state["epoch"], "epochs": total or a.epochs,
                                        "map": state["map"]}), flush=True)

    # RF-DETR exposes a callbacks dict; the payload shape varies by version, so parse defensively.
    try:
        def on_epoch(payload=None, **kw):
            d = payload if isinstance(payload, dict) else kw
            ep = d.get("epoch")
            mp = (d.get("test_coco_eval_bbox") or [None])[0] if isinstance(d.get("test_coco_eval_bbox"), list) \
                else (d.get("map") or d.get("mAP") or d.get("coco_eval_bbox"))
            _emit(epoch=(ep + 1) if isinstance(ep, int) else state["epoch"] + 1, mp=mp)
        cbs = getattr(net, "callbacks", None)
        if isinstance(cbs, dict) and "on_fit_epoch_end" in cbs:
            cbs["on_fit_epoch_end"].append(on_epoch)
    except Exception as e:
        print(f"[rfdetr] callback hookup skipped: {e}", flush=True)

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
    Model = _model_ctor(a.model)
    net = Model(pretrain_weights=a.weights)
    det = net.predict(Image.open(a.image).convert("RGB"), threshold=a.conf)
    # supervision.Detections: xyxy (abs px), class_id, confidence. Emit canonical normalized objects.
    W, H = Image.open(a.image).size
    names = list(a.classes or [])
    xyxy = getattr(det, "xyxy", [])
    cids = getattr(det, "class_id", None)
    confs = getattr(det, "confidence", None)
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
        cid = int(cids[i]) if cids is not None else 0
        cat = names[cid] if 0 <= cid < len(names) else str(cid)
        obj = {"category": cat,
               "bbox": {"x_center": (x1 + x2) / 2 / W, "y_center": (y1 + y2) / 2 / H,
                        "width": (x2 - x1) / W, "height": (y2 - y1) / H},
               "score": float(confs[i]) if confs is not None else None}
        print("OBJ " + json.dumps(obj), flush=True)


def export(a):
    Model = _model_ctor(a.model)
    net = Model(pretrain_weights=a.weights)
    out_dir = os.path.dirname(os.path.abspath(a.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    # RF-DETR writes an ONNX under output_dir; API varies, so try a couple of signatures.
    try:
        net.export(output_dir=out_dir)
    except TypeError:
        net.export()
    onnx = None
    for c in sorted(glob.glob(os.path.join(out_dir, "**", "*.onnx"), recursive=True),
                    key=os.path.getmtime, reverse=True):
        onnx = c
        break
    if not onnx:
        print("no onnx produced", file=sys.stderr)
        sys.exit(1)
    if os.path.abspath(onnx) != os.path.abspath(a.out):
        os.replace(onnx, a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--model", default="RFDETRBase")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--bs", type=int, default=4)
    t.add_argument("--cpu", action="store_true")
    t.add_argument("--classes", nargs="*", default=[])
    p = sub.add_parser("predict")
    p.add_argument("--weights", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--model", default="RFDETRBase")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--classes", nargs="*", default=[])
    e = sub.add_parser("export")
    e.add_argument("--weights", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--model", default="RFDETRBase")
    args = ap.parse_args()
    {"train": train, "predict": predict, "export": export}[args.mode](args)
