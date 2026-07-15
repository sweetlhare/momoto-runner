"""RF-DETR detection backend (Roboflow, Apache-2.0) — the preferred real-time DETR for detection.

OPTIONAL DEP: the model env must have the `rfdetr` package. When it's absent, `available()` is
False, so get_backend() skips RF-DETR for the default pick (falls back to D-FINE) and the arch
picker flags it unavailable; an EXPLICIT rfdetr arch still raises a clear error. RF-DETR eats a
Roboflow-COCO layout (<data>/{train,valid}/_annotations.coco.json), which prepare_dataset builds
from our universal DB annotations via the shared COCO builder.
"""
import importlib.util
import json
import os
import re
import shutil

from .base import ModelBackend, TrainConfig, TrainResult, register
from ._env import venv_python, run_stream, resolve_device

_IMPL = os.path.join(os.path.dirname(__file__), "rfdetr_impl.py")
_RE_PROGRESS = re.compile(r"^PROGRESS (\{.*\})\s*$")
_RE_RESULT = re.compile(r"^RESULT (\{.*\})\s*$")
# rf-detr prints its best mAP in lines like "Best EMA mAP improved to 0.9681" or "...value=0.9699".
_RE_MAP = re.compile(r"(?:mAP improved to|value=)\s*([0-9]*\.?[0-9]+)")
_META = "_rfdetr_meta.json"                        # stashed next to the checkpoint for predict/export

_SIZE_TO_CLASS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
                  "base": "RFDETRBase", "large": "RFDETRLarge"}


@register("detection")
class RFDetrDetector(ModelBackend):
    task = "detection"
    family = "rfdetr"
    label = "RF-DETR"
    sizes = ("nano", "small", "medium", "base", "large")
    default_size = "base"
    weights_glob = ("checkpoint_best_ema.pth", "checkpoint_best_regular.pth", "checkpoint.pth")

    @classmethod
    def available(cls) -> bool:
        # find_spec is cheap and does NOT import torch/the repo — safe in the light API process.
        try:
            return importlib.util.find_spec("rfdetr") is not None
        except Exception:
            return False

    def _model_class(self, size: str) -> str:
        return _SIZE_TO_CLASS.get(self.size_tier(size), "RFDETRBase")

    def prepare_dataset(self, samples, categories, out_dir, **kw):
        """Build COCO (shared builder) then reshape to RF-DETR's Roboflow layout: <out>/{train,valid}/
        with the images + an `_annotations.coco.json` inside each split dir."""
        from src.ml.datasets import build_coco_dataset
        tmp = os.path.join(out_dir, "_coco")
        build_coco_dataset(samples, categories, tmp, task=self.task, **kw)
        for split, rf_split in (("train", "train"), ("val", "valid")):
            src_img = os.path.join(tmp, split)
            src_ann = os.path.join(tmp, "annotations", f"instances_{split}.json")
            dst = os.path.join(out_dir, rf_split)
            os.makedirs(dst, exist_ok=True)
            if os.path.isdir(src_img):
                for fn in os.listdir(src_img):
                    shutil.copy(os.path.join(src_img, fn), os.path.join(dst, fn))
            if os.path.isfile(src_ann):
                shutil.copy(src_ann, os.path.join(dst, "_annotations.coco.json"))
        shutil.rmtree(tmp, ignore_errors=True)
        return {"out_dir": out_dir, "num_classes": len(categories), "categories": list(categories)}

    def train(self, cfg: TrainConfig, progress_cb=None, should_stop=None) -> TrainResult:
        out_dir = os.path.abspath(cfg.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        model_class = self._model_class(cfg.size)
        cmd = [venv_python(), _IMPL, "train", "--data", os.path.abspath(cfg.dataset_dir),
               "--out", out_dir, "--model", model_class, "--epochs", str(cfg.epochs),
               "--bs", str(cfg.batch), "--classes", *list(cfg.categories)]
        if resolve_device(cfg.device) == "cpu":
            cmd.append("--cpu")
        result = {}

        def on_line(line):
            mp = _RE_PROGRESS.match(line)
            if mp and progress_cb:
                p = json.loads(mp.group(1))
                progress_cb({"epoch": p.get("epoch"), "total_epochs": p.get("epochs"),
                             "metrics": {"map": p.get("map"), "map50_95": p.get("map")}})
            mr = _RE_RESULT.match(line)
            if mr:
                result.update(json.loads(mr.group(1)))
            # rf-detr logs its best mAP as "Best EMA mAP improved to X" / "...value=X"; my callback
            # can't see its internal metric dict, so scrape the number and keep the max seen.
            mm = _RE_MAP.search(line)
            if mm:
                result["map"] = max(float(result.get("map") or 0.0), float(mm.group(1)))
            if should_stop and should_stop():
                raise KeyboardInterrupt("training stop requested")

        rc, tail = run_stream(cmd, os.path.dirname(_IMPL), on_line=on_line)
        if rc != 0 or not result.get("weights"):
            raise RuntimeError(f"RF-DETR training failed (rc={rc}):\n{tail}")
        weights = result["weights"]
        # stash the model class + class names next to the checkpoint so predict/export can rebuild
        try:
            with open(os.path.join(os.path.dirname(weights), _META), "w") as f:
                json.dump({"model": model_class, "classes": list(cfg.categories)}, f)
        except Exception:
            pass
        mp = result.get("map", 0.0)
        return TrainResult(weights=weights, metrics={"map": mp, "map50_95": mp},
                           classes=result.get("classes", list(cfg.categories)), fmt=self.family)

    def _meta(self, weights):
        try:
            with open(os.path.join(os.path.dirname(weights), _META)) as f:
                return json.load(f)
        except Exception:
            return {"model": "RFDETRBase", "classes": []}

    def predict(self, weights: str, image_path: str, conf: float = 0.3):
        meta = self._meta(weights)
        cmd = [venv_python(), _IMPL, "predict", "--weights", weights, "--image", image_path,
               "--conf", str(conf), "--model", meta.get("model", "RFDETRBase"),
               "--classes", *list(meta.get("classes", []))]
        objs = []

        def on_line(l):
            if l.startswith("OBJ "):
                objs.append(json.loads(l[4:]))
        rc, tail = run_stream(cmd, os.path.dirname(_IMPL), on_line=on_line)
        if rc != 0:
            raise RuntimeError(f"RF-DETR predict failed (rc={rc}):\n{tail}")
        return objs

    def export_onnx(self, weights: str, out_path: str, **kw) -> str:
        meta = self._meta(weights)
        cmd = [venv_python(), _IMPL, "export", "--weights", weights, "--out", out_path,
               "--model", meta.get("model", "RFDETRBase")]
        rc, tail = run_stream(cmd, os.path.dirname(_IMPL))
        if rc != 0 or not os.path.exists(out_path):
            raise RuntimeError(f"RF-DETR ONNX export failed (rc={rc}):\n{tail}")
        return out_path
