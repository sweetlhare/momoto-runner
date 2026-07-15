"""RF-DETR keypoint/pose backend (Roboflow, Apache-2.0).

Registered SECOND for keypose so DETRPose stays the default; this gives users the CHOICE of families
(like detection has RF-DETR + D-FINE). Roboflow's keypoint model is a *preview* (`RFDETRKeypointPreview`,
pretrained on COCO person keypoints, K=17). Verified on a 4090 end to end: `.train()` reads a project's
own K from the COCO metadata and fine-tunes to it (bird K=5, not locked to person-17), produces a
checkpoint, and `.predict()` loads it and runs — see rfdetr_pose_impl.py for the checkpoint (.pth/.ckpt)
and KeyPoints-parsing details. The GPU run exercised the code path, not accuracy, so the label stays
"(preview)". All the real model work happens in rfdetr_pose_impl.py inside the model venv (where
`rfdetr` is installed); this module just orchestrates + streams progress, exactly like the detection
RF-DETR backend.

Dataset: same Roboflow-COCO layout as detection, but the annotations carry keypoints (db_to_coco
task='keypose' already emits [x,y,v]*K + num_keypoints), reshaped by prepare_dataset().
"""
import importlib.util
import json
import os
import re
import shutil

from .base import ModelBackend, TrainConfig, TrainResult, register
from ._env import venv_python, run_stream, resolve_device

_IMPL = os.path.join(os.path.dirname(__file__), "rfdetr_pose_impl.py")
_RE_PROGRESS = re.compile(r"^PROGRESS (\{.*\})\s*$")
_RE_RESULT = re.compile(r"^RESULT (\{.*\})\s*$")
_META = "_rfdetr_pose_meta.json"


@register("keypose")
class RFDetrPoseEstimator(ModelBackend):
    task = "keypose"
    family = "rfdetrpose"
    label = "RF-DETR Pose (preview)"
    sizes = ("preview",)                       # Roboflow ships a single RFDETRKeypointPreview class
    default_size = "preview"
    weights_glob = ("checkpoint_best_total.pth", "checkpoint_best_ema.pth",
                    "checkpoint_best_regular.pth", "checkpoint.pth", "last.ckpt")

    @classmethod
    def available(cls) -> bool:
        # find_spec is cheap + doesn't import torch. The impl raises a clear error if this rfdetr
        # build predates RFDETRKeypointPreview.
        try:
            return importlib.util.find_spec("rfdetr") is not None
        except Exception:
            return False

    def prepare_dataset(self, samples, categories, out_dir, **kw):
        """COCO (keypoints) via the shared builder, reshaped to RF-DETR's Roboflow layout:
        <out>/{train,valid}/ with images + `_annotations.coco.json` (annotations carry keypoints)."""
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
        cmd = [venv_python(), _IMPL, "train", "--data", os.path.abspath(cfg.dataset_dir),
               "--out", out_dir, "--epochs", str(cfg.epochs), "--bs", str(cfg.batch),
               "--classes", *list(cfg.categories)]
        if resolve_device(cfg.device) == "cpu":
            cmd.append("--cpu")
        result = {}

        def on_line(line):
            mp = _RE_PROGRESS.match(line)
            if mp and progress_cb:
                p = json.loads(mp.group(1))
                progress_cb({"epoch": p.get("epoch"), "total_epochs": p.get("epochs"),
                             "metrics": {"map": p.get("map"), "oks_ap": p.get("map")}})
            mr = _RE_RESULT.match(line)
            if mr:
                result.update(json.loads(mr.group(1)))
            if should_stop and should_stop():
                raise KeyboardInterrupt("training stop requested")

        rc, tail = run_stream(cmd, os.path.dirname(_IMPL), on_line=on_line)
        if rc != 0 or not result.get("weights"):
            raise RuntimeError(f"RF-DETR pose training failed (rc={rc}):\n{tail}")
        weights = result["weights"]
        try:
            with open(os.path.join(os.path.dirname(weights), _META), "w") as f:
                json.dump({"classes": list(cfg.categories)}, f)
        except Exception:
            pass
        mp = result.get("map", 0.0)
        return TrainResult(weights=weights, metrics={"map": mp, "oks_ap": mp},
                           classes=result.get("classes", list(cfg.categories)), fmt=self.family)

    def _meta(self, weights):
        try:
            with open(os.path.join(os.path.dirname(weights), _META)) as f:
                return json.load(f)
        except Exception:
            return {"classes": []}

    def predict(self, weights: str, image_path: str, conf: float = 0.3):
        meta = self._meta(weights)
        cmd = [venv_python(), _IMPL, "predict", "--weights", weights, "--image", image_path,
               "--conf", str(conf), "--classes", *list(meta.get("classes", []))]
        objs = []

        def on_line(l):
            if l.startswith("OBJ "):
                objs.append(json.loads(l[4:]))
        rc, tail = run_stream(cmd, os.path.dirname(_IMPL), on_line=on_line)
        if rc != 0:
            raise RuntimeError(f"RF-DETR pose predict failed (rc={rc}):\n{tail}")
        return objs

    def export_onnx(self, weights: str, out_path: str, **kw) -> str:
        cmd = [venv_python(), _IMPL, "export", "--weights", weights, "--out", out_path]
        rc, tail = run_stream(cmd, os.path.dirname(_IMPL))
        if rc != 0 or not os.path.exists(out_path):
            raise RuntimeError(f"RF-DETR pose ONNX export failed (rc={rc}):\n{tail}")
        return out_path
