"""D-FINE detection backend (Apache-2.0). Replaces YOLO detection.

train() + export_onnx() follow the recipe verified end-to-end on real data
(DB->COCO via db_to_coco -> generated config including custom/dfine_hgnetv2_<size>_custom.yml
-> train.py --use-amp -> best_stg1.pth -> export_onnx.py). predict() is implemented:
it subprocesses a helper in the model venv and parses 'OBJ <json>' lines into canonical
detections (see predict() below).
"""
import os
import re
import json
import shutil
import tempfile

from .base import ModelBackend, TrainConfig, TrainResult, register
from ._env import venv_python, repo, run_stream, resolve_device, find_weight


_RE_EPOCH = re.compile(r"Epoch:\s*\[(\d+)/(\d+)\]")
_RE_MAP = re.compile(r"IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([\d.]+)")


class _DFineFamily(ModelBackend):
    """Shared orchestration for the D-FINE config-driven repos (detection + segmentation)."""
    REPO = "D-FINE"
    CUSTOM_TMPL = "custom/dfine_hgnetv2_{size}_custom.yml"   # relative to configs/dfine/
    sizes = ("n", "s", "m", "l", "x")
    default_size = "n"
    weights_glob = ("best_stg1.pth", "best.pth", "last.pth")

    def _write_config(self, cfg: TrainConfig, size: str) -> str:
        ds = os.path.abspath(cfg.dataset_dir)
        gen_dir = os.path.join(repo(self.REPO), "configs", "dfine", "_gen")
        os.makedirs(gen_dir, exist_ok=True)
        cfg_path = os.path.join(gen_dir, f"proj_{cfg.project_id}_{self.task}.yml")
        out_dir = os.path.abspath(cfg.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        # ABSOLUTE include path: the config is stashed next to the checkpoint (out_dir) for
        # predict/export, where a '../custom/...' relative include would resolve outside the repo.
        # An absolute path works both in-place (configs/dfine/_gen/) at train time and from out_dir.
        custom_abs = os.path.join(repo(self.REPO), "configs", "dfine", self.CUSTOM_TMPL.format(size=size))
        body = f"""__include__: [ '{custom_abs}' ]

output_dir: {out_dir}
num_classes: {cfg.num_classes}
remap_mscoco_category: False
epochs: {cfg.epochs}
checkpoint_freq: {max(1, cfg.epochs)}

train_dataloader:
  total_batch_size: {cfg.batch}
  dataset:
    img_folder: {ds}/train
    ann_file: {ds}/annotations/instances_train.json

val_dataloader:
  total_batch_size: {cfg.batch}
  dataset:
    img_folder: {ds}/val
    ann_file: {ds}/annotations/instances_val.json
"""
        with open(cfg_path, "w") as f:
            f.write(body)
        # stash the config next to the checkpoints so export/predict can reuse it
        shutil.copy(cfg_path, os.path.join(out_dir, "_dfine_config.yml"))
        # stash the ordered category names too: predict maps the postprocessor's 0-indexed
        # label back to a name. (The COCO ann_file the config points at also carries these,
        # but it lives under the ephemeral dataset dir — the meta is self-contained.)
        with open(os.path.join(out_dir, "_dfine_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"size": size, "num_classes": cfg.num_classes,
                       "categories": list(cfg.categories)}, f)
        return cfg_path

    def train(self, cfg: TrainConfig, progress_cb=None, should_stop=None) -> TrainResult:
        size = cfg.size if cfg.size in self.sizes else self.default_size
        cfg_path = self._write_config(cfg, size)
        dev = resolve_device(cfg.device)
        cmd = [venv_python(), "train.py", "-c", cfg_path, "--seed", "0", "-d", dev]
        if dev == "cuda":
            cmd.append("--use-amp")
        state = {"epoch": 0, "map": 0.0}

        def on_line(line):
            m = _RE_EPOCH.search(line)
            if m:
                state["epoch"] = int(m.group(1))
            mm = _RE_MAP.search(line)
            if mm:
                state["map"] = float(mm.group(1))
                if progress_cb:
                    progress_cb({"epoch": state["epoch"], "total_epochs": cfg.epochs,
                                 "metrics": {"map": state["map"], "map50_95": state["map"]}})
            if should_stop and should_stop():
                raise KeyboardInterrupt("training stop requested")

        # PYTHONPATH="" so the agent app's regular `src` package can't shadow the repo's
        # namespace `src` (from src.core/src.solver import ...) — mirrors predict/export_onnx
        # and the DETRPose/seg backends; train() was the one entry point that omitted it.
        rc, tail = run_stream(cmd, repo(self.REPO), on_line=on_line, env={"PYTHONPATH": ""})
        if rc != 0:
            raise RuntimeError(f"D-FINE ({self.task}) training failed (rc={rc}):\n{tail}")
        weights = find_weight(os.path.abspath(cfg.out_dir), self.weights_glob)
        if not weights:
            raise RuntimeError(f"D-FINE produced no checkpoint in {cfg.out_dir}:\n{tail}")
        return TrainResult(weights=weights, metrics={"map": state["map"], "map50_95": state["map"]},
                           classes=list(cfg.categories), fmt=self.family)

    def export_onnx(self, weights: str, out_path: str, **kw) -> str:
        cfg_path = kw.get("config") or os.path.join(os.path.dirname(weights), "_dfine_config.yml")
        if not os.path.exists(cfg_path):
            raise RuntimeError(f"D-FINE export needs the training config; not found at {cfg_path}")
        # export_onnx.py builds Model(model.deploy() + postprocessor.deploy()) and writes
        # weights.replace('.pth','.onnx') in CWD. It sys.path.insert(0)s its own repo root, but we
        # still clear PYTHONPATH so the agent app's regular `src` package can't shadow the repo's
        # namespace `src` (mirrors the DETR-family convention). CPU-safe export.
        cmd = [venv_python(), "tools/deployment/export_onnx.py", "-c", cfg_path, "-r", weights]
        rc, tail = run_stream(cmd, repo(self.REPO),
                              env={"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": ""})
        produced = os.path.splitext(weights)[0] + ".onnx"
        if not os.path.exists(produced):
            raise RuntimeError(f"D-FINE ONNX export failed (rc={rc}):\n{tail}")
        if os.path.abspath(produced) != os.path.abspath(out_path):
            shutil.move(produced, out_path)
        return out_path

    # inference helper run in the model env. Mirrors tools/inference/torch_inf.py's model build
    # (YAMLConfig -> model.deploy() + postprocessor.deploy(), forward(images, orig_target_sizes)
    # returns labels, boxes(xyxy ABS px), scores), then normalizes to canonical 0..1 cxcywh and
    # emits one OBJ <json> line per kept detection. PYTHONPATH is cleared by the caller so the
    # repo's namespace `src` isn't shadowed by the agent app's `src` package; we also
    # sys.path.insert(0) the repo root like torch_inf.py does.
    _PREDICT_HELPER = r'''
import os, sys, json
REPO = sys.argv[1]; CFG = sys.argv[2]; WEIGHTS = sys.argv[3]; IMG = sys.argv[4]
CONF = float(sys.argv[5]); META = sys.argv[6] if len(sys.argv) > 6 else ""
sys.path.insert(0, REPO)
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from src.core import YAMLConfig

# ----- category names: meta json (preferred) -> COCO ann_file the config points at -> numeric
cats = []
if META and os.path.exists(META):
    try:
        cats = list(json.load(open(META, encoding="utf-8")).get("categories") or [])
    except Exception:
        cats = []

cfg = YAMLConfig(CFG, resume=WEIGHTS)
if "HGNetv2" in cfg.yaml_cfg:
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

if not cats:
    try:
        ann = (cfg.yaml_cfg.get("val_dataloader", {}) or {}).get("dataset", {}).get("ann_file")
        if ann and os.path.exists(ann):
            coco = json.load(open(ann, encoding="utf-8"))
            by_id = {c["id"]: c["name"] for c in coco.get("categories", [])}
            if by_id:
                cats = [by_id.get(i, str(i)) for i in range(max(by_id) + 1)]
    except Exception:
        cats = []

ckpt = torch.load(WEIGHTS, map_location="cpu")
state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
cfg.model.load_state_dict(state)

class _M(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
    def forward(self, images, orig_target_sizes):
        return self.postprocessor(self.model(images), orig_target_sizes)

device = "cuda" if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "" else "cpu"
net = _M().to(device).eval()

im = Image.open(IMG).convert("RGB")
W, H = im.size
orig_size = torch.tensor([[W, H]]).to(device)
tf = T.Compose([T.Resize((640, 640)), T.ToTensor()])
x = tf(im).unsqueeze(0).to(device)
with torch.no_grad():
    labels, boxes, scores = net(x, orig_size)
# batch of 1: take row 0. boxes are xyxy in ABSOLUTE pixels of the ORIGINAL image.
labels = labels[0].cpu().tolist()
boxes = boxes[0].cpu().tolist()
scores = scores[0].cpu().tolist()
for lab, (x1, y1, x2, y2), sc in zip(labels, boxes, scores):
    if sc < CONF:
        continue
    lab = int(lab)
    name = cats[lab] if 0 <= lab < len(cats) else str(lab)
    # clamp to image, normalize to 0..1 cxcywh
    x1 = max(0.0, min(float(x1), W)); x2 = max(0.0, min(float(x2), W))
    y1 = max(0.0, min(float(y1), H)); y2 = max(0.0, min(float(y2), H))
    if x2 <= x1 or y2 <= y1:
        continue
    cx = ((x1 + x2) / 2.0) / W; cy = ((y1 + y2) / 2.0) / H
    bw = (x2 - x1) / W;        bh = (y2 - y1) / H
    print("OBJ " + json.dumps({
        "category": name, "score": float(sc),
        "bbox": {"x_center": cx, "y_center": cy, "width": bw, "height": bh},
    }))
'''

    def predict(self, weights: str, image_path: str, conf: float = 0.3):
        """Run single-image D-FINE detection in the model env; return canonical objects.

        Subprocesses _PREDICT_HELPER in the model venv: it builds the model from the stashed
        _dfine_config.yml + weights (same as tools/inference/torch_inf.py), runs model+postprocessor
        (which returns labels, ABSOLUTE-pixel xyxy boxes, scores), filters by `conf`, maps the
        0-indexed label to a category name (from _dfine_meta.json, else the COCO ann_file the config
        points at), normalizes the box to 0..1 cxcywh, and prints one ``OBJ <json>`` line per
        detection. Returns the parsed list of canonical detection objects.
        """
        cfg_path = os.path.join(os.path.dirname(weights), "_dfine_config.yml")
        if not os.path.exists(cfg_path):
            raise RuntimeError(f"D-FINE predict needs the training config; not found at {cfg_path}")
        meta_path = os.path.join(os.path.dirname(weights), "_dfine_meta.json")
        repo_dir = repo(self.REPO)
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(self._PREDICT_HELPER)
            hp = f.name
        objs = []

        def on_line(line):
            if line.startswith("OBJ "):
                try:
                    objs.append(json.loads(line[4:]))
                except Exception:
                    pass

        try:
            # CPU-safe predict (CUDA_VISIBLE_DEVICES=""); the helper picks cpu accordingly.
            # PYTHONPATH="" so the agent app's `src` doesn't shadow the repo's namespace `src`
            # (the helper sys.path.insert(0)s the repo root for its own `src.core`).
            rc, tail = run_stream(
                [venv_python(), hp, repo_dir, cfg_path, weights, image_path, str(conf), meta_path],
                repo_dir, on_line=on_line,
                env={"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": ""},
            )
            if rc != 0:
                raise RuntimeError(f"D-FINE predict failed (rc={rc}):\n{tail}")
            return objs
        finally:
            os.unlink(hp)


@register("detection")
class DFineDetector(_DFineFamily):
    task = "detection"
    family = "dfine"
    label = "D-FINE"
    REPO = "D-FINE"
