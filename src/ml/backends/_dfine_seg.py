"""D-FINE-seg instance-segmentation backend (Apache-2.0 code; ArgoHA/D-FINE-seg).

PROVEN end-to-end on the server (val mAP_50 0.98 / mAP_50_mask 0.97 on the hands set),
this mirrors the recipe verified in ``~/D-FINE-seg-run/output/models/hands_n_2026-06-10``.

The seg repo is a SEPARATE checkout (``$MODEL_INTEGRATION_ROOT/D-FINE-seg``) that differs
from the D-FINE detection repo: it is uv/pyproject + a Hydra-driven top-level ``config.yaml``
(``@hydra.main(config_path="../../", config_name="config")`` in ``src/dl/train.py``) and a
YOLO-seg dataset layout (``images/`` + ``labels/<stem>.txt`` + ``train.csv``/``val.csv``),
NOT the COCO-json layout the D-FINE detector uses. So this backend:

  * prepare_dataset  -> writes the YOLO-seg layout from universal DB samples (polygons are
                        already 0..1 in the DB, exactly the ``<cls> x1 y1 x2 y2 ...`` line the
                        repo's ``parse_yolo_label_file`` expects);
  * train            -> writes a full ``config.yaml`` (the proven keys) into a per-project gen
                        dir, points Hydra at it with ``--config-path/--config-name``, and runs
                        ``uv run python -m src.dl.train`` in the repo, parsing the
                        ``Metrics on epoch N:`` + ``| val | mAP_50 | mAP_50_mask | ... |`` table
                        rows for progress; best checkpoint is ``model.pt`` (last is ``last.pt``).

ENV NOTE: the seg repo manages its OWN uv environment (Python 3.11, torch 2.9, hydra 1.3.2 —
all pinned in ``uv.lock``); the shared ``$MODEL_INTEGRATION_ROOT/.venv`` does NOT have hydra.
We therefore launch via ``uv run`` (``~/.local/bin/uv``) so the repo's locked env is used.
A one-time ``uv sync`` is required before the first train (see the verify commands).

LICENSE NOTE: training initialises from ``pretrained/dfine_<size>_coco.pt`` — the COCO-pretrained
D-FINE *detection* backbone (Apache-2.0, auto-downloaded from HF ``ArgoSA/D-FINE-seg`` and loaded
non-strictly so the seg head trains FROM SCRATCH on our data). This is NOT the released D-FINE-seg
segmentation checkpoint (which has no stated license). We never fine-tune that. If the COCO init
weight is unavailable, set ``imagenet_backbone: True`` to train the whole net from scratch.

export_onnx: shells the repo's ``src.dl.export`` (Hydra) at the stashed config with
``export.formats=[onnx]`` so it loads ``<dir>/model.pt`` and writes ``<dir>/model.onnx``
(the ExportWrapper graph: fused box postprocessor, masks passed through with sigmoid),
which we move to ``out_path``.

predict: shells a small helper in the repo's uv env that rebuilds the model from the bare
state_dict (``Torch_model`` -> ``build_model`` + ``load_state_dict(strict=False)``), runs the
proven postprocess (NMS, masks binarized + resized to original size), and emits one canonical
object per detection — normalized cxcywh bbox + normalized 0..1 ``polygon`` (via
``Torch_model.mask2poly``) — as ``OBJ <json>`` lines we parse back.
"""
import os
import re
import shutil

from .base import ModelBackend, TrainConfig, TrainResult, register
from ._env import venv_python, repo, run_stream, resolve_device, find_weight


# The repo logs metrics as two lines: a header `Metrics on epoch N:` followed (a few lines
# later, after the tabulate "pretty" frame) by a `| val | ... |` row. We track the most recent
# epoch header, then read the mAPs off the next `| val |` row.
_RE_EPOCH = re.compile(r"Metrics on epoch\s+(\d+)\s*:")
# REAL column order (verified from the trainer's live output):
#   | val | mAP_50 | f1 | precision | recall | iou | mAP_50_95 | TPs | FPs | FNs |
# So mAP_50 is column 1 and mAP_50_95 is column 6 — NOT column 2 (column 2 is f1, which is why
# the old `\| val \| c1 \| c2 \|` regex registered mask_map=f1=0.0 for every seg model). Capture
# column 1 (group1 = mAP_50) and column 6 (group2 = mAP_50_95).
_RE_VALROW = re.compile(
    r"\|\s*val\s*\|\s*([\d.]+)\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|")


@register("segmentation")
class DFineSegmentor(ModelBackend):
    task = "segmentation"
    family = "dfine_seg"
    label = "D-FINE-Seg"
    REPO = "D-FINE-seg"
    sizes = ("n", "s", "m", "l", "x")
    default_size = "n"
    # best checkpoint is model.pt; last.pt is the rolling/last-epoch save.
    weights_glob = ("model.pt", "best.pt", "last.pt")

    # -- recommended LRs per size, copied verbatim from the proven config (train.lrs.<size>) --
    _LRS = {
        "n": {"backbone_lr": 0.0004, "base_lr": 0.0008},
        "s": {"backbone_lr": 0.00006, "base_lr": 0.00025},
        "m": {"backbone_lr": 0.00002, "base_lr": 0.00015},
        "l": {"backbone_lr": 0.00001, "base_lr": 0.00016},
        "x": {"backbone_lr": 0.000002, "base_lr": 0.0002},
    }

    # ------------------------------------------------------------------ dataset --
    def prepare_dataset(self, samples, categories, out_dir, val_ratio=0.2, seed=0,
                        copy_images=True, **kw) -> dict:
        """Write the repo's YOLO-seg layout from universal DB samples.

        Produces::

            out_dir/images/<file_name>                 (the image files; symlink if not copying)
            out_dir/labels/<stem>.txt                  ("<cls> x1 y1 x2 y2 ..." normalized polys)
            out_dir/train.csv  out_dir/val.csv          (image filenames, no header)

        DB polygons are already normalized (0..1), matching parse_yolo_label_file. Objects
        without a >=3-point polygon are skipped (segmentation needs a polygon). Category ids
        are 0-indexed and contiguous (label_to_name in the config must match).
        """
        import random
        cat_id = {c: i for i, c in enumerate(categories)}
        samples = list(samples)
        rng = random.Random(seed)
        rng.shuffle(samples)
        nval = max(1, int(len(samples) * val_ratio)) if len(samples) > 1 else 0
        splits = {"val": samples[:nval], "train": samples[nval:]}

        img_dir = os.path.join(out_dir, "images")
        lbl_dir = os.path.join(out_dir, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        result = {"out_dir": out_dir, "splits": {}, "num_classes": len(categories),
                  "categories": list(categories)}
        for split, ss in splits.items():
            names, n_obj = [], 0
            for s in ss:
                fname = s["file_name"]
                names.append(fname)
                # image
                dst = os.path.join(img_dir, fname)
                src = s.get("src")
                if src and os.path.abspath(src) != os.path.abspath(dst) and not os.path.exists(dst):
                    if copy_images:
                        shutil.copy(src, dst)
                    else:
                        os.symlink(os.path.abspath(src), dst)
                # label
                stem = os.path.splitext(fname)[0]
                lines = []
                for o in s.get("objects", []):
                    c = o.get("category")
                    if c not in cat_id:
                        continue
                    poly = o.get("polygon")
                    if not poly or len(poly) < 6:
                        continue  # segmentation requires a polygon (>=3 points)
                    coords = " ".join(f"{float(v):.6f}" for v in poly)
                    lines.append(f"{cat_id[c]} {coords}")
                    n_obj += 1
                with open(os.path.join(lbl_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                    if lines:
                        f.write("\n")
            csv_path = os.path.join(out_dir, f"{split}.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("\n".join(names))
                if names:
                    f.write("\n")
            result["splits"][split] = {"csv": csv_path, "images": len(names), "annotations": n_obj}
        return result

    # ------------------------------------------------------------------- config --
    def _label_to_name(self, categories) -> str:
        return "\n".join(f"    {i}: {c}" for i, c in enumerate(categories)) or "    0: object"

    def _write_config(self, cfg: TrainConfig, size: str) -> tuple:
        """Write a full Hydra config.yaml mirroring the proven run; return (gen_dir, name, out_dir).

        ``exp`` is pinned to a literal (no ${now} date suffix) so ``path_to_save`` == out_dir
        is deterministic and the checkpoint/log land where we look for them.
        """
        data_path = os.path.abspath(cfg.dataset_dir)
        out_dir = os.path.abspath(cfg.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        gen_dir = os.path.join(out_dir, "_gen")
        os.makedirs(gen_dir, exist_ok=True)
        name = "config"  # Hydra --config-name (no .yaml)

        exp = f"proj{cfg.project_id}_{self.task}"          # literal -> path_to_save deterministic
        root = out_dir                                     # keep all repo outputs under out_dir
        lr = self._LRS.get(size, self._LRS[self.default_size])
        img = int(cfg.img_size)
        # respect an explicit init weight, else the COCO-pretrained D-FINE backbone (auto-DL).
        pretrained = cfg.weights_init or f"pretrained/dfine_{size}_coco.pt"
        dev = resolve_device(cfg.device)
        amp = "True" if dev == "cuda" else "False"
        label_to_name = self._label_to_name(cfg.categories)
        # we already split into train.csv/val.csv ourselves; the repo only re-splits via `make split`
        h, w = img, img

        body = f"""project_name: momoto
exp_name: {exp}
exp: {exp}

model_name: '{size}'
task: segment

train:
  root: {root}
  pretrained_dataset: coco
  pretrained_model_path: {pretrained}
  imagenet_backbone: False
  coco_dataset: False

  data_path: {data_path}
  path_to_test_data: {data_path}/test
  path_to_save: {out_dir}
  debug_img_path: {out_dir}/debug_images
  eval_preds_path: {out_dir}/eval_preds
  bench_img_path: {out_dir}/bench_imgs
  infer_path: {out_dir}/infer

  use_wandb: False
  device: {dev}
  label_to_name:
{label_to_name}
  use_one_class: False

  ddp:
    enabled: False
    n_gpus: 2

  decision_metrics:
  - f1
  - mAP_50
  - iou

  img_size: [{h}, {w}]
  in_channels: 3
  keep_ratio: False
  to_visualize_eval: False
  debug_img_processing: False

  amp_enabled: {amp}
  clip_max_norm: 0.1

  batch_size: {int(cfg.batch)}
  b_accum_steps: 1
  epochs: {int(cfg.epochs)}
  max_walltime_min: null
  early_stopping: 0
  ignore_background_epochs: 0
  num_workers: 4
  mask_batch_size: 150

  conf_thresh: 0.5
  iou_thresh: 0.5

  use_ema: True
  ema_momentum: 0.9998

  use_scheduler: True
  base_lr: {lr['base_lr']}
  backbone_lr: {lr['backbone_lr']}
  cycler_pct_start: 0.1
  weight_decay: 0.000125
  betas: [0.9, 0.999]
  label_smoothing: 0.0

  mosaic_augs:
    mosaic_prob: 0.0
    no_mosaic_epochs: 5
    mosaic_scale: [0.5, 1.5]
    degrees: 0.0
    translate: 0.2
    shear: 2.0

  augs:
    rotation_degree: 10
    rotation_p: 0.05
    multiscale_prob: 0.0
    rotate_90: 0.05
    left_right_flip: 0.3
    up_down_flip: 0.0
    to_gray: 0.01
    blur: 0.01
    gamma: 0.02
    brightness: 0.02
    noise: 0.01
    coarse_dropout: 0.0

  seed: {int(getattr(cfg, 'extra', {}).get('seed', 42)) if isinstance(getattr(cfg, 'extra', {}), dict) else 42}
  cudnn_fixed: False

  lrs:
    'n': {{backbone_lr: 0.0004, base_lr: 0.0008}}
    s: {{backbone_lr: 0.00006, base_lr: 0.00025}}
    m: {{backbone_lr: 0.00002, base_lr: 0.00015}}
    l: {{backbone_lr: 0.00001, base_lr: 0.00016}}
    x: {{backbone_lr: 0.000002, base_lr: 0.0002}}

split:
  ignore_negatives: False
  shuffle: True
  train_split: 0.85
  val_split: 0.15

export:
  from_pretrained: False
  half: True
  max_batch_size: 1
  dynamic_input: False
  formats: null
  ov_int8_max_drop: 0.02
  trt_int8_workspace_gb: 4
  trt_int8_validate: True

bench:
  formats: [torch, onnx]

infer:
  to_crop: True
  to_track: False
  paddings:
    w: 0.05
    h: 0.05

defaults:
  - _self_
  - override hydra/hydra_logging: disabled
  - override hydra/job_logging: disabled

hydra:
  output_subdir: null
  run:
    dir: .

now_dir: ${{now:%Y-%m-%d}}
"""
        cfg_path = os.path.join(gen_dir, f"{name}.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(body)
        return gen_dir, name, out_dir

    # -------------------------------------------------------------------- launch --
    def _python_cmd(self) -> list:
        """Launch python for the seg repo's OWN uv env (the shared .venv lacks hydra).

        Prefer ``uv run`` (the repo is uv/pyproject + uv.lock). Fall back to a repo-local
        ``.venv`` python, then to the shared model env python as a last resort.
        """
        repo_dir = repo(self.REPO)
        repo_py = os.path.join(repo_dir, ".venv", "bin", "python")
        if os.path.exists(repo_py):
            return [repo_py]
        for uv in (os.environ.get("UV_BIN"),
                   os.path.expanduser("~/.local/bin/uv"),
                   os.path.expanduser("~/.cargo/bin/uv"),
                   shutil.which("uv")):
            if uv and os.path.exists(uv):
                return [uv, "run", "--project", repo_dir, "python"]
        return [venv_python()]

    # --------------------------------------------------------------------- train --
    def train(self, cfg: TrainConfig, progress_cb=None, should_stop=None) -> TrainResult:
        size = cfg.size if cfg.size in self.sizes else self.default_size
        gen_dir, name, out_dir = self._write_config(cfg, size)
        # stash the resolved config next to the checkpoints for export/predict reuse + debugging
        shutil.copy(os.path.join(gen_dir, f"{name}.yaml"),
                    os.path.join(out_dir, "_dfine_seg_config.yaml"))

        cmd = self._python_cmd() + [
            "-m", "src.dl.train",
            "--config-path", gen_dir,   # Hydra: override the @hydra.main config_path/name
            "--config-name", name,
        ]
        state = {"epoch": 0, "map50": 0.0, "mask_map": 0.0}

        def on_line(line):
            me = _RE_EPOCH.search(line)
            if me:
                state["epoch"] = int(me.group(1))
            mv = _RE_VALROW.search(line)
            if mv:
                # task==segment => decision metric is mask; table is `| val | mAP_50 | mAP_50_mask | ...`
                state["map50"] = float(mv.group(1))
                state["mask_map"] = float(mv.group(2))
                if progress_cb:
                    progress_cb({"epoch": state["epoch"], "total_epochs": int(cfg.epochs),
                                 "metrics": {"map": state["mask_map"], "map50": state["map50"],
                                             "mask_map": state["mask_map"]}})
            if should_stop and should_stop():
                # the trainer catches KeyboardInterrupt and still evaluates/saves the best model.pt
                raise KeyboardInterrupt("training stop requested")

        # Clear PYTHONPATH so the agent app's regular `src` package can't shadow the repo's own
        # `src.dl` (same collision the DETRPose backend hits); the repo finds its src via cwd.
        env = {"HYDRA_FULL_ERROR": "1", "PYTHONPATH": ""}
        if resolve_device(cfg.device) != "cuda":
            env["CUDA_VISIBLE_DEVICES"] = ""
        # A cooperative stop propagates as KeyboardInterrupt out of run_stream (it SIGINTs the
        # child), so it never reaches here. On a normal return we decide on rc + what was produced.
        rc, tail = run_stream(cmd, repo(self.REPO), on_line=on_line, env=env)
        weights = find_weight(out_dir, self.weights_glob)
        ran_eval = (state["mask_map"] > 0) or (state["map50"] > 0)
        # Fail when the run errored AND we don't have a usable, validated checkpoint. We tolerate a
        # non-zero exit only when training reached at least one validation (real metrics) AND saved
        # a checkpoint — i.e. a benign late-stage error (post-save eval/teardown) — rather than
        # throwing away a full GPU run. A mid-training crash (no eval yet → metrics 0) is a genuine
        # failure and must NOT be reported as a (zero-metric) success. (out_dir is fresh per agent
        # job, so any checkpoint here is from THIS run, not a stale leftover.)
        if rc != 0 and not (weights and ran_eval):
            raise RuntimeError(f"D-FINE-seg training failed (rc={rc}):\n{tail}")
        if not weights:
            raise RuntimeError(
                f"D-FINE-seg training produced no checkpoint in {out_dir} (rc={rc}).\n"
                f"Last log lines:\n{tail}")
        if rc != 0:
            print(f"[D-FINE-seg] exited rc={rc} but a validated checkpoint exists "
                  f"(mask_map={state['mask_map']}) — returning it.", flush=True)
        return TrainResult(
            weights=weights,
            metrics={"map": state["mask_map"], "map50": state["map50"],
                     "mask_map": state["mask_map"]},
            classes=list(cfg.categories),
            fmt=self.family,
        )

    # ------------------------------------------------------------ stashed config --
    def _stashed_config(self, weights: str) -> str:
        """Locate the config.yaml stashed next to the checkpoint by train().

        train() copies the generated Hydra config to ``<out_dir>/_dfine_seg_config.yaml``
        (literal paths/lrs, standalone-resolvable). predict()/export receive only a
        ``weights`` path, so we rediscover the config in the checkpoint's directory.
        Falls back to the repo-saved ``config.yaml`` if our stash is missing.
        """
        d = os.path.dirname(os.path.abspath(weights))
        for name in ("_dfine_seg_config.yaml", "config.yaml"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(
            f"No D-FINE-seg config.yaml next to {weights} (looked for "
            f"_dfine_seg_config.yaml / config.yaml in {d}); was this trained by this backend?")

    # -------------------------------------------------------------------- export --
    def export_onnx(self, weights: str, out_path: str, **kw) -> str:
        """Export the trained model.pt to ONNX via the repo's ``src.dl.export``.

        ``src.dl.export`` is Hydra-driven (``@hydra.main(config_path="../../",
        config_name="config")``). We point Hydra at the stashed config (literal
        ``path_to_save`` == the checkpoint dir) so ``prepare_model`` loads
        ``<dir>/model.pt`` (export.from_pretrained=False) and writes the ExportWrapper
        graph (fused box postprocessor; masks passed through with sigmoid) to
        ``<dir>/model.onnx`` (``model_path.with_suffix('.onnx')``). We override
        ``export.formats=[onnx]`` so only ONNX is built (skip TRT/OpenVINO/CoreML/LiteRT)
        and ``export.half=False`` to keep an fp32 graph (portable; avoids the float16
        IO-type conversion). The produced .onnx is then moved to ``out_path``.
        """
        weights = os.path.abspath(weights)
        ckpt_dir = os.path.dirname(weights)
        cfg_path = self._stashed_config(weights)
        cfg_dir = os.path.dirname(cfg_path)
        cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]  # Hydra: no .yaml

        # The repo always exports the dir's `model.pt` (prepare_model reads
        # Path(path_to_save)/"model.pt"); make sure that's the checkpoint we were given.
        model_pt = os.path.join(ckpt_dir, "model.pt")
        if os.path.basename(weights) != "model.pt" and not os.path.exists(model_pt):
            shutil.copy(weights, model_pt)

        dev = "cuda" if resolve_device(kw.get("device", "auto")) == "cuda" else "cpu"
        cmd = self._python_cmd() + [
            "-m", "src.dl.export",
            "--config-path", cfg_dir,
            "--config-name", cfg_name,
            # Hydra dotlist overrides: only build ONNX, fp32, force the export device.
            "export.formats=[onnx]",
            "export.from_pretrained=False",
            "export.half=False",
            f"train.device={dev}",
            # The stashed config's literal train.path_to_save is the (now-deleted) TRAINING
            # tmpdir; export runs as a SEPARATE job in a fresh ckpt_dir. Repoint it so
            # get_latest_experiment_name sees an existing dir (returns exp, skips the
            # parent.iterdir() that FileNotFound'd) and prepare_model reads ckpt_dir/model.pt.
            f"train.path_to_save={ckpt_dir}",
        ]
        env = {"HYDRA_FULL_ERROR": "1", "PYTHONPATH": ""}
        if dev != "cuda":
            env["CUDA_VISIBLE_DEVICES"] = ""
        rc, tail = run_stream(cmd, repo(self.REPO), env=env)

        produced = os.path.join(ckpt_dir, "model.onnx")
        if not os.path.exists(produced):
            raise RuntimeError(
                f"D-FINE-seg ONNX export produced no model.onnx in {ckpt_dir} (rc={rc}).\n"
                f"Last log lines:\n{tail}")
        if os.path.abspath(produced) != os.path.abspath(out_path):
            os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
            shutil.move(produced, out_path)
        return out_path

    # ------------------------------------------------------------------- predict --
    # Helper run in the seg repo's uv env: builds the model from the bare state_dict
    # (build_model + load_state_dict strict=False, exactly like Torch_model._load_model),
    # runs the proven Torch_model postprocess (NMS + masks->binarized->original-size),
    # converts boxes (abs xyxy @ orig size) -> normalized cxcywh and masks -> normalized
    # polygons via Torch_model.mask2poly, and prints one OBJ <json> line per object.
    _PREDICT_HELPER = (
        "import os, sys, json\n"
        # run_stream launches this as a script file (sys.path[0] is the helper's temp dir,
        # NOT the repo). Prepend cwd (= the repo root we pass to run_stream) so the repo's
        # own `src` package resolves — same trick as launching `-m src.dl.*` from cwd.
        "sys.path.insert(0, os.getcwd())\n"
        "import yaml, cv2, numpy as np\n"
        "from src.infer.torch_model import Torch_model\n"
        "cfg_path, weights, image_path, conf = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])\n"
        "cfg = yaml.safe_load(open(cfg_path))\n"
        "tr = cfg['train']\n"
        "l2n = tr['label_to_name'] or {}\n"
        "names = {int(k): v for k, v in l2n.items()}\n"
        "ih, iw = (int(tr['img_size'][0]), int(tr['img_size'][1])) if tr.get('img_size') else (640, 640)\n"
        "tm = Torch_model(\n"
        "    model_name=str(cfg['model_name']),\n"
        "    model_path=weights,\n"
        "    n_outputs=len(names) or 1,\n"
        "    input_width=iw, input_height=ih,\n"
        "    conf_thresh=conf,\n"
        "    rect=False, keep_ratio=bool(tr.get('keep_ratio', False)),\n"
        "    enable_mask_head=(cfg.get('task') == 'segment'),\n"
        "    binarize_masks=True, mask_threshold=0.5,\n"
        "    device='cpu', channels=int(tr.get('in_channels', 3)),\n"
        ")\n"
        "img = cv2.imread(image_path)\n"
        "if img is None:\n"
        "    raise SystemExit('PREDICT_ERR could not read image: ' + image_path)\n"
        "H, W = img.shape[0], img.shape[1]\n"
        "res = tm(img, bgr=True)[0]\n"
        "boxes = res['boxes'].cpu().numpy()\n"
        "labels = res['labels'].cpu().numpy()\n"
        "scores = res['scores'].cpu().numpy()\n"
        "polys = None\n"
        "if 'masks' in res and res['masks'] is not None and len(res['masks']):\n"
        "    polys = tm.mask2poly(res['masks'].cpu().numpy(), img.shape)\n"
        "for i in range(len(boxes)):\n"
        "    x1, y1, x2, y2 = [float(v) for v in boxes[i]]\n"
        "    bw, bh = (x2 - x1), (y2 - y1)\n"
        "    obj = {\n"
        "        'category': names.get(int(labels[i]), str(int(labels[i]))),\n"
        "        'score': float(scores[i]),\n"
        "        'bbox': {\n"
        "            'x_center': (x1 + bw / 2) / W, 'y_center': (y1 + bh / 2) / H,\n"
        "            'width': bw / W, 'height': bh / H,\n"
        "        },\n"
        "    }\n"
        "    if polys is not None:\n"
        "        p = polys[i]\n"
        "        if p is not None and len(p) >= 3:\n"
        "            flat = []\n"
        "            for pt in np.asarray(p).reshape(-1, 2):\n"
        "                flat.append(float(pt[0])); flat.append(float(pt[1]))\n"
        "            obj['polygon'] = flat\n"
        "    print('OBJ ' + json.dumps(obj))\n"
    )

    def predict(self, weights: str, image_path: str, conf: float = 0.3):
        """Run the trained D-FINE-seg model on one image; return canonical objects.

        Mirrors the ConvNeXtV2 pattern: shells a small helper in the seg repo's uv env
        (which has hydra/torch 2.9) via run_stream, with PYTHONPATH="" so the repo's own
        ``src`` package isn't shadowed by the agent app's ``src`` and CUDA_VISIBLE_DEVICES=""
        for CPU-safe predict. Each detection becomes a canonical object with a normalized
        cxcywh bbox plus (for segmentation) a normalized 0..1 ``polygon`` flat list.
        """
        import json
        import tempfile
        cfg_path = self._stashed_config(weights)
        objs = []

        def on_line(l):
            if l.startswith("OBJ "):
                try:
                    objs.append(json.loads(l[4:]))
                except Exception:
                    pass

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(self._PREDICT_HELPER)
            hp = f.name
        try:
            cmd = self._python_cmd() + [hp, cfg_path, os.path.abspath(weights),
                                        os.path.abspath(image_path), str(conf)]
            env = {"PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": ""}
            rc, tail = run_stream(cmd, repo(self.REPO), on_line=on_line, env=env)
            if rc != 0 and not objs:
                raise RuntimeError(f"D-FINE-seg predict failed (rc={rc}):\n{tail}")
            return objs
        finally:
            os.unlink(hp)
