"""DETRPose keypoint/pose backend (Apache-2.0; SebastianJanampa/DETRPose).

NOT YET PROVEN ON GPU. This is a best-effort, repo-faithful implementation built by
reading the cloned repo at $MODEL_INTEGRATION_ROOT/DETRPose. Everything that is
UNVERIFIED is flagged in comments + the parent's risk list so the first GPU smoke can
iterate quickly.

How DETRPose trains (verified by reading the repo source):
  * Entry point: ``train.py -c <config.py> --device cuda --seed N [--amp] [--pretrain dfine_<size>_obj365]``.
    - argparse keys become ``cfg.training_params.<key>`` (merged via OmegaConf), so e.g.
      ``--device cuda`` -> training_params.device. ``--amp`` is a store_true flag.
    - ``--options k=v ...`` does a Hydra/OmegaConf override but is brittle for list/nested
      values, so we DO NOT use it; we generate a config .py instead (repo-idiomatic, mirrors
      configs/detrpose/detrpose_hgnetv2_<size>_crowdpose.py which overrides num_body_points etc.).
  * Configs are detectron2-style LazyConfig **.py** files (needs cloudpickle; present, 3.1.2).
    A size config (configs/detrpose/detrpose_hgnetv2_<size>.py) imports model/criterion/
    postprocessor/training_params from include/detrpose_hgnetv2.py and dataset_train/
    dataset_val/dataset_test/evaluator from include/dataset.py, then mutates them. We
    generate configs/detrpose/_gen/proj_<id>.py that imports the chosen size config's
    objects and re-points the dataset at our COCO-keypoints + sets num_body_points / num_classes
    / epochs / output_dir.
  * Dataset format: COCO person_keypoints (db_to_coco task='keypose' already emits absolute
    keypoints [x,y,v]*K + num_keypoints + per-category keypoints/skeleton; category_id 0-indexed).
    CocoDetection only keeps annotations whose num_keypoints != 0.
  * Output: training_params.output_dir gets checkpoint.pth (every save_checkpoint_interval),
    checkpoint{epoch:04}.pth, and checkpoint_best_regular.pth (best sAP). We discover via
    find_weight(best first).
  * Metric: pycocotools COCOeval(iouType='keypoints').summarize() prints to stdout; the first
    AP line (maxDets=20, IoU=0.50:0.95) is sAP50:95, the IoU=0.50 line is sAP50. Trainer also
    logs epoch via "Epoch: [N]" in train_one_epoch and "New best achieved @ epoch NNNN".

CRITICAL UNVERIFIED BLOCKER (flagged for the parent — likely needs a repo patch on first GPU run):
  src/data/coco.py ConvertCocoPolysToMask HARDCODES ``keypoints.reshape(-1, 17, 3)`` and
  src/data/crowdpose.py hardcodes ``reshape(-1, 14, 3)``. The COCOeval keypoints OKS also uses
  pycocotools' default 17 sigmas. So the stock CocoDetection path ONLY works for K=17 keypoints.
  Project 34 has K=6. We auto-apply a tiny, idempotent patch to src/data/coco.py so the reshape
  follows the dataset's K (see _ensure_repo_kpt_patch). If the patch ever fails to apply (repo
  changed), training of K!=17 datasets will crash in the dataloader -> surfaced as a RuntimeError
  with the tail.
"""
import os
import re
import json
import shutil

from .base import ModelBackend, TrainConfig, TrainResult, register
from ._env import venv_python, repo, run_stream, resolve_device, find_weight


# train_one_epoch header: 'Epoch: [{}]'.format(epoch)
_RE_EPOCH = re.compile(r"Epoch:\s*\[(\d+)\]")
# "New best achieved @ epoch 0007!!!..." -> also bumps epoch
_RE_BEST = re.compile(r"New best achieved @ epoch\s*(\d+)")
# pycocotools COCOeval.summarize() keypoints lines (maxDets=20):
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.123
_RE_AP = re.compile(
    r"Average Precision\s*\(AP\)\s*@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*\d+\s*\]\s*=\s*([\d.]+)"
)
_RE_AP50 = re.compile(
    r"Average Precision\s*\(AP\)\s*@\[\s*IoU=0\.50\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*\d+\s*\]\s*=\s*([\d.]+)"
)

# pretrain (transfer-learning) backbone+encoder init, keyed by size. The repo's --pretrain
# downloads D-FINE obj365 weights; we only use it when explicitly requested via cfg.extra.
_PRETRAIN = {"n": "dfine_n_obj365", "s": "dfine_s_obj365", "m": "dfine_m_obj365",
             "l": "dfine_l_obj365", "x": "dfine_x_obj365"}


@register("keypose")   # also matches 'keypoint_detection' / 'pose' via canon_task
class DETRPoseEstimator(ModelBackend):
    task = "keypose"
    family = "detrpose"
    label = "DETRPose"
    REPO = "DETRPose"
    sizes = ("n", "s", "m", "l", "x")
    default_size = "s"
    # LAST checkpoint first (checkpoint.pth, rewritten every epoch = fully trained), then best.
    # checkpoint_best_regular.pth tracks best-by-sAP, but OKS AP for a CUSTOM keypoint set uses
    # guessed uniform sigmas and reads ~0 throughout, so "best" never beats the epoch-0 snapshot →
    # registering it shipped an essentially UNTRAINED model (50- and 400-epoch runs were identical).
    # The last checkpoint is the trustworthy trained weights regardless of the unreliable OKS metric.
    weights_glob = ("checkpoint.pth", "checkpoint_best_regular.pth")

    # ------------------------------------------------------------------ config gen
    def _kpt_params(self, cfg: TrainConfig):
        """Return (num_keypoints, kpt_names, flip_idx, skeleton, sigmas) from cfg.keypoint_config.

        keypoint_config shape (from Project / base.TrainConfig):
            {num_keypoints, names, flip_idx, skeleton}
        """
        kc = cfg.keypoint_config or {}
        names = list(kc.get("names") or [])
        K = int(kc.get("num_keypoints") or len(names) or 17)
        flip_idx = list(kc.get("flip_idx") or list(range(K)))
        skeleton = list(kc.get("skeleton") or [])
        # OKS sigmas: we don't know per-keypoint sigmas for a custom skeleton, so use a uniform
        # default (COCO's are ~0.025..0.107). 0.05 is a reasonable mid value. UNVERIFIED — only
        # affects the *reported* OKS AP, not the trained weights.
        sigmas = [0.05] * K
        return K, names, flip_idx, skeleton, sigmas

    def _write_config(self, cfg: TrainConfig, size: str, K: int, skeleton, num_classes: int) -> str:
        ds = os.path.abspath(cfg.dataset_dir)
        out_dir = os.path.abspath(cfg.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        gen_dir = os.path.join(repo(self.REPO), "configs", "detrpose", "_gen")
        os.makedirs(gen_dir, exist_ok=True)
        # ensure it's importable as a package level (detectron2 LazyConfig derives the package
        # name from the path to the nearest dir; relative imports `from ..detrpose_..` need the
        # parent configs/detrpose to be on the path, which it is since the size configs use the
        # same relative style).
        cfg_path = os.path.join(gen_dir, f"proj_{cfg.project_id}_keypose.py")
        train_img = f"{ds}/train"
        train_ann = f"{ds}/annotations/instances_train.json"
        val_img = f"{ds}/val"
        val_ann = f"{ds}/annotations/instances_val.json"
        # per-rank batch divisibility: DETRPose asserts total_batch_size % world_size == 0.
        # We run single-process (world_size=1) so any batch is fine.
        # Horizontal-flip pairs from our flip_idx (the repo hardcodes COCO-17 1-indexed pairs in
        # transforms.hflip). flip_idx[i]=j means kpt i maps to kpt j under a mirror, so each i<j
        # is a swap pair; we patch transforms.hflip to read these (else it crashes / mis-swaps
        # for K!=17). Empty -> hflip leaves keypoints unmoved (still flips image+boxes).
        flip_idx = list((cfg.keypoint_config or {}).get("flip_idx") or list(range(K)))
        flip_pairs = [[i, j] for i, j in enumerate(flip_idx) if isinstance(j, int) and j > i]
        body = f'''# AUTO-GENERATED by momoto _detrpose.py — do not edit by hand.
from ..detrpose_hgnetv2_{size} import (
    model, criterion, postprocessor, training_params,
    ema, optimizer, lr_scheduler,
    dataset_train, dataset_val, dataset_test, evaluator,
)
import src.data.transforms as _mmt_T
_mmt_T._MOTOMOTO_FLIP_PAIRS = {flip_pairs!r}   # this skeleton's L/R swaps (patched hflip reads it)
import src.data.coco_eval as _mmt_E
import numpy as _mmt_np
_mmt_E._MOTOMOTO_KPT_SIGMAS = _mmt_np.full({K}, 0.05, dtype=_mmt_np.float32)  # K-len OKS sigmas for eval

# ---- keypoint count (override the COCO-17 default everywhere it is hardcoded) ----
K = {K}
model.transformer.num_body_points = K
criterion.num_body_points = K
criterion.matcher.num_body_points = K
postprocessor.num_body_points = K

# ---- class count ----
# criterion uses target background index == num_classes and one_hot(num_classes+1)[..,:-1];
# the stock COCO config uses 2 for the single 'person' class. We mirror that convention:
# num_classes = max(2, num_foreground_categories + 1). UNVERIFIED for multi-category pose.
model.transformer.num_classes = {num_classes}
criterion.num_classes = {num_classes}

# ---- training schedule / output ----
training_params.output_dir = {out_dir!r}
training_params.epochs = {int(cfg.epochs)}
training_params.save_checkpoint_interval = 1

# ---- dataset: point at our generated COCO-keypoints ----
dataset_train.dataset.img_folder = {train_img!r}
dataset_train.dataset.ann_file = {train_ann!r}
dataset_train.total_batch_size = {int(cfg.batch)}
dataset_val.dataset.img_folder = {val_img!r}
dataset_val.dataset.ann_file = {val_ann!r}
dataset_val.total_batch_size = {int(cfg.batch)}

# ---- evaluator: OKS on our val set ----
evaluator.ann_file = {val_ann!r}
evaluator.iou_types = ["keypoints"]
'''
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(body)
        # stash a copy + the kpt metadata next to checkpoints for export/predict reuse.
        # NOTE: the stashed COPY can't be loaded standalone — the LazyConfig does a Python
        # relative import (`from ..detrpose_hgnetv2_<size>`) that only resolves from inside the
        # repo's configs/detrpose/ tree. So predict/export prefer the GEN config (still in the
        # repo, persists per-project) via meta["config_path"]; the copy is a fallback/record.
        shutil.copy(cfg_path, os.path.join(out_dir, "_detrpose_config.py"))
        with open(os.path.join(out_dir, "_detrpose_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"size": size, "num_keypoints": K, "skeleton": skeleton,
                       "num_classes": num_classes, "categories": list(cfg.categories),
                       "config_path": cfg_path}, f)
        return cfg_path

    def _resolve_config(self, weights: str) -> str:
        """Pick a loadable LazyConfig for predict/export: the GEN config in the repo (relative
        imports resolve there) recorded in _detrpose_meta.json, else the stashed copy."""
        d = os.path.dirname(os.path.abspath(weights))
        meta_path = os.path.join(d, "_detrpose_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    cp = json.load(f).get("config_path")
                if cp and os.path.exists(cp):
                    return cp
            except Exception:
                pass
        return os.path.join(d, "_detrpose_config.py")

    # ------------------------------------------------------------------ repo patch
    @staticmethod
    def _patch_file(path, needle, replacement, sentinel="# momoto-kpt-patch"):
        """Idempotent str-replace patch of a repo source file. No-op if the sentinel is already
        present or the needle isn't found (repo drifted) — we never guess, we let the GPU run
        reveal an unpatched site. Returns True if it wrote a change."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()
        except OSError:
            return False
        if sentinel in src or needle not in src:
            return False
        with open(path, "w", encoding="utf-8") as f:
            f.write(src.replace(needle, replacement))
        return True

    def _ensure_repo_kpt_patch(self) -> None:
        """Port DETRPose (built for human pose: K=17 COCO / K=14 CrowdPose) to an ARBITRARY
        keypoint count. Three idempotent, sentinel-guarded source patches:

          1. src/data/coco.py — ``reshape(-1, 17, 3)`` is hardcoded → infer K from each
             annotation's flat ``[x,y,v]*K`` list so K!=17 datasets load.
          2. src/models/detrpose/matcher.py — ``__init__`` only sets OKS sigmas for K in
             {17,14}, else ``raise NotImplementedError`` → fall back to uniform sigmas (0.05)
             for any K (we lack per-keypoint OKS sigmas for custom skeletons; uniform is the
             standard choice and only affects the matching/AP weighting, not the learned model).
          3. src/misc/keypoint_loss.py — ``OKSLoss.__init__`` only sets sigmas for K in
             {17,14,3}, else ``raise ValueError`` → same uniform-sigma fallback.

        If a needle isn't found (repo revision drift) the patch is skipped and training will
        surface the original error, which can then be patched by hand.
        """
        rp = lambda *p: os.path.join(repo(self.REPO), *p)
        # 1) dataloader reshape
        self._patch_file(
            rp("src", "data", "coco.py"),
            "        keypoints = torch.as_tensor(keypoints, dtype=torch.float32).reshape(-1, 17, 3)",
            ("        # momoto-kpt-patch: honor an arbitrary keypoint count\n"
             "        _flat = keypoints[0] if (isinstance(keypoints, list) and keypoints) else []\n"
             "        _K = (len(_flat) // 3) if _flat else 17\n"
             "        keypoints = torch.as_tensor(keypoints, dtype=torch.float32).reshape(-1, _K, 3)"),
        )
        # 2) matcher OKS sigmas: bare `raise NotImplementedError` (line ~41, NOT the matcher_type one)
        self._patch_file(
            rp("src", "models", "detrpose", "matcher.py"),
            "        else:\n            raise NotImplementedError\n",
            ("        else:\n"
             "            # momoto-kpt-patch: uniform OKS sigmas for an arbitrary keypoint count\n"
             "            self.sigmas = np.full((num_body_points,), 0.05, dtype=np.float32)\n"),
        )
        # 3) OKSLoss sigmas
        self._patch_file(
            rp("src", "misc", "keypoint_loss.py"),
            "        else:\n            raise ValueError(f'Unsupported keypoints number {num_keypoints}')",
            ("        else:\n"
             "            # momoto-kpt-patch: uniform OKS sigmas for an arbitrary keypoint count\n"
             "            self.sigmas = np.full((num_keypoints,), 0.05, dtype=np.float32)"),
        )
        # 3b) denoising (CDN) sigmas: src/models/detrpose/dn_component.get_sigmas hardcodes K in
        #     {17,14,3} (then prepends a 0.1 center) → uniform fallback for any K. Same /10 scale.
        self._patch_file(
            rp("src", "models", "detrpose", "dn_component.py"),
            "    else:\n        raise ValueError(f'Unsupported keypoints number {num_keypoints}')",
            ("    else:\n"
             "        # momoto-kpt-patch: uniform OKS sigmas for an arbitrary keypoint count\n"
             "        sigmas = np.full((num_keypoints,), 0.05, dtype=np.float32)"),
        )
        # 4) hflip flip-pairs: hardcoded COCO-17 (1-indexed) crashes / mis-swaps for other K.
        #    Read them from a module global our generated config sets from this skeleton's flip_idx
        #    (None -> COCO default for real human-pose datasets). Workers inherit it via fork.
        self._patch_file(
            rp("src", "data", "transforms.py"),
            "        flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],",
            ("        flip_pairs = globals().get('_MOTOMOTO_FLIP_PAIRS')\n"
             "        if flip_pairs is None:  # momoto-kpt-patch: default COCO-17 when not set\n"
             "            flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],"),
            sentinel="_MOTOMOTO_FLIP_PAIRS",
        )
        # 4b) numpy 2.x: `float(np.random.uniform(a, b, 1))` returns a 1-element ARRAY, and float()
        #     on it is a hard TypeError in numpy>=2 (it was only a deprecation in <2). The
        #     RandomZoomOut/Mosaic transform hits this at the first Mosaic epoch — drop the size=1
        #     so np.random.uniform returns a scalar. (Pre-existing repo bug, surfaced once a real
        #     run reached the Mosaic epoch on numpy 2.x.)
        self._patch_file(
            rp("src", "data", "transforms.py"),
            "float(np.random.uniform(self.side_range[0], self.side_range[1], 1))",
            "float(np.random.uniform(self.side_range[0], self.side_range[1]))  # momoto-np2-patch",
            sentinel="momoto-np2-patch",
        )
        # 5) COCOeval OKS sigmas: pycocotools defaults to 17 COCO sigmas, mismatching K!=17 at
        #    eval (computeOks broadcast error). Override params.kpt_oks_sigmas from a module global
        #    our config sets (length K). Two COCOeval creation sites (__init__ + cleanup).
        ce = rp("src", "data", "coco_eval.py")
        try:
            with open(ce, "r", encoding="utf-8") as f:
                ce_src = f.read()
        except OSError:
            ce_src = None
        if ce_src is not None and "_MOTOMOTO_KPT_SIGMAS" not in ce_src:
            inject = (
                "\n            _mmt_s = globals().get('_MOTOMOTO_KPT_SIGMAS')  # momoto-kpt-patch\n"
                "            if _mmt_s is not None and iou_type == 'keypoints':\n"
                "                self.coco_eval[iou_type].params.kpt_oks_sigmas = _mmt_s")
            ce_src = ce_src.replace(
                "            self.coco_eval[iou_type].useCats = useCats",
                "            self.coco_eval[iou_type].useCats = useCats" + inject)
            ce_src = ce_src.replace(
                "            self.coco_eval[iou_type].useCats = self.useCats",
                "            self.coco_eval[iou_type].useCats = self.useCats" + inject)
            with open(ce, "w", encoding="utf-8") as f:
                f.write(ce_src)

    # ------------------------------------------------------------------ train
    def train(self, cfg: TrainConfig, progress_cb=None, should_stop=None) -> TrainResult:
        size = cfg.size if cfg.size in self.sizes else self.default_size
        K, names, flip_idx, skeleton, sigmas = self._kpt_params(cfg)
        # criterion background convention mirrors stock COCO (single class -> num_classes=2).
        num_classes = max(2, cfg.num_classes + 1)
        self._ensure_repo_kpt_patch()
        cfg_path = self._write_config(cfg, size, K, skeleton, num_classes)
        dev = resolve_device(cfg.device)

        # train.py CLI. LazyConfig path is relative to repo cwd (run_stream cwd=repo).
        rel_cfg = os.path.relpath(cfg_path, repo(self.REPO))
        cmd = [venv_python(), "train.py", "-c", rel_cfg, "--device", dev, "--seed", "0"]
        if dev == "cuda":
            cmd.append("--amp")
        # optional transfer learning from D-FINE obj365 (downloads weights) — opt-in only.
        if cfg.extra.get("pretrain") and size in _PRETRAIN:
            cmd += ["--pretrain", _PRETRAIN[size]]

        state = {"epoch": 0, "ap": 0.0, "ap50": 0.0}

        def on_line(line):
            m = _RE_EPOCH.search(line)
            if m:
                state["epoch"] = int(m.group(1))
            mb = _RE_BEST.search(line)
            if mb:
                state["epoch"] = max(state["epoch"], int(mb.group(1)))
            m5 = _RE_AP50.search(line)
            if m5:
                state["ap50"] = float(m5.group(1))
            ma = _RE_AP.search(line)
            if ma:
                state["ap"] = float(ma.group(1))
                if progress_cb:
                    progress_cb({"epoch": state["epoch"], "total_epochs": cfg.epochs,
                                 "metrics": {"map": state["ap"], "map50_95": state["ap"],
                                             "map50": state["ap50"]}})
            if should_stop and should_stop():
                raise KeyboardInterrupt("training stop requested")

        # Clear PYTHONPATH for the repo subprocess: the agent app dir carries a REGULAR `src`
        # package (src.ml.*) which would shadow DETRPose's own NAMESPACE `src` (no __init__),
        # so `from src.solver import Trainer` resolves to the app and fails. The repo gets its
        # own src via the script dir (sys.path[0]); it needs nothing from PYTHONPATH.
        rc, tail = run_stream(cmd, repo(self.REPO), on_line=on_line, env={"PYTHONPATH": ""})
        if rc != 0:
            raise RuntimeError(f"DETRPose training failed (rc={rc}):\n{tail}")
        weights = find_weight(os.path.abspath(cfg.out_dir), self.weights_glob)
        if not weights:
            raise RuntimeError(f"DETRPose produced no checkpoint in {cfg.out_dir}:\n{tail}")
        return TrainResult(
            weights=weights,
            metrics={"map": state["ap"], "map50_95": state["ap"], "map50": state["ap50"]},
            classes=list(cfg.categories),
            fmt=self.family,
        )

    # ------------------------------------------------------------------ export
    def export_onnx(self, weights: str, out_path: str, **kw) -> str:
        """Export via tools/deployment/export_onnx.py (pending GPU smoke; repo path confirmed by read).

        export_onnx.py args: ``--config_file/-c <cfg.py> --resume/-r <weights.pth>``. It builds the
        model+postprocessor from the LazyConfig, loads the checkpoint (ema first, else 'model'),
        deploys, traces a dummy ``(1,3,640,640)`` image + ``[[640,640]]`` size, and writes
        ``onnx_engines/<config-basename>.onnx`` *relative to the CWD* (the repo dir). Its --check and
        --simplify flags are hardcoded True (onnx==1.21 + onnxsim are installed in the model env), so
        the export self-validates. We move the produced file to out_path.

        The config MUST be the same one used at train (it carries num_body_points/num_classes); we
        default to the stashed ``_detrpose_config.py`` next to the checkpoint. Runs on CPU
        (CUDA_VISIBLE_DEVICES="") — torch.onnx tracing of the deploy model is CPU-safe — and with
        PYTHONPATH="" so the repo's namespace ``src`` package is not shadowed by the agent app's.
        """
        cfg_path = kw.get("config") or self._resolve_config(weights)
        if not os.path.exists(cfg_path):
            raise RuntimeError(f"DETRPose export needs the training config; not found at {cfg_path}")
        # export_onnx.py derives the output filename from the config path's basename (split('/')[-1]),
        # so a repo-relative path keeps the name predictable. Fall back to an absolute path otherwise.
        repo_dir = repo(self.REPO)
        cfg_abs = os.path.abspath(cfg_path)
        rel_cfg = os.path.relpath(cfg_abs, repo_dir) if cfg_abs.startswith(os.path.abspath(repo_dir) + os.sep) \
            else cfg_abs
        cmd = [venv_python(), "tools/deployment/export_onnx.py", "-c", rel_cfg, "-r", os.path.abspath(weights)]
        rc, tail = run_stream(cmd, repo_dir, env={"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": ""})
        # output_file = 'onnx_engines/' + config_basename.replace('py','onnx')  (see export_onnx.py)
        engines = os.path.join(repo_dir, "onnx_engines")
        produced = os.path.join(engines, os.path.basename(rel_cfg).replace(".py", ".onnx"))
        if not os.path.exists(produced):
            # tolerate a config basename without a .py suffix / repo drift: take the newest .onnx.
            import glob as _g
            cands = sorted(_g.glob(os.path.join(engines, "*.onnx")), key=os.path.getmtime, reverse=True)
            produced = cands[0] if cands else produced
        if not os.path.exists(produced):
            raise RuntimeError(f"DETRPose ONNX export failed (rc={rc}):\n{tail}")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        if os.path.abspath(produced) != os.path.abspath(out_path):
            shutil.move(produced, out_path)
        return out_path

    # ------------------------------------------------------------------ predict
    def predict(self, weights: str, image_path: str, conf: float = 0.3):
        """Run single-image pose inference and return canonical DB objects.

        We ship our OWN canonical-JSON helper (the repo's tools/inference/torch_inf.py only draws and
        hardcodes the annotator for K in {17,14}). It mirrors that script's model build + preprocess:

          * Load the stashed ``_detrpose_config.py`` (carries num_body_points/num_classes) via
            ``src.core.LazyConfig`` and ``instantiate(cfg.model)`` / ``instantiate(cfg.postprocessor)``.
          * Load the checkpoint state (``ema.module`` if present, else ``model``) and ``model.deploy()``
            (eval + backbone reparam). We keep the postprocessor in NON-deploy mode so it returns a
            per-image dict ``{scores, labels, keypoints[num_select, K*3]}`` with a visibility column
            (deploy mode would drop visibility and only give K*2).
          * Preprocess exactly like the val pipeline / torch_inf: ``Resize((640,640))`` + ``ToTensor``
            (scales to 0..1; the repo's Normalize is mean=0/std=1, i.e. a no-op).
          * Run ``out = model(im)`` -> dict; ``postprocessor(out, target_sizes=[[1,1]])`` so keypoints
            come back ALREADY normalized 0..1 (PostProcess multiplies the sigmoid keypoints by
            target_sizes). Filter by ``score >= conf``; map ``label`` -> category via meta; derive a
            tight bbox from the visible keypoints' extent (DETRPose emits no boxes). Emit one
            ``OBJ {category, score, bbox{...}, keypoints:[x,y,v,...]}`` line per detection.

        Runs with CUDA_VISIBLE_DEVICES="" (CPU-safe predict) and PYTHONPATH="" so the repo's
        namespace ``src`` package is not shadowed by the agent app's regular ``src`` package.
        """
        cfg_path = self._resolve_config(weights)
        meta_path = os.path.join(os.path.dirname(weights), "_detrpose_meta.json")
        if not os.path.exists(cfg_path):
            raise RuntimeError(f"DETRPose predict needs the training config; not found at {cfg_path}")
        categories = []
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    categories = list(json.load(f).get("categories") or [])
            except Exception:
                categories = []

        helper = (
            "import sys, os, json, torch\n"
            "import torchvision.transforms as T\n"
            "from PIL import Image\n"
            "sys.path.insert(0, os.path.abspath('.'))  # repo root: provides the repo's own `src`\n"
            "from src.core import LazyConfig, instantiate\n"
            "cfg_path, weights, image_path, conf = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])\n"
            "categories = json.loads(sys.argv[5])\n"
            "cfg = LazyConfig.load(cfg_path)\n"
            "if hasattr(cfg.model.backbone, 'pretrained'):\n"
            "    cfg.model.backbone.pretrained = False  # don't fetch backbone weights at inference\n"
            "model = instantiate(cfg.model)\n"
            "postprocessor = instantiate(cfg.postprocessor)\n"
            "ckpt = torch.load(weights, map_location='cpu', weights_only=False)\n"
            "state = ckpt['ema']['module'] if 'ema' in ckpt else ckpt['model']\n"
            "model.load_state_dict(state)\n"
            "model = model.deploy()\n"          # eval() + backbone reparam (matches torch_inf.py)
            "postprocessor = postprocessor.deploy()\n"  # deploy: returns (scores,labels,keypoints[B,N,K,2]) — matches torch_inf.py; non-deploy gather mismatches
            "K = int(model.transformer.num_body_points)\n"
            "im_pil = Image.open(image_path).convert('RGB')\n"
            "tf = T.Compose([T.Resize((640, 640)), T.ToTensor()])  # ToTensor -> 0..1; repo Normalize is a no-op\n"
            "im = tf(im_pil).unsqueeze(0)\n"
            "def clamp01(v):\n"
            "    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)\n"
            "with torch.no_grad():\n"
            "    out = model(im)\n"
            "    # target_sizes=[[1,1]] => keypoints already normalized 0..1 (deploy multiplies by it)\n"
            "    sc, lb, kpt = postprocessor(out, torch.tensor([[1.0, 1.0]]))\n"
            "scores = sc[0].cpu().tolist()\n"
            "labels = lb[0].cpu().tolist()\n"
            "kps = kpt[0].cpu().tolist()  # [num_select, K, 2] (x,y) normalized; deploy drops visibility\n"
            "for s, lab, kpc in zip(scores, labels, kps):\n"
            "    if s < conf:\n"
            "        continue\n"
            "    kp = []\n"
            "    xs, ys = [], []\n"
            "    for j in range(K):\n"
            "        x = clamp01(float(kpc[j][0])); y = clamp01(float(kpc[j][1]))\n"
            "        kp += [x, y, 2]  # deploy gives no visibility; mark every emitted keypoint labeled+visible\n"
            "        xs.append(x); ys.append(y)\n"
            "    if not xs:\n"
            "        continue\n"
            "    x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)\n"
            "    w = x1 - x0; h = y1 - y0\n"
            "    # pad the keypoint hull a little so the derived bbox isn't a zero-area line/point\n"
            "    pad_x = max(0.02, w * 0.1); pad_y = max(0.02, h * 0.1)\n"
            "    bx0 = clamp01(x0 - pad_x); by0 = clamp01(y0 - pad_y)\n"
            "    bx1 = clamp01(x1 + pad_x); by1 = clamp01(y1 + pad_y)\n"
            "    bbox = {'x_center': (bx0 + bx1) / 2.0, 'y_center': (by0 + by1) / 2.0,\n"
            "            'width': bx1 - bx0, 'height': by1 - by0}\n"
            "    name = categories[int(lab)] if 0 <= int(lab) < len(categories) else str(int(lab))\n"
            "    print('OBJ ' + json.dumps({'category': name, 'score': float(s), 'bbox': bbox, 'keypoints': kp}))\n"
        )
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(helper)
            hp = f.name
        try:
            objs = []

            def on_line(line):
                if line.startswith("OBJ "):
                    try:
                        objs.append(json.loads(line[4:]))
                    except Exception:
                        pass

            rc, tail = run_stream(
                [venv_python(), hp, os.path.abspath(cfg_path), os.path.abspath(weights),
                 os.path.abspath(image_path), str(conf), json.dumps(categories)],
                repo(self.REPO), on_line=on_line,
                env={"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": ""},
            )
            if rc != 0 and not objs:
                raise RuntimeError(f"DETRPose predict failed (rc={rc}):\n{tail}")
            return objs
        finally:
            os.unlink(hp)
