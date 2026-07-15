"""Pluggable model backends — route CV training / inference by task to a permissive
model family, replacing Ultralytics YOLO (AGPL-3.0):

    detection     -> D-FINE        (Apache-2.0)
    segmentation  -> D-FINE-seg    (Apache-2.0)
    keypose       -> DETRPose      (Apache-2.0)
    classification-> ConvNeXtV2    (timm, MIT/Apache)

One canonical DB annotation format is the single source of truth; `ml.datasets.db_to_coco`
generates COCO on demand for the DETR family (no YOLO-txt anymore). ImageFolder for cls.

Backends are intended to run the heavy model code in a SEPARATE environment (dedicated
venv / training agent / container, like the LocateAnything microservice) and are LAZY:
importing this package must stay light for the API process — torch / the model repos are
only touched inside train()/predict()/export_onnx() (typically via subprocess)."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional


# task aliases -> canonical task
_TASK_ALIAS = {"keypoint_detection": "keypose", "pose": "keypose"}


def canon_task(task: str) -> str:
    return _TASK_ALIAS.get(task, task)


@dataclass
class TrainConfig:
    project_id: int
    task: str                              # detection | segmentation | keypose | classification
    categories: List[str]                  # ordered class names; list index == 0-based category id
    dataset_dir: str                       # prepared COCO dir (DETR) or ImageFolder (cls)
    out_dir: str                           # where to write checkpoints/onnx
    size: str = "n"                        # det/seg/pose: n|s|m|l|x ; cls: atto|femto|pico|nano|tiny
    epochs: int = 50
    batch: int = 8
    img_size: int = 640
    device: str = "auto"                   # auto|cuda|cpu (backend resolves)
    mode: str = "final"                    # final | few_shot
    loss: str = "ce"                       # classification only: ce|ce_smooth|focal|class_balanced
    keypoint_config: Optional[dict] = None # pose: {num_keypoints, names, flip_idx, skeleton}
    weights_init: Optional[str] = None     # optional checkpoint to fine-tune from
    extra: dict = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.categories)


@dataclass
class TrainResult:
    weights: str                           # best checkpoint path (framework-native; NOT assumed .pt)
    onnx: Optional[str] = None
    metrics: dict = field(default_factory=dict)   # normalized: {map, map50, precision, recall, val_acc, ...}
    classes: List[str] = field(default_factory=list)
    fmt: str = ""                          # backend format tag (e.g. 'dfine','convnextv2') for load/export


class ModelBackend(ABC):
    """A per-task model backend. Inputs/outputs use the canonical DB annotation shape
    ({"objects":[{category, bbox{x_center,y_center,width,height}, polygon?, keypoints?}]},
    normalized) so the rest of the platform stays format-agnostic."""

    task: str = None
    family: str = None
    sizes: tuple = ()                      # valid size tiers, e.g. ('n','s','m','l','x')
    default_size: str = ""
    weights_glob: tuple = ()               # checkpoint filenames to discover, e.g. ('best.pth','best.pt')

    # ---- lifecycle ----
    @abstractmethod
    def train(self, cfg: TrainConfig, progress_cb: Optional[Callable[[dict], None]] = None,
              should_stop: Optional[Callable[[], bool]] = None) -> TrainResult:
        ...

    @abstractmethod
    def export_onnx(self, weights: str, out_path: str, **kw) -> str:
        ...

    @abstractmethod
    def predict(self, weights: str, image_path: str, conf: float = 0.3) -> List[dict]:
        """Return canonical objects ({category, bbox{x_center,...}, polygon?, keypoints?}) for one image."""
        ...

    # ---- dataset (DB universal -> backend layout); default = COCO via db_to_coco ----
    def prepare_dataset(self, samples, categories, out_dir, **kw) -> dict:
        """samples: [{file_name, src, w, h, objects}] (universal DB shape). Returns layout info."""
        from src.ml.datasets import build_coco_dataset
        return build_coco_dataset(samples, categories, out_dir, task=self.task, **kw)

    # ---- metadata used by routes/models_config/export ----
    label: str = ""                        # human-facing name for the arch picker (defaults to family)

    @classmethod
    def valid_archs(cls) -> list:
        return [f"{cls.family}_{s}" for s in cls.sizes]

    @classmethod
    def size_tier(cls, arch: str) -> str:
        a = (arch or "")
        for s in cls.sizes:
            if a.endswith(s) or a == s:
                return s
        return cls.default_size

    # sizes are ALWAYS listed smallest→largest, so the last is the biggest model.
    @classmethod
    def billing_tier(cls, size: str) -> str:
        """Map a family size tier onto the n|s|m|l|x scale the billing gate
        (require_training_access) understands. Normalized by position so the SMALLEST size → 'n'
        and the LARGEST → 'x' regardless of how many tiers a family has — never undercharges a big
        model to a cheap tier."""
        scale = ("n", "s", "m", "l", "x")
        tier = cls.size_tier(size)
        try:
            i = list(cls.sizes).index(tier)
        except ValueError:
            i = 0
        k = max(1, len(cls.sizes) - 1)
        return scale[min(4, round(i * 4 / k))]

    @classmethod
    def matches_arch(cls, arch: str) -> bool:
        """Does this backend own `arch`? Accepts the bare family ("rfdetr") or a family-qualified
        arch ("rfdetr_s" / "convnextv2_tiny")."""
        a = (arch or "").strip().lower()
        fam = (cls.family or "").lower()
        # accept the bare family, a separated arch (rfdetr_s), an exact valid_arch, OR the stored
        # registry form fmt+size with no separator (e.g. "rfdetrbase"/"dfinen" from rm.arch).
        return bool(a) and (a == fam or a.startswith(fam + "_") or a.startswith(fam + "-")
                            or a.startswith(fam)
                            or a in [x.lower() for x in cls.valid_archs()])

    @classmethod
    def available(cls) -> bool:
        """Whether this backend's heavy deps are importable in THIS environment. Overridden by
        backends whose package is optional (e.g. RF-DETR's `rfdetr`), so the default-backend pick
        can skip an unavailable primary and the arch picker can flag it. Kept light — never imports
        torch/the model repo here."""
        return True


# task -> ordered list of backend classes (index 0 = primary/preferred default)
_REGISTRY = {}


def register(task: str):
    def deco(cls):
        _REGISTRY.setdefault(canon_task(task), []).append(cls)
        return cls
    return deco


def backends_for(task: str) -> list:
    return list(_REGISTRY.get(canon_task(task), []))


def get_backend(task: str, arch: str = None) -> ModelBackend:
    """Resolve a backend for a task. With `arch` (family or family-qualified) → the backend that
    owns it (explicit choice — errors later if its deps are missing, with a clear message). Without
    arch → the preferred backend that is actually AVAILABLE here (so a primary whose optional
    package is absent, e.g. RF-DETR, gracefully falls back to the next, e.g. D-FINE)."""
    t = canon_task(task)
    backends = _REGISTRY.get(t)
    if not backends:
        raise ValueError(f"no model backend registered for task '{task}' (have {sorted(_REGISTRY)})")
    if arch:
        for cls in backends:
            if cls.matches_arch(arch):
                return cls()
    for cls in backends:
        try:
            if cls.available():
                return cls()
        except Exception:
            continue
    return backends[0]()   # nothing reported available → fall back to the primary anyway


# A ModelBackend handles both training and inference, so both factories are the same.
get_trainer_backend = get_backend
get_inference_backend = get_backend


def available_tasks() -> list:
    return sorted(_REGISTRY)


def billing_tier(task: str, arch: str, size: str) -> str:
    """The n|s|m|l|x tier the billing/quota gate should charge for (task, arch, size). Authoritative
    map owned by the registry so callers never guess a family's size→tier and undercharge."""
    return get_backend(task, arch).billing_tier(size)


def architectures(task: str = None) -> dict:
    """Arch catalogue for the training UI: {task: [{family, label, sizes, default_size, primary,
    available}]}. `primary` marks the preferred default; `available` is whether the backend's deps
    are present in the API process (informational — the agent is what actually trains)."""
    tasks = [canon_task(task)] if task else sorted(_REGISTRY)
    out = {}
    for t in tasks:
        rows = []
        for i, c in enumerate(_REGISTRY.get(t, [])):
            try:
                avail = c.available()
            except Exception:
                avail = False
            rows.append({"family": c.family, "label": c.label or c.family,
                         "sizes": list(c.sizes), "default_size": c.default_size,
                         "primary": (i == 0), "available": avail})
        out[t] = rows
    return out
