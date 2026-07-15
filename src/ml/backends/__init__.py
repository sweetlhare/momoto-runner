"""Pluggable, permissive model backends (replace Ultralytics YOLO / AGPL).

    detection      -> RF-DETR (_rfdetr, primary) + D-FINE (_dfine, fallback)   [Apache-2.0]
    classification -> ConvNeXtV2 (default) / MobileNetV4 / EfficientViT (_timm_cls) [timm; MIT/Apache]
    segmentation   -> D-FINE-seg   (_dfine_seg.DFineSegmentor)   [scaffold — verify on GPU]
    keypose        -> DETRPose (_detrpose, default) + RF-DETR Pose (_rfdetr_pose, preview) [verify on GPU]

A task can now have SEVERAL backends (user picks the arch family); the first registered is the
preferred default, and get_backend() skips a primary whose optional deps are absent.

Importing this package is LIGHT (no torch): backends shell out to the model env lazily.
"""
from .base import (
    ModelBackend, TrainConfig, TrainResult,
    get_backend, get_trainer_backend, get_inference_backend,
    register, available_tasks, canon_task, architectures, backends_for, billing_tier,
)

# import concrete backends so their @register decorators run (self-registration).
# ORDER matters: the FIRST registered backend for a task is its preferred default.
from . import _rfdetr         # noqa: F401  detection (primary)
from . import _dfine          # noqa: F401  detection (fallback) + shared D-FINE family
from . import _dfine_seg      # noqa: F401  segmentation
from . import _detrpose       # noqa: F401  keypose (primary/default)
from . import _rfdetr_pose    # noqa: F401  keypose (RF-DETR keypoint preview — second choice)
from . import _timm_cls       # noqa: F401  classification (ConvNeXtV2 / MobileNetV4 / EfficientViT)

__all__ = [
    "ModelBackend", "TrainConfig", "TrainResult",
    "get_backend", "get_trainer_backend", "get_inference_backend",
    "available_tasks", "canon_task", "architectures", "backends_for", "billing_tier",
]
