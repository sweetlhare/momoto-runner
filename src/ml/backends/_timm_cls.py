"""timm image-classification backends (MIT/Apache-2.0): ConvNeXtV2 (default), MobileNetV4,
EfficientViT — user-selectable, each in several sizes.

They all share ONE trainer (convnextv2_impl.py is model-agnostic: `timm.create_model(--model)`)
and one predict path (`timm.create_model(ckpt['model'])`), so a family is just its name + a
`size tier -> exact timm model name` map. Dataset is ImageFolder (train/<class>/*, val/<class>/*).
"""
import os
import re
import json
import shutil
import tempfile

from .base import ModelBackend, TrainConfig, TrainResult, register
from ._env import venv_python, run_stream, resolve_device

_IMPL = os.path.join(os.path.dirname(__file__), "convnextv2_impl.py")
_RE_PROGRESS = re.compile(r"^PROGRESS (\{.*\})\s*$")
_RE_RESULT = re.compile(r"^RESULT (\{.*\})\s*$")


class _TimmClassifier(ModelBackend):
    """Shared ImageFolder + timm-subprocess orchestration. A subclass sets `family`, `label`,
    `default_size` and `TIMM_MODELS` (size tier -> exact timm model name); `sizes` is derived."""
    task = "classification"
    weights_glob = ("best.pt",)
    TIMM_MODELS: dict = {}

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls.TIMM_MODELS:                       # derive the size tiers from the model map
            cls.sizes = tuple(cls.TIMM_MODELS.keys())

    @classmethod
    def valid_archs(cls):
        return [f"{cls.family}_{s}" for s in cls.sizes]

    @classmethod
    def size_tier(cls, arch: str) -> str:
        a = (arch or "")
        if a.startswith(cls.family + "_"):
            a = a[len(cls.family) + 1:]
        return a if a in cls.TIMM_MODELS else cls.default_size

    def _timm_name(self, size: str) -> str:
        return self.TIMM_MODELS[self.size_tier(size)]

    def prepare_dataset(self, samples, categories, out_dir, **kw):
        """ImageFolder (train/<class>/*, val/<class>/*). samples: [{file_name, src, category|label}]."""
        import random
        rng = random.Random(kw.get("seed", 0))
        val_ratio = kw.get("val_ratio", 0.2)
        by_class = {}
        for s in samples:
            by_class.setdefault(s.get("category") or s.get("label"), []).append(s)
        counts = {"train": 0, "val": 0}
        for cls_name, items in by_class.items():
            rng.shuffle(items)
            nval = max(1, int(len(items) * val_ratio)) if len(items) > 1 else 0
            for split, group in (("val", items[:nval]), ("train", items[nval:])):
                d = os.path.join(out_dir, split, str(cls_name))
                os.makedirs(d, exist_ok=True)
                for s in group:
                    if s.get("src"):
                        shutil.copy(s["src"], os.path.join(d, s["file_name"]))
                    counts[split] += 1
        return {"out_dir": out_dir, "num_classes": len(by_class),
                "categories": list(by_class.keys()), "counts": counts}

    def train(self, cfg: TrainConfig, progress_cb=None, should_stop=None) -> TrainResult:
        model = self._timm_name(cfg.size)
        out_dir = os.path.abspath(cfg.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        dev_cpu = resolve_device(cfg.device) == "cpu"
        cmd = [venv_python(), _IMPL, "--data", os.path.abspath(cfg.dataset_dir), "--out", out_dir,
               "--model", model, "--loss", cfg.loss, "--epochs", str(cfg.epochs),
               "--bs", str(cfg.batch), "--img", str(cfg.img_size)]
        if dev_cpu:
            cmd.append("--cpu")
        result = {}

        def on_line(line):
            mp = _RE_PROGRESS.match(line)
            if mp and progress_cb:
                p = json.loads(mp.group(1))
                progress_cb({"epoch": p.get("epoch"), "total_epochs": p.get("epochs"),
                             "metrics": {"val_acc": p.get("val_acc"), "loss": p.get("loss")}})
            mr = _RE_RESULT.match(line)
            if mr:
                result.update(json.loads(mr.group(1)))
            if should_stop and should_stop():
                raise KeyboardInterrupt("training stop requested")

        rc, tail = run_stream(cmd, os.path.dirname(_IMPL), on_line=on_line)
        if rc != 0 or not result.get("weights"):
            raise RuntimeError(f"{self.label or self.family} training failed (rc={rc}):\n{tail}")
        return TrainResult(weights=result["weights"], onnx=result.get("onnx"),
                           metrics={"val_acc": result.get("val_acc")},
                           classes=result.get("classes", list(cfg.categories)), fmt=self.family)

    def export_onnx(self, weights: str, out_path: str, **kw) -> str:
        sibling = os.path.join(os.path.dirname(weights), "model.onnx")
        if os.path.exists(sibling):
            if os.path.abspath(sibling) != os.path.abspath(out_path):
                shutil.copy(sibling, out_path)
            return out_path
        raise RuntimeError(f"{self.label or self.family} ONNX not found next to checkpoint; re-run train.")

    def predict(self, weights: str, image_path: str, conf: float = 0.3):
        """Run the timm classifier in the model env; return one canonical classification object.
        The checkpoint stashes its own timm model name, so this works for any family."""
        helper = (
            "import sys,json,torch,timm\n"
            "from PIL import Image\n"
            "from torchvision import transforms\n"
            "ck=torch.load(sys.argv[1],map_location='cpu')\n"
            "net=timm.create_model(ck['model'],pretrained=False,num_classes=len(ck['classes']))\n"
            "net.load_state_dict(ck['state_dict']);net.eval()\n"
            "sz=ck.get('img',224)\n"
            "tf=transforms.Compose([transforms.Resize((sz,sz)),transforms.ToTensor(),"
            "transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])\n"
            "x=tf(Image.open(sys.argv[2]).convert('RGB')).unsqueeze(0)\n"
            "p=torch.softmax(net(x),1)[0];i=int(p.argmax())\n"
            "print('OBJ '+json.dumps({'category':ck['classes'][i],'score':float(p[i])}))\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(helper)
            hp = f.name
        try:
            out = {}

            def on_line(l):
                if l.startswith("OBJ "):
                    out.update(json.loads(l[4:]))
            rc, tail = run_stream([venv_python(), hp, weights, image_path], os.path.dirname(_IMPL),
                                  on_line=on_line, env={"CUDA_VISIBLE_DEVICES": ""})
            if not out:
                raise RuntimeError(f"{self.label or self.family} predict failed (rc={rc}):\n{tail}")
            return [{"category": out["category"], "score": out.get("score")}]
        finally:
            os.unlink(hp)


@register("classification")
class ConvNeXtV2Classifier(_TimmClassifier):
    """Default classifier — strong accuracy/size trade-off across tiny→base."""
    family = "convnextv2"
    label = "ConvNeXtV2"
    default_size = "femto"
    TIMM_MODELS = {
        "atto": "convnextv2_atto", "femto": "convnextv2_femto", "pico": "convnextv2_pico",
        "nano": "convnextv2_nano", "tiny": "convnextv2_tiny", "base": "convnextv2_base",
    }


@register("classification")
class MobileNetV4Classifier(_TimmClassifier):
    """Mobile/edge-first — smallest + fastest for on-device / CPU inference."""
    family = "mobilenetv4"
    label = "MobileNetV4"
    default_size = "small"
    TIMM_MODELS = {
        "small": "mobilenetv4_conv_small",
        "medium": "mobilenetv4_conv_medium",
        "large": "mobilenetv4_conv_large",
    }


@register("classification")
class EfficientViTClassifier(_TimmClassifier):
    """Transformer-efficient — strong throughput on GPU at higher accuracy tiers."""
    family = "efficientvit"
    label = "EfficientViT"
    default_size = "b1"
    TIMM_MODELS = {
        "b0": "efficientvit_b0", "b1": "efficientvit_b1",
        "b2": "efficientvit_b2", "b3": "efficientvit_b3",
    }
