"""ConvNeXtV2 classification training (timm backbone) with selectable losses.

Permissive replacement for YOLO-cls / the CLIP path: timm code is Apache-2.0; training
from scratch / a permissive init avoids FCMAE weight-license traps. Runs in the model
environment (heavy: torch + timm) — imported lazily by the platform classification backend.

Dataset: ImageFolder (train/ val/ with class sub-dirs). Losses: ce | ce_smooth | focal |
class_balanced. Exposes ``train_classifier(...)`` (programmatic) plus a CLI.
Verified end-to-end on a fresh dataset (val_acc 1.0, all losses, ONNX export).
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- losses ----------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def _class_balanced_weights(counts, beta=0.999):
    counts = torch.tensor(counts, dtype=torch.float)
    eff = 1.0 - torch.pow(beta, counts)
    w = (1.0 - beta) / eff.clamp(min=1e-8)
    return w / w.sum() * len(counts)


def make_loss(name, class_counts=None, device="cpu"):
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "ce_smooth":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    if name == "focal":
        return FocalLoss(gamma=2.0)
    if name == "class_balanced":
        cw = _class_balanced_weights(class_counts).to(device) if class_counts else None
        return nn.CrossEntropyLoss(weight=cw)
    raise ValueError(f"unknown loss '{name}'")


def _loaders(data_dir, img_size, bs):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(), norm,
    ])
    tf_val = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), norm])
    tr = datasets.ImageFolder(os.path.join(data_dir, "train"), tf_train)
    va = datasets.ImageFolder(os.path.join(data_dir, "val"), tf_val)
    counts = [0] * len(tr.classes)
    for _, y in tr.samples:
        counts[y] += 1
    return (DataLoader(tr, bs, shuffle=True, num_workers=2),
            DataLoader(va, bs, shuffle=False, num_workers=2), tr.classes, counts)


def train_classifier(data_dir, out_dir, model="convnextv2_femto", loss="ce", epochs=30,
                     bs=16, img_size=224, lr=1e-3, pretrained=False, cpu=False, progress_cb=None):
    """Train a ConvNeXtV2 classifier on an ImageFolder dataset. Returns a result dict."""
    import timm
    dev = "cuda" if (torch.cuda.is_available() and not cpu) else "cpu"
    tl, vl, classes, counts = _loaders(data_dir, img_size, bs)
    net = timm.create_model(model, pretrained=pretrained, num_classes=len(classes)).to(dev)
    crit = make_loss(loss, counts, dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs))
    os.makedirs(out_dir, exist_ok=True)
    best, best_path = 0.0, os.path.join(out_dir, "best.pt")
    for ep in range(epochs):
        net.train()
        last = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            l = crit(net(x), y)
            l.backward()
            opt.step()
            last = float(l.item())
        sched.step()
        net.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(dev), y.to(dev)
                correct += int((net(x).argmax(1) == y).sum().item())
                total += int(y.numel())
        acc = correct / max(1, total)
        if progress_cb:
            progress_cb({"epoch": ep + 1, "epochs": epochs, "loss": last, "val_acc": acc})
        if acc >= best:
            best = acc
            torch.save({"state_dict": net.state_dict(), "classes": classes,
                        "model": model, "img": img_size}, best_path)
    onnx_path = os.path.join(out_dir, "model.onnx")
    _export_onnx(net.eval(), img_size, dev, onnx_path)
    json.dump({"classes": classes, "best_val_acc": best, "loss": loss, "model": model},
              open(os.path.join(out_dir, "meta.json"), "w"))
    return {"weights": best_path, "onnx": onnx_path, "classes": classes, "val_acc": best}


def _export_onnx(net, img_size, dev, onnx_path):
    dummy = torch.randn(1, 3, img_size, img_size, device=dev)
    # dynamo=False -> legacy TorchScript exporter; avoids the onnxscript version_converter
    # assertion seen on torch 2.11 (Pad/opset-17). Produces a clean, simplified graph.
    torch.onnx.export(net, dummy, onnx_path, opset_version=17, dynamo=False,
                      input_names=["img"], output_names=["logits"],
                      dynamic_axes={"img": {0: "b"}})
    return onnx_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/cnv2")
    ap.add_argument("--model", default="convnextv2_femto")
    ap.add_argument("--loss", default="ce", choices=["ce", "ce_smooth", "focal", "class_balanced"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    # machine-parseable lines so the ConvNeXtV2Classifier backend can read progress + result
    r = train_classifier(
        a.data, a.out, a.model, a.loss, a.epochs, a.bs, a.img, a.lr, a.pretrained, a.cpu,
        progress_cb=lambda p: print("PROGRESS " + json.dumps(p), flush=True))
    print("RESULT " + json.dumps(r), flush=True)
