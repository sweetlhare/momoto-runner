"""Universal DB annotation format -> COCO dataset for DETR-family training.

THE single source of truth is the DB (`Annotation.data`), a normalized canonical
shape::

    {"objects": [
        {"category": "<name>",
         "bbox": {"x_center", "y_center", "width", "height"},   # all 0..1
         "polygon": [x, y, x, y, ...],                          # 0..1, optional (seg)
         "keypoints": [x, y, v, ...]}                           # 0..1 + visibility, optional (pose)
    ]}

COCO is generated ON DEMAND for training D-FINE / DETRPose / D-FINE-seg. We do NOT
keep a YOLO-txt format anymore — the DB is the only annotation format.

IMPORTANT: category ids are **0-indexed**. D-FINE / DETRPose / D-FINE-seg run with
``remap_mscoco_category: False`` and use ``category_id`` DIRECTLY as the class label,
so labels must be 0..num_classes-1 (1-indexed ids trigger a CUDA device-side assert).
"""
import os
import json
import shutil


def universal_to_coco(samples, categories, task="detection", kpt_names=None, skeleton=None):
    """Convert universal DB samples to a COCO dict.

    samples: iterable of {"file_name", "w", "h", "objects": [...]}.
    categories: ordered list of class names (index == 0-based category_id).
    """
    cat_id = {c: i for i, c in enumerate(categories)}  # 0-indexed (see module docstring)
    is_pose = task in ("keypose", "keypoint_detection", "pose")
    cats = []
    for c in categories:
        entry = {"id": cat_id[c], "name": c, "supercategory": "object"}
        if is_pose and kpt_names:
            entry["keypoints"] = kpt_names
            entry["skeleton"] = skeleton or []
        cats.append(entry)
    coco = {"images": [], "annotations": [], "categories": cats}
    ann_id = 1
    for img_id, s in enumerate(samples, 1):
        W, H = s["w"], s["h"]
        coco["images"].append({"id": img_id, "file_name": s["file_name"], "width": W, "height": H})
        for o in s.get("objects", []):
            c = o.get("category")
            if c not in cat_id:
                continue
            e = {"id": ann_id, "image_id": img_id, "category_id": cat_id[c], "iscrowd": 0}
            b = o.get("bbox")
            if b:
                w = float(b["width"]) * W
                h = float(b["height"]) * H
                x = float(b["x_center"]) * W - w / 2
                y = float(b["y_center"]) * H - h / 2
                e["bbox"] = [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]
                e["area"] = round(w * h, 2)
            poly = o.get("polygon")
            if poly and len(poly) >= 6:
                ap = [round((float(poly[i]) * W) if i % 2 == 0 else (float(poly[i]) * H), 2)
                      for i in range(len(poly))]
                e["segmentation"] = [ap]
                if "bbox" not in e:
                    xs, ys = ap[0::2], ap[1::2]
                    e["bbox"] = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                    e["area"] = e["bbox"][2] * e["bbox"][3]
            if o.get("text"):
                e["text"] = o.get("text")          # OCR transcription
            if o.get("obb"):
                e["obb"] = o.get("obb")            # oriented box params (cx,cy,w,h,angle)
            if o.get("point"):
                e["point"] = o.get("point")        # counting point
            if o.get("attributes"):
                e["attributes"] = o.get("attributes")
            kp = o.get("keypoints")
            if kp:
                ak, n = [], 0
                for i in range(0, len(kp) - 2, 3):
                    v = int(kp[i + 2])
                    ak += [round(float(kp[i]) * W, 2), round(float(kp[i + 1]) * H, 2), v]
                    if v > 0:
                        n += 1
                e["keypoints"] = ak
                e["num_keypoints"] = n
            # Append if the object carries ANY geometry (box/poly/kps/obb/point).
            if "bbox" in e or "segmentation" in e or "keypoints" in e or "obb" in e or "point" in e:
                # For OBB give a synthetic axis-aligned bbox (COCO consumers expect one).
                if "bbox" not in e and "obb" in e:
                    ob = e["obb"]
                    import math as _m
                    c, sang = _m.cos(float(ob["angle"])), _m.sin(float(ob["angle"]))
                    hw, hh = float(ob["w"]) * W / 2, float(ob["h"]) * H / 2
                    cxp, cyp = float(ob["cx"]) * W, float(ob["cy"]) * H
                    xs = [cxp + lx * c - ly * sang for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
                    ys = [cyp + lx * sang + ly * c for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
                    e["bbox"] = [round(min(xs), 2), round(min(ys), 2), round(max(xs) - min(xs), 2), round(max(ys) - min(ys), 2)]
                    e["area"] = round(e["bbox"][2] * e["bbox"][3], 2)
                coco["annotations"].append(e)
                ann_id += 1
    return coco


def build_coco_dataset(samples, categories, out_dir, task="detection",
                       val_ratio=0.2, seed=0, copy_images=True, kpt_names=None, skeleton=None):
    """Materialize a COCO dataset directory from universal DB samples.

    Each sample needs {"file_name", "src" (abs image path), "w", "h", "objects"}.
    Produces::

        out_dir/train/<images>            out_dir/val/<images>
        out_dir/annotations/instances_train.json    .../instances_val.json

    Returns a dict of the produced paths + counts (for wiring into the trainer config).
    """
    import random
    rng = random.Random(seed)
    samples = list(samples)
    rng.shuffle(samples)
    nval = max(1, int(len(samples) * val_ratio)) if len(samples) > 1 else 0
    splits = {"val": samples[:nval], "train": samples[nval:]}
    os.makedirs(os.path.join(out_dir, "annotations"), exist_ok=True)
    result = {"out_dir": out_dir, "splits": {}}
    for split, ss in splits.items():
        img_dir = os.path.join(out_dir, split)
        os.makedirs(img_dir, exist_ok=True)
        if copy_images:
            for s in ss:
                dst = os.path.join(img_dir, s["file_name"])
                if s.get("src") and os.path.abspath(s["src"]) != os.path.abspath(dst):
                    shutil.copy(s["src"], dst)
        coco = universal_to_coco(ss, categories, task=task, kpt_names=kpt_names, skeleton=skeleton)
        ann_path = os.path.join(out_dir, "annotations", f"instances_{split}.json")
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)
        result["splits"][split] = {"img_folder": img_dir, "ann_file": ann_path,
                                    "images": len(coco["images"]), "annotations": len(coco["annotations"])}
    result["num_classes"] = len(categories)
    result["categories"] = list(categories)
    return result
