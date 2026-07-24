#!/usr/bin/env python3
"""momoto Runner (local-first BYO-compute).

Runs on the USER's GPU (their machine / a rented box). Connects OUTBOUND to the platform
with the account token, long-polls for queued training tasks, builds the dataset locally
from the platform's annotations + the user's S3 images, trains via the permissive stack
(src/ml/backends — D-FINE / ConvNeXtV2), streams progress, writes the weights to the user's
S3, and reports the result. No inbound ports.

    docker run --gpus all \
      -e MOTOMOTO_URL=https://app.momoto.example \
      -e MOTOMOTO_TOKEN=<account token> \
      momoto/agent

Env: MOTOMOTO_URL, MOTOMOTO_TOKEN (required); AGENT_ID (default: hostname);
     MODEL_INTEGRATION_ROOT (where the model repos+venv live, default /opt/model-integration).
"""
import os
import sys
import time
import socket
import shutil
import tempfile
import traceback

import requests

BASE = os.environ["MOTOMOTO_URL"].rstrip("/")
TOKEN = os.environ.get("MOTOMOTO_TOKEN")
AGENT_ID = os.environ.get("AGENT_ID") or socket.gethostname()
MODEL_ROOT = os.environ.get("MODEL_INTEGRATION_ROOT", "/opt/model-integration")
# Optional credentials: when set, the agent (re-)logs in to mint a fresh token — both at startup
# if no MOTOMOTO_TOKEN was passed, and automatically when the current token expires (401). This
# lets a long-running agent outlive the JWT TTL without a manual restart. (Prod should prefer a
# long-lived account API key over username/password.)
USER = os.environ.get("MOTOMOTO_USER")
PASSWORD = os.environ.get("MOTOMOTO_PASS")
# Persistent local model store: a model the agent trains is kept here (weights + stashed config
# + meta) so the SAME agent can later serve inference (agent_infer) without re-downloading,
# keyed by the S3 weights key. (A different agent downloads the weights from the user's S3.)
MODEL_STORE = os.path.join(MODEL_ROOT, "agent_models")
S = requests.Session()
if TOKEN:
    S.headers["Authorization"] = f"Bearer {TOKEN}"


def _login():
    """Mint a fresh token from MOTOMOTO_USER/MOTOMOTO_PASS; returns it (and sets the session
    header) or None if no creds / login failed."""
    if not (USER and PASSWORD):
        return None
    try:
        r = requests.post(BASE + "/auth/login-json",
                          json={"username": USER, "password": PASSWORD}, timeout=30)
        if r.ok:
            tok = r.json().get("access_token")
            if tok:
                S.headers["Authorization"] = f"Bearer {tok}"
                print("[agent] (re)authenticated", flush=True)
                return tok
    except Exception as e:
        print(f"[agent] login failed: {e}", flush=True)
    return None


def api(method, path, _retry=True, **kw):
    timeout = kw.pop("timeout", 60)
    r = S.request(method, BASE + path, timeout=timeout, **kw)
    # token expired → re-login once and retry (long-running agents outlive the JWT TTL)
    if r.status_code == 401 and _retry and _login():
        return api(method, path, _retry=False, timeout=timeout, **kw)
    r.raise_for_status()
    return r.json() if r.content else {}


def download_media(file_path, local):
    """Fetch a platform-stored image (no BYO S3 — e.g. the local-storage community edition) via the
    authenticated media endpoint, using the agent's account token. Follows the S3 302→presigned
    redirect too, so it works for BYO-S3 users as well. Returns True on success."""
    try:
        r = S.get(BASE + "/data/" + file_path.lstrip("/"), timeout=120, stream=True, allow_redirects=True)
        if r.status_code != 200:
            return False
        with open(local, "wb") as f:
            for chunk in r.iter_content(65536):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"[agent] media fetch {file_path} failed: {e}", flush=True)
        return False


def upload_weights(job_key, path):
    """Stream a trained checkpoint to the platform (community / local-storage — no BYO S3) so the
    model is durable, downloadable, and servable to a fresh agent. Returns the server-assigned
    weights_key, or None on failure. The read side is download_media()."""
    try:
        with open(path, "rb") as f:
            r = S.post(BASE + f"/agent/tasks/{job_key}/weights",
                       files={"file": (os.path.basename(path), f, "application/octet-stream")},
                       timeout=1200)
        if r.status_code != 200:
            print(f"[agent] weights upload failed: HTTP {r.status_code}", flush=True)
            return None
        return (r.json() or {}).get("weights_key")
    except Exception as e:
        print(f"[agent] weights upload {path} failed: {e}", flush=True)
        return None


def _model_store_dir(weights_key):
    """Deterministic local dir for a model, derived from its S3 weights key."""
    safe = (weights_key or "").strip("/").replace("/", "__")
    return os.path.join(MODEL_STORE, safe)


# ───────────────────────── dataset assembly ─────────────────────────

def _s3_client(cfg):
    import boto3
    return boto3.client(
        "s3", endpoint_url=cfg.get("endpoint") or None,
        aws_access_key_id=cfg.get("access_key"), aws_secret_access_key=cfg.get("secret_key"),
        # Prefix-scoped platform creds are TEMPORARY STS creds — invalid without the session token.
        # None for static/BYO creds (backward-compatible).
        aws_session_token=cfg.get("session_token") or None,
        region_name=cfg.get("region") or None,
    )


def build_samples(job_key, payload, work):
    """Download the project's images from the user's S3, pair them with the universal DB
    annotations, and return samples [{file_name, src, w, h, objects | category}] for the
    backend's prepare_dataset."""
    import json
    from PIL import Image

    unique = payload.get("project_unique_name") or f"project_{payload['project_id']}"
    cfg = api("GET", f"/agent/tasks/{job_key}/credentials").get("storage")
    anns = api("GET", f"/agent/tasks/{job_key}/annotations").get("annotations", [])
    is_cls = payload.get("task") == "classification"

    img_dir = os.path.join(work, "images")
    os.makedirs(img_dir, exist_ok=True)
    s3 = _s3_client(cfg) if cfg else None
    prefix = (cfg.get("path_prefix").strip("/") + "/") if (cfg and cfg.get("path_prefix")) else ""
    bucket = cfg.get("bucket") if cfg else None

    samples = []
    for a in anns:
        raw = a["file_path"]
        # COMBINED datasets tag each annotation with a full image key '<unique>/images/<file>' so
        # images from several projects don't collide; single-project stays a bare basename.
        if "/images/" in raw:
            key = raw
            fn = raw.replace("/", "_")            # collision-safe local + dataset name
        else:
            fn = os.path.basename(raw)
            key = f"{unique}/images/{fn}"
        data = json.loads(a["data"]) if isinstance(a["data"], str) else a["data"]
        if is_cls and not data.get("label"):
            continue
        if not is_cls and not (data.get("objects")):
            continue
        local = os.path.join(img_dir, fn)
        if not os.path.exists(local):
            if s3:
                try:
                    s3.download_file(bucket, f"{prefix}{key}", local)
                except Exception as e:
                    # The platform's S3 endpoint isn't always reachable from the agent's network (a
                    # host-local published port, a private VPC, a proxy that breaks SigV4). Fall back
                    # to the media API we're already authenticated against — it follows the 302 to a
                    # presigned URL for BYO-S3 users, so it covers both storage modes.
                    if not download_media(key, local):
                        print(f"[agent] skip {fn}: {e}", flush=True)
                        continue
            elif not download_media(key, local):
                # No BYO S3 → platform stores images locally; fetch via the authenticated media API.
                print(f"[agent] skip {fn}: media fetch failed", flush=True)
                continue
        if not os.path.exists(local):
            continue
        try:
            with Image.open(local) as im:
                w, h = im.size
        except Exception:
            w = h = 0
        s = {"file_name": fn, "src": local, "w": w, "h": h}
        if is_cls:
            s["category"] = data["label"]
        else:
            s["objects"] = data["objects"]
        samples.append(s)
    return samples


# ───────────────────────── task execution ─────────────────────────

def run_task(task):
    job_key = task["job_key"]
    payload = task["payload"]
    print(f"[agent] claimed {job_key}: {payload.get('task')} size={payload.get('size')}", flush=True)

    def post_progress(d):
        try:
            r = api("POST", f"/agent/tasks/{job_key}/progress", json=d)
            return bool(r.get("cancel_requested"))
        except Exception as e:
            print(f"[agent] progress post failed: {e}", flush=True)
            return False

    cancelled = {"v": False}
    post_progress({"stage": "Preparing dataset", "progress": 5})

    work = tempfile.mkdtemp(prefix=f"agent_{job_key}_")
    try:
        from src.ml.backends import get_trainer_backend, TrainConfig
        samples = build_samples(job_key, payload, work)
        if not samples:
            api("POST", f"/agent/tasks/{job_key}/result",
                json={"ok": False, "error": "no usable samples (images/annotations) for the project"})
            return
        backend = get_trainer_backend(payload["task"], payload.get("arch"))
        ds_dir = os.path.join(work, "dataset")
        backend.prepare_dataset(samples, payload.get("categories") or [], ds_dir)
        cfg = TrainConfig(
            project_id=payload["project_id"], task=payload["task"],
            categories=payload.get("categories") or [], dataset_dir=ds_dir,
            out_dir=os.path.join(work, "out"), size=payload.get("size", "n"),
            epochs=int(payload.get("epochs", 50)), batch=int(payload.get("batch", 8)),
            img_size=int(payload.get("img_size", 640)), loss=payload.get("loss", "ce"),
            keypoint_config=payload.get("keypoint_config"),
        )

        def progress_cb(p):
            ep, tot = p.get("epoch"), p.get("total_epochs")
            pct = int(5 + 90 * (ep / tot)) if (ep and tot) else None
            if post_progress({"stage": f"Training epoch {ep}/{tot}", "progress": pct,
                              "epoch": ep, "total_epochs": tot, "metrics": p.get("metrics", {})}):
                cancelled["v"] = True

        res = backend.train(cfg, progress_cb=progress_cb, should_stop=lambda: cancelled["v"])

        # Persist the trained weights so the model survives and can be served later. BYO-S3 users
        # write to THEIR bucket; local-storage (community) users have no S3, so stream the checkpoint
        # to the platform, which stores it in the owner's blob store (local FS) — mirror of image
        # download_media. Same "<unique>/models/agent/<file>" key shape either way.
        weights_key = None
        unique = payload.get("project_unique_name") or f"project_{payload['project_id']}"
        cfg_s3 = api("GET", f"/agent/tasks/{job_key}/credentials").get("storage")
        if res.weights and os.path.exists(res.weights):
            if cfg_s3:
                prefix = (cfg_s3.get("path_prefix").strip("/") + "/") if cfg_s3.get("path_prefix") else ""
                weights_key = f"{unique}/models/agent/{os.path.basename(res.weights)}"
                try:
                    _s3_client(cfg_s3).upload_file(res.weights, cfg_s3["bucket"], f"{prefix}{weights_key}")
                except Exception as e:
                    # Write-side mirror of the image-download fallback: the platform's S3 endpoint is
                    # not always reachable from the agent's network. Stream the checkpoint to the
                    # platform instead — it stores it and returns the server-derived key. Losing the
                    # weights of a finished run to a network detail is the worst possible outcome.
                    print(f"[agent] weights S3 upload failed ({e}); streaming to the platform", flush=True)
                    weights_key = upload_weights(job_key, res.weights)
            else:
                weights_key = upload_weights(job_key, res.weights)

        # Keep the model locally (weights + stashed config + meta) so THIS agent can serve
        # inference later without re-downloading — keyed by the S3 weights_key. The train out_dir
        # holds the framework-native checkpoint + the config/meta the backend.predict() needs.
        if weights_key:
            try:
                store = _model_store_dir(weights_key)
                if os.path.isdir(store):
                    shutil.rmtree(store, ignore_errors=True)
                os.makedirs(os.path.dirname(store), exist_ok=True)
                shutil.move(os.path.join(work, "out"), store)
            except Exception as e:
                print(f"[agent] model store failed: {e}", flush=True)

        api("POST", f"/agent/tasks/{job_key}/result", json={
            "ok": True, "weights": weights_key, "metrics": res.metrics or {},
            "classes": res.classes or [], "fmt": res.fmt or "",
        })
        print(f"[agent] {job_key} done: {res.metrics}", flush=True)
    except KeyboardInterrupt:
        # Cooperative stop: the backend raises KeyboardInterrupt out of run_stream when the user
        # cancels training. It is a BaseException, so `except Exception` below would miss it —
        # letting it crash the whole agent process AND leave the job non-terminal. Catch it here,
        # report a terminal 'cancelled' result, and let the agent keep serving other work.
        print(f"[agent] {job_key} cancelled by user", flush=True)
        try:
            api("POST", f"/agent/tasks/{job_key}/result",
                json={"ok": False, "cancelled": True, "error": "stopped by user"})
        except Exception:
            pass
    except Exception as e:
        traceback.print_exc()
        try:
            api("POST", f"/agent/tasks/{job_key}/result", json={"ok": False, "error": str(e)[:500]})
        except Exception:
            pass
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _zeroshot_model():
    """Load GroundingDINO once (HF transformers, on the GPU) and cache it. `base` by default — bigger
    and more accurate than the browser's `tiny`, affordable on the agent's GPU. Env ZEROSHOT_MODEL
    overrides (e.g. IDEA-Research/grounding-dino-tiny for a lighter download)."""
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    model_id = os.environ.get("ZEROSHOT_MODEL", "IDEA-Research/grounding-dino-base")
    if _ZS["model"] is None or _ZS["id"] != model_id:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        proc = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        _ZS.update(model=model, proc=proc, id=model_id, device=device)
        print(f"[agent] zero-shot model ready: {model_id} on {device}", flush=True)
    return _ZS["model"], _ZS["proc"], _ZS["device"]


def _zs_iou(a, b):
    """IoU of two normalized xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    uni = ua + ub - inter
    return inter / uni if uni > 0 else 0.0


def _zs_nms(objs, same_iou=0.5, cross_iou=0.55, max_keep=300):
    """Greedy NMS over canonical cxcywh objects — open-vocab detectors emit many near-duplicate boxes
    (and the same object fires under several prompts). Mirrors the browser's browserDetect nms()."""
    def xyxy(o):
        b = o["bbox"]
        return (b["x_center"] - b["width"] / 2, b["y_center"] - b["height"] / 2,
                b["x_center"] + b["width"] / 2, b["y_center"] + b["height"] / 2)
    kept = []
    for o in sorted(objs, key=lambda o: o.get("score") or 0.0, reverse=True):
        if len(kept) >= max_keep:
            break
        ob = xyxy(o)
        dup = False
        for k in kept:
            thr = same_iou if k["category"] == o["category"] else cross_iou
            if _zs_iou(ob, xyxy(k)) > thr:
                dup = True
                break
        if not dup:
            kept.append(o)
    return kept


def _zeroshot_detect(model, proc, device, image_path, queries, box_th, text_th):
    """Run GroundingDINO once per prompt (batch-1 text encoder, like the browser) so each box's
    category is unambiguous; return canonical cxcywh objects, NMS-deduped."""
    import torch
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    objs = []
    for category, prompt in queries:
        inputs = proc(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        try:
            res = proc.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=box_th, text_threshold=text_th,
                target_sizes=[(H, W)])[0]
        except TypeError:   # older transformers used box_threshold=
            res = proc.post_process_grounded_object_detection(
                out, inputs.input_ids, box_threshold=box_th, text_threshold=text_th,
                target_sizes=[(H, W)])[0]
        for box, score in zip(res["boxes"].tolist(), res["scores"].tolist()):
            x1, y1, x2, y2 = box
            bw, bh = (x2 - x1) / W, (y2 - y1) / H
            if bw <= 0 or bh <= 0:
                continue
            objs.append({"category": category, "score": float(score),
                         "bbox": {"x_center": ((x1 + x2) / 2) / W, "y_center": ((y1 + y2) / 2) / H,
                                  "width": bw, "height": bh}})
    return _zs_nms(objs)


# ── segmentation: GroundingDINO boxes → SAM masks → polygons (the old GroundedSAM, on the GPU) ──
_SAM = {"model": None, "proc": None, "id": None, "device": "cpu"}


def _sam_model():
    """Load SAM once (HF transformers, on the GPU) and cache it. Box-prompted: each GroundingDINO box
    becomes a SAM prompt → a mask we trace to a polygon. Env ZEROSHOT_SAM_MODEL overrides."""
    import torch
    from transformers import SamModel, SamProcessor
    model_id = os.environ.get("ZEROSHOT_SAM_MODEL", "facebook/sam-vit-base")
    if _SAM["model"] is None or _SAM["id"] != model_id:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _SAM["proc"] = SamProcessor.from_pretrained(model_id)
        _SAM["model"] = SamModel.from_pretrained(model_id).to(device).eval()
        _SAM.update(id=model_id, device=device)
        print(f"[agent] zero-shot SAM ready: {model_id} on {device}", flush=True)
    return _SAM["model"], _SAM["proc"], _SAM["device"]


def _mask_to_polygon(mask_uint8, W, H, eps_frac=0.004):
    """Largest external contour of a binary mask → simplified, normalized flat polygon [x1,y1,...]
    (the editor's polygon shape). Mirrors the browser's maskToPolygon."""
    import cv2
    cnts, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 4:
        return None
    approx = cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    flat = []
    for x, y in approx:
        flat.append(float(x) / W)
        flat.append(float(y) / H)
    return flat


def _sam_segment(sam, sam_proc, device, image_path, objs):
    """Add a normalized `polygon` to each detection by prompting SAM with its box. Boxes with no
    usable mask keep just their box. All boxes go in one forward (batched prompts)."""
    if not objs:
        return objs
    import torch
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    boxes_abs = [[(o["bbox"]["x_center"] - o["bbox"]["width"] / 2) * W,
                  (o["bbox"]["y_center"] - o["bbox"]["height"] / 2) * H,
                  (o["bbox"]["x_center"] + o["bbox"]["width"] / 2) * W,
                  (o["bbox"]["y_center"] + o["bbox"]["height"] / 2) * H] for o in objs]
    inputs = sam_proc(image, input_boxes=[boxes_abs], return_tensors="pt").to(device)
    with torch.no_grad():
        out = sam(**inputs)
    masks = sam_proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
    scores = out.iou_scores.cpu()[0]   # (N, 3) — SAM emits 3 masks/box; pick the highest-IoU one
    result = []
    for n, o in enumerate(objs):
        try:
            best = int(scores[n].argmax())
            m = masks[n, best].numpy().astype("uint8")
            poly = _mask_to_polygon(m, W, H)
        except Exception:
            poly = None
        result.append({**o, "polygon": poly} if poly else o)
    return result


# ── classification: CLIP zero-shot over the category names (the old server CLIP, on the GPU) ──
_CLIP = {"pipe": None, "id": None}


def _clip_classify(image_path, categories):
    """Top-1 zero-shot classification of an image against the category names (CLIP). Returns a single
    canonical object [{category, score}] — the backend stores it as {label}. Env ZEROSHOT_CLS_MODEL."""
    import torch
    from PIL import Image
    from transformers import pipeline
    model_id = os.environ.get("ZEROSHOT_CLS_MODEL", "openai/clip-vit-base-patch32")
    if _CLIP["pipe"] is None or _CLIP["id"] != model_id:
        _CLIP["pipe"] = pipeline("zero-shot-image-classification", model=model_id,
                                 device=0 if torch.cuda.is_available() else -1)
        _CLIP["id"] = model_id
        print(f"[agent] zero-shot CLIP ready: {model_id}", flush=True)
    out = _CLIP["pipe"](Image.open(image_path).convert("RGB"), candidate_labels=list(categories))
    if not out:
        return []
    top = out[0]
    return [{"category": top["label"], "score": float(top["score"])}]


def run_zeroshot(task):
    """Open-vocabulary zero-shot detection on the agent's GPU (GroundingDINO). No trained weights:
    detect the project's category prompts in each image and return canonical objects — written back
    as annotations by the backend (source='model', never clobbering manual work), like /auto-label."""
    job_key = task["job_key"]
    payload = task["payload"]
    print(f"[agent] claimed {job_key}: zero-shot {payload.get('task')}", flush=True)

    def post_progress(d):
        try:
            r = api("POST", f"/agent/tasks/{job_key}/progress", json=d)
            return bool(r.get("cancel_requested"))
        except Exception:
            return False

    tt = payload.get("task") or "detection"
    is_cls = tt == "classification"
    is_seg = tt in ("segmentation", "semantic")
    post_progress({"stage": "Loading model", "progress": 3})
    work = tempfile.mkdtemp(prefix=f"agent_zeroshot_{job_key}_")
    try:
        cats = payload.get("categories") or []
        conf = float(payload.get("conf", 0.3))
        text_th = float(payload.get("text_threshold", 0.25))
        # Classification → CLIP over the category names (no boxes). Everything else → GroundingDINO
        # boxes (+ SAM masks for segmentation). Load only what this task needs.
        model = proc = device = sam = sam_proc = None
        queries = []
        if is_cls:
            if not cats:
                api("POST", f"/agent/tasks/{job_key}/result",
                    json={"ok": False, "error": "no categories for zero-shot classification"})
                return
        else:
            model, proc, device = _zeroshot_model()
            # Prompts: category + optional per-category prompt override. GroundingDINO wants a lowercase
            # phrase ending with " ." — one prompt per forward keeps each box's category unambiguous.
            overrides = payload.get("prompts") or {}     # {category: "prompt override"}
            for c in cats:
                raw = (overrides.get(c) or c or "").strip()
                if raw:
                    queries.append((c, raw.lower().rstrip(".").strip() + " ."))
            if not queries:
                api("POST", f"/agent/tasks/{job_key}/result",
                    json={"ok": False, "error": "no categories/prompts for zero-shot"})
                return
            if is_seg:
                sam, sam_proc, _ = _sam_model()

        cfg = api("GET", f"/agent/tasks/{job_key}/credentials").get("storage")
        s3 = _s3_client(cfg) if cfg else None
        prefix = (cfg.get("path_prefix").strip("/") + "/") if (cfg and cfg.get("path_prefix")) else ""
        bucket = cfg.get("bucket") if cfg else None
        unique = payload.get("project_unique_name") or f"project_{payload['project_id']}"

        img_dir = os.path.join(work, "images")
        os.makedirs(img_dir, exist_ok=True)
        images = payload.get("images") or []
        results, cancelled = [], False
        for i, fn in enumerate(images):
            # eval / combined datasets pass a full key '<unique>/images/<file>'; legacy = basename.
            base = fn.replace("/", "_") if "/images/" in fn else os.path.basename(fn)
            key = fn if "/images/" in fn else f"{unique}/images/{os.path.basename(fn)}"
            local = os.path.join(img_dir, base)
            if not os.path.exists(local):
                if s3:
                    try:
                        s3.download_file(bucket, f"{prefix}{key}", local)
                    except Exception as e:
                        # Same S3-unreachable fallback as the training dataset build.
                        if not download_media(key, local):
                            print(f"[agent] zs skip {base}: {e}", flush=True)
                            continue
                elif not download_media(key, local):
                    print(f"[agent] zs skip {base}: media fetch failed", flush=True)
                    continue
            if not os.path.exists(local):
                continue
            try:
                if is_cls:
                    objs = _clip_classify(local, cats)
                else:
                    objs = _zeroshot_detect(model, proc, device, local, queries, conf, text_th)
                    if tt == "count":     # each box → a point at its centre (counting shape)
                        objs = [{"category": o["category"], "score": o["score"],
                                 "point": {"x": o["bbox"]["x_center"], "y": o["bbox"]["y_center"]}}
                                for o in objs]
                    elif tt == "obb":     # axis-aligned box → oriented box at angle 0 (annotator rotates)
                        objs = [{"category": o["category"], "score": o["score"],
                                 "obb": {"cx": o["bbox"]["x_center"], "cy": o["bbox"]["y_center"],
                                         "w": o["bbox"]["width"], "h": o["bbox"]["height"], "angle": 0}}
                                for o in objs]
                    elif is_seg:          # box → SAM mask → polygon
                        objs = _sam_segment(sam, sam_proc, device, local, objs)
            except Exception as e:
                print(f"[agent] zero-shot failed on {base}: {e}", flush=True)
                objs = []
            results.append({"file_path": fn, "objects": objs})
            if post_progress({"stage": f"Zero-shot {i + 1}/{len(images)}",
                              "progress": int(5 + 90 * (i + 1) / max(1, len(images)))}):
                cancelled = True
                break

        api("POST", f"/agent/tasks/{job_key}/result", json={
            "ok": True, "predictions": results, "count": len(results),
            "cancelled": cancelled, "task": payload.get("task")})
        print(f"[agent] {job_key} zero-shot done: {len(results)} image(s)", flush=True)
    except Exception as e:
        traceback.print_exc()
        try:
            api("POST", f"/agent/tasks/{job_key}/result", json={"ok": False, "error": str(e)[:500]})
        except Exception:
            pass
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_infer(task):
    """Run a trained permissive model on a set of images and return canonical objects per image.
    The model is located in the agent's local store (it trained it) or downloaded from the user's
    S3; images are pulled from the user's S3; backend.predict() yields the universal DB shape."""
    job_key = task["job_key"]
    payload = task["payload"]
    # Open-vocabulary zero-shot (GroundingDINO) — no trained weights. Same JOB_TYPE_INFER queue /
    # result path (predictions → annotations), but a different model, so it forks here.
    if payload.get("zero_shot"):
        return run_zeroshot(task)
    print(f"[agent] claimed {job_key}: infer {payload.get('task')}", flush=True)

    def post_progress(d):
        try:
            r = api("POST", f"/agent/tasks/{job_key}/progress", json=d)
            return bool(r.get("cancel_requested"))
        except Exception:
            return False

    post_progress({"stage": "Loading model", "progress": 5})
    work = tempfile.mkdtemp(prefix=f"agent_infer_{job_key}_")
    try:
        from src.ml.backends import get_inference_backend
        weights_key = payload["weights_key"]
        unique = payload.get("project_unique_name") or f"project_{payload['project_id']}"
        cfg = api("GET", f"/agent/tasks/{job_key}/credentials").get("storage")
        s3 = _s3_client(cfg) if cfg else None
        prefix = (cfg.get("path_prefix").strip("/") + "/") if (cfg and cfg.get("path_prefix")) else ""
        bucket = cfg.get("bucket") if cfg else None

        # locate the model: this agent's persistent store (it trained it), else pull the weights —
        # from the user's S3, or (local-storage community) from the platform via the media route.
        store = _model_store_dir(weights_key)
        weights = os.path.join(store, os.path.basename(weights_key))
        if not os.path.exists(weights):
            os.makedirs(store, exist_ok=True)
            if s3:
                s3.download_file(bucket, f"{prefix}{weights_key}", weights)
            else:
                download_media(weights_key, weights)
        if not os.path.exists(weights):
            api("POST", f"/agent/tasks/{job_key}/result",
                json={"ok": False, "error": f"model weights not found (key {weights_key})"})
            return

        backend = get_inference_backend(payload["task"], payload.get("arch"))
        conf = float(payload.get("conf", 0.3))
        img_dir = os.path.join(work, "images")
        os.makedirs(img_dir, exist_ok=True)
        images = payload.get("images") or []
        # Active-learning scoring: same predict pass, but instead of writing predicted annotations
        # we report a per-image UNCERTAINTY (1 - top confidence; no detection = 1.0 = most uncertain)
        # so the UI can surface the least-confident images to label next.
        score_mode = payload.get("mode") == "score"
        results, scores, cancelled = [], [], False
        # perf telemetry (used by held-out eval to compare candidates on latency + size)
        try:
            model_size_mb = round(os.path.getsize(weights) / 1e6, 2)
        except Exception:
            model_size_mb = None
        predict_ms_total, predict_n = 0.0, 0
        for i, fn in enumerate(images):
            # eval / combined datasets pass a full image key '<unique>/images/<file>' (fetch by it,
            # report it back so metric matching is collision-safe); legacy passes a bare basename.
            if "/images/" in fn:
                key = fn
                base = fn.replace("/", "_")
            else:
                base = os.path.basename(fn)
                key = f"{unique}/images/{base}"
            local = os.path.join(img_dir, base)
            if not os.path.exists(local):
                if s3:
                    try:
                        s3.download_file(bucket, f"{prefix}{key}", local)
                    except Exception as e:
                        # Same S3-unreachable fallback as the training dataset build above.
                        if not download_media(key, local):
                            print(f"[agent] infer skip {base}: {e}", flush=True)
                            continue
                elif not download_media(key, local):
                    print(f"[agent] infer skip {base}: media fetch failed", flush=True)
                    continue
            if not os.path.exists(local):
                continue
            try:
                _t0 = time.perf_counter()
                objs = backend.predict(weights, local, conf=conf)
                predict_ms_total += (time.perf_counter() - _t0) * 1000.0
                predict_n += 1
            except Exception as e:
                print(f"[agent] predict failed on {base}: {e}", flush=True)
                objs = []
            if score_mode:
                top = max((float(o.get("score") or 0.0) for o in objs), default=0.0)
                scores.append({"file_path": fn, "uncertainty": round(1.0 - top, 4)})
            else:
                results.append({"file_path": fn, "objects": objs})
            if post_progress({"stage": f"{'Scoring' if score_mode else 'Inference'} {i + 1}/{len(images)}",
                              "progress": int(5 + 90 * (i + 1) / max(1, len(images)))}):
                cancelled = True
                break

        if score_mode:
            api("POST", f"/agent/tasks/{job_key}/result", json={
                "ok": True, "scores": scores, "count": len(scores), "cancelled": cancelled})
            print(f"[agent] {job_key} score done: {len(scores)} image(s)", flush=True)
        else:
            api("POST", f"/agent/tasks/{job_key}/result", json={
                "ok": True, "predictions": results, "count": len(results),
                "cancelled": cancelled, "task": payload.get("task"),
                "latency_ms": round(predict_ms_total / predict_n, 2) if predict_n else None,
                "model_size_mb": model_size_mb,
            })
            print(f"[agent] {job_key} infer done: {len(results)} image(s)", flush=True)
    except Exception as e:
        traceback.print_exc()
        try:
            api("POST", f"/agent/tasks/{job_key}/result", json={"ok": False, "error": str(e)[:500]})
        except Exception:
            pass
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_export(task):
    """Export a trained permissive model to ONNX and upload it next to the weights in the user's
    S3. The model is located in the agent's local store (it trained it) or downloaded from S3."""
    job_key = task["job_key"]
    payload = task["payload"]
    print(f"[agent] claimed {job_key}: export {payload.get('task')}", flush=True)

    def post_progress(d):
        try:
            api("POST", f"/agent/tasks/{job_key}/progress", json=d)
        except Exception:
            pass

    post_progress({"stage": "Exporting ONNX", "progress": 10})
    work = tempfile.mkdtemp(prefix=f"agent_export_{job_key}_")
    try:
        from src.ml.backends import get_inference_backend
        weights_key = payload["weights_key"]
        cfg = api("GET", f"/agent/tasks/{job_key}/credentials").get("storage")
        s3 = _s3_client(cfg) if cfg else None
        prefix = (cfg.get("path_prefix").strip("/") + "/") if (cfg and cfg.get("path_prefix")) else ""
        bucket = cfg.get("bucket") if cfg else None

        store = _model_store_dir(weights_key)
        store_weights = os.path.join(store, os.path.basename(weights_key))
        if not os.path.exists(store_weights) and s3:
            os.makedirs(store, exist_ok=True)
            s3.download_file(bucket, f"{prefix}{weights_key}", store_weights)
        if not os.path.exists(store_weights):
            api("POST", f"/agent/tasks/{job_key}/result",
                json={"ok": False, "error": f"model weights not found (key {weights_key})"})
            return

        # Export from a CLEAN dir: the store dir name embeds the weights filename ('...best_stg1.pth'),
        # and some repo export tools do a global '.pth'->'.onnx' path rewrite that would corrupt the
        # directory component. Copy the model FILES (weights + stashed config/meta) into work/model/.
        model_dir = os.path.join(work, "model")
        os.makedirs(model_dir, exist_ok=True)
        for f in os.listdir(store):
            sp = os.path.join(store, f)
            if os.path.isfile(sp):
                shutil.copy(sp, os.path.join(model_dir, f))
        weights = os.path.join(model_dir, os.path.basename(weights_key))

        backend = get_inference_backend(payload["task"], payload.get("arch"))
        out_onnx = os.path.join(work, os.path.splitext(os.path.basename(weights_key))[0] + ".onnx")
        backend.export_onnx(weights, out_onnx)
        post_progress({"stage": "Uploading ONNX", "progress": 80})

        onnx_key = None
        if os.path.exists(out_onnx):
            if s3:
                onnx_key = os.path.splitext(weights_key)[0] + ".onnx"
                try:
                    s3.upload_file(out_onnx, bucket, f"{prefix}{onnx_key}")
                except Exception as e:
                    print(f"[agent] ONNX S3 upload failed ({e}); streaming to the platform", flush=True)
                    onnx_key = upload_weights(job_key, out_onnx)
            else:
                # No BYO S3 (community / local storage). The weights endpoint places an export job's
                # upload next to the model's existing weights, so the ONNX is durable there too —
                # without this the export "succeeded" while storing nothing.
                onnx_key = upload_weights(job_key, out_onnx)

        api("POST", f"/agent/tasks/{job_key}/result", json={
            "ok": True, "onnx": onnx_key, "fmt": payload.get("task"),
        })
        print(f"[agent] {job_key} export done: {onnx_key}", flush=True)
    except Exception as e:
        traceback.print_exc()
        try:
            api("POST", f"/agent/tasks/{job_key}/result", json={"ok": False, "error": str(e)[:500]})
        except Exception:
            pass
    finally:
        shutil.rmtree(work, ignore_errors=True)


_DISPATCH = {"infer": run_infer, "export": run_export}


def main():
    if not TOKEN:
        _login()   # bootstrap a token from creds when none was passed
    print(f"[agent] runner up: id={AGENT_ID} url={BASE}", flush=True)
    while True:
        try:
            r = api("POST", "/agent/poll", json={"agent_id": AGENT_ID}, timeout=40)
            task = r.get("task")
            if task:
                _DISPATCH.get(task.get("kind"), run_task)(task)
            # else: long-poll returned empty → loop immediately (server held the connection)
        except requests.HTTPError as e:
            print(f"[agent] poll http error: {e}", flush=True)
            time.sleep(10)
        except Exception as e:
            print(f"[agent] poll error: {e}", flush=True)
            time.sleep(10)


if __name__ == "__main__":
    main()
