# momoto Runner — self-contained BYO-compute image (runs training/inference on YOUR GPU).
#
# Bundles the permissive training/inference stack (RF-DETR / D-FINE / DETRPose + ConvNeXtV2 /
# MobileNetV4 / EfficientViT via timm in a shared venv + D-FINE-seg in its own uv-locked env) +
# the platform's src/ml backends + the outbound runner. The user picks the model family per
# task. The user runs this on THEIR GPU; the host needs only the NVIDIA driver + Container
# Toolkit (the torch wheels bundle their CUDA runtime):
#
#   docker run -d --gpus all --ipc=host \
#     -e MOTOMOTO_URL=https://app.example.com -e MOTOMOTO_TOKEN=<token> \
#     -e MOTOMOTO_USER=<login> -e MOTOMOTO_PASS=<pass>   # for 401 token refresh \
#     momoto-runner
#
# Versions are pinned to the EXACT stack verified on the platform's GPU box:
#   shared venv: Python 3.12 + torch 2.11.0+cu130 (D-FINE / DETRPose / ConvNeXtV2)
#   D-FINE-seg : its own uv env — Python 3.11 + torch 2.9.0+cu128 + hydra (from uv.lock)
#
# Build (context = repo root):  docker build -t momoto-runner .
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    MODEL_INTEGRATION_ROOT=/opt/model-integration \
    VENV=/opt/model-integration/.venv \
    PYTHONPATH=/app
ENV PATH=$VENV/bin:/root/.local/bin:$PATH

# System deps: Python 3.12 (ubuntu24.04 default), git, OpenGL/glib for opencv, curl for uv.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip git ca-certificates curl \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# uv — used ONLY to build the D-FINE-seg repo's own locked env (the shared venv lacks hydra).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 1) Permissive model repos. ConvNeXtV2 uses timm directly (no repo).
WORKDIR /opt/model-integration
RUN git clone --depth 1 https://github.com/Peterande/D-FINE.git D-FINE && \
    git clone --depth 1 https://github.com/SebastianJanampa/DETRPose.git DETRPose && \
    git clone --depth 1 https://github.com/ArgoHA/D-FINE-seg.git D-FINE-seg

# 2a) Shared venv + build tooling + torch (own layer so later dep fixes don't re-download the
#     ~3 GB torch). venv_python() resolves to $MODEL_INTEGRATION_ROOT/.venv (platform layout).
#     torch 2.11+cu130 wheels bundle CUDA 13 — only the host NVIDIA driver is needed at runtime.
#     numpy + cython are installed FIRST so the source-built coco tools can find them.
RUN python3 -m venv "$VENV" && pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy==2.4.6 cython && \
    pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 \
        --index-url https://download.pytorch.org/whl/cu130

# 2a') C toolchain + Python headers for the source-built Cython extensions (xtcocotools'
#      _mask.pyx needs gcc AND Python.h from python3-dev). Kept as its own layer AFTER torch so
#      it doesn't invalidate the (large) torch layer's cache.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential python3-dev && \
    rm -rf /var/lib/apt/lists/*

# 2b) Remaining shared-stack deps. xtcocotools has no wheel for this combo → it builds from
#     source (Cython + C) and needs numpy/cython at build time, so install it WITHOUT build
#     isolation (numpy is already present from 2a) — fixes the earlier 'No module named numpy'.
RUN pip install --no-cache-dir \
        timm==1.0.26 transformers==5.7.0 faster-coco-eval==1.7.2 pycocotools==2.0.11 \
        cloudpickle==3.1.2 calflops==0.3.2 omegaconf==2.3.0 iopath==0.1.10 scipy==1.17.1 \
        PyYAML pillow==12.2.0 opencv-python-headless==4.13.0.92 onnx==1.21.0 \
        onnxruntime-gpu==1.25.0 onnxscript onnxsim loguru tensorboard requests boto3 && \
    pip install --no-cache-dir --no-build-isolation xtcocotools==1.14.3 && \
    (pip install --no-cache-dir -r D-FINE/requirements.txt || true) && \
    (pip install --no-cache-dir -r DETRPose/requirements.txt || true)

# 2c) RF-DETR (Roboflow, Apache-2.0) — the preferred real-time DETR for detection. The `[train]`
#     extra pulls pytorch_lightning (its trainer) + torchmetrics/albumentations/peft; the heavy
#     wandb/mlflow/clearml `[loggers]` extra is intentionally OMITTED (tensorboard is already in
#     2b). Then PRE-DOWNLOAD the COCO-pretrained checkpoints so an OFFLINE / air-gapped agent can
#     fine-tune without reaching Roboflow at runtime (the agent has no outbound). Weights cache
#     under $HOME/.roboflow/models (root → /root); the runtime user is root, same HOME. Large is
#     left to download on first use (rarely picked, biggest file). Own layer for cache locality.
RUN pip install --no-cache-dir "rfdetr[train]==1.8.3" && \
    CUDA_VISIBLE_DEVICES="" python -c "import rfdetr; [getattr(rfdetr, c)() for c in \
        ('RFDETRNano','RFDETRSmall','RFDETRMedium','RFDETRBase')]" || \
    echo "[build] RF-DETR pretrain prefetch incomplete — will fetch on first use if online"

# cmake — the seg repo's locked onnxsim==0.4.36 has no wheel and builds its C++ ext via cmake.
#   Own layer (after 2b) so it doesn't re-run the shared-deps install.
RUN apt-get update && apt-get install -y --no-install-recommends cmake && \
    rm -rf /var/lib/apt/lists/*

# 3) D-FINE-seg's OWN locked env (its own torch 2.9 cu128 + hydra, which the shared venv lacks),
#    built at image-build time so the first seg training isn't slow. The seg backend prefers
#    <repo>/.venv. uv reuses the image's system Python 3.12 — one Python version across the whole
#    image (the seg repo's requires-python is >=3.11,<3.14, so 3.12 is in range).
RUN cd /opt/model-integration/D-FINE-seg && uv sync --frozen --no-dev

# 4) Platform ML package (permissive backends + db_to_coco) + the outbound runner.
COPY src/ml/backends /app/src/ml/backends
COPY src/ml/datasets /app/src/ml/datasets
COPY src/ml/__init__.py /app/src/ml/__init__.py
COPY src/__init__.py /app/src/__init__.py
COPY runner.py /app/runner.py

WORKDIR /app
ENTRYPOINT ["/opt/model-integration/.venv/bin/python", "/app/runner.py"]
