# momoto Runner

Runs **training, inference, auto-labelling and ONNX export on your own GPU** (laptop,
workstation, or a rented box) instead of ours. The runner connects **outbound** to the platform
— no inbound ports, works behind NAT/firewall — claims your queued jobs, runs the fully
**permissive** stack (all four task types — Apache/MIT, **no AGPL YOLO**), writes weights to
**your** S3, and streams progress back:

| task | backend | licence |
|------|---------|---------|
| detection | RF-DETR (primary) · D-FINE | Apache-2.0 |
| segmentation | D-FINE-seg | Apache-2.0 |
| keypoints / pose | DETRPose · RF-DETR Pose | Apache-2.0 |
| classification | ConvNeXtV2 · MobileNetV4 · EfficientViT (timm) | MIT/Apache |

The image is **self-contained**: it bundles the model repos + a shared venv (Python 3.12 /
torch 2.11 cu130 for D-FINE / DETRPose / ConvNeXtV2) and D-FINE-seg's own uv-locked env
(Python 3.11 / torch 2.9 cu128 / hydra). The host needs only the NVIDIA driver + Container
Toolkit — the torch wheels bundle their CUDA runtime, so there are no host torch conflicts.

## Run

```bash
docker run -d --gpus all --ipc=host --restart unless-stopped \
  -e MOTOMOTO_URL=https://app.momoto.example \
  -e MOTOMOTO_TOKEN=<your account token> \
  -e MOTOMOTO_USER=<login> -e MOTOMOTO_PASS=<password> \
  momoto-runner
```

- `--ipc=host` — PyTorch DataLoader shared memory (the default 64 MB `/dev/shm` hangs the loader).
- `MOTOMOTO_USER`/`MOTOMOTO_PASS` — let the runner re-authenticate when the 7-day token expires
  (long-running agents); omit to run with just the token until it lapses.

No GPU? Drop `--gpus all` to run on CPU (slow), or rent a GPU box and run the same command there.

### Idle cost: none

A connected runner that has no work just holds an outbound long-poll — **no GPU use and no model
in memory while idle**. Models are loaded only when a job is claimed (in a subprocess) and freed
when it finishes, so `nvidia-smi` shows the runner at ~0 between jobs. Leave it running
`--restart unless-stopped`; it wakes up when you queue a job and goes back to idle after.

## Build

```bash
# from the repo root (build context = repo root so the ml/ package + runner are bundled)
docker build -t momoto-runner .
```

## How it works

```
platform  ──queue──►  BackgroundJob(job_type=agent_train, status=queued)
   ▲                            │
   │ POST /agent/poll (long-poll, account token)
   │                            ▼
 runner  ──claim── runs:  GET /agent/tasks/{key}/annotations  (universal DB labels)
   │                     GET /agent/tasks/{key}/credentials  (your S3 creds, just-in-time)
   │                     download images from your S3 → db_to_coco / ImageFolder
   │                     src/ml/backends.get_trainer_backend(task).train(...)
   │  POST /agent/tasks/{key}/progress  (epoch/metrics; response carries the cancel flag)
   │                     upload weights → your S3
   └─ POST /agent/tasks/{key}/result   (metrics + S3 weights key)
```

- **Auth:** the account token (same as the web app), scoped to your user. The runner only
  ever sees your own tasks / data.
- **Stop:** the platform sets a cooperative cancel flag; the runner reads it in each progress
  response and stops between epochs.
- **Durability:** tasks are durable DB rows; a dead runner's task is reaped and re-queued.
- **`'server'` in Compute settings** routes training to your connected runner (this), keeping
  the platform free of GPU/compute cost (local-first).

## Status

Verified end-to-end on a real GPU: the protocol (queue / claim / progress / result / cancel),
training (all backends), inference, auto-label, and ONNX export all run through the runner.
A cooperative **Stop** kills the training child and the runner keeps serving (it does not crash).
Weights + exported ONNX land in your S3; the platform's deploy/microservice export pulls that
ONNX back to bundle a permissive (onnxruntime) deployable artifact.
