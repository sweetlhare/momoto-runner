"""Shared execution environment for the model backends.

The heavy model code (D-FINE / DETRPose / D-FINE-seg repos + their torch env, and the
ConvNeXtV2/timm trainer) lives in a dedicated "model integration" environment so the
API process stays light and free of torch-version conflicts. Backends shell out to that
env's python. Location is configurable via MODEL_INTEGRATION_ROOT (default: the dev
sandbox); in production this points at the training-agent install (venv + cloned repos).
"""
import os
import sys
import subprocess
import collections

MODEL_ROOT = os.environ.get("MODEL_INTEGRATION_ROOT", "/opt/model-integration")


def venv_python() -> str:
    p = os.path.join(MODEL_ROOT, ".venv", "bin", "python")
    return p if os.path.exists(p) else sys.executable


def repo(name: str) -> str:
    return os.path.join(MODEL_ROOT, name)


def run_stream(cmd, cwd, on_line=None, env=None, tail=120):
    """Run a subprocess, streaming combined stdout/stderr line-by-line to on_line.
    Returns (returncode, tail_text)."""
    buf = collections.deque(maxlen=tail)
    # AGENT_LOG_TRAINER=1 echoes the child trainer's stdout/stderr to the agent log — invaluable
    # for diagnosing a training that runs but never reports progress (a progress-regex mismatch),
    # or any silent failure on a customer's box. Off by default to keep the agent log clean.
    log_trainer = bool(os.environ.get("AGENT_LOG_TRAINER"))
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env={**os.environ, **(env or {})},
    )
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            buf.append(line)
            if log_trainer:
                print(f"[trainer] {line}", flush=True)
            if on_line:
                try:
                    on_line(line)
                except Exception:
                    pass
        proc.wait()
    except BaseException:
        # A cooperative stop surfaces as KeyboardInterrupt raised from on_line. It is a
        # BaseException, so it escapes the inner Exception guard above and lands here. We must
        # STOP the heavy child — the old `finally: proc.wait()` blocked until training finished
        # on its own, making every 'stop' a no-op that hung the worker for the whole run.
        # SIGINT first so a well-behaved trainer (e.g. D-FINE-seg) can save its best checkpoint
        # and exit; escalate to SIGKILL if it ignores the interrupt.
        import signal as _sig
        try:
            proc.send_signal(_sig.SIGINT)
        except Exception:
            pass
        try:
            proc.wait(timeout=15)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        raise
    return proc.returncode, "\n".join(buf)


def resolve_device(device: str) -> str:
    if device and device != "auto":
        return device
    try:
        import torch  # noqa
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def find_weight(out_dir: str, globs) -> str:
    """Return the first existing checkpoint among the backend's preferred filenames."""
    for name in globs:
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            return p
    # fallback: newest *.pth/*.pt in the dir
    import glob as _g
    cands = sorted(_g.glob(os.path.join(out_dir, "*.pth")) + _g.glob(os.path.join(out_dir, "*.pt")),
                   key=os.path.getmtime, reverse=True)
    return cands[0] if cands else None
