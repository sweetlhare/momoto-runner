"""Agent-side regression test for Batch B: the GPU agent's S3 client MUST thread the STS
session_token, or every scoped-cred S3 op 403s (the hardened path would be dead-on-arrival).

runner/runner.py is a standalone script (reads MOTOMOTO_URL at import); we load it in isolation
by file path so this lives with the backend unit suite without making `runner` a package."""
import importlib.util
import os
import sys

os.environ.setdefault("SECRET_KEY", "unit-test-secret-agent-s3")
os.environ.setdefault("MOTOMOTO_URL", "http://localhost:0")  # runner.py reads this at import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUNNER_PY = os.path.join(_REPO, "runner.py")

_spec = importlib.util.spec_from_file_location("agent_runner_under_test", _RUNNER_PY)
runner = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = runner
_spec.loader.exec_module(runner)


def _capture_client(monkeypatch):
    import boto3
    cap = {}
    monkeypatch.setattr(boto3, "client", lambda svc, **kw: cap.update(kw) or object())
    return cap


def test_agent_s3_client_threads_session_token(monkeypatch):
    cap = _capture_client(monkeypatch)
    runner._s3_client({"endpoint": "http://e:9000", "access_key": "TMP",
                       "secret_key": "sec", "region": "r", "session_token": "tok123"})
    assert cap["aws_session_token"] == "tok123"


def test_agent_s3_client_none_token_for_static_creds(monkeypatch):
    cap = _capture_client(monkeypatch)
    runner._s3_client({"endpoint": "http://e:9000", "access_key": "AK", "secret_key": "sec"})
    assert cap.get("aws_session_token") is None
