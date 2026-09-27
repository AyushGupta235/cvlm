"""Minimal RunPod REST client (https://rest.runpod.io/v1) and credential loading.

The API key is read from the environment or the project's gitignored .env, held only in the
session's Authorization header, and never printed or written anywhere.
"""
from __future__ import annotations

import os
from pathlib import Path

import requests

BASE = "https://rest.runpod.io/v1"
ROOT = Path(__file__).resolve().parent.parent
POD_PREFIX = "cvlm-"             # every pod this project creates is named cvlm-<run>; `pods.py` only touches these


class RunPodError(RuntimeError):
    pass


def load_env(path: Path = ROOT / ".env") -> None:
    """KEY=VALUE lines from .env into os.environ; exported variables win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if v.strip():
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def api_key() -> str:
    load_env()
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise RunPodError("RUNPOD_API_KEY is not set. Put it in .env (see .env.example) or export it.")
    return key


class RunPod:
    def __init__(self, key: str, session: requests.Session | None = None, timeout: float = 30.0):
        self.s = session or requests.Session()
        self.s.headers["Authorization"] = f"Bearer {key}"
        self.timeout = timeout

    def __repr__(self) -> str:                 # never show the key
        return "RunPod(<key hidden>)"

    def _req(self, method: str, path: str, **kw):
        r = self.s.request(method, BASE + path, timeout=self.timeout, **kw)
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            raise RunPodError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json() if r.content and r.headers.get("content-type", "").startswith("application/json") else {}

    def create_pod(self, payload: dict) -> dict:
        pod = self._req("POST", "/pods", json=payload)
        if not pod or "id" not in pod:
            raise RunPodError(f"create pod returned no id: {pod}")
        return pod

    def get_pod(self, pod_id: str) -> dict | None:
        return self._req("GET", f"/pods/{pod_id}")

    def list_pods(self) -> list[dict]:
        out = self._req("GET", "/pods")
        return out if isinstance(out, list) else (out or {}).get("pods", [])

    def delete_pod(self, pod_id: str) -> None:
        self._req("DELETE", f"/pods/{pod_id}")
