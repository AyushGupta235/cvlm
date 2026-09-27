"""RunPod orchestration against a fake API and a fake SSH transport: no network, no pods."""
import argparse
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runpod"))
import launch  # noqa: E402
import rp_api  # noqa: E402


def args(**kw):
    base = dict(max_price=1.0, max_minutes=90, poll=45, image=launch.IMAGE, cloud="COMMUNITY", gpu=launch.GPUS,
                disk=80, min_download=500, spot=False)
    base.update(kw)
    return argparse.Namespace(**base)


class FakeAPI:
    def __init__(self, cost=0.5, fail_deletes=0, gone_after=None):
        self.cost, self.fail_deletes, self.gone_after = cost, fail_deletes, gone_after
        self.created, self.deleted, self.gets = [], [], 0

    def create_pod(self, payload):
        self.created.append(payload)
        return {"id": "pod1", "costPerHr": self.cost}

    def get_pod(self, pid):
        self.gets += 1
        if pid in self.deleted or (self.gone_after is not None and self.gets > self.gone_after):
            return None
        return {"desiredStatus": "RUNNING", "publicIp": "203.0.113.7", "portMappings": {"22": 2222},
                "costPerHr": self.cost}

    def delete_pod(self, pid):
        if self.fail_deletes:
            self.fail_deletes -= 1
            raise rp_api.RunPodError("blip")
        self.deleted.append(pid)

    def list_pods(self):
        return []


class FakeShell:
    """Each poll pops the next scripted response: (log text, status tokens) or 'ERR'."""

    def __init__(self, polls, fail_upload=False, interrupt_on_poll=None):
        self.polls, self.fail_upload, self.interrupt_on_poll = list(polls), fail_upload, interrupt_on_poll
        self.uploads, self.downloads, self.commands, self.n_polls = [], [], [], 0

    def __call__(self, ip, port, known_hosts):          # acts as the shell factory
        return self

    def run(self, cmd, timeout=120):
        self.commands.append(cmd)
        if "tail -c" not in cmd:
            return 0, ""
        self.n_polls += 1
        if self.interrupt_on_poll == self.n_polls:
            raise KeyboardInterrupt
        nxt = self.polls.pop(0) if self.polls else ("", "")
        if nxt == "ERR":
            return 255, "connection refused"
        text, status = nxt
        return 0, f"{text}\n@@@\n{status}\n"

    def upload(self, files, dirs):
        if self.fail_upload:
            raise rp_api.RunPodError("upload failed")
        self.uploads.append((len(files), dirs))

    def download(self, remote_dir, local_dir, excludes=()):
        self.downloads.append((remote_dir, tuple(excludes)))


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def go(api, shell, tmp_path, **kw):
    clock = Clock()
    return launch.run_on_pod(api, "e1-test", "configs/e1/full.yaml", {"name": "cvlm-e1-test"}, args(**kw),
                             shell_factory=shell, clock=clock, sleep=clock.sleep, runs_root=tmp_path)


def test_success_pulls_stages_once_and_terminates(tmp_path):
    api, shell = FakeAPI(), FakeShell([("[embed-vl] done\n", "vl/manifest.json"),
                                       ("[eval] done\n", "vl/manifest.json text/manifest.json DONE")])
    assert go(api, shell, tmp_path) == "DONE"
    assert api.deleted == ["pod1"]
    pulled = [d for d, ex in shell.downloads if ex == ()]
    assert [p.rsplit("/", 1)[1] for p in pulled] == ["vl", "text"]
    final = shell.downloads[-1]
    assert final[0].endswith("runs/e1-test") and set(final[1]) == {"./data", "./vl", "./text"}
    assert (tmp_path / "e1-test" / "remote.log").read_text() == "[embed-vl] done\n[eval] done\n"
    assert '"status": "terminated"' in (tmp_path / "e1-test" / "pod.json").read_text()


@pytest.mark.parametrize("failure", ["upload", "interrupt", "price"])
def test_pod_is_terminated_on_failure(tmp_path, failure):
    api = FakeAPI(cost=5.0 if failure == "price" else 0.5)
    shell = FakeShell([], fail_upload=failure == "upload", interrupt_on_poll=1 if failure == "interrupt" else None)
    with pytest.raises((rp_api.RunPodError, KeyboardInterrupt)):
        go(api, shell, tmp_path)
    assert api.deleted == ["pod1"]


def test_time_cap_terminates(tmp_path):
    api, shell = FakeAPI(), FakeShell([("working\n", "")] * 1000)
    assert go(api, shell, tmp_path, max_minutes=10) == "TIMEOUT"
    assert api.deleted == ["pod1"]


def test_gone_pod_is_detected(tmp_path):
    api, shell = FakeAPI(gone_after=3), FakeShell(["ERR"] * 5)
    with pytest.raises(rp_api.RunPodError, match="gone"):
        go(api, shell, tmp_path)


def test_terminate_retries_through_api_errors(tmp_path):
    api, shell = FakeAPI(fail_deletes=2), FakeShell([("", "DONE")])
    assert go(api, shell, tmp_path) == "DONE"
    assert api.deleted == ["pod1"]


def test_payload_carries_only_public_material():
    p = launch.build_payload("e1-full", args(spot=True), "ssh-ed25519 AAAA test")
    assert p["name"] == "cvlm-e1-full" and p["volumeInGb"] == 0 and p["ports"] == ["22/tcp"]
    assert set(p["env"]) == {"PUBLIC_KEY", "CVLM_MAX_MINUTES"} and p["interruptible"] is True
    assert p["gpuTypePriority"] == "custom" and p["gpuTypeIds"][0] == "NVIDIA GeForce RTX 4090"


def test_upload_list_respects_gitignore_and_refuses_env(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text(".env\n")
    (tmp_path / ".env").write_text("RUNPOD_API_KEY=not-a-real-key\n")
    (tmp_path / "a.py").write_text("x = 1\n")
    assert sorted(launch.upload_files(tmp_path)) == [".gitignore", "a.py"]
    (tmp_path / ".gitignore").write_text("")
    with pytest.raises(rp_api.RunPodError, match="credential"):
        launch.upload_files(tmp_path)


def test_client_hides_key_and_maps_404(monkeypatch):
    class Resp:
        def __init__(self, code):
            self.status_code, self.content, self.headers, self.text = code, b"", {}, ""

    class Session:
        headers = {}

        def request(self, method, url, **kw):
            return Resp(404)

    api = rp_api.RunPod("secret-value", session=Session())
    assert "secret-value" not in repr(api)
    assert api.get_pod("nope") is None
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr(rp_api, "load_env", lambda *a, **k: None)
    with pytest.raises(rp_api.RunPodError, match="not set"):
        rp_api.api_key()


def test_dry_run_creates_nothing(monkeypatch, capsys):
    monkeypatch.setattr(rp_api, "RunPod", lambda *a, **k: pytest.fail("dry run must not call the API"))
    monkeypatch.setattr(launch, "RunPod", lambda *a, **k: pytest.fail("dry run must not call the API"))
    assert launch.main(["run", "--config", "configs/e1/full.yaml", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert '"volumeInGb": 0' in out and "no pod created" in out
