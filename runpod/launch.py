#!/usr/bin/env python3
"""Run a CVLM config on a RunPod GPU pod, paying only for the GPU forward passes.

    python runpod/launch.py run --config configs/e1/full.yaml [--dry-run] [--yes] [--spot]
    python runpod/launch.py list
    python runpod/launch.py terminate <pod-id> | --all

`run` does everything that needs no GPU on the laptop first: tests, dataset preparation and
image extraction. Then it:

1. creates one pod (no persistent volume, so nothing is billed after termination)
2. uploads the tracked code plus the prepared data over SSH
3. runs a CUDA smoke test, then the config, streaming the log
4. pulls each finished stage back as soon as it completes
5. terminates the pod

Teardown runs on success, on any error, on Ctrl-C / SIGTERM / SIGHUP, and at the time cap. On
the pod, a watchdog deletes the pod at the same cap in case this laptop disappears.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rp_api import POD_PREFIX, ROOT, RunPod, RunPodError, api_key  # noqa: E402

IMAGE = "runpod/pytorch:1.3.3-cu1281-torch2130-ubuntu2204"
GPUS = ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000", "NVIDIA A40", "NVIDIA L40S", "NVIDIA GeForce RTX 3090"]
SSH_KEY = Path.home() / ".ssh" / "runpod_ed25519"
REMOTE = "/root/cvlm"
STAGE_DIRS = ["data", "vl", "text", "fit", "eval"]
PY = str(ROOT / ".venv" / "bin" / "python")


def log(msg: str) -> None:
    print(f"[launch {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------- pod request
def build_payload(run_name: str, a, public_key: str) -> dict:
    return {
        "name": f"{POD_PREFIX}{run_name}",
        "imageName": a.image,
        "cloudType": a.cloud,
        "computeType": "GPU",
        "gpuTypeIds": a.gpu,
        "gpuTypePriority": "custom",           # try the GPUs in the order given
        "gpuCount": 1,
        "containerDiskInGb": a.disk,
        "volumeInGb": 0,                        # no persistent volume: nothing is billed once terminated
        "ports": ["22/tcp"],                    # full SSH (public IP + TCP), which scp/tar/rsync need
        "supportPublicIp": True,
        "minDownloadMbps": a.min_download,      # 34 GB of weights over a slow link would burn GPU minutes
        "allowedCudaVersions": ["12.8", "12.9", "13.0"],
        "interruptible": a.spot,
        # PUBLIC_KEY is how RunPod's official images authorise SSH. Only the public half leaves this machine.
        "env": {"PUBLIC_KEY": public_key, "CVLM_MAX_MINUTES": str(a.max_minutes)},
    }


def upload_files(root: Path = ROOT) -> list[str]:
    """Tracked plus untracked-but-not-ignored files: .gitignore keeps .env, runs/ and .venv out."""
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                         cwd=root, check=True, capture_output=True).stdout.decode()
    files = [f for f in out.split("\0") if f and (root / f).is_file()]
    bad = [f for f in files if Path(f).name.startswith(".env") and Path(f).name != ".env.example"]
    if bad:
        raise RunPodError(f"refusing to upload credential files: {bad}")
    return files


# ---------------------------------------------------------------------------- transport
@dataclass
class Shell:
    """SSH and tar-over-SSH to one pod. A per-run known_hosts file keeps ~/.ssh/known_hosts clean."""
    ip: str
    port: int
    known_hosts: Path

    def _ssh(self) -> list[str]:
        return ["ssh", "-i", str(SSH_KEY), "-p", str(self.port), "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={self.known_hosts}",
                "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-o", "LogLevel=ERROR",
                f"root@{self.ip}"]

    def run(self, cmd: str, timeout: float = 120) -> tuple[int, str]:
        p = subprocess.run(self._ssh() + [cmd], capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr

    def upload(self, files: list[str], extra_dirs: list[str]) -> None:
        """tar the listed files and directories (relative to ROOT) into REMOTE on the pod."""
        listing = "\0".join(files + extra_dirs).encode()
        # macOS tar would otherwise add AppleDouble/xattr entries that Linux tar warns about.
        tar = subprocess.Popen(["tar", "--null", "--no-mac-metadata", "--no-xattrs", "-czf", "-", "-T", "-"],
                               cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               env={**os.environ, "COPYFILE_DISABLE": "1"})
        ssh = subprocess.Popen(self._ssh() + [f"mkdir -p {REMOTE} && tar -xzf - -C {REMOTE}"],
                               stdin=tar.stdout)
        tar.stdin.write(listing)
        tar.stdin.close()
        tar.stdout.close()
        if ssh.wait(timeout=1800) != 0 or tar.wait() != 0:
            raise RunPodError("upload failed")

    def download(self, remote_dir: str, local_dir: Path, excludes: tuple[str, ...] = ()) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        ex = " ".join(f"--exclude={shlex.quote(e)}" for e in excludes)
        ssh = subprocess.Popen(self._ssh() + [f"cd {shlex.quote(remote_dir)} && tar -czf - {ex} ."],
                               stdout=subprocess.PIPE)
        tar = subprocess.run(["tar", "-xzf", "-", "-C", str(local_dir)], stdin=ssh.stdout)
        ssh.stdout.close()
        if ssh.wait(timeout=1800) != 0 or tar.returncode != 0:
            raise RunPodError(f"download of {remote_dir} failed")


# ---------------------------------------------------------------------------- pod lifecycle
class PodRun:
    def __init__(self, api: RunPod, run_name: str, config: str, a, shell_factory=Shell, clock=time.time,
                 sleep=time.sleep, runs_root: Path = ROOT / "runs"):
        self.api, self.run_name, self.config, self.a = api, run_name, config, a
        self.run_dir = runs_root / run_name
        self.shell_factory, self.clock, self.sleep = shell_factory, clock, sleep
        self.pod: dict | None = None
        self.shell: Shell | None = None
        self.t_created: float | None = None
        self.cost_per_hr = 0.0
        self.pulled: set[str] = set()
        self.log_offset = 0

    # -- bookkeeping
    def _record(self, **kw) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        p = self.run_dir / "pod.json"
        state = json.loads(p.read_text()) if p.exists() else {}
        state.update(kw)
        p.write_text(json.dumps(state, indent=1))

    # -- steps
    def create(self, payload: dict) -> None:
        self.pod = self.api.create_pod(payload)
        self.t_created = self.clock()
        self.cost_per_hr = float(self.pod.get("costPerHr") or 0.0)
        self._record(id=self.pod["id"], name=payload["name"], created=time.strftime("%FT%T"),
                     cost_per_hr=self.cost_per_hr, status="created")
        log(f"pod {self.pod['id']} created ({self.pod.get('machine', {}).get('gpuDisplayName', 'GPU pending')}, "
            f"${self.cost_per_hr:.3f}/hr)")
        if self.cost_per_hr > self.a.max_price:
            raise RunPodError(f"pod costs ${self.cost_per_hr:.3f}/hr, above --max-price {self.a.max_price}")

    def wait_ready(self, timeout: float = 900) -> None:
        deadline = self.clock() + timeout
        while self.clock() < deadline:
            pod = self.api.get_pod(self.pod["id"]) or {}
            ip, port = pod.get("publicIp"), (pod.get("portMappings") or {}).get("22")
            if pod.get("costPerHr"):
                self.cost_per_hr = float(pod["costPerHr"])
            if pod.get("desiredStatus") == "RUNNING" and ip and port:
                self.shell = self.shell_factory(ip, int(port), self.run_dir / "known_hosts")
                log(f"pod running at {ip}:{port}")
                break
            self.sleep(10)
        else:
            raise RunPodError(f"pod not ready within {timeout:.0f}s (image pull or no public IP)")
        while self.clock() < deadline:
            rc, _ = self.shell.run("true", timeout=30)
            if rc == 0:
                log("ssh ready")
                return
            self.sleep(10)
        raise RunPodError("sshd did not come up")

    def upload(self) -> None:
        dirs = [f"runs/{self.run_name}"]
        if (ROOT / "runs" / "e1-tiny" / "data" / "manifest.json").exists():
            dirs.append("runs/e1-tiny/data")            # for the on-pod CUDA smoke test
        files = upload_files()
        log(f"uploading {len(files)} files + {', '.join(dirs)}")
        self.shell.upload(files, dirs)

    def start(self) -> None:
        cmd = (f"cd {REMOTE} && mkdir -p runs/{self.run_name} && "
               f"nohup bash runpod/remote_run.sh {shlex.quote(self.config)} {self.run_name} "
               f"> runs/{self.run_name}/remote.log 2>&1 < /dev/null &")
        rc, out = self.shell.run(cmd)
        if rc != 0:
            raise RunPodError(f"could not start the remote run: {out[-500:]}")
        self._record(status="running")

    def _poll(self) -> str | None:
        """Print new log output, pull finished stages; return DONE/FAILED when the run ends."""
        remote = f"{REMOTE}/runs/{self.run_name}"
        rc, out = self.shell.run(
            f"tail -c +{self.log_offset + 1} {remote}/remote.log 2>/dev/null; echo; echo '@@@'; "
            f"cd {remote} && ls -d */manifest.json 2>/dev/null; ls DONE FAILED 2>/dev/null", timeout=60)
        if rc != 0:
            return "SSH_ERROR"
        text, _, status = out.rpartition("@@@")
        text = text[:-1] if text.endswith("\n") else text
        if text:
            self.log_offset += len(text.encode())
            with open(self.run_dir / "remote.log", "a") as f:
                f.write(text)
            for line in text.splitlines():
                print(f"  | {line}", flush=True)
        lines = status.split()
        for d in [x.split("/")[0] for x in lines if x.endswith("/manifest.json")]:
            if d in STAGE_DIRS and d != "data" and d not in self.pulled:
                log(f"stage {d} finished on the pod; pulling it")
                self.shell.download(f"{remote}/{d}", self.run_dir / d)
                self.pulled.add(d)
        if "FAILED" in lines:
            return "FAILED"
        if "DONE" in lines:
            return "DONE"
        return None

    def monitor(self) -> str:
        deadline = self.t_created + self.a.max_minutes * 60 - 180      # leave time to pull results and tear down
        errors = 0
        while self.clock() < deadline:
            state = self._poll()
            if state in ("DONE", "FAILED"):
                return state
            if state == "SSH_ERROR":
                errors += 1
                pod = self.api.get_pod(self.pod["id"])
                if not pod or pod.get("desiredStatus") != "RUNNING":
                    raise RunPodError("pod is gone (preempted, or stopped by the watchdog)")
                if errors >= 10:
                    raise RunPodError("lost SSH to the pod")
            else:
                errors = 0
            self.sleep(self.a.poll)
        log(f"time cap of {self.a.max_minutes} min reached")
        return "TIMEOUT"

    def pull_final(self) -> None:
        remote = f"{REMOTE}/runs/{self.run_name}"
        log("pulling results")
        self.shell.download(remote, self.run_dir, excludes=("./data", *(f"./{d}" for d in sorted(self.pulled))))

    def terminate(self) -> None:
        if not self.pod:
            return
        pid = self.pod["id"]
        for attempt in range(6):
            try:
                self.api.delete_pod(pid)
                break
            except Exception as e:           # network blips must not leave a pod running
                log(f"terminate attempt {attempt + 1} failed: {e}")
                self.sleep(min(2 ** attempt, 20))
        else:
            log(f"!!! COULD NOT TERMINATE POD {pid}. Run: python runpod/launch.py terminate {pid}")
            self._record(status="TERMINATE_FAILED")
            return
        for _ in range(12):
            pod = self.api.get_pod(pid)
            if not pod or pod.get("desiredStatus") in ("TERMINATED", "EXITED"):
                break
            self.sleep(5)
        hours = (self.clock() - self.t_created) / 3600 if self.t_created else 0.0
        self._record(status="terminated", terminated=time.strftime("%FT%T"), hours=round(hours, 3),
                     est_cost=round(hours * self.cost_per_hr, 3))
        log(f"pod {pid} terminated after {hours * 60:.1f} min, about ${hours * self.cost_per_hr:.2f}")
        self.pod = None


def run_on_pod(api: RunPod, run_name: str, config: str, payload: dict, a, **kw) -> str:
    pr = PodRun(api, run_name, config, a, **kw)
    state = "ERROR"
    try:
        pr.create(payload)
        pr.wait_ready()
        pr.upload()
        pr.start()
        state = pr.monitor()
        pr.pull_final()
    finally:
        pr.terminate()
    return state


# ---------------------------------------------------------------------------- preflight
def preflight(a, cfg) -> str:
    if not SSH_KEY.exists():
        raise RunPodError(f"{SSH_KEY} missing; run scripts/setup_local.sh")
    import requests
    repo, tag = a.image.split(":")
    r = requests.get(f"https://hub.docker.com/v2/repositories/{repo}/tags/{tag}", timeout=20)
    if r.status_code != 200:
        raise RunPodError(f"image {a.image} not found on Docker Hub")
    if not a.skip_tests:
        log("running tests")
        subprocess.run([PY, "-m", "pytest", "-q"], cwd=ROOT, check=True)
    for c in ("configs/e1/tiny.yaml", a.config):
        log(f"preparing data locally for {c}")
        subprocess.run([PY, "-m", "cvlm", "prepare", "--config", c], cwd=ROOT, check=True)
    return (SSH_KEY.with_suffix(".pub")).read_text().strip()


def _caffeinate() -> None:
    """Keep the Mac awake for the whole run (a sleeping laptop cannot tear the pod down)."""
    if sys.platform == "darwin" and not os.environ.get("CVLM_CAFFEINATED"):
        os.environ["CVLM_CAFFEINATED"] = "1"
        os.execvp("caffeinate", ["caffeinate", "-i", sys.executable, *sys.argv])


def cmd_run(a) -> int:
    sys.path.insert(0, str(ROOT))
    from cvlm import config as config_mod
    cfg = config_mod.load(a.config)
    if cfg.device not in ("cuda", "auto"):
        raise RunPodError(f"{a.config} has device={cfg.device}; the pod runs CUDA")
    worst = a.max_minutes / 60 * a.max_price
    if a.dry_run:
        payload = build_payload(cfg.name, a, "<~/.ssh/runpod_ed25519.pub>")
        print(json.dumps(payload, indent=1))
        print(f"\nupload: {len(upload_files())} code files + runs/{cfg.name} (+ runs/e1-tiny/data)")
        log(f"dry run: no pod created. Worst case {a.max_minutes} min x ${a.max_price}/hr = ${worst:.2f}")
        return 0
    _caffeinate()
    public_key = preflight(a, cfg)
    payload = build_payload(cfg.name, a, public_key)
    api = RunPod(api_key())
    clash = [p for p in api.list_pods() if p.get("name") == payload["name"]]
    if clash:
        raise RunPodError(f"a pod named {payload['name']} already exists ({clash[0].get('id')}); "
                          "terminate it first: python runpod/launch.py terminate " + str(clash[0].get("id")))
    log(f"about to create a PAID pod: {payload['name']}, GPUs {a.gpu[0]} (+{len(a.gpu) - 1} fallbacks), "
        f"{a.cloud.lower()} cloud{', spot' if a.spot else ''}, cap {a.max_minutes} min, "
        f"max ${a.max_price}/hr (worst case ${worst:.2f})")
    if not a.yes and input("type 'yes' to create it: ").strip() != "yes":
        log("aborted; nothing created")
        return 1

    def _interrupt(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _interrupt)
    state = run_on_pod(api, cfg.name, a.config, payload, a)
    report = ROOT / "runs" / cfg.name / "report.md"
    log(f"remote run {state}; " + (f"report: {report}" if report.exists() else "no report yet"))
    if state != "DONE":
        log("finished stages were pulled; re-running the same command resumes from them")
    return 0 if state == "DONE" else 1


def cmd_list(a) -> int:
    pods = RunPod(api_key()).list_pods()
    if not pods:
        print("no pods")
    for p in pods:
        print(f"{p.get('id')}  {p.get('name'):<28} {p.get('desiredStatus'):<10} ${p.get('costPerHr')}/hr  "
              f"{(p.get('machine') or {}).get('gpuDisplayName', '')}")
    return 0


def cmd_terminate(a) -> int:
    api = RunPod(api_key())
    if a.all:
        ids = [p["id"] for p in api.list_pods() if str(p.get("name", "")).startswith(POD_PREFIX)]
    else:
        ids = [a.pod_id]
    if not ids:
        print("nothing to terminate")
        return 0
    if not a.yes and input(f"terminate {ids}? type 'yes': ").strip() != "yes":
        return 1
    for pid in ids:
        api.delete_pod(pid)
        print(f"terminated {pid}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a config on a fresh pod, then terminate it")
    r.add_argument("--config", required=True)
    r.add_argument("--dry-run", action="store_true", help="print the pod request; create nothing")
    r.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    r.add_argument("--spot", action="store_true", help="interruptible pricing; re-run to resume if preempted")
    r.add_argument("--gpu", nargs="+", default=GPUS, help="GPU type ids, in order of preference")
    r.add_argument("--cloud", choices=["COMMUNITY", "SECURE"], default="COMMUNITY")
    r.add_argument("--image", default=IMAGE)
    r.add_argument("--disk", type=int, default=80, help="container disk GB (models are ~34 GB)")
    r.add_argument("--max-minutes", type=int, default=90, help="hard cap; the pod is terminated at this point")
    r.add_argument("--max-price", type=float, default=1.0, help="$/hr; a pricier pod is terminated at once")
    r.add_argument("--min-download", type=int, default=500, help="minimum pod download Mbps")
    r.add_argument("--poll", type=int, default=45, help="seconds between status polls")
    r.add_argument("--skip-tests", action="store_true")
    sub.add_parser("list", help="list your pods")
    t = sub.add_parser("terminate", help="terminate a pod, or every cvlm-* pod")
    g = t.add_mutually_exclusive_group(required=True)
    g.add_argument("pod_id", nargs="?")
    g.add_argument("--all", action="store_true")
    t.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    return {"run": cmd_run, "list": cmd_list, "terminate": cmd_terminate}[a.cmd](a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RunPodError as e:
        log(f"error: {e}")
        sys.exit(2)
