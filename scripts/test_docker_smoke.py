#!/usr/bin/env python3
"""Smoke-test a local CPU image over HTTP, then verify graceful Docker shutdown."""

import argparse
import json
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid


def docker(*args, timeout=30, check=True):
    result = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout
    )
    if check and result.returncode:
        raise RuntimeError(f"docker {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def inspect(name):
    return json.loads(docker("inspect", name).stdout)[0]


def diagnostics(name):
    for args in (("inspect", name), ("logs", "--tail", "200", name)):
        try:
            result = docker(*args, check=False)
            print(result.stdout, file=sys.stderr)
            print(result.stderr, file=sys.stderr)
        except (OSError, subprocess.SubprocessError) as error:
            print(f"Unable to collect Docker diagnostics: {error}", file=sys.stderr)


def wait_ready(name, timeout):
    deadline = time.monotonic() + timeout
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_error = "No HTTP response yet"
    while time.monotonic() < deadline:
        container = inspect(name)
        state = container["State"]
        if not state["Running"]:
            raise RuntimeError(f"Container exited before readiness: {state}")
        ports = container["NetworkSettings"]["Ports"].get("11451/tcp")
        if not ports:
            raise RuntimeError("Container has no published HTTP port")
        binding = ports[0]
        if binding["HostIp"] != "127.0.0.1":
            raise RuntimeError(f"HTTP port is not loopback-only: {binding}")
        url = f"http://127.0.0.1:{binding['HostPort']}/v0/infer/models"
        try:
            with opener.open(url, timeout=min(3, max(0.1, deadline - time.monotonic()))) as response:
                if response.status != 200:
                    raise RuntimeError(f"Unexpected HTTP status: {response.status}")
                if response.headers.get_content_type() != "application/json":
                    raise RuntimeError("Model discovery did not return application/json")
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise RuntimeError(f"Invalid model catalog: {payload!r}")
            for kind in ("yolo", "ocr"):
                models = payload.get(kind)
                if not isinstance(models, list) or not models or not all(
                    isinstance(model, str) and model for model in models
                ):
                    raise RuntimeError(f"Invalid {kind} model list: {models!r}")
            health = state.get("Health", {}).get("Status")
            if health is None:
                raise RuntimeError("Image has no Docker HEALTHCHECK")
            if health == "healthy":
                print(f"HTTP model discovery and Docker health passed: {json.dumps(payload)}")
                return
            last_error = f"HTTP passed, Docker health is {health}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last_error = str(error)
        time.sleep(min(0.5, max(0, deadline - time.monotonic())))
    raise RuntimeError(f"Container readiness timed out after {timeout}s: {last_error}")


def stop_cleanly(name):
    if not inspect(name)["State"]["Running"]:
        raise RuntimeError("Container exited unexpectedly before normal shutdown")
    docker("stop", "--time", "15", name, timeout=25)
    state = inspect(name)["State"]
    # main.cpp handles SIGTERM, cleans up the server, and calls std::exit(SIGTERM).
    if state["Running"] or state["OOMKilled"] or state["ExitCode"] not in (0, 15):
        raise RuntimeError(f"Container did not shut down cleanly: {state}")


def interrupt(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Already built local linux/amd64 image")
    parser.add_argument("--timeout", type=int, default=120, help="Readiness deadline in seconds")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    signal.signal(signal.SIGTERM, interrupt)
    name = f"vision-simple-smoke-{uuid.uuid4().hex}"
    created = False
    passed = False
    try:
        # --pull=never prevents accidentally testing an unrelated remote image.
        # A UUID name allows cleanup even if the create command times out.
        created = True
        docker(
            "create", "--pull=never", "--platform", "linux/amd64",
            "--name", name, "--publish", "127.0.0.1::11451", args.image,
        )
        docker("start", name)
        wait_ready(name, args.timeout)
        stop_cleanly(name)
        passed = True
    except (Exception, KeyboardInterrupt) as error:
        print(f"Docker smoke failed: {error}", file=sys.stderr)
        if created:
            diagnostics(name)
    finally:
        # Do not interrupt the bounded cleanup if the runner cancels the job twice.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if created:
            try:
                if passed:
                    docker("rm", name)
                else:
                    # Force removal is failure recovery only, never the success path.
                    docker("rm", "--force", name)
                remaining = docker("container", "ls", "--all", "--quiet", "--filter", f"name=^/{name}$")
                if remaining.stdout.strip():
                    raise RuntimeError(f"Owned container survived removal: {name}")
            except (Exception, KeyboardInterrupt) as error:
                print(f"Container cleanup failed: {error}", file=sys.stderr)
                diagnostics(name)
                passed = False
    if passed:
        print("Docker smoke passed: HTTP JSON, healthy container, graceful stop, and removal")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
