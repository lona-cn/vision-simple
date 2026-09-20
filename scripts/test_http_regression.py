#!/usr/bin/env python3
"""Exercise a real CPU server, isolated from developer configuration and listeners."""

import argparse
import base64
import contextlib
import http.client
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time


class RegressionFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise RegressionFailure(message)


def fixture(root, relative):
    path = root / relative
    require(path.is_file() and path.stat().st_size > 0, f"Required fixture missing: {path}")
    with path.open("rb") as stream:
        require(not stream.read(128).startswith(b"version https://git-lfs.github.com/spec/"),
                f"Fixture is a Git LFS pointer, not downloaded content: {path}")
    return path.resolve()


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def listeners(pid):
    """Query the OS socket table, not server logs or reachability."""
    if os.name == "nt":
        command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                   "$ErrorActionPreference='Stop'; "
                   "@(Get-NetTCPConnection -State Listen -ErrorAction Stop | "
                   f"Where-Object {{ $_.OwningProcess -eq {pid} }} | "
                   "Select-Object LocalAddress,LocalPort) | ConvertTo-Json -Compress"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
        rows = json.loads(result.stdout) if result.stdout.strip() else []
        if isinstance(rows, dict):
            rows = [rows]
        return [(row["LocalAddress"], int(row["LocalPort"])) for row in rows]
    require(sys.platform.startswith("linux"), "Socket inspection supports Windows and Linux only")
    require(shutil.which("ss"), "Linux regression requires ss (install iproute2)")
    result = subprocess.run(["ss", "-H", "-ltnp"], capture_output=True, text=True,
                            timeout=10, check=True)
    endpoints = []
    for line in result.stdout.splitlines():
        if re.search(rf"\bpid={pid}(?:,|\))", line):
            address, port = line.split()[3].rsplit(":", 1)
            endpoints.append((address.strip("[]"), int(port)))
    return endpoints


class Server:
    def __init__(self, executable, root, model_config, *, host="127.0.0.1", port=None,
                 framework="kONNXRUNTIME", ep="kCPU"):
        self.executable = executable
        self.root = root
        self.model_config = model_config
        self.host, self.port = host, port if port is not None else free_port()
        self.framework, self.ep = framework, ep
        self.process = None
        self.temp = None
        self.output = None
        self.expected_start_failure = False

    def __enter__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vision-simple-http-")
        self.cwd = Path(self.temp.name)
        try:
            (self.cwd / "logs").mkdir()
            (self.cwd / "config").mkdir()
            shutil.copyfile(self.root / "app/config/base/log.properties",
                            self.cwd / "config/log.properties")
            # JSON string quoting is also valid YAML; forward slashes avoid Windows escapes.
            options = {"static_path": (self.root / "doc/openapi").as_posix(),
                       "infer_framework": self.framework, "infer_ep": self.ep, "infer_device": "0"}
            text = f"host: {json.dumps(self.host)}\nport: {self.port}\noptions:\n"
            text += "".join(f"  {key}: {json.dumps(value)}\n" for key, value in options.items())
            (self.cwd / "config/server.yaml").write_text(text, encoding="utf-8")
            if self.model_config is not None:
                (self.cwd / "config/models.yaml").write_text(self.model_config, encoding="utf-8")
            self.output = (self.cwd / "server-output.log").open("w+b")
            environment = os.environ.copy()
            # xmake copies shared libraries beside its target. The temporary cwd must not
            # accidentally make Linux's historical './' rpath lose those dependencies.
            library_var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
            environment[library_var] = str(self.executable.parent) + os.pathsep + environment.get(library_var, "")
            self.process = subprocess.Popen(
                [str(self.executable)], cwd=self.cwd, env=environment, stdin=subprocess.PIPE,
                stdout=self.output, stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def diagnostics(self):
        if self.output is None:
            return "Server did not start"
        self.output.flush()
        captured = (self.cwd / "server-output.log").read_text(encoding="utf-8", errors="replace")
        code = self.process.poll() if self.process else None
        return f"server={self.executable}\ncwd={self.cwd}\nport={self.port} exit={code}\n{captured}"

    def wait_ready(self, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            require(self.process.poll() is None, "Server exited before readiness")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.3):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise RegressionFailure("Server did not listen before startup deadline")
        rows = listeners(self.process.pid)
        require(("127.0.0.1", self.port) in rows,
                f"Expected owned loopback listener, actual OS sockets: {rows}")
        require(all(ipaddress.ip_address(address).is_loopback for address, _ in rows),
                f"Server opened a non-loopback/wildcard listener: {rows}")

    def request(self, route, payload=None, *, raw=None, method="POST"):
        require(self.process.poll() is None, "Server exited between requests")
        body = raw if raw is not None else json.dumps(payload).encode("utf-8")
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        try:
            connection.request(method, route, body=None if method == "GET" else body,
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            data = response.read()
            require(response.getheader("Content-Type", "").split(";", 1)[0].lower() == "application/json",
                    f"{route}: expected JSON content type, got {response.getheaders()}")
            try:
                parsed = json.loads(data)
            except (ValueError, UnicodeDecodeError) as exc:
                raise RegressionFailure(f"{route}: non-JSON response {data[:1000]!r}") from exc
            if isinstance(parsed, dict) and "error" in parsed:
                message = str(parsed["error"])
                require(all(path not in message for path in
                            (str(self.root), self.root.as_posix(), str(self.cwd), self.cwd.as_posix())),
                        f"{route}: error response leaked an internal filesystem path")
            return response.status, parsed
        finally:
            connection.close()

    def expect_start_failure(self):
        try:
            code = self.process.wait(timeout=30)
        except subprocess.TimeoutExpired as exc:
            raise RegressionFailure("Invalid startup configuration left server running") from exc
        require(code != 0, "Invalid startup configuration exited successfully")
        require(0 < code < 256, f"Startup crashed instead of controlled failure: exit={code}")
        output = self.diagnostics()
        require(not re.search(r"\blistening on\b", output, re.IGNORECASE),
                "Startup failure advertised successful listening")
        require(not listeners(self.process.pid), "Failed startup left an owned listener")
        self.expected_start_failure = True

    def __exit__(self, exc_type, exc, traceback):
        cleanup_error = None
        try:
            if exc_type:
                print(self.diagnostics(), file=sys.stderr)
            if self.process is not None:
                if self.process.poll() is None:
                    try:
                        self.process.stdin.write(b"\n")
                        self.process.stdin.flush()
                        self.process.wait(timeout=15)
                    except (OSError, subprocess.TimeoutExpired):
                        cleanup_error = "Server did not shut down cleanly via stdin newline"
                        if os.name == "nt":
                            subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                                           capture_output=True, timeout=15, check=False)
                        else:
                            with contextlib.suppress(ProcessLookupError):
                                os.killpg(self.process.pid, signal.SIGTERM)
                            try:
                                self.process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                with contextlib.suppress(ProcessLookupError):
                                    os.killpg(self.process.pid, signal.SIGKILL)
                        self.process.wait(timeout=10)
                    if self.process.returncode != 0 and cleanup_error is None:
                        cleanup_error = f"Server failed during normal shutdown: {self.process.returncode}"
                elif not self.expected_start_failure and exc_type is None:
                    cleanup_error = f"Server exited unexpectedly before cleanup: {self.process.returncode}"
                if self.process.stdin:
                    self.process.stdin.close()
                require(not listeners(self.process.pid), "Owned listener survived process cleanup")
            if cleanup_error:
                print(self.diagnostics(), file=sys.stderr)
        finally:
            if self.output:
                self.output.close()
            if self.temp:
                self.temp.cleanup()
        if cleanup_error and exc_type is None:
            raise RegressionFailure(cleanup_error)


def error_response(response, status, code, index):
    actual_status, body = response
    require(actual_status == status, f"Expected HTTP {status}, got {actual_status}: {body}")
    require(isinstance(body, dict) and set(body) == {"error"}, f"Not an atomic error response: {body}")
    error = body["error"]
    require(isinstance(error, dict) and set(error) == {"code", "message", "image_index"},
            f"Wrong error schema: {body}")
    require(error["code"] == code and error["image_index"] == index,
            f"Expected {code} index={index}, got {body}")
    require(index is None or type(error["image_index"]) is int, f"Noninteger error index: {body}")
    require(isinstance(error["message"], str) and error["message"], f"Missing error message: {body}")


def infer(server, kind, model, images):
    status, response = server.request(f"/v0/infer/{kind}", {"model": model, "images": images})
    require(status == 200 and isinstance(response, dict) and "error" not in response,
            f"{kind}/{model}: expected successful inference, got {status}: {response}")
    results = response.get("results")
    require(isinstance(results, list) and len(results) == len(images),
            f"{kind}/{model}: input/output count mismatch: {response}")
    require(all(isinstance(frame, list) for frame in results), f"Invalid result frames: {response}")
    if kind == "yolo":
        require(isinstance(response.get("class_names"), list), "YOLO class_names missing")
    return response


def close_values(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(close_values(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close_values(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, (float, int)):
        return math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-6)
    return type(left) is type(right) and left == right


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0


def check_targets(response, references, model):
    detections = response["results"][0]
    for target in references:
        require(any(detection["class_id"] == target["class_id"] and
                    iou(detection["bbox"], target["bbox"]) >= target["min_iou"]
                    for detection in detections),
                f"{model}: missing frozen significant target {target}; detections={detections}")


def model_yaml(root):
    assets = root / "app/assets/test"
    quote = lambda path: json.dumps(path.as_posix())
    return (
        "yolo:\n"
        f"  - name: hd2-fp32\n    version: kV11\n    path: {quote(assets / 'hd2-yolo11n-fp32.onnx')}\n"
        f"  - name: hd2-fp16\n    version: kV11\n    path: {quote(assets / 'hd2-yolo11n-fp16.onnx')}\n"
        f"  - name: missing-yolo\n    version: kV11\n    path: {quote(assets / 'does-not-exist.onnx')}\n"
        f"  - name: corrupt-yolo\n    version: kV11\n    path: {quote(assets / 'hd2.png')}\n"
        f"  - name: version-yolo\n    version: not-a-yolo-version\n    path: {quote(assets / 'hd2-yolo11n-fp32.onnx')}\n"
        f"  - name: runtime-yolo\n    version: kV10\n    path: {quote(assets / 'yolo_runtime_failure.onnx')}\n"
        "ocr:\n"
        f"  - name: corrupt-ocr\n    version: kPPOCRv4\n    det_path: {quote(assets / 'hd2.png')}\n"
        f"    rec_path: {quote(assets / 'ppocr_rec.onnx')}\n    char_dict_path: {quote(assets / 'ppocr_keys_v1.txt')}\n"
        f"  - name: version-ocr\n    version: not-an-ocr-version\n    det_path: {quote(assets / 'ppocr_det.onnx')}\n"
        f"    rec_path: {quote(assets / 'ppocr_rec.onnx')}\n    char_dict_path: {quote(assets / 'ppocr_keys_v1.txt')}\n"
        f"  - name: runtime-ocr\n    version: kPPOCRv4\n    det_path: {quote(assets / 'ocr_det_runtime_failure.onnx')}\n"
        f"    rec_path: {quote(assets / 'ppocr_rec.onnx')}\n    char_dict_path: {quote(assets / 'ppocr_keys_v1.txt')}\n"
        f"  - name: ppocr-v4\n    version: kPPOCRv4\n    det_path: {quote(assets / 'ppocr_det.onnx')}\n"
        f"    rec_path: {quote(assets / 'ppocr_rec.onnx')}\n    char_dict_path: {quote(assets / 'ppocr_keys_v1.txt')}\n"
        f"  - name: missing-ocr\n    version: kPPOCRv4\n    det_path: {quote(assets / 'does-not-exist.onnx')}\n"
        f"    rec_path: {quote(assets / 'ppocr_rec.onnx')}\n    char_dict_path: {quote(assets / 'ppocr_keys_v1.txt')}\n")


def request_matrix(server, kind, model, images):
    route = f"/v0/infer/{kind}"
    singles = [infer(server, kind, model, [image]) for image in images]
    expected = [single["results"][0] for single in singles]
    require(expected[0] != expected[1], f"{kind}: image fixtures must yield distinct results to prove order")
    if kind == "ocr":
        require(any(isinstance(line.get("line"), str) and line["line"].strip() for line in expected[0]),
                "Normal OCR smoke must recognize text, not just return HTTP 200")
    batch = infer(server, kind, model, images)
    require(close_values(batch["results"], expected), f"{kind}: batch results differ from corresponding singles")
    reverse = infer(server, kind, model, list(reversed(images)))
    require(close_values(reverse["results"], list(reversed(expected))), f"{kind}: reversed batch lost input order")
    infer(server, kind, model, [])

    def bad(payload=None, *, raw=None, status=400, code="invalid_request", index=None):
        error_response(server.request(route, payload, raw=raw), status, code, index)
        recovered = infer(server, kind, model, [images[0]])
        require(close_values(recovered, singles[0]), f"{kind}: valid request changed after error")

    for raw in (b"{", b"", b"null", b"[]", b'"text"', b"42"):
        bad(raw=raw)
    for payload in ({}, {"model": model}, {"images": []}, {"model": "", "images": []},
                    {"model": None, "images": []}, {"model": 1, "images": []},
                    {"model": [], "images": []}, {"model": {}, "images": []},
                    {"model": model, "images": None}, {"model": model, "images": "text"},
                    {"model": model, "images": {}}, {"model": model, "images": [1]},
                    {"model": model, "images": [None]}, {"model": model, "images": [images[0], {}]}):
        bad(payload)
    # Structure wins over lookup; lookup/load wins over image decoding (including [] batches).
    bad({"model": "unknown-model", "images": [1]})
    for content in ([], ["%%%"]):
        bad({"model": "unknown-model", "images": content}, code="unknown_model")
        for failure in ("missing", "corrupt", "version"):
            bad({"model": f"{failure}-{kind}", "images": content}, status=500, code="model_load_failed")
    nonimage = base64.b64encode(b"This is valid base64, but not an image.").decode("ascii")
    for encoded in ("", "%%%", "A", "AAAA=", "AA=A", nonimage):
        bad({"model": model, "images": [encoded]}, code="invalid_image", index=0)
    for contents, index in (([images[0], "%%%", nonimage], 1),
                            ([nonimage, images[0], "%%%"], 0),
                            ([images[0], images[1], nonimage], 2)):
        bad({"model": model, "images": contents}, code="invalid_image", index=index)
    print(f"PASS {kind}: request/error/recovery and ordered batch matrix", flush=True)


def inference_failure_matrix(server, kind):
    # Binary PPM is decoded by OpenCV without a Python image dependency. Uniform
    # inputs survive preprocessing; fixture Gather fails only for the white input.
    black, white = [base64.b64encode(b"P6\n32 32\n255\n" + bytes([value]) * (32 * 32 * 3)).decode("ascii")
                    for value in (0, 255)]
    model = f"runtime-{kind}"
    expected = infer(server, kind, model, [black])
    if kind == "yolo":
        check_targets(expected, [{"class_id": 0, "bbox": [4, 4, 16, 16], "min_iou": 1}], model)
    else:
        require(expected["results"] == [[]], "Black OCR fixture should yield no text")
    for contents, status, code, index in (
            ([white], 500, "inference_failed", 0),
            ([black, white, white], 500, "inference_failed", 1),
            ([white, black, white], 500, "inference_failed", 0),
            # All images must be decoded before any inference starts.
            ([white, "%%%"], 400, "invalid_image", 1)):
        response = server.request(f"/v0/infer/{kind}", {"model": model, "images": contents})
        error_response(response, status, code, index)
        require(close_values(infer(server, kind, model, [black]), expected),
                f"{kind}: same model instance did not recover from a real ORT runtime failure")
    print(f"PASS {kind}: real ORT failure, stage/index precedence, same-instance recovery", flush=True)


def run(args):
    root = args.project_root.resolve(strict=True)
    executable = args.server.resolve(strict=True)
    require(executable.is_file(), f"Server executable missing: {executable}")
    for path in ("app/assets/test/hd2-yolo11n-fp32.onnx", "app/assets/test/hd2-yolo11n-fp16.onnx",
                 "app/assets/test/ppocr_det.onnx", "app/assets/test/ppocr_rec.onnx",
                 "app/assets/test/ppocr_keys_v1.txt", "app/config/base/log.properties",
                 "app/assets/test/yolo_runtime_failure.onnx", "app/assets/test/ocr_det_runtime_failure.onnx"):
        fixture(root, path)
    reference = json.loads(fixture(root, "app/assets/test/http_yolo_reference.json").read_text(encoding="utf-8"))
    require(isinstance(reference.get("provenance"), str) and reference["provenance"],
            "Frozen YOLO references must document their independent provenance")
    targets = reference.get("targets")
    require(isinstance(targets, list) and targets, "Frozen significant YOLO targets are required")
    for target in targets:
        require(type(target.get("class_id")) is int and target["class_id"] >= 0 and
                isinstance(target.get("bbox"), list) and len(target["bbox"]) == 4 and
                all(type(value) in (int, float) and math.isfinite(value) for value in target["bbox"]) and
                target["bbox"][2] > 0 and target["bbox"][3] > 0 and 0 < target.get("min_iou", 0) <= 1,
                f"Invalid frozen target: {target}")
    image_paths = [fixture(root, reference["image"]), fixture(root, "doc/images/ppocr.png")]
    images = [base64.b64encode(path.read_bytes()).decode("ascii") for path in image_paths]
    require(images[0] != images[1], "Integration fixtures must contain distinct image bytes")
    config = model_yaml(root)
    with Server(executable, root, config) as server:
        server.wait_ready()
        status, models = server.request("/v0/infer/models", method="GET")
        require(status == 200 and "error" not in models, f"Models endpoint failed: {models}")
        for model in ("hd2-fp32", "hd2-fp16"):
            check_targets(infer(server, "yolo", model, [images[0]]), targets, model)
        print("PASS FP32/FP16: independently frozen class/IoU reference targets", flush=True)
        request_matrix(server, "yolo", "hd2-fp32", images)
        # The text image first also makes every OCR error's recovery exercise real recognition.
        request_matrix(server, "ocr", "ppocr-v4", list(reversed(images)))
        inference_failure_matrix(server, "yolo")
        inference_failure_matrix(server, "ocr")
    for bad_config in (None, 'yolo:\n  - name: "unterminated\n'):
        for route, payload, method, code in (
                ("/v0/infer/models", None, "GET", "model_config_failed"),
                ("/v0/infer/yolo", {"model": "hd2-fp32", "images": []}, "POST", "model_config_failed"),
                ("/v0/infer/ocr", {"model": "ppocr-v4", "images": []}, "POST", "model_config_failed")):
            # Each error runs before singleton config initialization, then a real
            # request proves recovery immediately after that error.
            with Server(executable, root, bad_config) as server:
                server.wait_ready()
                error_response(server.request(route, payload, method=method), 500, code, None)
                (server.cwd / "config/models.yaml").write_text(config, encoding="utf-8")
                infer(server, "ocr", "ppocr-v4", [images[1]])
                infer(server, "yolo", "hd2-fp32", [images[0]])
            print(f"PASS model configuration error and recovery: {route}", flush=True)
    for options in ({"host": ""}, {"host": "invalid host!"}, {"host": "x" * 256},
                    {"framework": "not-a-framework"}, {"ep": "not-an-ep"}):
        with Server(executable, root, config, **options) as server:
            server.expect_start_failure()
        print(f"PASS controlled startup failure: {options}", flush=True)
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        with Server(executable, root, config, port=occupied.getsockname()[1]) as server:
            server.expect_start_failure()
    print("PASS occupied port; all owned server processes stopped", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args)
    except (RegressionFailure, OSError, ValueError, subprocess.SubprocessError, http.client.HTTPException) as exc:
        print(f"FAIL HTTP regression: {exc}", file=sys.stderr)
        return 1
    print("PASS HTTP reliability and inference correctness regression", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
