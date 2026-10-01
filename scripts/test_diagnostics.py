#!/usr/bin/env python3
"""Exercise selected-model diagnostics as real children, without HTTP or a listener."""

import argparse
import base64
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib


class RegressionFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise RegressionFailure(message)


TOKEN = "DiagnosticPrivateToken-61-AbCd0123456789"
TEXT = "DiagnosticRecognizedTextSentinel"


def ppm(value, width=32, height=32):
    return f"P6\n{width} {height}\n255\n".encode("ascii") + bytes([value]) * (width * height * 3)


def model_yaml(rows):
    text = "models:\n"
    for row in rows:
        task, name, version, files = row[:4]
        text += f"  - task: {json.dumps(task)}\n    name: {json.dumps(name)}\n"
        text += f"    version: {json.dumps(version)}\n    files:\n"
        text += "".join(f"      {role}: {json.dumps(path.as_posix())}\n" for role, path in files.items())
        if len(row) == 5:
            text += f"    ocr_detection: {row[4]}\n"
    return text


class Driver:
    def __init__(self, executable, cwd):
        self.executable, self.cwd = executable, cwd
        self.forbidden = [str(cwd), cwd.as_posix(), json.dumps(str(cwd))[1:-1], TOKEN, TEXT]
        self.canary_kind = {value: "cwd" for value in self.forbidden}
        self.canary_kind.update({TOKEN: "token", TEXT: "recognition"})
        self.environment = os.environ.copy()
        self.environment["VISION_SIMPLE_DIAGNOSTIC_PRIVATE_TOKEN"] = TOKEN
        library_var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        self.environment[library_var] = str(executable.parent) + os.pathsep + self.environment.get(library_var, "")

    def configure(self, rows, **overrides):
        config = self.cwd / "config"
        config.mkdir(exist_ok=True)
        options = {"infer_framework": "kONNXRUNTIME", "infer_ep": "kCPU", "infer_device": "0",
                   "infer_idle_timeout_ms": "1", "infer_sweep_interval_ms": "1",
                   "infer_pipeline_capacity": "1", "infer_pipeline_max_batches": "1",
                   "infer_max_batch_images": "128", "infer_timeout_ms": "60000",
                   # Local diagnostics must not require the HTTP management secret or log properties.
                   "http_management_token_env": "VISION_SIMPLE_DIAGNOSTIC_UNSET_SECRET"}
        options.update(overrides)
        self.environment.pop("VISION_SIMPLE_DIAGNOSTIC_UNSET_SECRET", None)
        (config / "server.yaml").write_text(
            'host: "127.0.0.1"\nport: 11451\noptions:\n' +
            "".join(f"  {key}: {json.dumps(value)}\n" for key, value in options.items()), encoding="utf-8")
        (config / "models.yaml").write_text(model_yaml(rows), encoding="utf-8")

    def output_failure(self):
        full = Path("/dev/full")
        if not sys.platform.startswith("linux") or not full.is_char_device():
            return
        errors = []
        for arguments in (["--help"], ["--diagnose", "preflight", "--model", "yolo:shared"]):
            with full.open("wb", buffering=0) as output:
                child = subprocess.run([str(self.executable), *arguments], cwd=self.cwd,
                                       env=self.environment, stdout=output, stderr=subprocess.PIPE, timeout=180)
            stderr = child.stderr.decode("utf-8", errors="strict")
            mode = "help" if arguments == ["--help"] else "preflight"
            for private in self.forbidden:
                require(private not in stderr,
                        f"{mode} output_failure: private canary kind={self.canary_kind.get(private, 'fixture')} plane=stderr")
            require(child.returncode == 1, "Unwritable diagnostic stdout was reported as success")
            require(stderr.strip(), "Output failure lacked a safe stderr reason")
            errors.append(child.stderr)
        require(errors[0] == errors[1], "Output failure reason depended on configuration/model contents")

    def call(self, arguments, expected, *, help_text=False, label="argument validation"):
        if help_text:
            label = "help selected=0 fixtures=0"
        try:
            child = subprocess.run([str(self.executable), *map(str, arguments)], cwd=self.cwd,
                                   env=self.environment, capture_output=True, timeout=180)
        except subprocess.TimeoutExpired as error:
            raise RegressionFailure("Diagnostic child did not exit (possible server fallthrough)") from error
        stdout, stderr = (value.decode("utf-8", errors="strict") for value in (child.stdout, child.stderr))
        # Check both native stderr and the JSON channel before presenting any failure.
        for private in self.forbidden:
            for plane, captured in (("stdout", stdout), ("stderr", stderr)):
                require(private not in captured,
                        f"{label}: private canary kind={self.canary_kind.get(private, 'fixture')} plane={plane}")
        if child.returncode != expected:
            summary = "unparseable report"
            try:
                report = json.loads(stdout)
                states = {"not_configured", "missing_files", "load_failed", "loadable", "smoke_failed", "smoke_tested"}
                codes = {"invalid_arguments", "configuration_failed", "context_failed", "fixture_failed",
                         "diagnostic_failed", "inference_failed", "invalid_image", "image_limit_exceeded",
                         "service_overloaded", "request_timeout", "model_load_failed"}
                safe_code = lambda value: value if isinstance(value, str) and value in codes else "other"
                if isinstance(report, dict) and "models" in report:
                    rows = []
                    for row in report["models"]:
                        state = row.get("state")
                        state = state if isinstance(state, str) and state in states else "other"
                        passes = [{"phase": entry.get("phase") if entry.get("phase") in ("preflight", "cold", "warm") else "other",
                                   "success": entry.get("success") is True,
                                   "error": safe_code((entry.get("error") or {}).get("code"))}
                                  for entry in row.get("passes", [])]
                        rows.append({"state": state, "unloaded": row.get("unloaded") is True, "passes": passes})
                    summary = json.dumps(rows)
                elif isinstance(report, dict):
                    summary = "fatal=" + safe_code((report.get("error") or {}).get("code"))
            except (ValueError, TypeError, AttributeError):
                pass
            raise RegressionFailure(f"{label}: child exit {child.returncode}; expected {expected}; {summary}")
        require(not stderr.strip(), f"{label}: unexpected native/logging output plane=stderr")
        if help_text:
            require("--diagnose" in stdout and "--model" in stdout and "--image" in stdout,
                    "Help omitted diagnostic usage")
            return None
        try:
            body = json.loads(stdout)
        except (ValueError, UnicodeError) as error:
            raise RegressionFailure("Diagnostic stdout was not one complete JSON object") from error
        require(isinstance(body, dict) and body.get("schema_version") == 1, "Invalid diagnostic envelope")
        return body

    def diagnose(self, mode, selections, images=(), expected=0, extra=()):
        args = ["--diagnose", mode]
        for selection in selections:
            args += ["--model", selection]
        for image in images:
            args += ["--image", str(image)]
        label = f"{mode} selected={len(selections)} fixtures={len(images)}"
        body = self.call([*args, *extra], expected, label=label)
        if "models" in body:
            require(body["mode"] == mode and body["validity"] == "this_invocation_only" and
                    body["http_readiness"] == "not_assessed", "Diagnostic overstated readiness/validity")
            capabilities = body["capabilities"]
            require(capabilities["framework"] == "kONNXRUNTIME" and capabilities["requested_ep"] == "kCPU" and
                    capabilities["device_id"] == 0 and capabilities["context_created"] is True and
                    capabilities["cpu_fallback_allowed"] is True and
                    "kCPU" in capabilities["compiled_execution_providers"] and
                    "CPUExecutionProvider" in capabilities["available_execution_providers"] and
                    isinstance(capabilities["runtime_version"], str) and capabilities["runtime_version"],
                    "Diagnostic omitted actual CPU/runtime capabilities")
            require([(row["task"] + ":" + row["model"]) for row in body["models"]] == list(selections),
                    "Selected task:name identity/order changed or unselected model was inspected")
            require(body["limits"]["selection_max"] == 16 and body["limits"]["batch_max"] <= 128 and
                    body["limits"]["encoded_bytes_max"] == 67108864 and
                    body["limits"]["passes_per_model"] == (1 if mode == "preflight" else 2),
                    "Diagnostic resource/pass limits are incorrect")
        return body


def stage(record, *, completed=None, minimum=1):
    require(isinstance(record, dict) and type(record["elapsed_ns"]) is int and record["elapsed_ns"] >= 0,
            "Reached service stage has invalid elapsed time")
    require(record["calls"] >= minimum and 0 <= record["completed_calls"] <= record["calls"],
            "Service stage attempt/completion accounting is invalid")
    if completed is not None:
        require(record["completed_calls"] == completed, "Incorrect completed service work count")


def pipeline(record, frames, *, success, ocr=False):
    require(isinstance(record, dict) and record["input_frames"] == frames and record["complete"] is success,
            "Incorrect pipeline batch/completion accounting")
    for name in ("wall_ns", "capacity_wait_ns", "setup_ns"):
        require(type(record[name]) is int and record[name] >= 0, "Invalid pipeline elapsed time")
    for name in ("preprocess", "inference", "postprocess"):
        lane = record[name]
        require(isinstance(lane, dict), "Pipeline stage must preserve its fixed timing record")
        for field in ("execution_ns", "queue_ns", "calls", "completed_calls"):
            require(type(lane[field]) is int and lane[field] >= 0, "Invalid native stage accounting")
        require(lane["completed_calls"] <= lane["calls"], "Stage completed nonexistent work")
        if success:
            require(lane["calls"] == lane["completed_calls"] and lane["calls"] >= frames,
                    "Successful pipeline did not account for all frame work")
    if success:
        require(record["completed_frames"] == frames, "Pipeline lost successful frames")
        require(record["inference"]["calls"] >= frames * (2 if ocr else 1),
                "Native timing omitted OCR recognition or frame inference")
    else:
        require(record["inference"]["calls"] > record["inference"]["completed_calls"],
                "Native exception was marked as completed inference")
    # Wall, queue and frame sums overlap; there is intentionally no sum identity or speed threshold.


def warm_success(row, frames=1, *, ocr=False, count=1):
    require(row["configured"] and row["state"] == "smoke_tested" and row["loadable"] is True and
            row["smoke_tested"] and row["unloaded"], "Successful selected model not smoke-tested/unloaded")
    require([entry["phase"] for entry in row["passes"]] == ["cold", "warm"], "Missing actual cold/warm passes")
    for index, entry in enumerate(row["passes"]):
        require(entry["success"] and entry["error"] is None and entry["batch_size"] == frames and
                entry["result_counts"] == [count] * frames, "Real fixture results were not preserved as counts")
        require(entry["cache_hit"] is bool(index), "Warm model lease did not retain the cold session")
        timing = entry["timing"]
        require(type(timing["wall_ns"]) is int and timing["wall_ns"] >= 0,
                "Invalid request elapsed time")
        stage(timing["model_acquire"], completed=1)
        if index:
            require(timing["model_load"] is None, "Warm cache hit performed a model load")
        else:
            stage(timing["model_load"], completed=1)
        stage(timing["input_prepare"], completed=frames, minimum=frames)
        stage(timing["decode"], completed=frames, minimum=frames)
        pipeline(timing["pipeline"], frames, success=True, ocr=ocr and count > 0)


def configuration_cases(driver, rows, black):
    ocr = rows[1]
    # Scalar/type/range validation is covered by test_config_load; these children
    # prove accepted encodings retain the actual factory/inference default.
    for leaf in ("null", "{}", "{kernel_size: 2, dilation_iterations: 3, min_box_area: 64}",
                 '{kernel_size: "2", dilation_iterations: "3", min_box_area: "64"}'):
        driver.configure([(*ocr, leaf)])
        warm_success(driver.diagnose("warmup", ["ocr:shared"], [black])["models"][0], ocr=True)
    legacy = 'yolo:\n  - name: shared\n    version: kV10\n    path: ' + json.dumps(rows[0][3]["model"].as_posix())
    (driver.cwd / "config/models.yaml").write_text(legacy + '\n', encoding="utf-8")
    driver.diagnose("preflight", ["yolo:shared"])
    (driver.cwd / "config/models.yaml").write_text(legacy + '\n    ocr_detection: {}\n', encoding="utf-8")
    require(driver.diagnose("preflight", ["yolo:shared"], expected=1)["error"]["code"] ==
            "configuration_failed", "Legacy YOLO accepted OCR morphology")
    # Independent named sessions, in both selection orders and repeated fresh children.
    configured = [("ocr", "ordinary", ocr[2], ocr[3]),
                  ("ocr", "filtered", ocr[2], ocr[3], "{min_box_area: 1048576}")]
    driver.configure(configured)
    before = {entry.name: entry.stat().st_mtime_ns for entry in driver.cwd.iterdir()}
    for selections in (["ocr:ordinary", "ocr:filtered"], ["ocr:filtered", "ocr:ordinary"]):
        for _ in range(2):
            body = driver.diagnose("warmup", selections, [black])
            require("debug" not in body, "Ordinary diagnostics unexpectedly enabled export")
            for row in body["models"]:
                warm_success(row, ocr=True, count=0 if row["model"] == "filtered" else 1)
    require(before == {entry.name: entry.stat().st_mtime_ns for entry in driver.cwd.iterdir()},
            "Non-debug diagnostics created or changed working-directory entries")
    driver.configure(rows)


def decode_png(path):
    data = path.read_bytes()
    require(data[:8] == b"\x89PNG\r\n\x1a\n", "Export is not a PNG")
    offset, compressed, header = 8, bytearray(), None
    while offset < len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        require(len(payload) == length and offset + 12 + length <= len(data), "Truncated PNG chunk")
        crc = struct.unpack_from(">I", data, offset + 8 + length)[0]
        require(zlib.crc32(kind + payload) & 0xffffffff == crc, "Corrupt PNG checksum")
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", payload)
        elif kind == b"IDAT":
            compressed.extend(payload)
        offset += length + 12
        if kind == b"IEND":
            break
    require(header is not None, "PNG lacks dimensions")
    width, height, depth, color, compression, filtering, interlace = header
    require(depth == 8 and color == 2 and compression == filtering == interlace == 0,
            "Export must decode as noninterlaced 8-bit RGB")
    raw, stride = zlib.decompress(compressed), width * 3
    require(len(raw) == height * (stride + 1), "PNG scanline dimensions disagree")
    pixels, previous = bytearray(), bytearray(stride)
    for y in range(height):
        start = y * (stride + 1)
        method, line = raw[start], bytearray(raw[start + 1:start + 1 + stride])
        require(method <= 4, "Invalid PNG filter")
        for x in range(stride):
            left, up = (line[x - 3] if x >= 3 else 0), previous[x]
            corner = previous[x - 3] if x >= 3 else 0
            predictors = (0, left, up, (left + up) // 2)
            if method == 4:
                p = left + up - corner
                distances = (abs(p - left), abs(p - up), abs(p - corner))
                predictor = (left, up, corner)[distances.index(min(distances))]
            else:
                predictor = predictors[method]
            line[x] = (line[x] + predictor) & 255
        pixels.extend(line)
        previous = line
    return width, height, pixels


def debug_cases(driver, rows, black, white):
    driver.configure(rows)
    warm = ["--diagnose", "warmup", "--model", "ocr:shared", "--image", str(black)]
    before = set(driver.cwd.iterdir())
    for name in ("", ".", "..", "../escape", "a/b", "a\\b", "/absolute", "C:drive",
                 "a.b", "-bad", "a" * 65, "CON", "prn", "AuX", "NUL", "COM1", "lpt9"):
        driver.call([*warm, "--debug-dir", name], 2)
    for flag, maximum in (("--debug-max-bytes", 67108864), ("--debug-max-files", 64)):
        driver.call([*warm, flag, "1"], 2)
        for value in ("0", "-1", "1.5", "true", str(maximum + 1), ""):
            driver.call([*warm, "--debug-dir", "Rejected", flag, value], 2)
    for arguments in (["--debug-dir", "Rejected"],
                      ["--diagnose", "preflight", "--model", "ocr:shared", "--debug-dir", "Rejected"],
                      ["--diagnose", "warmup", "--model", "yolo:shared", "--image", str(black), "--debug-dir", "Rejected"],
                      [*warm, "--model", "ocr:missing", "--debug-dir", "Rejected"],
                      [*warm, *sum((["--image", str(black)] for _ in range(16)), []), "--debug-dir", "Rejected"]):
        driver.call(arguments, 2)
    require(set(driver.cwd.iterdir()) == before, "Rejected syntax created export artifacts")
    owned = driver.cwd / "ProtectedDirectory"
    owned.mkdir()
    canary = owned / "user-file"
    canary.write_bytes(TOKEN.encode())
    file = driver.cwd / "ProtectedFile"
    driver.forbidden += ["QuotaRollback", "InferenceRollback", "SixteenFrames", "StdoutRollback", "WriteRollback",
                         "ProtectedDirectory", "ProtectedFile", "ProtectedLink"]
    file.write_bytes(TOKEN.encode())
    for path in (owned, file):
        driver.call([*warm, "--debug-dir", path.name], 1)
    require(canary.read_bytes() == file.read_bytes() == TOKEN.encode(), "Existing user files changed")
    link = driver.cwd / "ProtectedLink"
    try:
        link.symlink_to(owned, target_is_directory=True)
    except OSError:
        if os.name == "nt":
            child = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(owned)], capture_output=True)
            require(child.returncode == 0, "Cannot exercise Windows junction rejection")
        else:
            raise
    driver.call([*warm, "--debug-dir", link.name], 1)
    require(canary.read_bytes() == TOKEN.encode() and link.is_dir(), "Link target changed on rejection")
    # A one-file cap necessarily fails after writing the original first PNG.
    for flag in ("--debug-max-bytes", "--debug-max-files"):
        body = driver.call([*warm, "--debug-dir", "QuotaRollback", flag, "1"], 1)
        require(body["debug"]["retained"] is False and body["debug"]["error"] == "debug_quota",
                "Quota failure was not safely classified")
        warm_success(body["models"][0], ocr=True)
        require(not (driver.cwd / "QuotaRollback").exists(), "Quota failure left owned files")
        verify_debug(driver, warm, "QuotaRollback", 1)
    body = driver.call(["--diagnose", "warmup", "--model", "ocr:shared", "--image", str(white),
                        "--debug-dir", "InferenceRollback"], 1)
    require(body["models"][0]["unloaded"] and not (driver.cwd / "InferenceRollback").exists(),
            "Inference failure retained export artifacts/session")
    verify_debug(driver, [*warm, *sum((["--image", str(black)] for _ in range(15)), [])], "SixteenFrames", 16)
    size = verify_debug(driver, warm, "QuotaRollback", 1)
    verify_debug(driver, [*warm, "--debug-max-files", "3", "--debug-max-bytes", str(size)], "QuotaRollback", 1)
    if sys.platform.startswith("linux"):
        import resource
        import signal

        def restrict_file_size():
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE, (32, 32))

        child = subprocess.run([str(driver.executable), *warm, "--debug-dir", "WriteRollback"],
                               cwd=driver.cwd, env=driver.environment, capture_output=True,
                               timeout=180, preexec_fn=restrict_file_size)
        require(child.returncode == 1 and not (driver.cwd / "WriteRollback").exists(),
                "Actual EFBIG write failure did not roll back owned artifacts")
        captured = child.stdout.decode("utf-8", errors="strict") + child.stderr.decode("utf-8", errors="strict")
        require(all(value not in captured for value in driver.forbidden), "Write failure leaked private values")
        report = json.loads(child.stdout)
        require(report["debug"]["retained"] is False and all(row["unloaded"] for row in report["models"]),
                "Write failure leaked model lease or claimed retained artifacts")
        warm_success(report["models"][0], ocr=True)
        verify_debug(driver, warm, "WriteRollback", 1)
        # Restore the normal SIGPIPE disposition in the exec'd child: Python
        # otherwise ignores it and would conceal a native rollback bypass.
        driver.forbidden.append("ClosedReaderRollback")
        read_fd, write_fd = os.pipe()
        os.close(read_fd)
        try:
            child = subprocess.run([str(driver.executable), *warm, "--debug-dir", "ClosedReaderRollback"],
                                   cwd=driver.cwd, env=driver.environment, stdout=write_fd,
                                   stderr=subprocess.PIPE, timeout=180, restore_signals=True)
        finally:
            os.close(write_fd)
        require(child.returncode == 1 and not (driver.cwd / "ClosedReaderRollback").exists(),
                "Closed stdout consumer bypassed artifact rollback")
        stderr = child.stderr.decode("utf-8", errors="strict")
        require(all(value not in stderr for value in driver.forbidden), "Broken-pipe failure leaked private values")
        verify_debug(driver, warm, "ClosedReaderRollback", 1)
    full = Path("/dev/full")
    if sys.platform.startswith("linux") and full.is_char_device():
        with full.open("wb", buffering=0) as output:
            child = subprocess.run([str(driver.executable), *warm, "--debug-dir", "StdoutRollback"],
                                   cwd=driver.cwd, env=driver.environment, stdout=output,
                                   stderr=subprocess.PIPE, timeout=180)
        require(child.returncode == 1 and not (driver.cwd / "StdoutRollback").exists(),
                "Failed stdout commit retained images")
        stderr = child.stderr.decode("utf-8", errors="strict")
        require(all(value not in stderr for value in driver.forbidden), "Stdout failure leaked private values")


def verify_debug(driver, warm, name, frames):
    body = driver.call([*warm, "--debug-dir", name], 0)
    warm_success(body["models"][0], frames, ocr=True)
    directory = driver.cwd / name
    if os.name == "nt":
        environment = driver.environment.copy()
        environment["VISION_SIMPLE_ACL_TARGET"] = str(directory)
        command = ('$ErrorActionPreference="Stop"; $a=Get-Acl -LiteralPath $env:VISION_SIMPLE_ACL_TARGET; '
                   '$sid=[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value; '
                   'if (!$a.AreAccessRulesProtected -or $a.Access.Count -ne 1) {exit 1}; '
                   '$r=$a.Access[0]; '
                   'if ($r.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value -ne $sid '
                   '-or $r.AccessControlType -ne "Allow" -or $r.IsInherited) {exit 1}')
        checked = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                                 env=environment, capture_output=True, timeout=30)
        require(checked.returncode == 0, "Debug root DACL is not protected/current-user-only")
    if os.name != "nt":
        require(directory.stat().st_mode & 0o777 == 0o700, "Debug root is not owner-only")
    files = list(directory.iterdir())
    expected = {"manifest.json"} | {f"{prefix}-{index:03}.png" for index in range(frames)
                                   for prefix in ("input", "boxes")}
    require({file.name for file in files} == expected, "Export omitted images or created unexpected files")
    summary = body["debug"]
    require(summary == {"files": len(files), "bytes": sum(file.stat().st_size for file in files),
                        "retained": True, "error": None}, "Export summary disagrees with actual disk contents")
    manifest_text = (directory / "manifest.json").read_text(encoding="utf-8")
    require(all(private not in manifest_text for private in driver.forbidden), "Manifest leaked private values")
    manifest = json.loads(manifest_text)
    require(manifest["ocr_detection"] == {"kernel_size": 2, "dilation_iterations": 3, "min_box_area": 64} and
            manifest["recognition_confidence"] == 0.125, "Manifest omitted effective parameters")
    require(len(manifest["frames"]) == frames, "Manifest lost a frame")
    for index, frame in enumerate(manifest["frames"]):
        width, height, original = decode_png(directory / f"input-{index:03}.png")
        ow, oh, overlay = decode_png(directory / f"boxes-{index:03}.png")
        require((width, height) == (ow, oh) == (32, 32) and
                (frame["index"], frame["width"], frame["height"], frame["box_count"]) == (index, 32, 32, 1),
                "Manifest/PNG geometry disagrees with actual fixture")
        require(original == bytes(32 * 32 * 3), "Export did not preserve original RGB pixels")
        box, = frame["boxes"]
        x, y, w, h = box["bbox"]
        require(box["index"] == 0 and 0 <= box["confidence"] <= 1 and
                0 <= x < x + w <= width and 0 <= y < y + h <= height, "Invalid exported bounding box")
        changed = {(p % width, p // width) for p in range(width * height)
                   if overlay[p * 3:p * 3 + 3] != original[p * 3:p * 3 + 3]}
        require(any(x - 1 <= px <= x + w + 1 and y - 1 <= py <= y + h + 1 and
                    (abs(px - x) <= 1 or abs(px - (x + w)) <= 1 or
                     abs(py - y) <= 1 or abs(py - (y + h)) <= 1) for px, py in changed),
                "Overlay did not draw the manifest bounding rectangle")
    # Successful export is explicitly retained; the fixture owner performs manual cleanup.
    for file in files:
        file.unlink()
    directory.rmdir()
    return summary["bytes"]



def run(args):
    root, executable = args.project_root.resolve(strict=True), args.server.resolve(strict=True)
    assets = root / "app/assets/test"
    names = ("yolo_runtime_failure.onnx", "ocr_det_box.onnx", "ocr_rec_batch.onnx")
    for name in names:
        contents = (assets / name).read_bytes()
        require(contents and not contents.startswith(b"version https://git-lfs.github.com/spec"),
                "Tiny genuine ONNX diagnostic fixture is not materialized")
    with tempfile.TemporaryDirectory(prefix="vision-simple-diagnostics-private-") as temporary:
        cwd = Path(temporary)
        driver = Driver(executable, cwd)
        driver.forbidden += [str(root), root.as_posix(), json.dumps(str(root))[1:-1]]
        driver.canary_kind.update({value: "project" for value in (str(root), root.as_posix(), json.dumps(str(root))[1:-1])})
        driver.call(["--help"], 0, help_text=True)
        invalid = (["--unknown"], ["--diagnose"], ["--diagnose", "bad"],
                   ["--diagnose", "preflight"], ["--model", "yolo:shared"],
                   ["--help", "--unknown"], ["--diagnose", "warmup", "--model", "yolo:shared"],
                   ["--diagnose", "preflight", "--model", "shared"],
                   ["--diagnose", "preflight", "--model", "no-task:shared"],
                   ["--diagnose", "preflight", "--model", "yolo:"],
                   ["--diagnose", "preflight", "--model", "yolo:shared", "--timeout-ms"],
                   ["--diagnose", "warmup", "--model", "yolo:shared", "--image"],
                   ["--diagnose", "preflight", "--model"],
                   ["--diagnose", "preflight", "--model", "yolo:shared", "--image", "absent"],
                   ["--diagnose", "preflight", "--model", "yolo:shared", "--model", "yolo:shared"])
        for arguments in invalid:
            require(driver.call(arguments, 2)["error"]["code"] == "invalid_arguments", "Misuse was not normalized")
        for timeout in ("0", "300001", "-1", "1.5", "abc", ""):
            driver.call(["--diagnose", "preflight", "--model", "yolo:shared", "--timeout-ms", timeout], 2)
        driver.call(["--diagnose", "preflight", *sum((["--model", f"yolo:model{i}"] for i in range(17)), [])], 2)
        driver.call(["--diagnose", "warmup", "--model", "yolo:shared",
                     *sum((["--image", "absent"] for _ in range(129)), [])], 2)
        require(driver.diagnose("preflight", ["yolo:shared"], expected=1)["error"]["code"] ==
                "configuration_failed", "Missing local configuration was not operational failure")

        dictionary = cwd / "private-dictionary.txt"
        # File-based PP-OCR appends its space class; three entries + space + blank match C5.
        dictionary.write_text(TEXT + "\nB\nC\n", encoding="utf-8")
        corrupt = cwd / "private-corrupt.onnx"
        corrupt.write_bytes(b"not an ONNX model " + TOKEN.encode())
        rows = [("yolo", "shared", "kV10", {"model": assets / names[0]}),
                ("ocr", "shared", "kPPOCRv4", {"det": assets / names[1], "rec": assets / names[2],
                                               "dictionary": dictionary}),
                ("yolo", "missing", "kV10", {"model": cwd / "private-missing.onnx"}),
                ("ocr", "missing", "kPPOCRv4", {"det": cwd / "absent-det", "rec": cwd / "absent-rec",
                                                "dictionary": cwd / "absent-dictionary"}),
                ("yolo", "corrupt", "kV10", {"model": corrupt})]
        driver.configure(rows)
        driver.output_failure()
        black, white = cwd / "private-black.ppm", cwd / "private-white.ppm"
        for image, value in ((black, 0), (white, 255)):
            payload = ppm(value)
            image.write_bytes(payload)
            driver.forbidden += [base64.b64encode(payload).decode("ascii"), str(image), image.as_posix()]
            driver.canary_kind.update({base64.b64encode(payload).decode("ascii"): "base64",
                                       str(image): "fixture", image.as_posix(): "fixture"})
        configuration_cases(driver, rows, black)
        debug_cases(driver, rows, black, white)
        body = driver.diagnose("preflight", ["yolo:unknown", "yolo:missing", "ocr:missing", "yolo:corrupt"], expected=1)
        unknown, missing, missing_ocr, broken = body["models"]
        require(not unknown["configured"] and unknown["state"] == "not_configured" and
                unknown["loadable"] is None and unknown["passes"] == [], "Unknown model attempted loading")
        for row, roles in ((missing, {"model"}), (missing_ocr, {"det", "rec", "dictionary"})):
            require(row["configured"] and row["state"] == "missing_files" and row["loadable"] is None and
                    set(row["missing_roles"]) == roles and row["passes"] == [], "Missing file roles/load state conflated")
        require(broken["state"] == "load_failed" and broken["loadable"] is False and
                not broken["smoke_tested"], "Corrupt existing model not distinguished from missing file")
        stage(broken["passes"][0]["timing"]["model_load"], completed=0)

        body = driver.diagnose("preflight", ["yolo:shared", "ocr:shared"], extra=("--timeout-ms", "300000"))
        require(body["limits"]["timeout_ms"] == 300000, "Explicit diagnostic timeout not honored")
        for row in body["models"]:
            require(row["state"] == "loadable" and row["loadable"] is True and not row["smoke_tested"] and
                    row["unloaded"], "Preflight claimed a smoke test or failed unloading")
            entry, = row["passes"]
            require(entry["phase"] == "preflight" and entry["success"] and entry["batch_size"] == 0 and
                    entry["result_counts"] == [] and not entry["cache_hit"], "Preflight performed frame inference")
            stage(entry["timing"]["model_load"], completed=1)
            require(entry["timing"]["input_prepare"] is None and entry["timing"]["decode"] is None,
                    "Preflight decoded nonexistent images")
            native = entry["timing"]["pipeline"]
            if native is not None:
                require(all(native[name]["calls"] == 0 for name in ("preprocess", "inference", "postprocess")),
                        "Preflight ran native stages")

        for _ in range(2):
            body = driver.diagnose("warmup", ["yolo:shared", "ocr:shared"], [black, black])
            for row in body["models"]:
                warm_success(row, 2, ocr=row["task"] == "ocr")
        for selection in ("yolo:shared", "ocr:shared"):
            row, = driver.diagnose("warmup", [selection], [white], expected=1)["models"]
            require(row["state"] == "smoke_failed" and row["loadable"] is True and
                    not row["smoke_tested"] and row["unloaded"], "Load/smoke/native error states conflated")
            entry, = row["passes"]
            require(entry["phase"] == "cold" and not entry["success"] and
                    entry["error"] == {"code": "inference_failed", "image_index": 0},
                    "Failed cold pass fabricated warm success or exposed native error")
            pipeline(entry["timing"]["pipeline"], 1, success=False)

        malformed = cwd / "private-malformed.ppm"
        malformed.write_bytes(b"P6\n32 32\n255\n")
        for image, overrides, code in ((malformed, {}, "invalid_image"),
                                       (black, {"infer_max_image_pixels": "16"}, "image_limit_exceeded"),
                                       (black, {"infer_max_batch_decoded_bytes": "16"}, "image_limit_exceeded"),
                                       (black, {"infer_max_inflight_decoded_bytes": "16"}, "service_overloaded")):
            driver.configure(rows, **overrides)
            row, = driver.diagnose("warmup", ["yolo:shared"], [image], expected=1)["models"]
            entry, = row["passes"]
            require(entry["error"]["code"] == code and not entry["success"] and row["unloaded"],
                    "Malformed/resource-limited fixture escaped the shared service budgets")
            require(entry["timing"]["pipeline"] is None, "Rejected input entered inference")
        driver.configure(rows, infer_max_batch_images="1")
        driver.diagnose("warmup", ["yolo:shared"], [black, black], expected=2)
        driver.configure(rows)
        driver.diagnose("warmup", ["yolo:shared"], [cwd / "private-absent.ppm"], expected=1)
        oversized = cwd / "private-oversized.ppm"
        with oversized.open("wb") as output:
            output.truncate(48 * 1024 * 1024 + 1)
        require(driver.diagnose("warmup", ["yolo:shared"], [oversized], expected=1)["error"]["code"] ==
                "fixture_failed", "Raw diagnostic fixture limit not enforced")
        # Two individually bounded reads must obey the invocation-wide raw/encoded sums.
        first, second = cwd / "private-large-first.ppm", cwd / "private-large-second.ppm"
        for file, size in ((first, 24 * 1024 * 1024 + 1), (second, 24 * 1024 * 1024 - 1)):
            with file.open("wb") as output:
                output.truncate(size)
        require(driver.diagnose("warmup", ["yolo:shared"], [first, second], expected=1)["error"]["code"] ==
                "fixture_failed", "Encoded fixture sum exceeded 64 MiB without rejection")
    print("PASS diagnostics: arguments, selected preflight, real cold/warm/OCR/failure, leases, budgets and safe output")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        run(args)
    except (RegressionFailure, OSError, UnicodeError, KeyError, TypeError, ValueError) as error:
        print(f"FAIL diagnostic regression: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
