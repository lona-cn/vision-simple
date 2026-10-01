#!/usr/bin/env python3
"""Exercise selected-model diagnostics as real children, without HTTP or a listener."""

import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


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
    for task, name, version, files in rows:
        text += f"  - task: {json.dumps(task)}\n    name: {json.dumps(name)}\n"
        text += f"    version: {json.dumps(version)}\n    files:\n"
        text += "".join(f"      {role}: {json.dumps(path.as_posix())}\n" for role, path in files.items())
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


def warm_success(row, frames=1, *, ocr=False):
    require(row["configured"] and row["state"] == "smoke_tested" and row["loadable"] is True and
            row["smoke_tested"] and row["unloaded"], "Successful selected model not smoke-tested/unloaded")
    require([entry["phase"] for entry in row["passes"]] == ["cold", "warm"], "Missing actual cold/warm passes")
    for index, entry in enumerate(row["passes"]):
        require(entry["success"] and entry["error"] is None and entry["batch_size"] == frames and
                entry["result_counts"] == [1] * frames, "Real fixture results were not preserved as counts")
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
        pipeline(timing["pipeline"], frames, success=True, ocr=ocr)


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
