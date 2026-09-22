#!/usr/bin/env python3
"""Real FFmpeg/PP-OCR subtitle HTTP regression; owns its server and temporary media."""

import argparse
import base64
import contextlib
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote

from test_http_regression import RegressionFailure, Server, fixture, infer, require


ROOT = "/v1/subtitle/jobs"
TERMINAL = {"completed", "failed", "cancelled"}
OPTIONS = {"model": "ppocr-v4", "roi": [0, 0, 1, 1],
           "sample_interval_ms": 100, "stable_samples": 2, "gap_samples": 2,
           "min_confidence": 0.8}


def ffmpeg_run(executable, cwd, *arguments):
    result = subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-nostdin",
                             "-y", *arguments], cwd=cwd, capture_output=True, text=True,
                            timeout=120, check=False)
    require(result.returncode == 0, f"FFmpeg fixture generation failed: {result.stderr}")


def make_media(executable, directory, font):
    # A relative font name avoids Windows drive-colon escaping in FFmpeg filters.
    shutil.copyfile(font, directory / "font.ttf")
    draw = "drawtext=fontfile=font.ttf:fontsize=80:fontcolor=black:x=(w-tw)/2:y=(h-th)/2"
    for text in ("HELLO", "WORLD", "NOISE"):
        ffmpeg_run(executable, directory, "-f", "lavfi", "-i", "color=white:s=640x160:r=10",
                   "-vf", f"{draw}:text={text}", "-frames:v", "1", "-threads", "1", f"{text}.png")
    filters = (f"{draw}:text=HELLO:enable='between(n,10,49)*not(eq(n,30))',"
               f"{draw}:text=NOISE:enable='eq(n,30)',"
               f"{draw}:text=WORLD:enable='between(n,50,69)'")
    ffmpeg_run(executable, directory, "-f", "lavfi", "-i", "color=white:s=640x160:r=10:d=8",
               "-vf", filters, "-an", "-c:v", "mjpeg", "-q:v", "2", "-threads", "1", "semantic.avi")
    # Drop alternating frames only within one interval: presentation spacing is
    # genuinely variable, not just a CFR stream stored under a different suffix.
    # WORLD extends to video EOF, so using the 12s audio/container duration fails.
    ffmpeg_run(executable, directory, "-i", "semantic.avi", "-f", "lavfi", "-i",
               "sine=frequency=440:sample_rate=48000:duration=12", "-vf",
               f"{draw}:text=WORLD:enable='gte(t,7)',select='not(between(n,20,39)*mod(n,2))'",
               "-fps_mode", "vfr", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12",
               "-pix_fmt", "yuv420p", "-video_track_timescale", "10000", "-c:a", "aac",
               "-movflags", "+faststart", "vfr-audio-tail.mp4")
    ffmpeg_run(executable, directory, "-stream_loop", "14", "-i", "semantic.avi", "-t", "120",
               "-an", "-c:v", "copy", "cancellation.avi")
    # A second observable transcript exposes cross-job buffer/result ownership.
    ffmpeg_run(executable, directory, "-loop", "1", "-framerate", "10", "-i", "WORLD.png",
               "-t", "2", "-an", "-c:v", "mjpeg", "-q:v", "2", "-threads", "1", "world.avi")
    return {name: (directory / name).read_bytes() for name in
            ("semantic.avi", "vfr-audio-tail.mp4", "cancellation.avi", "world.avi")}


def request(server, method, route, body=None, media=None, headers=None, timeout=20):
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=timeout)
    values = dict(headers or {})
    if media:
        values["Content-Type"] = media
    try:
        connection.request(method, route, body=body, headers=values)
        response = connection.getresponse()
        data = response.read()
        content_type = response.getheader("Content-Type", "").split(";", 1)[0].lower()
        parsed = json.loads(data) if content_type == "application/json" and data else data
        return response.status, parsed
    finally:
        connection.close()


def expect(response, status, label):
    require(response[0] == status, f"{label}: expected HTTP {status}, got {response}")
    return response[1]


def create(server, options=None):
    job = expect(request(server, "POST", ROOT, json.dumps(options or OPTIONS).encode(),
                         "application/json"), 201, "create")
    require(isinstance(job, dict) and re.fullmatch(r"[0-9a-f]{32}", job.get("id", "")),
            f"Create did not return a usable job identifier: {job}")
    return job["id"]


def info(server, identifier):
    return expect(request(server, "GET", f"{ROOT}/{identifier}"), 200, "job status")


def wait_state(server, identifier, states=TERMINAL, timeout=180, history=None):
    deadline = time.monotonic() + timeout
    previous = None
    while time.monotonic() < deadline:
        current = info(server, identifier)
        if history is not None:
            history.append(current)
        if previous:
            for field in ("uploaded_bytes", "decoded_frames", "sampled_frames", "position_ms"):
                require(current[field] >= previous[field], f"Progress regressed: {previous} -> {current}")
        if current["state"] in states:
            return current
        require(current["state"] not in TERMINAL, f"Job terminated before {states}: {current}")
        previous = current
        time.sleep(0.03)
    raise RegressionFailure(f"Job {identifier} did not reach {states}: {info(server, identifier)}")


def delete(server, identifier):
    expect(request(server, "DELETE", f"{ROOT}/{identifier}"), 204, "delete")
    expect(request(server, "GET", f"{ROOT}/{identifier}"), 404, "deleted job")


def cancel(server, identifier):
    expect(request(server, "POST", f"{ROOT}/{identifier}/cancel", b"{}", "application/json"),
           202, "cancel")
    require(wait_state(server, identifier)["state"] == "cancelled", "Cancellation did not terminate")
    expect(request(server, "GET", f"{ROOT}/{identifier}/subtitles.srt"), 409, "cancelled result")


def upload(server, identifier, data):
    return request(server, "PUT", f"{ROOT}/{identifier}/video", data, "application/octet-stream")


def no_source_files(temp_root, identifiers):
    deadline = time.monotonic() + 5
    while True:
        leftovers = [path for path in temp_root.rglob("*") if path.is_file() and
                     any(identifier in path.name for identifier in identifiers)]
        if not leftovers:
            return
        require(time.monotonic() < deadline, f"Terminal jobs retained uploaded source files: {leftovers}")
        time.sleep(0.05)


def partial_upload(server, identifier, data, *, chunked=False):
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=20)
    connection.putrequest("PUT", f"{ROOT}/{identifier}/video")
    connection.putheader("Content-Type", "application/octet-stream")
    connection.putheader("Transfer-Encoding" if chunked else "Content-Length",
                         "chunked" if chunked else str(len(data)))
    connection.endheaders()
    return connection


def send_piece(connection, piece, chunked):
    connection.send((f"{len(piece):x}\r\n".encode() + piece + b"\r\n") if chunked else piece)


def incremental_upload(server, identifier, data, temp_root, *, chunked):
    connection = partial_upload(server, identifier, data, chunked=chunked)
    try:
        first = min(16384, len(data) // 2)
        send_piece(connection, data[:first], chunked)
        deadline = time.monotonic() + 10
        while True:
            current = info(server, identifier)
            if current["uploaded_bytes"] == first:
                break
            require(time.monotonic() < deadline, f"Upload was buffered instead of incremental: {current}")
            time.sleep(0.03)
        require(current["state"] == "uploading", f"Partial body was prematurely scheduled: {current}")
        expect(upload(server, identifier, b"not another owner's video"), 409, "duplicate active upload")
        require(info(server, identifier)["state"] == "uploading", "Duplicate upload aborted the owner")
        expect(request(server, "GET", f"{ROOT}/{identifier}/subtitles.srt"), 409, "result during upload")
        expect(request(server, "DELETE", f"{ROOT}/{identifier}"), 409, "delete during upload")
        for offset in range(first, len(data), 16384):
            send_piece(connection, data[offset:offset + 16384], chunked)
        if chunked:
            connection.send(b"0\r\n\r\n")
        response = connection.getresponse()
        body = response.read()
        require(response.status == 202, f"Upload completion: {response.status} {body!r}")
    finally:
        connection.close()
    expect(upload(server, identifier, b"duplicate"), 409, "duplicate scheduled upload")


def timestamp(text, separator):
    match = re.fullmatch(r"(\d{2,}):(\d{2}):(\d{2})" + re.escape(separator) + r"(\d{3})", text)
    require(match is not None, f"Not a millisecond subtitle timestamp: {text!r}")
    hours, minutes, seconds, millis = map(int, match.groups())
    require(minutes < 60 and seconds < 60, f"Invalid clock timestamp: {text}")
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def parse_cues(data, webvtt):
    text = data.decode("utf-8-sig").replace("\r\n", "\n")
    if webvtt:
        require(text.startswith("WEBVTT\n\n"), "Missing WEBVTT signature")
        text = text[len("WEBVTT\n\n"):]
    cues = []
    for number, block in enumerate(text.strip().split("\n\n"), 1):
        lines = block.splitlines()
        if not webvtt:
            require(lines and lines.pop(0) == str(number), f"Invalid SRT cue numbering: {block}")
        require(len(lines) >= 2, f"Incomplete subtitle cue: {block}")
        times = lines.pop(0).split(" --> ")
        require(len(times) == 2, f"Invalid cue interval: {block}")
        start, end = (timestamp(value, "." if webvtt else ",") for value in times)
        require(0 <= start < end and (not cues or start >= cues[-1][1]),
                f"Invalid/overlapping cue bounds: {block}")
        cues.append((start, end, "\n".join(lines)))
    return cues


def transcript(server, identifier, expected, tolerance=1, video_end=8000):
    formats = []
    for suffix in ("srt", "vtt"):
        data = expect(request(server, "GET", f"{ROOT}/{identifier}/subtitles.{suffix}"),
                      200, f"download {suffix}")
        formats.append(parse_cues(data, suffix == "vtt"))
    require(formats[0] == formats[1], f"SRT/VTT semantics differ: {formats}")
    cues = formats[0]
    require([cue[2] for cue in cues] == [cue[2] for cue in expected],
            f"Wrong semantic cues (noise, duplicates, glyph crop, or ownership): {cues}")
    for actual, target in zip(cues, expected):
        require(abs(actual[0] - target[0]) <= tolerance and abs(actual[1] - target[1]) <= tolerance,
                f"Wrong presentation timing: {actual}, expected {target}, tolerance {tolerance}ms")
        require(actual[1] <= video_end, f"Cue extends past video EOF: {actual}")
    return cues


def completed(server, identifier, data, temp_root, *, chunked=False, expected=None,
              tolerance=1, video_end=8000):
    incremental_upload(server, identifier, data, temp_root, chunked=chunked)
    history = []
    final = wait_state(server, identifier, history=history)
    require(final["state"] == "completed", f"Valid video failed: {final}")
    require(any(row["state"] == "running" and row["sampled_frames"] > 0 for row in history),
            f"No observable incremental decoding/OCR progress: {history}")
    transcript(server, identifier, expected or [(1000, 5000, "HELLO"), (5000, 7000, "WORLD")],
               tolerance, video_end)
    no_source_files(temp_root, [identifier])


def read_status(sock):
    data = bytearray()
    while not data.endswith(b"\r\n\r\n"):
        piece = sock.recv(1)
        require(piece, f"Connection closed before response headers: {data!r}")
        data.extend(piece)
        require(len(data) <= 65536, "Unbounded HTTP response headers")
    match = re.match(rb"HTTP/1\.[01] (\d{3})\b", data)
    require(match is not None, f"Malformed HTTP response: {data!r}")
    return int(match[1])


def header_only(server, method, route, headers, expected):
    with socket.create_connection(("127.0.0.1", server.port), timeout=5) as sock:
        wire = f"{method} {route} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n"
        wire += "".join(f"{key}: {value}\r\n" for key, value in headers.items()) + "\r\n"
        sock.sendall(wire.encode("ascii"))
        require(read_status(sock) == expected, f"Header-only rejection did not return {expected}: {headers}")


def validation_matrix(server):
    for patch in ({"sample_interval_ms": 0}, {"sample_interval_ms": 99},
                  {"sample_interval_ms": 5001}, {"sample_interval_ms": True},
                  {"stable_samples": 1}, {"gap_samples": 11}, {"min_confidence": 1.1},
                  {"roi": [0.8, 0, 0.5, 1]}, {"roi": [0, 0, 0, 1]},
                  {"roi": [0, 0, 1]}, {"url": "https://example.invalid/video"},
                  {"path": "video.avi"}, {"model": 7}):
        expect(request(server, "POST", ROOT, json.dumps(OPTIONS | patch).encode(), "application/json"),
               400, f"strict options {patch}")
    for raw in (b"{}", b"[]", b"{", b"null"):
        expect(request(server, "POST", ROOT, raw, "application/json"), 400, "invalid create JSON")
    expect(request(server, "POST", ROOT, json.dumps(OPTIONS | {"model": "absent"}).encode(),
                   "application/json"), 404, "unknown OCR model")
    absent = "0" * 32
    for method, suffix, body, media in (("GET", "", None, None), ("DELETE", "", None, None),
            ("GET", "/subtitles.srt", None, None), ("POST", "/cancel", b"{}", "application/json"),
            ("PUT", "/video", b"invalid", "application/octet-stream")):
        expect(request(server, method, f"{ROOT}/{absent}{suffix}", body, media), 404, "missing ID")
    for query in ("limit=0", "limit=101", "limit=x", "cursor=bad"):
        expect(request(server, "GET", f"{ROOT}?{query}"), 400, "invalid pagination")
    job = create(server)
    expect(request(server, "GET", f"{ROOT}/{job}/subtitles.srt"), 409, "result before ready")
    expect(request(server, "PUT", f"{ROOT}/{job}/video", b"x", "video/avi"), 415, "wrong media")
    for method, route in (("GET", ROOT), ("GET", f"{ROOT}/{job}"),
                          ("GET", f"{ROOT}/{job}/subtitles.srt"), ("DELETE", f"{ROOT}/{job}")):
        header_only(server, method, route, {"Content-Length": "1"}, 400)
        header_only(server, method, route, {"Transfer-Encoding": "chunked"}, 400)
        header_only(server, method, route, {"Expect": "100-continue"}, 417)
    for method, route, media, length in (("PUT", f"{ROOT}/{job}/video", "application/octet-stream", 64 * 1024 * 1024 + 1),
                                        ("POST", ROOT, "application/json", 65537)):
        for expectation in (None, "100-continue"):
            headers = {"Content-Type": media, "Content-Length": str(length)}
            if expectation:
                headers["Expect"] = expectation
            header_only(server, method, route, headers, 413)
    header_only(server, "PUT", f"{ROOT}/{job}/video",
                {"Content-Type": "application/octet-stream", "Content-Length": "32", "Expect": "other"}, 417)
    header_only(server, "PUT", f"{ROOT}/{job}/video",
                {"Content-Type": "application/octet-stream", "Content-Length": "32",
                 "Origin": "https://example.invalid", "Expect": "100-continue"}, 403)
    header_only(server, "GET", ROOT, {"Origin": "https://example.invalid"}, 403)
    require(info(server, job)["state"] == "created", "Rejected headers claimed or changed upload ownership")
    delete(server, job)


def expect_continue(server, data, temp_root):
    job = create(server)
    with socket.create_connection(("127.0.0.1", server.port), timeout=10) as sock:
        sock.sendall((f"PUT {ROOT}/{job}/video HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                      f"Content-Type: application/octet-stream\r\nContent-Length: {len(data)}\r\n"
                      "Expect: 100-CoNtInUe\r\nConnection: close\r\n\r\n").encode())
        require(read_status(sock) == 100, "Case-insensitive 100-continue was not acknowledged before body")
        sock.sendall(data)
        require(read_status(sock) == 202, "Continued upload was not admitted")
    require(wait_state(server, job)["state"] == "completed", "Continued upload did not complete")
    transcript(server, job, [(0, 2000, "WORLD")], video_end=2000)
    no_source_files(temp_root, [job])
    delete(server, job)


def oversized_jpeg_avi():
    # Small, structurally complete AVI with a header-only JPEG. Its AVI header
    # says 640x160, but the embedded SOF claims 8192x8192 (>16MP). No oversized
    # pixel/entropy buffer is generated or sent.
    jpeg = (b"\xff\xd8\xff\xc0\x00\x11\x08\x20\x00\x20\x00\x03"
            b"\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xd9")

    def chunk(tag, payload):
        return tag + len(payload).to_bytes(4, "little") + payload + (b"\0" if len(payload) % 2 else b"")

    def header(size, fields):
        result = bytearray(size)
        for offset, value in fields.items():
            result[offset:offset + 4] = value.to_bytes(4, "little")
        return result

    main = header(56, {0: 100000, 16: 1, 24: 1, 28: len(jpeg), 32: 640, 36: 160})
    stream = header(56, {20: 1, 24: 10, 32: 1, 36: len(jpeg), 40: 0xffffffff})
    stream[:8] = b"vidsMJPG"
    stream[52:56] = b"\x80\x02\xa0\x00"
    bitmap = header(40, {0: 40, 4: 640, 8: 160, 20: len(jpeg)})
    bitmap[12:20] = b"\x01\x00\x18\x00MJPG"
    stream_list = chunk(b"LIST", b"strl" + chunk(b"strh", stream) + chunk(b"strf", bitmap))
    headers = chunk(b"LIST", b"hdrl" + chunk(b"avih", main) + stream_list)
    frames = chunk(b"LIST", b"movi" + chunk(b"00dc", jpeg))
    return chunk(b"RIFF", b"AVI " + headers + frames)


def invalid_media_matrix(server, media, temp_root):
    truncated = media[:len(media) // 2]
    # Also retain a self-consistent outer RIFF length, so header sniffing alone
    # cannot pass as complete decode validation (the movi chunk is truncated).
    forged = bytearray(truncated)
    forged[4:8] = (len(forged) - 8).to_bytes(4, "little")
    for label, data in (("empty", b""), ("invalid", b"not a video"),
                        ("truncated", truncated), ("truncated body", bytes(forged)),
                        ("oversized embedded JPEG", oversized_jpeg_avi())):
        job = create(server)
        status, body = upload(server, job, data)
        require(status in (202, 415), f"{label}: unexpected upload response {status} {body}")
        final = wait_state(server, job)
        require(final["state"] == "failed", f"{label}: invalid video produced successful subtitles: {final}")
        if label == "oversized embedded JPEG":
            require(final["error_code"] == "frame_dimensions" and final["decoded_frames"] == 0,
                    f"Embedded JPEG dimensions were not rejected before producing a frame: {final}")
        expect(request(server, "GET", f"{ROOT}/{job}/subtitles.srt"), 409, "failed decode result")
        no_source_files(temp_root, [job])
        delete(server, job)


def disconnect_matrix(server, media, temp_root):
    for chunked in (False, True):
        job = create(server)
        connection = partial_upload(server, job, media, chunked=chunked)
        try:
            send_piece(connection, media[:16384], chunked)
            wait_state(server, job, {"uploading"}, timeout=10)
            # Observe an actual private file before asserting that it disappears.
            deadline = time.monotonic() + 5
            while not any(job in path.name for path in temp_root.rglob("*")):
                require(time.monotonic() < deadline, "Owned TMP environment did not expose the upload source")
                time.sleep(0.03)
        finally:
            connection.close()
        final = wait_state(server, job, timeout=15)
        require(final["state"] == "failed" and final["error_code"] == "upload_interrupted",
                f"Disconnected upload was not terminally aborted: {final}")
        no_source_files(temp_root, [job])
        delete(server, job)


def cancellation_and_isolation(server, media, temp_root):
    running = create(server)
    expect(upload(server, running, media["cancellation.avi"]), 202, "long upload")
    wait_state(server, running, {"running"}, timeout=30)
    queued = create(server)
    expect(upload(server, queued, media["semantic.avi"]), 202, "queued upload")
    require(info(server, queued)["state"] == "queued", "Fixture did not hold worker for queued cancellation")
    cancel(server, queued)
    require(info(server, running)["state"] == "running", "Cancelling queued job disturbed running owner")
    cancel(server, running)
    no_source_files(temp_root, [running, queued])
    delete(server, queued)
    delete(server, running)
    first, second = create(server), create(server)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(upload, server, first, media["semantic.avi"]),
                   pool.submit(upload, server, second, media["world.avi"])]
        for future in futures:
            expect(future.result(timeout=30), 202, "concurrent upload")
    for job in (first, second):
        require(wait_state(server, job)["state"] == "completed", "Isolated concurrent job failed")
    transcript(server, first, [(1000, 5000, "HELLO"), (5000, 7000, "WORLD")])
    transcript(server, second, [(0, 2000, "WORLD")], video_end=2000)
    no_source_files(temp_root, [first, second])
    delete(server, first)
    # Deleting one result must not release another job's transcript.
    transcript(server, second, [(0, 2000, "WORLD")], video_end=2000)
    delete(server, second)


def capacity_and_pagination(server):
    jobs = [create(server) for _ in range(8)]
    expect(request(server, "POST", ROOT, json.dumps(OPTIONS).encode(), "application/json"),
           503, "ninth retained job")
    seen, cursor = [], None
    for _ in range(5):
        route = ROOT + "?limit=3" + ("&cursor=" + quote(cursor, safe="") if cursor else "")
        page = expect(request(server, "GET", route), 200, "page")
        require(0 < len(page["jobs"]) <= 3, f"Pagination ignored page size: {page}")
        seen.extend(job["id"] for job in page["jobs"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    require(cursor is None and len(seen) == 8 and len(set(seen)) == 8 and set(seen) == set(jobs),
            f"Pagination skipped/duplicated jobs: {seen}")
    delete(server, jobs.pop(0))
    recovered = create(server)
    require(recovered not in jobs, "Deleted capacity returned a live job's identifier")
    for job in jobs + [recovered]:
        delete(server, job)
    require(expect(request(server, "GET", ROOT), 200, "empty list")["jobs"] == [],
            "Deleted jobs remain listed")


@contextlib.contextmanager
def child_temp_environment(directory):
    # Server copies this environment only at Popen. Restore immediately after
    # startup; never inspect/delete a developer's shared system temp directory.
    previous = {key: os.environ.get(key) for key in ("TMP", "TEMP", "TMPDIR")}
    try:
        for key in previous:
            os.environ[key] = str(directory)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run(args):
    root = args.project_root.resolve(strict=True)
    executable = args.server.resolve(strict=True)
    require(executable.is_file(), f"Missing server executable: {executable}")
    ffmpeg = shutil.which(args.ffmpeg)
    require(ffmpeg is not None, f"FFmpeg executable not found: {args.ffmpeg}")
    candidates = ([Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/arial.ttf"] if os.name == "nt"
                  else [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                        Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf")])
    font = args.font.resolve(strict=True) if args.font else next((path for path in candidates if path.is_file()), None)
    require(font is not None, "Arial/DejaVuSans font missing; supply --font")
    assets = [fixture(root, "app/assets/test/" + name) for name in
              ("ppocr_det.onnx", "ppocr_rec.onnx", "ppocr_keys_v1.txt")]
    config = ("ocr:\n  - name: ppocr-v4\n    version: kPPOCRv4\n" +
              "".join(f"    {key}: {json.dumps(path.as_posix())}\n" for key, path in
                      zip(("det_path", "rec_path", "char_dict_path"), assets)))
    with tempfile.TemporaryDirectory(prefix="vision-subtitle-regression-") as temporary:
        directory = Path(temporary)
        media = make_media(ffmpeg, directory, font)
        temp_root = directory / "server-temp"
        temp_root.mkdir()
        with contextlib.ExitStack() as stack:
            with child_temp_environment(temp_root):
                server = stack.enter_context(Server(executable, root, config))
            server.wait_ready()
            # Verify the actual recognition stage, not just the post-processor.
            words = ("HELLO", "WORLD", "NOISE")
            images = [base64.b64encode((directory / f"{word}.png").read_bytes()).decode("ascii") for word in words]
            result = infer(server, "ocr", "ppocr-v4", images)
            for word, detections in zip(words, result["results"]):
                require([detection["line"] for detection in detections] == [word],
                        f"PP-OCR whole-glyph regression: expected {word}, got {detections}")
                require(all(detection["confidence"] >= 0.9 for detection in detections),
                        f"PP-OCR fixture confidence is insufficient: {detections}")
            validation_matrix(server)
            capacity_and_pagination(server)
            disconnect_matrix(server, media["semantic.avi"], temp_root)
            invalid_media_matrix(server, media["semantic.avi"], temp_root)
            for chunked in (False, True):
                job = create(server)
                completed(server, job, media["semantic.avi"], temp_root, chunked=chunked)
                delete(server, job)
            expect_continue(server, media["world.avi"], temp_root)
            cancellation_and_isolation(server, media, temp_root)
            job = create(server)
            if os.name == "nt":
                completed(server, job, media["vfr-audio-tail.mp4"], temp_root, chunked=True,
                          expected=[(1000, 5000, "HELLO"), (5000, 8000, "WORLD")], tolerance=100)
                final = info(server, job)
                require(final["duration_ms"] is not None and abs(final["duration_ms"] - 8000) <= 100,
                        f"Audio tail incorrectly defined video duration: {final}")
            else:
                expect(upload(server, job, media["vfr-audio-tail.mp4"]), 202, "unsupported codec upload")
                final = wait_state(server, job)
                require(final["state"] == "failed" and final["error_code"] == "unsupported_video",
                        f"Linux unsupported H264 must fail explicitly, not fake a transcript: {final}")
                expect(request(server, "GET", f"{ROOT}/{job}/subtitles.srt"), 409, "unsupported result")
            no_source_files(temp_root, [job])
            delete(server, job)
        require(not list(temp_root.iterdir()),
                "Owned server left its private temporary directory after shutdown")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable (requires drawtext, MJPEG, libx264, AAC)")
    parser.add_argument("--font", type=Path, help="Override Windows Arial / Linux DejaVuSans font")
    args = parser.parse_args()
    try:
        run(args)
    except (RegressionFailure, OSError, ValueError, KeyError, subprocess.SubprocessError,
            http.client.HTTPException) as exc:
        print(f"FAIL subtitle regression: {exc}", file=sys.stderr)
        return 1
    print("PASS real-video subtitle semantics, decoder timing, HTTP ownership, cancellation and cleanup", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
