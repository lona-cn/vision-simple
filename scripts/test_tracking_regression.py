#!/usr/bin/env python3
"""Exercise real tracking HTTP routes; owns and stops its server and sessions."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import math
from pathlib import Path
import socket
import struct
import sys
import zlib

from test_http_regression import Server, RegressionFailure, require

ROOT = "/v1/tracking/sessions"


def png():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    # Textured, deterministic BGR-independent RGB frame for actual GMC input.
    rows = b"".join(b"\0" + bytes(v for x in range(96) for v in
                                 ((x * 13 + y * 7) % 256, (x * y) % 256, (x ^ y) * 2))
                    for y in range(96))
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 96, 96, 8, 2, 0, 0, 0))
    return base64.b64encode(data + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")).decode()


def sized_png(width, height, *, header_only=False):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    header = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    compressor = zlib.compressobj()
    # Only one scanline is resident, including for the just-over-limit image.
    rows = 1 if header_only else height
    row = b"\0" if header_only else b"\0" * (1 + width * 3)
    compressed = b"".join(compressor.compress(row) for _ in range(rows)) + compressor.flush()
    return base64.b64encode(header + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")).decode()


def request(server, method, path, body=None, headers=None, raw=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=30)
    try:
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        connection.request(method, path, data, headers or {"Content-Type": "application/json"})
        response = connection.getresponse()
        payload = response.read()
        status, result_headers = response.status, dict(response.getheaders())
        if status == 204:
            require(not payload, "DELETE 204 must have no body")
            return status, None, result_headers
        require(response.getheader("Content-Type", "").split(";", 1)[0] == "application/json", "Expected JSON response")
        result = json.loads(payload)
        if "error" in result:
            require(result["error"].get("image_index", "missing") is None, "Expected null image_index")
            require(str(server.root) not in json.dumps(result) and str(server.cwd) not in json.dumps(result), "Error leaked path")
        return status, result, result_headers
    finally:
        connection.close()


def error(response, status, code):
    require(response[0] == status and response[1].get("error", {}).get("code") == code,
            f"Expected {status}/{code}: {response}")


def transport_bounds(server):
    # Header-only oversized declaration must reject before waiting for a body.
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    try:
        connection.putrequest("POST", ROOT)
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(4 * 1024 * 1024 + 1))
        connection.endheaders()
        response = connection.getresponse()
        error((response.status, json.loads(response.read())), 413, "invalid_request")
    finally:
        connection.close()
    # A normal header error drains a separately uploaded body without resetting
    # the connection. No Expect header means it must not finish early.
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    try:
        body = b'{"algorithm":"bytetrack"}'
        connection.putrequest("POST", ROOT)
        connection.putheader("Content-Type", "text/plain")
        connection.putheader("Content-Length", str(len(body)))
        connection.endheaders()
        connection.send(body[:5])
        connection.send(body[5:])
        response = connection.getresponse()
        error((response.status, json.loads(response.read())), 400, "invalid_request")
        original_socket = connection.sock
        connection.request("GET", ROOT)
        response = connection.getresponse()
        require(response.status == 200, "Normal POST header error broke keepalive")
        response.read()
        require(connection.sock is original_socket, "POST error silently replaced the connection")
        connection.request("POST", ROOT, b"{", {"Content-Type": "application/json"})
        response = connection.getresponse()
        error((response.status, json.loads(response.read())), 400, "invalid_request")
        connection.request("GET", ROOT)
        response = connection.getresponse()
        require(response.status == 200, "Normal POST JSON error broke keepalive")
        response.read()
        require(connection.sock is original_socket, "JSON error silently replaced the connection")
    finally:
        connection.close()
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    try:
        connection.putrequest("POST", ROOT)
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Transfer-Encoding", "chunked")
        connection.endheaders()
        chunk = b" " * (64 * 1024)
        for _ in range(65):
            connection.send(b"10000\r\n" + chunk + b"\r\n")
        response = connection.getresponse()
        error((response.status, json.loads(response.read())), 413, "invalid_request")
    finally:
        connection.close()
    session = request(server, "POST", ROOT, {"algorithm": "bytetrack"})[1]["id"]
    try:
        for method, path in (("GET", ROOT), ("GET", ROOT + "/" + session),
                             ("DELETE", ROOT + "/" + session)):
            for header, value in (("Content-Length", "1"), ("Content-Length", str(16 * 1024 * 1024)),
                                  ("Transfer-Encoding", "chunked")):
                connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
                try:
                    connection.putrequest(method, path)
                    connection.putheader(header, value)
                    connection.endheaders()
                    response = connection.getresponse()
                    require(response.getheader("Connection", "").lower() == "close", "Bodyless rejection must close")
                    error((response.status, json.loads(response.read())), 400, "invalid_request")
                finally:
                    connection.close()
            connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
            try:
                connection.putrequest(method, path)
                connection.putheader("Origin", "https://example.com")
                connection.putheader("Content-Length", "1")
                connection.endheaders()
                response = connection.getresponse()
                require(response.getheader("Connection", "").lower() == "close", "Origin rejection must close")
                error((response.status, json.loads(response.read())), 403, "invalid_request")
            finally:
                connection.close()

        # Wait for the interim response before sending any body; HTTPConnection
        # otherwise consumes 100 silently and cannot prove the handshake works.
        with socket.create_connection(("127.0.0.1", server.port), timeout=5) as sock:
            sock.sendall((f"POST {ROOT}/{session}/reset HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                          "Content-Type: application/json\r\nContent-Length: 2\r\n"
                          "Expect: \t100-CoNtInUe \t\r\n\r\n").encode())
            interim = b""
            while not interim.endswith(b"\r\n\r\n"):
                part = sock.recv(1)
                require(bool(part), "Connection closed before 100 Continue")
                interim += part
                require(len(interim) <= 4096, "Unbounded interim response")
            require(interim == b"HTTP/1.1 100 Continue\r\n\r\n", f"Missing interim response: {interim!r}")
            sock.sendall(b"{}")
            response = http.client.HTTPResponse(sock)
            response.begin()
            require(response.status == 200 and json.loads(response.read()) == {"reset": True}, "Mixed-case Expect failed")
        for expectation, content_type, status in ((" \t100-CoNtInUe \t", "text/plain", 400),
                                                  ("something-else", "application/json", 417)):
            connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
            try:
                connection.putrequest("POST", ROOT)
                connection.putheader("Content-Type", content_type)
                connection.putheader("Content-Length", "100")
                connection.putheader("Expect", expectation)
                connection.endheaders()
                response = connection.getresponse()
                require(response.getheader("Connection", "").lower() == "close", "Expect error must close immediately")
                error((response.status, json.loads(response.read())), status, "invalid_request")
            finally:
                connection.close()
        require(request(server, "GET", ROOT + "/" + session)[0] == 200, "Rejected bodyless requests mutated session")
    finally:
        require(request(server, "DELETE", ROOT + "/" + session)[0] == 204, "Transport test leaked session")

def run(args):
    root, executable = args.project_root.resolve(strict=True), args.server.resolve(strict=True)
    config = "yolo: []\nocr: []\n"
    if args.yolo_model:
        require(args.yolo_image is not None, "--yolo-image required with --yolo-model")
        config = "yolo:\n  - name: tracking-source\n    version: kV11\n    path: " + json.dumps(args.yolo_model.resolve(strict=True).as_posix()) + "\nocr: []\n"
    with Server(executable, root, config) as server:
        server.wait_ready()
        owned = set()
        image = png()

        def create(algorithm="bytetrack", options=None):
            response = request(server, "POST", ROOT, {"algorithm": algorithm, "options": options or {"min_hits": 1}})
            require(response[0] == 201, f"Create failed: {response}")
            session = response[1]["id"]
            owned.add(session)
            require(response[2].get("Location") == ROOT + "/" + session, "Missing resource Location")
            require(response[1]["algorithm"] == algorithm, "Wrong algorithm")
            return session

        def delete(session):
            require(request(server, "DELETE", ROOT + "/" + session)[0] == 204, "Delete failed")
            owned.discard(session)

        def frame(session, index, confidence=0.9, *, timestamp=None, detections=None, **extra):
            payload = {"frame_index": index, "timestamp": float(index) if timestamp is None else timestamp,
                       "detections": detections if detections is not None else
                       [{"class_id": 2, "confidence": confidence, "bbox": [20 + index, 20, 16, 24]}]}
            payload.update(extra)
            return request(server, "POST", ROOT + "/" + session + "/frames", payload)

        def observed(response, track_id=1, score=0.9):
            require(response[0] == 200, f"Frame failed: {response}")
            tracks = response[1]["tracks"]
            require(len(tracks) == 1 and tracks[0]["track_id"] == track_id and tracks[0]["class_id"] == 2,
                    f"Wrong identity/class: {tracks}")
            require(math.isclose(tracks[0]["confidence"], score, abs_tol=1e-6), f"Wrong observed score: {tracks}")

        try:
            for algorithm in ("bytetrack", "botsort"):
                session = create(algorithm)
                kwargs = {"image": image} if algorithm == "botsort" else {}
                observed(frame(session, 0, **kwargs))
                observed(frame(session, 1, 0.8, **kwargs), score=0.8)
                observed(frame(session, 2, 0.3, **kwargs), score=0.3)
                before = request(server, "GET", ROOT + "/" + session)[1]["status"]
                error(frame(session, 2, **kwargs), 409, "frame_out_of_order")
                error(frame(session, 3, timestamp=2, **kwargs), 409, "frame_out_of_order")
                error(frame(session, 3, detections=[{"class_id": -1, "confidence": 0.9, "bbox": [1, 1, 2, 2]}], **kwargs), 400, "invalid_request")
                error(frame(session, 3, image="https://example.com/image.png"), 400, "invalid_image")
                error(frame(session, 2, image="not-base64"), 409, "frame_out_of_order")
                error(frame(session, 3, timestamp=2, image=""), 409, "frame_out_of_order")
                error(frame(session, 3, image=""), 400, "invalid_image")
                require(request(server, "GET", ROOT + "/" + session)[1]["status"] == before, "Rejected frame mutated sequence")
                observed(frame(session, 3, **kwargs))
                require(request(server, "POST", ROOT + "/" + session + "/reset", {})[:2] == (200, {"reset": True}), "Reset failed")
                status = request(server, "GET", ROOT + "/" + session)[1]["status"]
                require(status == {"last_frame_index": None, "last_timestamp": None, "active_tracks": 0, "lost_tracks": 0}, "Reset retained state")
                observed(frame(session, 0, **kwargs))
                delete(session)
                error(frame(session, 1, **kwargs), 404, "tracking_session_not_found")
                error(request(server, "DELETE", ROOT + "/" + session), 404, "tracking_session_not_found")
                error(frame(session, 1, image="not-base64"), 404, "tracking_session_not_found")

            session = create()
            oversized_images = (sized_png(0x7fffffff, 0x7fffffff, header_only=True),
                                sized_png(4097, 4096))
            before = request(server, "GET", ROOT + "/" + session)[1]["status"]
            for encoded in oversized_images:
                error(frame(session, 0, image=encoded), 400, "invalid_image")
                error(frame("0" * 32, 0, image=encoded), 404, "tracking_session_not_found")
            for encoded in (sized_png(0, 1, header_only=True),
                            base64.b64encode(b"GIF89a" + b"\0" * 32).decode(),
                            base64.b64encode(b"\xff\xd8\xff\xc0\x00\x11\x08").decode()):
                error(frame(session, 0, image=encoded), 400, "invalid_image")
            require(request(server, "GET", ROOT + "/" + session)[1]["status"] == before,
                    "Rejected image changed tracking state")
            observed(frame(session, 0, image=sized_png(32, 24)))
            delete(session)

            for body in ({"algorithm": "unknown"}, {"algorithm": "bytetrack", "extra": 1},
                         {"algorithm": "bytetrack", "options": {"algorithm": "botsort"}},
                         {"algorithm": "bytetrack", "options": {"max_tracks": 257}},
                         {"algorithm": "bytetrack", "options": {"high_threshold": -0.1}},
                         {"algorithm": "bytetrack", "options": {"min_hits": 0}}):
                error(request(server, "POST", ROOT, body), 400, "invalid_request")
            error(request(server, "POST", ROOT, raw=b"{"), 400, "invalid_request")
            error(request(server, "POST", ROOT, {"algorithm": "bytetrack"}, headers={"Content-Type": "text/plain"}), 400, "invalid_request")
            error(request(server, "POST", ROOT, {"algorithm": "bytetrack"}, headers={"Content-Type": "application/json", "Origin": "https://example.com"}), 403, "invalid_request")
            transport_bounds(server)
            for query in ("limit=0", "limit=101", "limit=1x", "cursor=garbage"):
                error(request(server, "GET", ROOT + "?" + query), 400, "invalid_request")
            session = create()
            for extra in ({"frame_index": 2**53}, {"timestamp": -1}, {"unexpected": True},
                          {"detections": [{"class_id": 0, "confidence": 1.1, "bbox": [1, 1, 2, 2]}]},
                          {"detections": [{"class_id": 0, "confidence": 0.9, "bbox": [1, 1, 0, 2]}]},
                          {"detections": [{"class_id": 0, "confidence": 0.9, "bbox": [1, 1, 2, 2], "embedding": [1] * 513}]},
                          {"detections": [{"class_id": 0, "confidence": 0.9, "bbox": [1, 1, 2, 2]}] * 257}):
                payload = {"frame_index": 0, "timestamp": 0, "detections": []}
                payload.update(extra)
                error(request(server, "POST", ROOT + "/" + session + "/frames", payload), 400, "invalid_request")
            observed(frame(session, 0))
            delete(session)

            streams = [create() for _ in range(32)]
            overflow = request(server, "POST", ROOT, {"algorithm": "bytetrack"})
            error(overflow, 503, "tracking_capacity")
            require(overflow[2].get("Retry-After") == "1", "Capacity lacks retry hint")
            seen, cursor = [], None
            while True:
                response = request(server, "GET", ROOT + "?limit=3" + ("&cursor=" + cursor if cursor else ""))
                require(response[0] == 200 and len(response[1]["sessions"]) <= 3, "Pagination failed")
                seen.extend(response[1]["sessions"])
                cursor = response[1]["next_cursor"]
                if cursor is None:
                    break
                require(len(seen) <= 32, "Pagination does not advance")
            require(seen == sorted(streams), "Pagination skipped/duplicated sessions")
            with ThreadPoolExecutor(max_workers=8) as workers:
                for result in workers.map(lambda sid: frame(sid, 0), streams):
                    observed(result)
                for result in workers.map(lambda sid: frame(sid, 1, 0.75), streams):
                    observed(result, score=0.75)
            delete(streams.pop())
            replacement = create("botsort", {"min_hits": 1, "camera_motion": False})
            observed(frame(replacement, 0))
            for sid in streams + [replacement]:
                delete(sid)

            if args.yolo_model:
                encoded = base64.b64encode(args.yolo_image.resolve(strict=True).read_bytes()).decode()
                response = server.request("/v0/infer/yolo", {"model": "tracking-source", "images": [encoded]})
                require(response[0] == 200, f"Real detector failed: {response}")
                detections = response[1]["results"][0]
                require(detections, "Real detector fixture yielded no detections")
                sid = create(options={"min_hits": 1, "high_threshold": 0.01, "low_threshold": 0.001, "new_track_threshold": 0.01})
                tracked = frame(sid, 0, detections=detections)
                require(tracked[0] == 200 and tracked[1]["tracks"], "Actual detector output was not trackable")
                delete(sid)
        finally:
            for session in list(owned):
                delete(session)
    print("PASS tracking algorithms, isolation, sequence atomicity, reset, bounds, capacity and pagination")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--yolo-model", type=Path)
    parser.add_argument("--yolo-image", type=Path)
    args = parser.parse_args()
    try:
        run(args)
    except (RegressionFailure, OSError, ValueError, http.client.HTTPException) as exc:
        print(f"FAIL tracking regression: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
