#!/usr/bin/env python3
"""Real CPU HTTP scheduling, health and shutdown regression (stdlib, no mocks)."""
import argparse
import base64
import http.client
import json
import math
from pathlib import Path
import socket
import struct
import time

from test_http_regression import Server, close_values, error_response, fixture, infer, model_yaml, require
from test_image_budget import budget, drained, options, ppm, wait_budget, mcp_error
from test_protocol_regression import Session, wire


def health(server, count=6, label='health', active=None, charge=None):
    samples = []
    for index in range(count):
        if active is not None:
            state = budget(server)
            require(state['active_requests'] == active and state['in_use_bytes'] == charge,
                    f'Health load witness ended before probe {index}: {state}')
        route, expected = ('/livez', 'alive') if index % 2 == 0 else ('/readyz', 'ready')
        started = time.perf_counter()
        status, headers, raw = wire_health(server, route)
        elapsed = (time.perf_counter() - started) * 1000
        require(status == 200 and json.loads(raw) == {'status': expected},
                f'{route}: {status} {raw!r}')
        require(headers.get('Content-Type', '').startswith('application/json'), 'Health JSON media type')
        require(elapsed < 2000, f'{route} exceeded Docker curl --max-time 2: {elapsed}ms')
        samples.append(elapsed)
        if active is not None:
            state = budget(server)
            require(state['active_requests'] == active and state['in_use_bytes'] == charge,
                    f'Health probe {index} did not span the full load witness: {state}')
    ordered = sorted(samples)
    quantiles = {f'p{p}': ordered[min(len(ordered)-1, math.ceil(p * len(ordered) / 100)-1)]
                 for p in (50, 95, 99)}
    print(json.dumps({'scenario': label, 'raw_rtt_ms': samples, **quantiles}))


def wire_health(server, route):
    connection = http.client.HTTPConnection('127.0.0.1', server.port, timeout=2)
    try:
        connection.request('GET', route)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def begin(server, images, *, timeout=300000, route='/v1/infer/ocr'):
    connection = http.client.HTTPConnection('127.0.0.1', server.port, timeout=360)
    try:
        connection.request('POST', route, json.dumps({'model': 'ppocr-v4', 'images': images,
                                                     'timeout_ms': timeout}),
                           {'Content-Type': 'application/json'})
        return connection
    except BaseException:
        connection.close()
        raise


def finish(connection, status=200, code=None, count=None):
    response = connection.getresponse()
    body = json.loads(response.read())
    if code:
        error_response((response.status, body), status, code, None)
    else:
        require(response.status == status, f'HTTP {response.status}: {body}')
        if count is not None:
            require(len(body['results']) == count, f'Incomplete OCR result: {body}')
    return body


def staged(server, image, count, connections, charge, workers=4):
    for active in range(1, workers + 1):
        connections.append(begin(server, [image] * count))
        state = wait_budget(server, lambda value: value['active_requests'] == active and
                            value['in_use_bytes'] == active * charge,
                            f'No full decoded admission for staged request {active}', timeout=30)
        print(json.dumps({'scenario': 'staged_native_admission', 'stage': active, 'image_budget': state}))
    return state


def overload(server):
    deadline = time.monotonic() + 10
    while True:
        status, headers, raw = wire(server, 'POST', '/v1/infer/ocr', raw=b'{not-json')
        if status == 503:
            error_response((status, json.loads(raw)), 503, 'service_overloaded', None)
            require(headers.get('Retry-After') == '1', f'Missing overload retry advice: {headers}')
            return
        # Independent IO loops may receive the probe before the queued upload.
        error_response((status, json.loads(raw)), 400, 'invalid_request', None)
        require(time.monotonic() < deadline, 'Could not observe full transport admission')


def load_matrix(executable, root, image, pixels, count):
    charge = pixels * 3 * count
    config = options(pixels, charge, charge * 4)
    config.update({'http_data_workers': '4', 'http_data_queue_capacity': '1',
                   'http_control_workers': '1', 'http_control_queue_capacity': '4'})
    with Server(executable, root, model_yaml(root), options=config) as server:
        server.wait_ready()
        health(server, label='idle')
        warm = infer(server, 'ocr', 'ppocr-v4', [image])
        require(any(line.get('line', '').strip() for line in warm['results'][0]),
                'Real OCR warmup must recognize fixture text')
        connections = []
        try:
            admitted = staged(server, image, count, connections, charge)
            health(server, count=18, label='four_real_native_requests', active=4, charge=charge * 4)
            for request, route in ((server.request, '/v0/infer/models'),
                                   (server.request, '/v1/models'),
                                   (server.admin_request, '/v0/infer/stats')):
                started = time.monotonic()
                status, body = request(route, method='GET', timeout=2)
                require(status == 200, f'Control route failed: {route}: {status} {body}')
                print(json.dumps({'control_route': route, 'rtt_ms': (time.monotonic()-started)*1000}))
            # The fifth request is accepted into the sole waiting slot. The sixth
            # malformed request must fail transport admission before business parse.
            queued = begin(server, [image], timeout=1)
            connections.append(queued)
            overload(server)
            state = budget(server)
            require(state['active_requests'] == 4 and state['in_use_bytes'] == admitted['in_use_bytes'],
                    f'Queued request consumed service image admission: {state}')
            health(server, label='full_transport_queue')
            finish(queued, 504, 'request_timeout')
            for connection in connections[:4]:
                finish(connection, count=count)
            after = drained(server, timeout=120)
            require(after['decode_calls'] == 1 + count * 4,
                    f'Expired queued request decoded input: {after}')
            require(after['peak_bytes'] == charge * 4, f'Four exact concurrent reservations not witnessed: {after}')
        finally:
            for connection in connections:
                connection.close()
        error_response(server.request('/v1/infer/yolo', {'model': 'runtime-yolo',
                                                        'images': [ppm(32, 32, 255)]}),
                       500, 'inference_failed', 0)
        drained(server)
        infer(server, 'ocr', 'ppocr-v4', [image])
        health(server, label='exception_recovery')
    print('PASS staged load, transport saturation, queue deadline, control lane and exception recovery')


def shared_mcp(executable, root, image, pixels, count):
    count = max(count, 64)  # Sustain a real native window after the whole batch finishes decoding.
    charge = pixels * 3 * count
    config = options(pixels, charge, charge, slots=4)
    with Server(executable, root, model_yaml(root), options=config) as server:
        server.wait_ready()
        infer(server, 'ocr', 'ppocr-v4', [image])
        infer(server, 'yolo', 'hd2-fp32', [])
        initial = budget(server)
        connection = begin(server, [image] * count)
        try:
            before = wait_budget(server, lambda state: state['in_use_bytes'] == charge and
                                 state['active_requests'] == 1 and
                                 state['decode_calls'] == initial['decode_calls'] + count,
                                 'Native batch did not reach full decoded admission')
            with Session(server) as session:
                session.initialize()
                mcp_error(session.tool('infer_ocr', {'model': 'ppocr-v4', 'images': [image]}),
                          'service_overloaded', None)
                rejected = budget(server)
                require(rejected['decode_calls'] == before['decode_calls'] and
                        rejected['in_use_bytes'] == charge and rejected['active_requests'] == 1,
                        f'MCP overload decoded or refunded admitted native input: {rejected}')
                # MCP tools retain minItems=1; only service/native accepts empty batches.
                mcp_error(session.tool('infer_yolo', {'model': 'hd2-fp32', 'images': []}),
                          'invalid_request', None)
                status, empty = server.request('/v1/infer/yolo', {'model': 'hd2-fp32', 'images': []})
                require(status == 200 and empty['results'] == [], f'Empty native result: {status} {empty}')
                after = budget(server)
                require(after['decode_calls'] == before['decode_calls'] and
                        after['in_use_bytes'] == charge and after['active_requests'] == 1,
                        f'Empty request consumed/refunded shared image quota: {after}')
                health(server, label='mcp_native_shared_budget', active=1, charge=charge)
            finish(connection, count=count)
        finally:
            connection.close()
        require(drained(server)['peak_bytes'] == charge, 'Shared byte boundary was not exact')
    print('PASS native/MCP shared decoded boundary, MCP empty validation and native empty exemption')


def slow_clients(executable, root, image, pixels, count):
    charge = pixels * 3 * count
    with Server(executable, root, model_yaml(root), options=options(pixels, charge, charge * 4)) as server:
        server.wait_ready()
        infer(server, 'ocr', 'ppocr-v4', [image])
        body = json.dumps({'model': 'ppocr-v4', 'images': [image]}).encode()
        header = (f'POST /v1/infer/ocr HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n'
                  f'Content-Length: {len(body)}\r\nConnection: close\r\n\r\n').encode()
        with socket.create_connection(('127.0.0.1', server.port), timeout=120) as sock:
            before = budget(server)
            sock.sendall(header[:25])
            health(server, label='split_headers')
            sock.sendall(header[25:] + body[:len(body)//2])
            health(server, label='split_body')
            middle = budget(server)
            require(middle['decode_calls'] == before['decode_calls'] and middle['active_requests'] == 0,
                    f'Incomplete upload started inference: {middle}')
            sock.sendall(body[len(body)//2:])
            response = http.client.HTTPResponse(sock)
            response.begin()
            require(response.status == 200 and len(json.loads(response.read())['results']) == 1,
                    'Split upload did not preserve request body')
        for partial in (header[:25], header + body[:len(body)//2]):
            before = budget(server)
            with socket.create_connection(('127.0.0.1', server.port), timeout=120) as sock:
                sock.sendall(partial)
                health(server, label='early_upload_pause')
            health(server, label='early_upload_disconnect')
            after = drained(server)
            require(after['decode_calls'] == before['decode_calls'],
                    f'Abandoned incomplete upload reached decode: {after}')
        connection = begin(server, [image] * count)
        try:
            wait_budget(server, lambda state: state['in_use_bytes'] == charge,
                        'Disconnect test was never admitted')
            connection.sock.shutdown(socket.SHUT_RDWR)
            connection.close()
            # Native calls cannot be preempted. Zero is allowed only once their
            # owned inputs actually drain; do not demand a timing-dependent sample.
            drained(server, timeout=120)
            infer(server, 'ocr', 'ppocr-v4', [image])
        finally:
            connection.close()
        # Sequential keepalive is supported; concurrent same-connection
        # pipelining is rejected by closing, never by reusing an active context.
        keepalive = begin(server, [image])
        try:
            original = keepalive.sock
            first = finish(keepalive, count=1)
            require(keepalive.sock is original and original is not None,
                    'Normal response unexpectedly disabled keepalive')
            keepalive.request('POST', '/v1/infer/ocr', json.dumps({'model': 'ppocr-v4', 'images': [image]}),
                              {'Content-Type': 'application/json'})
            second = finish(keepalive, count=1)
            require(close_values(first, second) and keepalive.sock is original,
                    'Sequential keepalive changed result or silently reconnected')
        finally:
            keepalive.close()
        payload = json.dumps({'model': 'ppocr-v4', 'images': [image] * count,
                              'timeout_ms': 300000}).encode()
        forbidden = json.dumps({'model': 'runtime-yolo', 'images': [ppm(32, 32, 255)]}).encode()
        second_request = (f'POST /v1/infer/yolo HTTP/1.1\r\nHost: localhost\r\n'
                          f'Content-Type: application/json\r\nContent-Length: {len(forbidden)}\r\n\r\n').encode() + forbidden
        for same_feed in (False, True):
            first_body = json.dumps({'model': 'ppocr-v4', 'images': []}).encode() if same_feed else payload
            first_request = (f'POST /v1/infer/ocr HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n'
                             f'Content-Length: {len(first_body)}\r\n\r\n').encode() + first_body
            with socket.create_connection(('127.0.0.1', server.port), timeout=120) as sock:
                if same_feed:
                    sock.sendall(first_request + second_request)
                else:
                    sock.sendall(first_request)
                    wait_budget(server, lambda state: state['active_requests'] == 1,
                                'First split-pipelined request was never active')
                    sock.sendall(second_request)
                try:
                    while sock.recv(65536):
                        pass  # A first reply may race rejection; EOF is mandatory.
                except ConnectionResetError:
                    pass  # RST is also safe close, not a reused writer/context.
            drained(server, timeout=120)
            status, snapshot = server.admin_request('/v0/infer/stats', method='GET')
            require(status == 200 and not any(model['name'] == 'runtime-yolo' for model in snapshot['models']),
                    f'Forbidden pipelined request executed (same_feed={same_feed}): {snapshot}')
        infer(server, 'ocr', 'ppocr-v4', [image])
        # Deliberately leave a response unread while issuing independent health.
        unread = begin(server, [image] * count)
        try:
            wait_budget(server, lambda state: state['active_requests'] == 1,
                        'Unread response scenario never entered inference')
            drained(server, timeout=360)
            health(server, label='unread_response')
        finally:
            unread.close()
        infer(server, 'ocr', 'ppocr-v4', [image])
    print('PASS split/abandoned uploads, active disconnect physical drain, keepalive/pipelining close, unread response and recovery')


def shutdown(executable, root, image, pixels, count):
    connections = []
    try:
        charge = pixels * 3 * count
        config = options(pixels, charge, charge * 4)
        config.update({'http_data_queue_capacity': '1'})
        with Server(executable, root, model_yaml(root), options=config) as server:
            server.wait_ready()
            infer(server, 'ocr', 'ppocr-v4', [image])
            # Leave both header-incomplete and body-incomplete clients open during shutdown.
            body = json.dumps({'model': 'ppocr-v4', 'images': [image]}).encode()
            header = (f'POST /v1/infer/ocr HTTP/1.1\r\nHost: localhost\r\n'
                      f'Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n').encode()
            for partial in (header[:25], header + body[:len(body)//2]):
                paused = socket.create_connection(('127.0.0.1', server.port), timeout=120)
                connections.append(paused)
                paused.sendall(partial)
            health(server, label='shutdown_paused_uploads')
            staged(server, image, count, connections, charge)
            connections.append(begin(server, [image]))
            overload(server)
            # Existing Server owns stdin shutdown, exit=0, deadline, diagnostics,
            # force-cleanup on failure and proof of no surviving owned listener.
    finally:
        for connection in connections:
            connection.close()
    print('PASS shutdown with four admitted and one queued native request, no owned listener')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--batch-images', type=int, default=16,
                        help='Real OCR batch used to sustain observed admission (1..128)')
    args = parser.parse_args()
    require(1 <= args.batch_images <= 128, '--batch-images must be 1..128')
    root, executable = args.project_root.resolve(), args.server.resolve()
    for name in ('hd2-yolo11n-fp32.onnx', 'ppocr_det.onnx', 'ppocr_rec.onnx',
                 'ppocr_keys_v1.txt', 'yolo_runtime_failure.onnx'):
        fixture(root, 'app/assets/test/' + name)
    raw = fixture(root, 'doc/images/ppocr.png').read_bytes()
    require(raw[:8] == b'\x89PNG\r\n\x1a\n', 'OCR fixture is not PNG')
    width, height = struct.unpack('>II', raw[16:24])
    image, pixels = base64.b64encode(raw).decode('ascii'), width * height
    load_matrix(executable, root, image, pixels, args.batch_images)
    shared_mcp(executable, root, image, pixels, args.batch_images)
    slow_clients(executable, root, image, pixels, args.batch_images)
    shutdown(executable, root, image, pixels, args.batch_images)
    print('PASS all real HTTP dispatch regressions')


if __name__ == '__main__':
    main()
