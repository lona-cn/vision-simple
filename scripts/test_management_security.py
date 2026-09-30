#!/usr/bin/env python3
"""Exercise management permission against real CPU inference/cache state (stdlib)."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
from pathlib import Path
import socket
import subprocess
import sys
import select
import threading
import time
import traceback

from test_http_regression import (Server, RegressionFailure, REPEATED_WORKLOAD_OPTIONS,
                                  close_values, error_response, fixture, infer,
                                  model_stats, model_yaml, require)
from test_protocol_regression import wire

STATS = '/v0/infer/stats'
UNLOAD = '/v0/infer/unload'
MODEL = {'kind': 'yolo', 'model': 'hd2-fp32'}


def denied(server, method, route, status, code, *, headers=None, raw=None):
    payload = raw if raw is not None else (json.dumps(MODEL).encode() if method == 'POST' else b'')
    header_only = method == 'POST' and code in {
        'management_unauthorized', 'management_forbidden', 'management_disabled',
        'unsupported_media_type', 'payload_too_large'}
    host = next((value for key, value in (headers or {}).items() if key.lower() == 'host'),
                f'127.0.0.1:{server.port}')
    transport_context = (f'{method} {route}: Host={host!r}, length={len(payload)}, '
                         f'expected={status}/{code}, mode={"headers-only" if header_only else "full-body"}')
    transport_context = transport_context.replace(server.MANAGEMENT_TOKEN, '[redacted]')
    try:
        if header_only:
            actual, response_headers, body = header_rejection(
                server, method=method, route=route, length=len(payload), expected=status, code=code,
                headers={'Host': f'127.0.0.1:{server.port}', **(headers or {})})
        else:
            actual, response_headers, body = wire(server, method, route, MODEL if method == 'POST' else None,
                                                  headers=headers, raw=raw)
    except (OSError, http.client.HTTPException) as exc:
        raise RegressionFailure(f'Management transport failed: {transport_context}') from exc
    context = (f'{method} {route}: status={actual}, headers={response_headers}, raw={body[:1000]!r}')
    context = context.replace(server.MANAGEMENT_TOKEN, '[redacted]')
    try:
        parsed = json.loads(body)
    except (ValueError, UnicodeDecodeError) as exc:
        raise RegressionFailure(f'Expected native JSON denial: {context}') from exc
    try:
        error_response((actual, parsed), status, code, None)
    except RegressionFailure as exc:
        raise RegressionFailure(f'{exc}; {context}') from exc
    lowered = {key.lower(): value for key, value in response_headers.items()}
    require(not any(key.startswith('access-control-') for key in lowered),
            f'Management denial exposed CORS headers: {response_headers}')
    if status == 401:
        require(lowered.get('www-authenticate') == 'Bearer realm="vision-simple-management"',
                f'Missing bearer challenge: {response_headers}')
    require(server.MANAGEMENT_TOKEN.encode() not in body, 'Error exposed management credential')


def malformed_host(server, method, route, host):
    # libhv parses this Host into an unmatched full URL before routing, so generic
    # upload/100 behavior remains possible. Require a final error, not its incidental status class.
    status, headers, raw = wire(server, method, route, MODEL if method == 'POST' else None,
                                headers={**server.admin_headers(), 'Host': host})
    require(400 <= status < 600,
            f'{method} {route} malformed Host {host}: {status}, headers={headers}, raw={raw[:1000]!r}')
    require(server.MANAGEMENT_TOKEN.encode() not in raw, 'Transport syntax error exposed credential')


def logs_redacted(server, *extra):
    server.output.flush()
    paths = [server.cwd / 'server-output.log', *sorted((server.cwd / 'logs').rglob('*'))]
    for path in paths:
        if path.is_file():
            content = path.read_bytes()
            for secret in (server.MANAGEMENT_TOKEN, *extra):
                if secret:
                    require(secret.encode() not in content, f'Credential leaked into {path.name}')


def header_rejection(server, *, headers=None, length=2, expected=401, code='management_unauthorized',
                     method='POST', route=UNLOAD, chunked=False):
    """Send only headers: rejection must arrive without body or interim 100."""
    with socket.create_connection(('127.0.0.1', server.port), timeout=3) as sock:
        fields = {'Host': f'localhost:{server.port}', 'Content-Type': 'application/json',
                  'Expect': '100-continue', **(headers or {})}
        fields['Transfer-Encoding' if chunked else 'Content-Length'] = 'chunked' if chunked else str(length)
        request = f'{method} {route} HTTP/1.1\r\n' + ''.join(f'{key}: {value}\r\n' for key, value in fields.items()) + '\r\n'
        sock.sendall(request.encode('ascii'))
        response = http.client.HTTPResponse(sock)
        # HTTPResponse skips 100; inspect the first bytes explicitly before parsing.
        first = sock.recv(4096, socket.MSG_PEEK)
        require(first.startswith(f'HTTP/1.1 {expected} '.encode()),
                f'Header-only request did not reject immediately (or emitted 100): {first!r}')
        response.begin()
        body = response.read()
        error_response((response.status, json.loads(body)), expected, code, None)
        require(response.getheader('Connection', '').lower() == 'close',
                'Incomplete rejected request did not close deterministically')
        require(sock.recv(1) == b'', 'Rejected incomplete request retained its connection')
        return response.status, dict(response.getheaders()), body


def authorized_continue(server):
    body = json.dumps({'kind': 'yolo', 'model': 'not-loaded'}).encode()
    with socket.create_connection(('127.0.0.1', server.port), timeout=5) as sock:
        sock.sendall((f'POST {UNLOAD} HTTP/1.1\r\nHost: localhost:{server.port}\r\n'
                      f'Authorization: Bearer {server.MANAGEMENT_TOKEN}\r\n'
                      f'Content-Type: application/json\r\nContent-Length: {len(body)}\r\n'
                      'Expect: 100-continue\r\n\r\n').encode())
        interim = b''
        while b'\r\n\r\n' not in interim:
            byte = sock.recv(1)
            require(byte, 'Authorized Expect closed before interim response')
            interim += byte
        require(interim.startswith(b'HTTP/1.1 100 '), f'Authorized Expect rejected: {interim!r}')
        sock.sendall(body)
        response = http.client.HTTPResponse(sock)
        response.begin()
        error_response((response.status, json.loads(response.read())), 404, 'model_not_loaded', None)


def control_saturation(server):
    # Stage real, maximum-sized authorized unload bodies before releasing their
    # final byte together. Every admitted body must run actual JSON validation;
    # no test-only worker delays, mocked operations, or cheap stats retry loops.
    body = json.dumps({'kind': 'yolo', 'model': [0] * 16000}).encode()
    body += b' ' * (65536 - len(body))
    sockets = []
    barrier = threading.Barrier(32)
    try:
        for _ in range(32):
            sock = socket.create_connection(('127.0.0.1', server.port), timeout=10)
            sockets.append(sock)
            sock.sendall((f'POST {UNLOAD} HTTP/1.1\r\nHost: localhost:{server.port}\r\n'
                          f'Authorization: Bearer {server.MANAGEMENT_TOKEN}\r\n'
                          'Content-Type: application/json\r\nContent-Length: 65536\r\n'
                          'Expect: 100-continue\r\n\r\n').encode())
            interim = b''
            while b'\r\n\r\n' not in interim:
                byte = sock.recv(1)
                require(byte, 'Staged authorized upload closed before 100 Continue')
                interim += byte
            require(interim.startswith(b'HTTP/1.1 100 '), f'Staged upload rejected: {interim!r}')
            sock.sendall(body[:-1])

        def release(sock):
            barrier.wait(timeout=10)
            sock.sendall(body[-1:])

        with ThreadPoolExecutor(max_workers=32) as pool:
            releases = [pool.submit(release, sock) for sock in sockets]
            pending = set(sockets)
            witnessed = False
            while pending:
                readable, _, _ = select.select(list(pending), [], [], 10)
                require(readable, 'Completed upload failed to produce a bounded control response')
                for sock in readable:
                    response = http.client.HTTPResponse(sock)
                    response.begin()
                    result = json.loads(response.read())
                    pending.remove(sock)
                    if response.status == 503:
                        error_response((response.status, result), 503, 'service_overloaded', None)
                        require(response.getheader('Retry-After') == '1', 'Control overload omitted retry advice')
                        if not witnessed:
                            ready, _, _ = select.select(list(pending), [], [], 0)
                            require(pending - set(ready), 'Overload did not overlap pending actual control responses')
                            denied(server, 'GET', STATS, 401, 'management_unauthorized')
                            header_rejection(server)
                            witnessed = True
                    else:
                        error_response((response.status, result), 400, 'invalid_request', None)
            require(witnessed, 'Staged actual unload work did not witness bounded control overload')
            require(server.admin_request(STATS, method='GET')[0] == 200, 'Control lane did not recover')
            for job in releases:
                job.result()
    finally:
        for sock in sockets:
            sock.close()




def enabled_matrix(executable, root, config, image, ocr_image):
    options = {**REPEATED_WORKLOAD_OPTIONS, 'infer_idle_timeout_ms': '0',
               'http_control_workers': '1', 'http_control_queue_capacity': '1'}
    with Server(executable, root, config, options=options) as server:
        server.wait_ready()
        baseline = infer(server, 'yolo', 'hd2-fp32', [image])
        require(baseline['results'][0], 'Genuine inference must detect fixture targets')
        before = model_stats(server)['yolo', 'hd2-fp32']
        control_saturation(server)
        token = server.MANAGEMENT_TOKEN
        for method, route in (('GET', STATS), ('POST', UNLOAD)):
            for credential in (None, '', 'Bearer wrong', 'Bearer ' + token[:-1],
                               'Bearer ' + token + 'extra', 'Bearer ' + token.swapcase(),
                               'Basic ' + token, 'Bearer'):
                denied(server, method, route, 401, 'management_unauthorized',
                       headers={} if credential is None else {'Authorization': credential})
            denied(server, method, route + '?access_token=' + token, 401, 'management_unauthorized',
                   headers={'Cookie': 'Authorization=Bearer ' + token,
                            'Forwarded': f'for=127.0.0.1;host=localhost:{server.port};proto=http',
                            'X-Forwarded-For': '127.0.0.1', 'X-Forwarded-Host': f'localhost:{server.port}'})
            for host in ('evil.example', f'0.0.0.0:{server.port}', f'*:{server.port}',
                         f'localhost.evil:{server.port}', 'localhost', f'localhost:{server.port + 1}',
                         f'user@localhost:{server.port}',
                         f'[localhost]:{server.port}', f'[127.0.0.1]:{server.port}'):
                denied(server, method, route, 403, 'management_forbidden',
                       headers={**server.admin_headers(), 'Host': host,
                                'X-Forwarded-Host': f'localhost:{server.port}'})
            malformed_host(server, method, route, f'localhost:{server.port}/')
            for origin in ('', 'null', 'https://evil.example', f'http://localhost:{server.port + 1}',
                           f'http://localhost:{server.port}/', 'not-an-origin',
                           f'http://user@localhost:{server.port}', f'http://localhost.evil:{server.port}',
                           f'http://[localhost]:{server.port}', f'http://[127.0.0.1]:{server.port}'):
                denied(server, method, route, 403, 'management_forbidden',
                       headers={**server.admin_headers(), 'Origin': origin})
        require(model_stats(server)['yolo', 'hd2-fp32'] == before,
                'Denied administration modified resident model counters/cache state')
        require(close_values(infer(server, 'yolo', 'hd2-fp32', [image]), baseline),
                'Denied unload changed genuine inference result')
        for media in ('text/plain', 'application/x-www-form-urlencoded', 'application/json; charset=utf-8'):
            denied(server, 'POST', UNLOAD, 415, 'unsupported_media_type',
                   headers={**server.admin_headers(), 'Content-Type': media})
        denied(server, 'POST', UNLOAD, 400, 'invalid_request', headers=server.admin_headers(), raw=b'{')
        denied(server, 'GET', STATS + '?limit=201', 400, 'invalid_request', headers=server.admin_headers())
        denied(server, 'GET', STATS + '?offset=-1', 400, 'invalid_request', headers=server.admin_headers())
        denied(server, 'POST', UNLOAD, 404, 'model_not_loaded', headers=server.admin_headers(),
               raw=b'{"kind":"yolo","model":"not-loaded"}')
        denied(server, 'GET', STATS, 400, 'invalid_request', headers=server.admin_headers(), raw=b'{}')
        denied(server, 'OPTIONS', UNLOAD, 403, 'management_forbidden',
               headers={**server.admin_headers(), 'Origin': f'http://localhost:{server.port}',
                        'Access-Control-Request-Method': 'POST'})
        denied(server, 'OPTIONS', STATS, 403, 'management_forbidden')
        # More incomplete denied uploads than total control residents. None may enter
        # dispatch or prevent authenticated observation after the headers are rejected.
        for _ in range(4):
            header_rejection(server)
        header_rejection(server, chunked=True)
        header_rejection(server, method='GET', route=STATS)
        header_rejection(server, headers=server.admin_headers(), method='GET', route=STATS,
                         expected=400, code='invalid_request')
        exact = b'{"kind":"yolo","model":"not-loaded"}'
        exact += b' ' * (65536 - len(exact))
        denied(server, 'POST', UNLOAD, 404, 'model_not_loaded', headers=server.admin_headers(), raw=exact)
        denied(server, 'POST', UNLOAD, 413, 'payload_too_large', headers=server.admin_headers(), raw=exact + b' ')
        status, ordinary_headers, _ = wire(server, 'OPTIONS', '/v0/infer/yolo',
                                           headers={'Origin': 'https://example.com',
                                                    'Access-Control-Request-Method': 'POST'})
        require(status in (200, 204) and any(key.lower() == 'access-control-allow-origin'
                                           for key in ordinary_headers), 'Ordinary inference CORS changed')
        header_rejection(server, headers=server.admin_headers(), length=65537,
                         expected=413, code='payload_too_large')
        header_rejection(server, headers={**server.admin_headers(), 'Expect': 'unsupported'},
                         expected=417, code='expectation_failed')
        authorized_continue(server)
        status, _, raw = wire(server, 'GET', STATS,
                              headers={'authorization': 'bEaReR ' + token, 'host': f'LOCALHOST:{server.port}'})
        require(status == 200 and json.loads(raw)['total'] == 1,
                'Case-insensitive header/scheme or no-Origin CLI administration failed')
        for scheme in ('http', 'https'):
            status, headers, _ = wire(server, 'GET', STATS,
                                       headers={**server.admin_headers(),
                                                'Origin': f'{scheme}://localhost:{server.port}'})
            require(status == 200 and not any(key.lower().startswith('access-control-') for key in headers),
                    'Trusted Origin incorrectly denied or received permissive management CORS')
        denied(server, 'POST', UNLOAD, 401, 'management_unauthorized',
               headers={'Host': 'evil.example', 'Origin': 'null', 'Content-Type': 'text/plain'}, raw=b'{')
        # Real active native work still protects its lease after authorization.
        infer(server, 'ocr', 'ppocr-v4', [ocr_image])
        with ThreadPoolExecutor(max_workers=1) as pool:
            job = pool.submit(infer, server, 'ocr', 'ppocr-v4', [ocr_image] * 16)
            deadline = time.monotonic() + 30
            while model_stats(server)['ocr', 'ppocr-v4']['active_requests'] == 0:
                require(not job.done() and time.monotonic() < deadline, 'Could not observe genuine active lease')
                time.sleep(.01)
            error_response(server.admin_request(UNLOAD, {'kind': 'ocr', 'model': 'ppocr-v4'}),
                           409, 'model_busy', None)
            denied(server, 'POST', UNLOAD, 401, 'management_unauthorized')
            job.result()
        status, result = server.admin_request(UNLOAD, MODEL)
        require(status == 200 and result == {**MODEL, 'unloaded': True}, f'CLI unload failed: {result}')
        require(('yolo', 'hd2-fp32') not in model_stats(server), 'Authorized unload retained cache entry')
        require(close_values(infer(server, 'yolo', 'hd2-fp32', [image]), baseline),
                'Reload after authorized unload changed genuine inference result')
        logs_redacted(server)
    print('PASS enabled management identity, authority, streaming/body limits, lease and genuine cache lifecycle')


def startup_matrix(executable, root, config, image):
    env = Server.MANAGEMENT_ENV
    for value in (None, '', 'UnsafeSecret-53 with-space', 'UnsafeSecret-53\tcontrol',
                  'UnsafeSecret-53\ncontrol', 'UnsafeSecret-53é', 'X' * 4097):
        with Server(executable, root, config, environment={env: value}) as server:
            server.expect_start_failure()
            logs_redacted(server, value)
    for name in ('bad-name', '1BAD', 'BAD NAME', 'X' * 129):
        with Server(executable, root, config, options={'http_management_token_env': name}) as server:
            server.expect_start_failure()
            logs_redacted(server)
    # Removing/emptying only the option is a fail-closed rollback, not a startup
    # failure; inference remains genuinely functional without administrator identity.
    for setting in ('', None):
        with Server(executable, root, config, options={'http_management_token_env': setting},
                    environment={env: None}) as server:
            server.wait_ready()
            infer(server, 'yolo', 'hd2-fp32', [image])
            for method, route in (('GET', STATS), ('POST', UNLOAD)):
                denied(server, method, route, 403, 'management_disabled', headers=server.admin_headers())
                denied(server, method, route, 403, 'management_disabled')
            logs_redacted(server)
    print('PASS unsafe/missing configured credentials fail startup; disabled rollback preserves real inference')


def wildcard_matrix(executable, root, config):
    # Deliberately opt into a wildcard listener: it does not confer wildcard Host trust.
    with Server(executable, root, config, host='0.0.0.0') as server:
        deadline = time.monotonic() + 30
        while True:
            require(server.process.poll() is None, 'Wildcard fixture failed startup')
            try:
                status, _, _ = wire(server, 'GET', '/livez')
                require(status == 200, 'Wildcard fixture not alive')
                break
            except ConnectionRefusedError:
                require(time.monotonic() < deadline, 'Wildcard fixture did not listen')
                time.sleep(.05)
        for host in (f'0.0.0.0:{server.port}', f'evil.example:{server.port}'):
            denied(server, 'GET', STATS, 403, 'management_forbidden',
                   headers={**server.admin_headers(), 'Host': host})
        denied(server, 'GET', STATS, 401, 'management_unauthorized')
        status, _ = server.admin_request(STATS, method='GET')
        require(status == 200, 'Wildcard bind denied authenticated loopback authority')
        logs_redacted(server)
    print('PASS explicit wildcard bind does not confer authority or credentials')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True, type=Path)
    parser.add_argument('--project-root', required=True, type=Path)
    args = parser.parse_args()
    try:
        root = args.project_root.resolve(strict=True)
        executable = args.server.resolve(strict=True)
        for path in ('app/assets/test/hd2-yolo11n-fp32.onnx', 'app/assets/test/ppocr_det.onnx',
                     'app/assets/test/ppocr_rec.onnx', 'app/assets/test/ppocr_keys_v1.txt'):
            fixture(root, path)
        image = base64.b64encode(fixture(root, 'app/assets/test/hd2.png').read_bytes()).decode()
        ocr_image = base64.b64encode(fixture(root, 'doc/images/ppocr.png').read_bytes()).decode()
        config = model_yaml(root)
        enabled_matrix(executable, root, config, image, ocr_image)
        startup_matrix(executable, root, config, image)
        wildcard_matrix(executable, root, config)
    except (RegressionFailure, OSError, ValueError, subprocess.SubprocessError, http.client.HTTPException) as exc:
        print(f'FAIL management security regression: {exc}', file=sys.stderr)
        traceback.print_exc()
        return 1
    print('PASS management security regression')
    return 0


if __name__ == '__main__':
    sys.exit(main())
