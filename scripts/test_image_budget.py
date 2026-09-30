#!/usr/bin/env python3
"""Real CPU image-admission boundaries and physical lifetime, no mocks (stdlib)."""
import argparse
import base64
import http.client
import json
from pathlib import Path
import socket
import struct
import time

from test_http_regression import Server, error_response, fixture, infer, model_yaml, require
from test_protocol_regression import Session, chat_payload, tool_data


def ppm(width, height, value=0):
    return base64.b64encode(f'P6\n{width} {height}\n255\n'.encode() +
                            bytes([value]) * (width * height * 3)).decode('ascii')


def budget(server):
    status, body = server.admin_request('/v0/infer/stats', method='GET', timeout=10)
    require(status == 200, f'Stats unavailable: {status} {body}')
    return body['image_budget']


def wait_budget(server, predicate, description, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = budget(server)
        if predicate(snapshot):
            return snapshot
        time.sleep(.01)
    raise AssertionError(f'{description}: {snapshot}')


def drained(server, timeout=60):
    return wait_budget(server, lambda state: state["in_use_bytes"] == 0 and
                       state["active_requests"] == 0, "Image quota/credit did not drain",
                       timeout=timeout)


def options(pixels, batch, global_bytes, slots=4):
    return {'infer_max_image_pixels': str(pixels),
            'infer_max_batch_decoded_bytes': str(batch),
            'infer_max_inflight_decoded_bytes': str(global_bytes),
            'infer_pipeline_max_batches': str(slots),
            'infer_pipeline_capacity': '1', 'infer_timeout_ms': '300000'}


def mcp_error(response, code, index):
    require(response.get('result', {}).get('isError') is True, f'MCP expected error: {response}')
    detail = json.loads(response['result']['content'][0]['text'])
    error_response((400, detail), 400, code, index)


def call_error(server, session, protocol, kind, model, images, status, code, index):
    if protocol == 'mcp':
        mcp_error(session.tool('infer_' + kind, {'model': model, 'images': images}), code, index)
    elif protocol == 'chat':
        actual, body = server.request('/v1/chat/completions', chat_payload(kind, model, images))
        require(actual == status and body["error"]["code"] == code and
                body["error"]["image_index"] == index, f"Chat error: {actual} {body}")
        # OpenAI-like uses its established envelope rather than the native envelope.
    else:
        error_response(server.request(f'/{protocol}/infer/{kind}',
                                      {'model': model, 'images': images}), status, code, index)


def predecode_matrix(executable, root):
    small, large = ppm(4, 4), ppm(5, 4)
    # 16px/48B boundary fits exactly; second image exceeds cumulative batch quota.
    with Server(executable, root, model_yaml(root), options=options(16, 48, 96)) as server:
        server.wait_ready()
        with Session(server) as session:
            session.initialize()
            for kind, model in (('yolo', 'hd2-fp32'), ('ocr', 'ppocr-v4')):
                for protocol in ('v0', 'v1', 'chat', 'mcp'):
                    for images, code, index in (([large], 'image_limit_exceeded', 0),
                                               ([small, small], 'image_limit_exceeded', 1),
                                               ([small, '%%%'], 'invalid_image', 1),
                                               (['%%%', large], 'invalid_image', 0),
                                               ([large, '%%%'], 'image_limit_exceeded', 0)):
                        before = budget(server)
                        call_error(server, session, protocol, kind, model, images, 400, code, index)
                        after = drained(server)
                        require(after['decode_calls'] == before['decode_calls'],
                                f'{protocol}/{kind}: rejected preflight invoked decoder')
                        require(after['peak_bytes'] == before['peak_bytes'],
                                f'{protocol}/{kind}: rejected preflight reserved bytes')
                        require(after["rejected_requests"] == before["rejected_requests"] +
                                (1 if code == "image_limit_exceeded" else 0),
                                f"{protocol}/{kind}: incorrect budget rejection accounting")
                    before = budget(server)
                    expected_status = 404 if protocol == 'v1' else 400
                    call_error(server, session, protocol, kind, 'missing-model', [large],
                               expected_status, 'unknown_model', None)
                    call_error(server, session, protocol, kind, 'corrupt-' + kind, ['%%%'],
                               500, 'model_load_failed', None)
                    require(drained(server)['decode_calls'] == before['decode_calls'],
                            'Lookup/load priority invoked image decoder')
                    require(drained(server)["rejected_requests"] == before["rejected_requests"],
                            "Lookup/load failure counted as image-budget rejection")
                before = budget(server)
                infer(server, kind, model, [small])
                after = drained(server)
                require(after['decode_calls'] == before['decode_calls'] + 1 and after['peak_bytes'] == 48,
                        f'{kind}: exact boundary did not decode once/charge 48 bytes: {after}')
            # A valid header with truncated raster reaches the codec and releases reservation on failure.
            broken = base64.b64encode(b'P6\n4 4\n255\n').decode('ascii')
            before = budget(server)
            error_response(server.request('/v0/infer/yolo', {'model': 'hd2-fp32', 'images': [broken]}),
                           400, 'invalid_image', 0)
            after = drained(server)
            require(after['decode_calls'] == before['decode_calls'] + 1,
                    'Codec failure was not observed/refunded')
    print('PASS cross-protocol predecode pixel/batch limits, ordered priority, exact boundary and codec refund')


def cumulative_boundary(executable, root):
    small = ppm(4, 4)
    with Server(executable, root, model_yaml(root), options=options(16, 96, 96)) as server:
        server.wait_ready()
        before = budget(server)
        infer(server, 'yolo', 'hd2-fp32', [small, small])
        after = drained(server)
        require(after['decode_calls'] == before['decode_calls'] + 2 and after['peak_bytes'] == 96,
                f'Exact batch/global byte boundary failed: {after}')
        before = after
        error_response(server.request('/v1/infer/yolo', {'model': 'hd2-fp32', 'images': [small] * 3}),
                       400, 'image_limit_exceeded', 2)
        require(drained(server)['decode_calls'] == before['decode_calls'], 'Batch rejection decoded inputs')
    print('PASS exact cumulative batch/global boundary and first-overflow index')


def active_tool(session, images, request_id):
    session.send('tools/call', {'name': 'infer_ocr', 'arguments': {
        'model': 'ppocr-v4', 'images': images, 'timeout_ms': 300000}}, request_id)
    return wait_budget(session.server, lambda state: state['in_use_bytes'] > 0,
                       'Could not observe real active input reservation')


def global_and_slot(executable, root, image, pixels):
    count = 64
    charge = pixels * 3 * count
    for slots in (4, 1):
        with Server(executable, root, model_yaml(root),
                    options=options(pixels, charge, charge, slots)) as server:
            server.wait_ready()
            infer(server, 'ocr', 'ppocr-v4', [image])  # Exclude lazy-load timing from overlap.
            infer(server, "yolo", "hd2-fp32", [])  # Warm probe model without input/decode quota.
            with Session(server) as session:
                session.initialize()
                active_tool(session, [image] * count, 'occupy')
                # Ensure all decoder starts precede the rejection comparison.
                baseline = wait_budget(server, lambda state: state['decode_calls'] == count + 1,
                                       'Batch did not finish decoding')
                require(baseline['in_use_bytes'] == charge and baseline['active_requests'] == 1,
                        f'Incorrect full-batch reservation: {baseline}')
                error_response(server.request('/v1/infer/yolo', {'model': 'hd2-fp32', 'images': [image]}),
                               503, 'service_overloaded', None)
                middle = budget(server)
                require(middle['in_use_bytes'] == charge and middle['decode_calls'] == baseline['decode_calls'],
                        f'Overload decoded or refunded active pixels: {middle}')
                require(middle['rejected_requests'] == baseline['rejected_requests'] + 1,
                        'Global/credit rejection not counted exactly once')
                error_response(server.request('/v0/infer/yolo', {'model': 'hd2-fp32', 'images': ['%%%']}),
                               503 if slots == 1 else 400,
                               'service_overloaded' if slots == 1 else 'invalid_image',
                               None if slots == 1 else 0)
                # Empty valid-model requests retain existing semantics even when quota/credit is full.
                infer(server, 'yolo', 'hd2-fp32', [])
                current = budget(server)
                require(current["decode_calls"] == baseline["decode_calls"] and
                        current["in_use_bytes"] == charge and current["active_requests"] == 1,
                        'Empty batch acquired quota or refunded an active request')
                tool_data(session.receive('occupy', timeout=300))
                after = drained(server)
                require(after['peak_bytes'] == charge, f'Global budget peak exceeded limit: {after}')
                before = after['decode_calls']
                infer(server, 'ocr', 'ppocr-v4', [image])
                require(drained(server)['decode_calls'] == before + 1, 'Quota did not recover after success')
    print('PASS shared global/slot overload, invalid-image precedence, saturated empty batch and recovery')


def lifetime_matrix(executable, root, image, pixels):
    count = 64
    charge = pixels * 3 * count
    with Server(executable, root, model_yaml(root), options=options(max(pixels, 1024), charge, charge)) as server:
        server.wait_ready()
        infer(server, 'ocr', 'ppocr-v4', [image])
        # Real ORT runtime exception must release the same quota as success.
        before = budget(server)['decode_calls']
        error_response(server.request('/v0/infer/yolo', {'model': 'runtime-yolo', 'images': [ppm(32, 32, 255)]}),
                       500, 'inference_failed', 0)
        require(drained(server)['decode_calls'] == before + 1, 'Runtime exception leaked input reservation')
        with Session(server) as session:
            session.initialize()
            active_tool(session, [image] * count, 'cancel')
            session.send('notifications/cancelled', {'requestId': 'cancel', 'reason': 'budget regression'})
            response = session.receive('cancel', timeout=300)
            if 'result' in response:
                mcp_error(response, 'request_cancelled', None)
            else:
                require(response['error']['code'] == -32800, f'Unexpected cancellation: {response}')
            drained(server)
            tool_data(session.tool('infer_ocr', {'model': 'ppocr-v4', 'images': [image]}))
            before = budget(server)['decode_calls']
            error = session.tool("infer_ocr", {"model": "ppocr-v4", "images": [image] * count, "timeout_ms": 1000})
            mcp_error(error, 'request_timeout', None)
            after = drained(server)
            require(after['decode_calls'] > before, 'Timeout never exercised admitted decoded input')
        with Session(server) as disconnected:
            disconnected.initialize()
            active_tool(disconnected, [image] * count, 'session-close')
        drained(server)
        # A native disconnect must not release physical work just because the client disappears.
        connection = http.client.HTTPConnection('127.0.0.1', server.port, timeout=120)
        try:
            connection.request('POST', '/v1/infer/ocr',
                               json.dumps({'model': 'ppocr-v4', 'images': [image] * count, 'timeout_ms': 300000}),
                               {'Content-Type': 'application/json'})
            wait_budget(server, lambda state: state['in_use_bytes'] == charge,
                        'Native request did not reserve input quota')
            connection.sock.shutdown(socket.SHUT_RDWR)
            connection.close()
        finally:
            connection.close()
        # Disconnect now requests cooperative cancellation. It may have already
        # physically drained when stats is sampled; an instantaneous zero is valid.
        # The pipeline/service lifetime tests prove refund follows native drain.
        drained(server, timeout=120)
        infer(server, 'ocr', 'ppocr-v4', [image])
        drained(server)
    print('PASS real runtime exception, cancellation, timeout, MCP/native disconnect and quota recovery')


def shutdown_active(executable, root, image, pixels):
    session = None
    try:
        with Server(executable, root, model_yaml(root),
                    options=options(pixels, pixels * 3 * 64, pixels * 3 * 64)) as server:
            server.wait_ready()
            infer(server, 'ocr', 'ppocr-v4', [image])
            session = Session(server).__enter__()
            session.initialize()
            active_tool(session, [image] * 64, 'shutdown')
            # Server.__exit__ sends stdin newline and requires clean physical shutdown,
            # no owned listener and exit=0 while this request is genuinely active.
    finally:
        if session:
            session.close()
    print('PASS shutdown with active input reservation drains cleanly')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True, type=Path)
    parser.add_argument('--project-root', default=Path(__file__).resolve().parents[1], type=Path)
    args = parser.parse_args()
    executable, root = args.server.resolve(), args.project_root.resolve()
    fixture(root, 'app/assets/test/hd2-yolo11n-fp32.onnx')
    fixture(root, 'app/assets/test/ppocr_det.onnx')
    fixture(root, 'app/assets/test/ppocr_rec.onnx')
    fixture(root, 'app/assets/test/ppocr_keys_v1.txt')
    fixture(root, 'app/assets/test/yolo_runtime_failure.onnx')
    encoded = fixture(root, 'doc/images/ppocr.png').read_bytes()
    require(encoded[:8] == b'\x89PNG\r\n\x1a\n', 'OCR fixture is not PNG')
    width, height = struct.unpack('>II', encoded[16:24])
    image, pixels = base64.b64encode(encoded).decode('ascii'), width * height
    predecode_matrix(executable, root)
    cumulative_boundary(executable, root)
    global_and_slot(executable, root, image, pixels)
    lifetime_matrix(executable, root, image, pixels)
    shutdown_active(executable, root, image, pixels)
    print('PASS all real image-budget regressions')


if __name__ == '__main__':
    main()
