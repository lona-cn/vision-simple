"""Real ORT segmentation/pose/OBB serialization, lifecycle and protocol parity."""
import argparse
import base64
import json
import math
from pathlib import Path
import struct
import zlib

from test_http_regression import Server, close_values, error_response, model_stats, require
from test_protocol_regression import Session, chat_payload, oversized_headers, tool_data, wire


def png_chunk(kind, data):
    return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))


def image(width=64, height=64, value=0):
    header = struct.pack('!IIBBBBB', width, height, 8, 2, 0, 0, 0)
    rows = (b'\0' + bytes([value]) * (width * 3)) * height
    png = b'\x89PNG\r\n\x1a\n' + png_chunk(b'IHDR', header)
    png += png_chunk(b'IDAT', zlib.compress(rows)) + png_chunk(b'IEND', b'')
    return base64.b64encode(png).decode('ascii')


def decode_mask(encoded):
    data = base64.b64decode(encoded, validate=True)
    require(data[:8] == b'\x89PNG\r\n\x1a\n', 'Mask is not a PNG')
    position, compressed, shape = 8, bytearray(), None
    while position < len(data):
        length = struct.unpack_from('!I', data, position)[0]
        kind = data[position + 4:position + 8]
        body = data[position + 8:position + 8 + length]
        crc = struct.unpack_from('!I', data, position + 8 + length)[0]
        require(zlib.crc32(kind + body) == crc, 'Corrupt PNG mask chunk')
        if kind == b'IHDR':
            width, height, bits, color, compression, filtering, interlace = struct.unpack('!IIBBBBB', body)
            require((bits, color, compression, filtering, interlace) == (8, 0, 0, 0, 0),
                    'Mask must be a noninterlaced 8-bit grayscale PNG')
            shape = width, height
        elif kind == b'IDAT':
            compressed.extend(body)
        position += length + 12
    require(shape is not None, 'PNG mask lacks dimensions')
    width, height = shape
    raw = zlib.decompress(compressed)
    require(len(raw) == height * (width + 1), 'PNG mask decoded size mismatch')
    rows = []
    for y in range(height):
        mode = raw[y * (width + 1)]
        row = bytearray(raw[y * (width + 1) + 1:(y + 1) * (width + 1)])
        require(mode <= 4, 'Unsupported PNG row filter')
        for x in range(width):
            left = row[x - 1] if x else 0
            above = rows[y - 1][x] if y else 0
            diagonal = rows[y - 1][x - 1] if x and y else 0
            predictors = (left, above, diagonal)
            paeth = min(predictors, key=lambda p: abs(left + above - diagonal - p))
            prediction = (0, left, above, (left + above) // 2, paeth)[mode]
            row[x] = (row[x] + prediction) & 255
        rows.append(row)
    return width, height, rows


def configuration(root):
    entries = [(kind, name, f'yolo_{kind}{suffix}.onnx', 'kV11')
               for kind in ('seg', 'pose', 'obb')
               for name, suffix in (('shared', ''), ('half', '_fp16'))]
    entries += [('pose', 'recover', 'yolo_pose_nan.onnx', 'kV11'),
                ('pose', 'exported-nms', 'yolo_pose_nms_python.onnx', 'kV11')]
    entries += [(kind, f'v26-{mode}{suffix}', f'yolo26_{kind}_{mode}{suffix}.onnx', 'kV26')
                for kind in ('seg', 'pose', 'obb')
                for mode in ('raw', 'e2e') for suffix in ('', '_fp16')]
    text = 'models:\n'
    for kind, name, filename, version in entries:
        path = (root / 'app/assets/test/reliability' / filename).as_posix()
        text += (f'  - task: {kind}\n    name: {name}\n    version: {version}\n'
                 f'    files:\n      model: {json.dumps(path)}\n')
    return text, {f'{kind}:{name}' for kind, name, _, _ in entries}


def infer_task(server, kind, model, images):
    status, result = server.request('/v1/infer/' + kind, {'model': model, 'images': images})
    require(status == 200, f'{kind} inference failed: {status} {result}')
    return result


def check_geometry(kind, body):
    square, wide = body['results']
    require(len(square) == len(wide) == 2, f'{kind} NMS/ordering changed')
    if kind == 'seg':
        require(square[0]['bbox'] == [40, 40, 16, 16] and square[1]['bbox'] == [4, 4, 24, 24],
                'Segmentation bbox serialization is not original-pixel XYWH')
        require(wide[0]['bbox'] == [80, 48, 32, 16], 'Segmentation batch geometry lost image identity')
        for obj in square:
            width, height, rows = decode_mask(obj['mask_png_base64'])
            require([width, height] == obj['bbox'][2:], 'Mask is not bbox-cropped')
            for row in rows:
                expected = bytes(255 if (x < 8 if obj['class_id'] == 1 else x >= 12) else 0
                                 for x in range(width))
                require(row == expected, 'Encoded mask changed prototype/coefficients/foreground')
    elif kind == 'pose':
        require(close_values(square[1]['keypoints'], [
            {'x': 10, 'y': 12, 'confidence': .9}, {'x': 22, 'y': 24, 'confidence': .2}]),
            'Pose keypoints or their independent confidence changed')
        require(wide[1]['keypoints'][0]['x'] == 20 and wide[1]['keypoints'][0]['y'] == -8,
                'Pose inverse padding must preserve out-of-frame coordinates')
    else:
        require(math.isclose(square[0]['angle'], -math.pi / 4, abs_tol=1e-5) and
                math.isclose(square[1]['angle'], math.pi / 4, abs_tol=1e-5),
                'OBB angles must distinguish intersecting rotations')
        for before, after in zip(square, wide):
            require(len(before['corners']) == 4 and all(len(p) == 2 for p in before['corners']),
                    'OBB corners must serialize as four XY pairs')
            for a, b in zip(before['corners'], after['corners']):
                require(math.isclose(b[0], a[0] * 2, abs_tol=1e-4) and
                        math.isclose(b[1], (a[1] - 16) * 2, abs_tol=1e-4),
                        'OBB coordinates lost per-image inverse transform')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True, type=Path)
    parser.add_argument('--project-root', required=True, type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config, expected_ids = configuration(root)
    images = [image(), image(128, 64)]
    with Server(args.server.resolve(), root, config,
                options={'infer_idle_timeout_ms': '0'}) as server:
        server.wait_ready()
        status, legacy = server.request('/v0/infer/models', method='GET')
        require(status == 200 and legacy == {'yolo': [], 'ocr': []},
                'New tasks must not change the v0 catalog shape')
        found, cursor = [], ''
        while True:
            route = '/v1/models?limit=2' + ('&after=' + cursor if cursor else '')
            status, page = server.request(route, method='GET')
            require(status == 200, f'Task catalog pagination failed: {page}')
            found.extend(row['id'] for row in page['data'])
            if not page['has_more']:
                break
            cursor = page['next_cursor']
        require(found == sorted(expected_ids), 'Task-qualified catalog skipped or duplicated models')
        native = {}
        with Session(server) as session:
            session.initialize()
            tools = {tool['name'] for tool in session.call('tools/list')['result']['tools']}
            require({'infer_seg', 'infer_pose', 'infer_obb'} <= tools, 'New task tools were not discovered')
            for kind in ('seg', 'pose', 'obb'):
                native[kind] = infer_task(server, kind, 'shared', images)
                check_geometry(kind, native[kind])
                half = infer_task(server, kind, 'half', images)
                require([[obj['class_id'] for obj in row] for row in half['results']] ==
                        [[obj['class_id'] for obj in row] for row in native[kind]['results']],
                        'FP16 task loading changed object identities/order')
                raw26 = infer_task(server, kind, 'v26-raw', images)
                require(close_values(raw26, native[kind]),
                        f'{kind} YOLO26 raw decoding changed task geometry')
                end26 = infer_task(server, kind, 'v26-e2e', images)
                for raw_frame, end_frame in zip(raw26['results'], end26['results']):
                    require(len(end_frame) == 3 and close_values(end_frame[:2], raw_frame),
                            f'{kind} YOLO26 e2e lost overlapping predictions or changed extras')
                for mode in ('raw', 'e2e'):
                    half26 = infer_task(server, kind, f'v26-{mode}_fp16', images)
                    full26 = raw26 if mode == 'raw' else end26
                    require([[obj['class_id'] for obj in row] for row in half26['results']] ==
                            [[obj['class_id'] for obj in row] for row in full26['results']],
                            f'{kind} YOLO26 {mode} FP16 changed selected predictions')
                status, completion = server.request('/v1/chat/completions',
                                                     chat_payload(kind, 'shared', images))
                require(status == 200 and close_values(
                    json.loads(completion['choices'][0]['message']['content']), native[kind]),
                    f'{kind} OpenAI result differs from native structured result')
                mcp = tool_data(session.tool('infer_' + kind, {'model': 'shared', 'images': images}))
                require(close_values(mcp, native[kind]), f'{kind} MCP result differs from native result')
        status, _ = server.request('/v0/infer/unload', {'kind': 'seg', 'model': 'shared'})
        require(status == 200 and ('seg', 'shared') not in model_stats(server) and
                ('pose', 'shared') in model_stats(server) and ('obb', 'shared') in model_stats(server),
                'Task-qualified unload affected another same-name model')
        require(close_values(infer_task(server, 'seg', 'shared', images), native['seg']),
                'Reloaded segmentation changed owning masks')
        error_response(server.request('/v1/infer/pose', {'model': 'recover', 'images': [image(), image(value=255)]}),
                       500, 'inference_failed', 1)
        require(close_values(infer_task(server, 'pose', 'recover', images), native['pose']),
                'Task runtime failure poisoned reusable model/pipeline state')
        error_response(server.request('/v1/infer/pose', {'model': 'exported-nms', 'images': [image()]}),
                       500, 'model_load_failed', None)
        oversized_headers(server, '/v1/infer/seg')
        status, _, _ = wire(server, 'POST', '/v1/infer/seg', raw='{}',
                             headers={'Content-Type': 'text/plain'})
        require(status == 415, 'Native v1 inference did not reject non-JSON transport')
    print('PASS real seg/pose/OBB masks and geometry, FP16, catalog, cache isolation, runtime recovery and native/OpenAI/MCP parity')


if __name__ == '__main__':
    main()
