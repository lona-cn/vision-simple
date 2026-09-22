"""Real-model regression for canonical config and task-qualified cache identity."""
import argparse
import base64
import json
from pathlib import Path

from test_http_regression import Server, close_values, error_response, infer, model_stats, require


def configurations(root):
    assets = root / "app/assets/test"
    paths = {key: (assets / filename).as_posix() for key, filename in (
        ("model", "hd2-yolo11n-fp32.onnx"), ("det", "ppocr_det.onnx"),
        ("rec", "ppocr_rec.onnx"), ("dictionary", "ppocr_keys_v1.txt"))}
    quote = json.dumps
    legacy = ('yolo:\n  - name: shared\n    version: kV11\n    path: ' + quote(paths['model']) +
              '\nocr:\n  - name: shared\n    version: kPPOCRv4\n    det_path: ' + quote(paths['det']) +
              '\n    rec_path: ' + quote(paths['rec']) + '\n    char_dict_path: ' + quote(paths['dictionary']) + '\n')
    unified = 'models:\n'
    for task, version, roles in [('yolo', 'kV11', ('model',)),
                                  ('ocr', 'kPPOCRv4', ('det', 'rec', 'dictionary'))]:
        unified += f'  - task: {task}\n    name: shared\n    version: {version}\n    files:\n'
        unified += ''.join(f'      {role}: {quote(paths[role])}\n' for role in roles)
    return legacy, unified


def exercise(executable, root, config, image):
    with Server(executable, root, config,
                options={'infer_idle_timeout_ms': '0', 'ocr_rec_batch_size': '4'}) as server:
        server.wait_ready()
        status, catalog = server.request('/v0/infer/models', method='GET')
        require(status == 200 and catalog == {'yolo': ['shared'], 'ocr': ['shared']},
                f'Task-qualified discovery failed: {status} {catalog}')
        results = {kind: infer(server, kind, 'shared', [image]) for kind in ('yolo', 'ocr')}
        require(set(model_stats(server)) == {('yolo', 'shared'), ('ocr', 'shared')},
                'Same-name tasks must own distinct loaded cache entries')
        status, discovery = server.request('/v1/models', method='GET')
        require(status == 200 and {item['id'] for item in discovery['data']} == {'yolo:shared', 'ocr:shared'},
                'Protocol model IDs must retain task identity')
        status, completion = server.request('/v1/chat/completions', {
            'model': 'ocr:shared',
            'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {
                'url': 'data:image/png;base64,' + image}}]}]})
        require(status == 200 and close_values(json.loads(completion['choices'][0]['message']['content']), results['ocr']),
                'OpenAI-like adapter must execute the same registered OCR model')
        status, body = server.request('/v0/infer/unload', {'kind': 'yolo', 'model': 'shared'})
        require(status == 200, f'YOLO unload failed: {body}')
        require(set(model_stats(server)) == {('ocr', 'shared')},
                'Unloading YOLO must not remove same-name OCR')
        error_response(server.request('/v0/infer/unload', {'kind': 'yolo', 'model': 'shared'}),
                       404, 'model_not_loaded', None)
        require(close_values(infer(server, 'ocr', 'shared', [image]), results['ocr']),
                'Surviving OCR cache entry changed after unrelated unload')
        require(close_values(infer(server, 'yolo', 'shared', [image]), results['yolo']),
                'Reloaded task changed inference')
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True, type=Path)
    parser.add_argument('--project-root', required=True, type=Path)
    args = parser.parse_args()
    executable, root = args.server.resolve(), args.project_root.resolve()
    image = base64.b64encode((root / 'app/assets/test/hd2.png').read_bytes()).decode('ascii')
    legacy, unified = configurations(root)
    old = exercise(executable, root, legacy, image)
    new = exercise(executable, root, unified, image)
    require(close_values(old, new), 'Legacy and unified config must preserve actual inference output')
    unknown = 'models:\n  - task: unregistered\n    name: unknown\n    version: none\n'
    with Server(executable, root, unknown) as server:
        server.wait_ready()
        error_response(server.request('/v0/infer/models', method='GET'), 500, 'model_config_failed', None)
        error_response(server.request('/v0/infer/yolo', {'model': 'unknown', 'images': []}),
                       500, 'model_config_failed', None)
    print('PASS canonical/legacy real inference parity, same-name task isolation, unload/reload, protocol IDs and unsupported registry entries')


if __name__ == '__main__':
    main()
