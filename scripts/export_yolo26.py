"""Export pinned YOLO26 ONNX models and record reproducible artifact contracts.

Development-only dependencies: ultralytics==8.4.159, onnx==1.20.1,
onnxruntime==1.24.3. Downloads official pretrained nano weights on first use.
Artifacts belong in an ignored build directory, not the source tree.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('build/yolo26'))
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--tasks', nargs='+', choices=['detect', 'seg', 'pose', 'obb'],
                        default=['detect', 'seg', 'pose', 'obb'])
    parser.add_argument('--precisions', nargs='+', type=int, choices=[32, 16], default=[32, 16])
    args = parser.parse_args()
    import onnx
    import onnxruntime as ort
    import torch
    import ultralytics
    from ultralytics import YOLO
    if ultralytics.__version__ != '8.4.159':
        raise RuntimeError('Export contract requires ultralytics==8.4.159')
    torch.set_num_threads(4)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(ultralytics=ultralytics.__version__, torch=torch.__version__,
                    onnx=onnx.__version__, onnxruntime=ort.__version__)
    artifacts = manifest.setdefault('artifacts', {})
    for task in args.tasks:
        stem = 'yolo26n' + ('' if task == 'detect' else '-' + task)
        weights = output / (stem + '.pt')
        # An absolute missing checkpoint path is supported by Ultralytics' downloader.
        for precision in args.precisions:
            for mode, nms in [('raw', None), ('e2e', False)]:
                name = f'{task}_{mode}_fp{precision}'
                target = output / (name + '.onnx')
                model = YOLO(str(weights))
                exported = Path(model.export(format='onnx', imgsz=args.imgsz, batch=1,
                                             dynamic=False, simplify=False, opset=17,
                                             quantize=precision, nms=nms, device='cpu'))
                shutil.move(str(exported), target)
                graph = onnx.load(target)
                # ORT's FP16 converter appends I/O casts after their consumers.
                # Normalize node order without altering tensor types or operations.
                if precision == 16:
                    from onnxruntime.transformers.onnx_model import OnnxModel
                    OnnxModel(graph).topological_sort()
                    onnx.save(graph, target)
                onnx.checker.check_model(graph)
                options = ort.SessionOptions()
                options.intra_op_num_threads = 4
                session = ort.InferenceSession(str(target), options, providers=['CPUExecutionProvider'])
                def spec(value):
                    return {'name': value.name, 'shape': value.shape, 'type': value.type}
                artifacts[name] = {
                    'path': target.name, 'weights': weights.name,
                    'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
                    'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                    'opset': 17, 'imgsz': args.imgsz, 'batch': 1, 'dynamic': False,
                    'simplify': False, 'nms': nms, 'quantize': precision,
                    'inputs': [spec(v) for v in session.get_inputs()],
                    'outputs': [spec(v) for v in session.get_outputs()],
                    'metadata': session.get_modelmeta().custom_metadata_map,
                }
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
                print(f'EXPORTED {name}: {artifacts[name]["outputs"]}', flush=True)
    print(f'Manifest: {manifest_path}')


if __name__ == '__main__':
    main()
