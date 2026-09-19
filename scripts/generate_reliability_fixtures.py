"""Regenerate tiny checked-in ONNX fixtures (development-only `onnx` required).

The input-dependent Gather intentionally fails on white pixels (index 1 into
one row), exercising a real ORT Run exception rather than a mock. Black pixels
select row 0 (YOLO/detection) or -1 (normalized OCR recognition) successfully.
No model weights or production inference assets are modified.
"""
from pathlib import Path

import onnx
from onnx import TensorProto as T, helper as h


DEST = Path(__file__).resolve().parents[1] / "app" / "assets" / "test"


def save(name, nodes, inputs, outputs, initializers, names=None):
    graph = h.make_graph(nodes, name, inputs, outputs, initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)], ir_version=8)
    model.producer_name = "vision-simple reliability regression"
    if names:
        h.set_model_props(model, {"names": names})
    onnx.checker.check_model(model)
    onnx.save(model, DEST / name)


def selector():
    return [h.make_node("ReduceMax", ["images"], ["maximum"], keepdims=0),
            h.make_node("Cast", ["maximum"], ["index"], to=T.INT64)]


if __name__ == "__main__":
    save("yolo_runtime_failure.onnx",
         selector() + [h.make_node("Gather", ["rows", "index"], ["detections"], axis=0)],
         [h.make_tensor_value_info("images", T.FLOAT, [1, 3, 32, 32])],
         [h.make_tensor_value_info("detections", T.FLOAT, [1, 1, 6])],
         [h.make_tensor("rows", T.FLOAT, [1, 1, 1, 6], [4, 4, 20, 20, .9, 0])],
         "{0: 'fixture'}")
    save("ocr_det_runtime_failure.onnx",
         selector() + [h.make_node("Gather", ["rows", "index"], ["offset"], axis=0),
                       h.make_node("ReduceMean", ["images"], ["mask"], axes=[1], keepdims=1),
                       h.make_node("Add", ["mask", "offset"], ["output"])],
         [h.make_tensor_value_info("images", T.FLOAT, [1, 3, "height", "width"])],
         [h.make_tensor_value_info("output", T.FLOAT, [1, 1, "height", "width"])],
         [h.make_tensor("rows", T.FLOAT, [1], [0])])
    save("ocr_det_box.onnx",
         [h.make_node("Identity", ["box"], ["output"])],
         [h.make_tensor_value_info("images", T.FLOAT, [1, 3, 32, 32])],
         [h.make_tensor_value_info("output", T.FLOAT, [1, 1, 32, 32])],
         [h.make_tensor("box", T.FLOAT, [1, 1, 32, 32], [1] * 1024)])
    save("ocr_rec_runtime_failure.onnx",
         selector() + [h.make_node("Gather", ["rows", "index"], ["output"], axis=0)],
         [h.make_tensor_value_info("images", T.FLOAT, [1, 3, 48, "width"])],
         [h.make_tensor_value_info("output", T.FLOAT, [1, 1, 2])],
         [h.make_tensor("rows", T.FLOAT, [1, 1, 1, 2], [.1, .9])])
