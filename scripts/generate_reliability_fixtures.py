"""Regenerate tiny checked-in ONNX fixtures (development-only `onnx` required).

The input-dependent Gather intentionally fails on white pixels (index 1 into
one row), exercising a real ORT Run exception rather than a mock. Black pixels
select row 0 (YOLO/detection) or -1 (normalized OCR recognition) successfully.
No model weights or production inference assets are modified.
"""
import argparse
import math
from pathlib import Path

import onnx
from onnx import TensorProto as T, helper as h


DEST = Path(__file__).resolve().parents[1] / "app" / "assets" / "test"


def save(name, nodes, inputs, outputs, initializers, names=None, metadata=None):
    graph = h.make_graph(nodes, name, inputs, outputs, initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)], ir_version=8)
    model.producer_name = "vision-simple reliability regression"
    properties = dict(metadata or {})
    if names:
        properties["names"] = names
    if properties:
        h.set_model_props(model, properties)
    onnx.checker.check_model(model)
    (DEST / name).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, DEST / name)


def selector():
    return [h.make_node("ReduceMax", ["images"], ["maximum"], keepdims=0),
            h.make_node("Cast", ["maximum"], ["index"], to=T.INT64)]


def recognition_batch_fixtures():
    # Four separated text regions with interleaved widths. Their source-image
    # intensities identify each crop, so incorrect packing/order changes text.
    height, width = 256, 384
    mask = [0.0] * (height * width)
    # Keep full DB-unclip margins inside four distinct source bands.
    for y, box_width in ((20, 64), (84, 112), (148, 64), (212, 112)):
        for row in range(y, y + 20):
            for col in range(32, 32 + box_width):
                mask[row * width + col] = 1.0
    save("ocr_det_batch.onnx",
         [h.make_node("Identity", ["mask"], ["output"])],
         [h.make_tensor_value_info("images", T.FLOAT, [1, 3, height, width])],
         [h.make_tensor_value_info("output", T.FLOAT, [1, 1, height, width])],
         [h.make_tensor("mask", T.FLOAT, [1, 1, height, width], mask)])

    for name, batch, rec_width, sar, require_batch in (
        ("ocr_rec_batch.onnx", "batch", "width", False, False),
        ("ocr_rec_batch_required.onnx", "batch", "width", False, True),
        ("ocr_rec_fixed_batch.onnx", 3, 240, False, False),
        ("ocr_rec_sar_batch.onnx", "batch", "width", True, False),
    ):
        classes, timesteps = (7, 4) if sar else (5, 2)
        scores = []
        for character in range(4):
            tokens = [character, character, 5, (character + 1) % 4] if sar else [
                character + 1, character + 1]
            for token in tokens:
                scores.extend(.9 if index == token else .01
                              for index in range(classes))
        constants = [
            h.make_tensor("zero_index", T.INT64, [], [0]),
            h.make_tensor("center_y", T.INT64, [], [24]),
            h.make_tensor("width_index", T.INT64, [], [3]),
            h.make_tensor("index_two", T.INT64, [], [2]),
            h.make_tensor("one", T.FLOAT, [], [1]),
            h.make_tensor("two", T.FLOAT, [], [2]),
            h.make_tensor("scores", T.FLOAT, [4, timesteps, classes], scores),
        ]
        nodes = [
            # Crop margins can cross a neighboring band after DB unclip.
            # The center identifies the source region independently of padding.
            h.make_node("Shape", ["images"], ["input_shape"]),
            h.make_node("Gather", ["input_shape", "width_index"], ["input_width"], axis=0),
            h.make_node("Div", ["input_width", "index_two"], ["center_x"]),
            h.make_node("Gather", ["images", "center_y"], ["center_row"], axis=2),
            h.make_node("Gather", ["center_row", "center_x"], ["pixel"], axis=2),
            h.make_node("ReduceMean", ["pixel"], ["gray"], axes=[1], keepdims=0),
            h.make_node("Add", ["gray", "one"], ["positive"]),
            h.make_node("Mul", ["positive", "two"], ["scaled"]),
            h.make_node("Floor", ["scaled"], ["floored"]),
            h.make_node("Cast", ["floored"], ["character"], to=T.INT64),
            h.make_node("Gather", ["scores", "character"], ["decoded"], axis=0),
        ]
        if require_batch:
            constants += [h.make_tensor("single", T.INT64, [], [1]),
                          h.make_tensor("guard_rows", T.FLOAT, [1], [0])]
            nodes += [
                h.make_node("Gather", ["input_shape", "zero_index"], ["count"], axis=0),
                h.make_node("Equal", ["count", "single"], ["is_single"]),
                h.make_node("Cast", ["is_single"], ["guard_index"], to=T.INT64),
                # N=1 addresses row1 of one row: a real ORT error. N>1 succeeds.
                h.make_node("Gather", ["guard_rows", "guard_index"], ["guard"], axis=0),
                h.make_node("Add", ["decoded", "guard"], ["output"]),
            ]
        else:
            nodes.append(h.make_node("Identity", ["decoded"], ["output"]))
        save(name, nodes,
             [h.make_tensor_value_info("images", T.FLOAT, [batch, 3, 48, rec_width])],
             [h.make_tensor_value_info("output", T.FLOAT, [batch, timesteps, classes])],
             constants)


def yolo_task_fixtures():
    boxes = [[16, 16, 48], [16, 16, 48], [24, 24, 16], [24, 24, 16]]
    segmentation = boxes + [[.9, .8, .05], [.1, .2, .95],
                            [1, 1, 0], [0, 0, 1]]
    pose = boxes + [[.9, .8, .95], [10, 10, 45], [12, 12, 47],
                    [.9, .9, .8], [22, 22, 52], [24, 24, 50], [.2, .2, .7]]
    obb = [[32] * 3, [32] * 3, [40] * 3, [8] * 3,
           [.9, .8, .95], [.1, .1, .05],
           [math.pi / 4, math.pi / 4 + .01, -math.pi / 4]]
    prototypes = [(-1 if x < 4 else 1) for y in range(16) for x in range(16)]
    prototypes += [(1 if x < 12 else -1) for y in range(16) for x in range(16)]
    names = "{0: 'first', 1: 'second'}"
    for suffix, dtype in (("", T.FLOAT), ("_fp16", T.FLOAT16)):
        for task, rows, metadata in (
            ("seg", segmentation, {"task": "segment"}),
            ("pose", pose, {"task": "pose", "kpt_shape": "[2, 3]"}),
            ("obb", obb, {"task": "obb"}),
        ):
            shape = [1, len(rows), 3]
            nodes = [h.make_node("Identity", ["predictions"], ["output0"])]
            outputs = [h.make_tensor_value_info("output0", dtype, shape)]
            constants = [h.make_tensor("predictions", dtype, shape,
                                       [value for row in rows for value in row])]
            if task == "seg":
                nodes.append(h.make_node("Identity", ["prototypes"], ["output1"]))
                outputs.append(h.make_tensor_value_info("output1", dtype, [1, 2, 16, 16]))
                constants.append(h.make_tensor("prototypes", dtype, [1, 2, 16, 16], prototypes))
            save(f"reliability/yolo_{task}{suffix}.onnx", nodes,
                 [h.make_tensor_value_info("images", dtype, [1, 3, 64, 64])],
                 outputs, constants,
                 "{0: 'person'}" if task == "pose" else names, metadata)
    for name, rows, metadata in (
        ("pose_d2", pose[:7] + pose[8:10], {"task": "pose", "kpt_shape": "[2,2]"}),
        ("pose_nan", pose[:5] + [[float("nan"), 10, 45]] + pose[6:],
         {"task": "pose", "kpt_shape": "[2,3]"}),
    ):
        shape = [1, len(rows), 3]
        nodes = [h.make_node("Identity", ["predictions"], ["output0"])]
        constants = [h.make_tensor("predictions", T.FLOAT, shape,
                                   [value for row in rows for value in row])]
        if name == "pose_nan":
            constants += [h.make_tensor("valid", T.FLOAT, shape,
                                        [value for row in pose for value in row]),
                          h.make_tensor("threshold", T.FLOAT, [], [.5])]
            nodes = [h.make_node("ReduceMean", ["images"], ["mean"], keepdims=0),
                     h.make_node("Greater", ["mean", "threshold"], ["bad"]),
                     h.make_node("Where", ["bad", "predictions", "valid"], ["output0"])]
        save(f"reliability/yolo_{name}.onnx", nodes,
             [h.make_tensor_value_info("images", T.FLOAT, [1, 3, 64, 64])],
             [h.make_tensor_value_info("output0", T.FLOAT, shape)], constants,
             "{0: 'person'}", metadata)
    quoted_names = r'''{"0": "worker's glove", '1': 'say \'hi\' \\ \u00e9 \U0001f600'}'''
    for name, rows, class_names, metadata in (
        ("names_obb", obb, quoted_names,
         {"task": "obb", "args": '''{'nms': False, 'note': "'nms': True", 'nested': [1, {'value': 'end2end: true'}]}'''}),
        ("names_detect", boxes + segmentation[4:6], quoted_names, {}),
        ("names_invalid", obb, "{0: 'first', 2: 'second'}", {"task": "obb"}),
        ("names_unclosed", obb, """{0: "worker's glove, 1: 'second'}""", {"task": "obb"}),
        ("names_escape_invalid", obb, r"{0: 'bad\q', 1: 'second'}", {"task": "obb"}),
        ("pose_nms_python", [[0] * 12 for _ in range(11)], "{0: 'person'}",
         {"task": "pose", "kpt_shape": "[2,3]", "args": "{'nms': True}"}),
        ("pose_nms_json", [[0] * 12 for _ in range(11)], "{0: 'person'}",
         {"task": "pose", "kpt_shape": "[2,3]", "args": '{"nms": true}'}),
        ("pose_end2end", [[0] * 12 for _ in range(11)], "{0: 'person'}",
         {"task": "pose", "kpt_shape": "[2,3]", "end2end": "true"}),
        ("pose_nms_direct", [[0] * 12 for _ in range(11)], "{0: 'person'}",
         {"task": "pose", "kpt_shape": "[2,3]", "nms": "True"}),
        ("pose_end2end_args", [[0] * 12 for _ in range(11)], "{0: 'person'}",
         {"task": "pose", "kpt_shape": "[2,3]", "args": "{'end2end': True}"}),
    ):
        shape = [1, len(rows), len(rows[0])]
        save(f"reliability/yolo_{name}.onnx",
             [h.make_node("Identity", ["predictions"], ["output0"])],
             [h.make_tensor_value_info("images", T.FLOAT, [1, 3, 64, 64])],
             [h.make_tensor_value_info("output0", T.FLOAT, shape)],
             [h.make_tensor("predictions", T.FLOAT, shape,
                            [value for row in rows for value in row])],
             class_names, metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-tasks-only", action="store_true")
    args = parser.parse_args()
    yolo_task_fixtures()
    if args.yolo_tasks_only:
        return
    recognition_batch_fixtures()
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


if __name__ == "__main__":
    main()
