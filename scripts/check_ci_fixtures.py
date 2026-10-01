#!/usr/bin/env python3
"""Require materialized fixtures before native or real-server CI regressions."""

import argparse
from pathlib import Path
import sys

from test_http_regression import RegressionFailure, fixture


# Explicit consumer manifests, including models intentionally rejected by the
# runtime. Missing negative fixtures must not masquerade as rejected model loads.
NATIVE_FIXTURES = (
    "app/assets/test/reliability/yolo26_detect_threshold_raw.onnx",
    "app/assets/test/reliability/yolo26_detect_threshold_e2e.onnx",
    "app/assets/test/hd2-yolo11n-fp32.onnx",
    "app/assets/test/ppocr_det.onnx",
    "app/assets/test/ppocr_rec.onnx",
    "app/assets/test/ppocr_keys_v1.txt",
    "app/assets/test/yolo_runtime_failure.onnx",
    "app/assets/test/ocr_det_runtime_failure.onnx",
    "app/assets/test/ocr_det_box.onnx",
    "app/assets/test/ocr_rec_runtime_failure.onnx",
    "app/assets/test/ocr_det_batch.onnx",
    "app/assets/test/ocr_rec_batch.onnx",
    "app/assets/test/ocr_rec_batch_required.onnx",
    "app/assets/test/ocr_rec_fixed_batch.onnx",
    "app/assets/test/ocr_rec_sar_batch.onnx",
    "app/assets/test/ocr_sar_dictionary.txt",
    "app/assets/test/reliability/yolo_names_obb.onnx",
    "app/assets/test/reliability/yolo_names_detect.onnx",
    "app/assets/test/reliability/yolo_names_invalid.onnx",
    "app/assets/test/reliability/yolo_names_unclosed.onnx",
    "app/assets/test/reliability/yolo_names_escape_invalid.onnx",
    "app/assets/test/reliability/yolo_pose_nms_python.onnx",
    "app/assets/test/reliability/yolo_pose_nms_json.onnx",
    "app/assets/test/reliability/yolo_pose_end2end.onnx",
    "app/assets/test/reliability/yolo_pose_nms_direct.onnx",
    "app/assets/test/reliability/yolo_pose_end2end_args.onnx",
    "app/assets/test/reliability/yolo_pose_d2.onnx",
    "app/assets/test/reliability/yolo_pose_nan.onnx",
    "app/assets/test/reliability/yolo_seg.onnx",
    "app/assets/test/reliability/yolo_seg_fp16.onnx",
    "app/assets/test/reliability/yolo26_seg_raw.onnx",
    "app/assets/test/reliability/yolo26_seg_raw_fp16.onnx",
    "app/assets/test/reliability/yolo26_seg_e2e.onnx",
    "app/assets/test/reliability/yolo26_seg_e2e_fp16.onnx",
    "app/assets/test/reliability/yolo_pose.onnx",
    "app/assets/test/reliability/yolo_pose_fp16.onnx",
    "app/assets/test/reliability/yolo26_pose_raw.onnx",
    "app/assets/test/reliability/yolo26_pose_raw_fp16.onnx",
    "app/assets/test/reliability/yolo26_pose_e2e.onnx",
    "app/assets/test/reliability/yolo26_pose_e2e_fp16.onnx",
    "app/assets/test/reliability/yolo_obb.onnx",
    "app/assets/test/reliability/yolo_obb_fp16.onnx",
    "app/assets/test/reliability/yolo26_obb_raw.onnx",
    "app/assets/test/reliability/yolo26_obb_raw_fp16.onnx",
    "app/assets/test/reliability/yolo26_obb_e2e.onnx",
    "app/assets/test/reliability/yolo26_obb_e2e_fp16.onnx",
    "app/assets/test/reliability/yolo26_detect_missing_task.onnx",
    "app/assets/test/reliability/yolo26_detect_wrong_task.onnx",
    "app/assets/test/reliability/yolo26_detect_missing_args.onnx",
    "app/assets/test/reliability/yolo26_detect_empty_args.onnx",
    "app/assets/test/reliability/yolo26_detect_missing_mode.onnx",
    "app/assets/test/reliability/yolo26_detect_invalid_mode.onnx",
    "app/assets/test/reliability/yolo26_detect_embedded_nms.onnx",
    "app/assets/test/reliability/yolo26_detect_conflicting_mode.onnx",
    "app/assets/test/reliability/yolo26_detect_raw_conflict.onnx",
    "app/assets/test/reliability/yolo26_detect_raw.onnx",
    "app/assets/test/reliability/yolo26_detect_raw_fp16.onnx",
    "app/assets/test/reliability/yolo26_detect_e2e.onnx",
    "app/assets/test/reliability/yolo26_detect_e2e_fp16.onnx",
    "app/assets/test/reliability/yolo26_pose_missing_task.onnx",
    "app/assets/test/reliability/yolo26_pose_missing_kpt_shape.onnx",
    "app/assets/test/reliability/yolo26_pose_bad_extra.onnx",
    "app/assets/test/reliability/yolo26_pose_missing_mode.onnx",
    "app/assets/test/reliability/yolo26_pose_conflicting_mode.onnx",
    "app/assets/test/reliability/yolo26_pose_embedded_nms.onnx",
    "app/assets/test/reliability/yolo26_seg_bad_proto.onnx",
    "app/assets/test/reliability/yolo26_obb_bad_extra.onnx",
    "app/assets/test/reliability/yolo26_pose_bad_class.onnx",
    "app/assets/test/reliability/yolo26_pose_out_of_range_class.onnx",
)

HTTP_FIXTURES = (
    "app/assets/test/reliability/yolo26_detect_threshold_raw.onnx",
    "app/assets/test/reliability/yolo26_detect_threshold_e2e.onnx",
    "app/assets/test/hd2-yolo11n-fp32.onnx",
    "app/assets/test/hd2-yolo11n-fp16.onnx",
    "app/assets/test/ppocr_det.onnx",
    "app/assets/test/ppocr_rec.onnx",
    "app/assets/test/ppocr_keys_v1.txt",
    "app/assets/test/hd2.png",
    "app/assets/test/http_yolo_reference.json",
    "app/assets/test/yolo_runtime_failure.onnx",
    "app/assets/test/ocr_det_runtime_failure.onnx",
    "app/assets/test/reliability/yolo_seg.onnx",
    "app/assets/test/reliability/yolo_seg_fp16.onnx",
    "app/assets/test/reliability/yolo_pose.onnx",
    "app/assets/test/reliability/yolo_pose_fp16.onnx",
    "app/assets/test/reliability/yolo_obb.onnx",
    "app/assets/test/reliability/yolo_obb_fp16.onnx",
    "app/assets/test/reliability/yolo_pose_nan.onnx",
    "app/assets/test/reliability/yolo_pose_nms_python.onnx",
    "app/assets/test/reliability/yolo26_seg_raw.onnx",
    "app/assets/test/reliability/yolo26_seg_raw_fp16.onnx",
    "app/assets/test/reliability/yolo26_seg_e2e.onnx",
    "app/assets/test/reliability/yolo26_seg_e2e_fp16.onnx",
    "app/assets/test/reliability/yolo26_pose_raw.onnx",
    "app/assets/test/reliability/yolo26_pose_raw_fp16.onnx",
    "app/assets/test/reliability/yolo26_pose_e2e.onnx",
    "app/assets/test/reliability/yolo26_pose_e2e_fp16.onnx",
    "app/assets/test/reliability/yolo26_obb_raw.onnx",
    "app/assets/test/reliability/yolo26_obb_raw_fp16.onnx",
    "app/assets/test/reliability/yolo26_obb_e2e.onnx",
    "app/assets/test/reliability/yolo26_obb_e2e_fp16.onnx",
    "doc/images/ppocr.png",
    "app/config/base/log.properties",
    "doc/openapi/server.yaml",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--layer", choices=("native", "http"), required=True)
    args = parser.parse_args()
    try:
        root = args.project_root.resolve(strict=True)
        fixtures = NATIVE_FIXTURES if args.layer == "native" else HTTP_FIXTURES
        for relative in fixtures:
            fixture(root, relative)
    except (RegressionFailure, OSError) as error:
        print(f"FAIL {args.layer} fixture preflight: {error}", file=sys.stderr)
        return 1
    print(f"PASS {args.layer} fixture preflight: {len(fixtures)} regular, nonempty, materialized files", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
