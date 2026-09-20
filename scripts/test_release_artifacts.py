#!/usr/bin/env python3
"""Behavior tests for release archive generation."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile


SCRIPT = Path(__file__).with_name("release_artifacts.py")


class ReleaseArtifactsTest(unittest.TestCase):
    def test_pack_includes_runtime_files_and_manifest(self):
        with tempfile.TemporaryDirectory(prefix="vision-simple-release-") as temporary:
            root = Path(temporary)
            target = root / "build" / "windows" / "x64" / "release"
            (target / "config").mkdir(parents=True)
            (target / "assets").mkdir()
            files = {
                "vision_simple-server.exe": b"server",
                "vsrt_windows_x64_release.lib": b"runtime",
                "onnxruntime.dll": b"dependency",
                "test_common.exe": b"test",
                "config/server.yaml": b"port: 11451\n",
                "assets/model.onnx": b"model",
                "assets/.gitkeep": b"",
            }
            for relative, content in files.items():
                path = target / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)

            archive = root / "vision_simple-windows-x64-20260920-1234-UTF8.zip"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "pack",
                    "--server",
                    str(target / "vision_simple-server.exe"),
                    "--runtime",
                    str(target / "vsrt_windows_x64_release.lib"),
                    "--output",
                    str(archive),
                    "--platform",
                    "windows",
                    "--arch",
                    "x64",
                    "--variant",
                    "msvc-cpu",
                    "--commit",
                    "0123456789abcdef",
                    "--timestamp",
                    "2026-09-20T12:34:00Z",
                    "--build-name",
                    "Windows x64 / MSVC / Release",
                    "--xmake-version",
                    "2.9.7",
                    "--configure",
                    "-p windows -a x64",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            with zipfile.ZipFile(archive) as bundle:
                names = set(bundle.namelist())
                self.assertEqual(
                    names,
                    {
                        "vision_simple-server.exe",
                        "vsrt_windows_x64_release.lib",
                        "onnxruntime.dll",
                        "config/server.yaml",
                        "assets/model.onnx",
                        "build-info.json",
                    },
                )
                manifest = json.loads(bundle.read("build-info.json"))

            self.assertEqual(manifest["platform"], "windows")
            self.assertEqual(manifest["arch"], "x64")
            self.assertEqual(manifest["variant"], "msvc-cpu")
            self.assertEqual(manifest["commit"], "0123456789abcdef")
            self.assertEqual(
                {entry["path"] for entry in manifest["files"]},
                names - {"build-info.json"},
            )

    def test_notes_include_commit_and_build_metadata(self):
        with tempfile.TemporaryDirectory(prefix="vision-simple-notes-") as temporary:
            root = Path(temporary)
            archives = [
                (
                    "vision_simple-linux-x86_64-20260920-1234-UTF8.zip",
                    {
                        "platform": "linux",
                        "arch": "x86_64",
                        "variant": "gcc14-cpu",
                        "xmake_version": "2.9.7",
                        "configure": "-p linux -a x86_64",
                        "files": [{"path": "vision_simple-server", "size": 6, "sha256": "a" * 64}],
                    },
                ),
                (
                    "vision_simple-windows-x64-20260920-1234-UTF8.zip",
                    {
                        "platform": "windows",
                        "arch": "x64",
                        "variant": "msvc-cpu",
                        "xmake_version": "2.9.7",
                        "configure": "-p windows -a x64",
                        "files": [{"path": "vision_simple-server.exe", "size": 6, "sha256": "b" * 64}],
                    },
                ),
            ]
            for name, manifest in archives:
                with zipfile.ZipFile(root / name, "w") as bundle:
                    bundle.writestr("build-info.json", json.dumps(manifest))

            commit_message = root / "commit-message.txt"
            commit_message.write_text("修复 <release>\\n第二行\\n", encoding="utf-8")
            notes = root / "release-notes.md"
            checksums = root / "SHA256SUMS.txt"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "notes",
                    "--assets-dir",
                    str(root),
                    "--commit-message",
                    str(commit_message),
                    "--output",
                    str(notes),
                    "--checksums",
                    str(checksums),
                    "--run-url",
                    "https://github.com/example/project/actions/runs/42",
                    "--run-number",
                    "42",
                    "--run-attempt",
                    "2",
                    "--timestamp",
                    "2026-09-20T12:34:00Z",
                    "--commit",
                    "0123456789abcdef",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            rendered = notes.read_text(encoding="utf-8")
            self.assertIn("修复 &lt;release&gt;", rendered)
            self.assertIn("0123456789abcdef", rendered)
            self.assertIn("gcc14-cpu", rendered)
            self.assertIn("msvc-cpu", rendered)
            self.assertIn("Run 42, attempt 2", rendered)
            checksum_text = checksums.read_text(encoding="utf-8")
            for name, _ in archives:
                self.assertIn(name, checksum_text)


if __name__ == "__main__":
    unittest.main()
