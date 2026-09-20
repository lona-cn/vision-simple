#!/usr/bin/env python3
"""Build release archives and release notes from verified build outputs."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import sys
import zipfile


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_shared_library(path):
    name = path.name.lower()
    return path.suffix.lower() in {".dll", ".dylib", ".so"} or ".so." in name


def collect_release_files(server, runtime):
    target_dir = server.parent
    selected = {server.resolve(): server.name, runtime.resolve(): runtime.name}

    for path in target_dir.iterdir():
        if path.is_file() and is_shared_library(path):
            selected[path.resolve()] = path.name

    for suffix in (".lib", ".exp", ".pdb"):
        companion = server.with_suffix(suffix)
        if companion.is_file():
            selected[companion.resolve()] = companion.name

    for directory_name in ("config", "assets"):
        directory = target_dir / directory_name
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.name != ".gitkeep":
                selected[path.resolve()] = path.relative_to(target_dir).as_posix()

    return sorted(selected.items(), key=lambda item: item[1])


def pack(args):
    server = Path(args.server).resolve()
    runtime = Path(args.runtime).resolve()
    output = Path(args.output).resolve()
    for label, path in (("server", server), ("runtime", runtime)):
        if not path.is_file():
            raise ValueError(f"Required {label} artifact does not exist: {path}")

    release_files = collect_release_files(server, runtime)
    manifest_files = [
        {
            "path": archive_path,
            "size": source.stat().st_size,
            "sha256": file_digest(source),
        }
        for source, archive_path in release_files
    ]
    manifest = {
        "format_version": 1,
        "platform": args.platform,
        "arch": args.arch,
        "variant": args.variant,
        "commit": args.commit,
        "timestamp": args.timestamp,
        "build_name": args.build_name,
        "xmake_version": args.xmake_version,
        "configure": args.configure,
        "files": manifest_files,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for source, archive_path in release_files:
            bundle.write(source, archive_path)
        bundle.writestr(
            "build-info.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8") + b"\n",
        )
    print(output)


def notes(args):
    assets_dir = Path(args.assets_dir).resolve()
    archives = sorted(assets_dir.glob("vision_simple-*-UTF8.zip"))
    if not archives:
        raise ValueError(f"No release archives found in {assets_dir}")
    if args.expected_count is not None and len(archives) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} release archives, found {len(archives)}"
        )

    builds = []
    checksum_lines = []
    for archive in archives:
        with zipfile.ZipFile(archive) as bundle:
            try:
                manifest = json.loads(bundle.read("build-info.json"))
            except KeyError as error:
                raise ValueError(f"{archive.name} has no build-info.json") from error
        for field in ("platform", "arch", "variant", "xmake_version", "configure", "files"):
            if field not in manifest:
                raise ValueError(f"{archive.name} manifest is missing {field}")
        digest = file_digest(archive)
        checksum_lines.append(f"{digest}  {archive.name}")
        builds.append((archive, manifest, digest))

    commit_message = Path(args.commit_message).read_text(encoding="utf-8").strip()
    lines = [
        "# vision-simple automated release",
        "",
        f"- Commit: `{args.commit}`",
        f"- Build timestamp (UTC): `{args.timestamp}`",
        f"- Workflow: [Run {args.run_number}, attempt {args.run_attempt}]({args.run_url})",
        "",
        "## Commit message",
        "",
        f"<pre>{html.escape(commit_message)}</pre>",
        "",
        "## Build artifacts",
        "",
        "| Asset | Platform | Architecture | Variant | Xmake | Files | SHA-256 |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for archive, manifest, digest in builds:
        lines.append(
            f"| `{archive.name}` | {manifest['platform']} | {manifest['arch']} | "
            f"{manifest['variant']} | {manifest['xmake_version']} | "
            f"{len(manifest['files'])} | `{digest}` |"
        )

    lines.extend(["", "## Build configurations", ""])
    for archive, manifest, _ in builds:
        lines.extend(
            [
                f"<details><summary>{html.escape(archive.name)}</summary>",
                "",
                f"<pre>{html.escape(manifest['configure'])}</pre>",
                "",
                "</details>",
                "",
            ]
        )

    output = Path(args.output)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    checksums = Path(args.checksums)
    checksums.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(output.resolve())


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    pack_parser = commands.add_parser("pack", help="Create one architecture release archive")
    for option in (
        "server",
        "runtime",
        "output",
        "platform",
        "arch",
        "variant",
        "commit",
        "timestamp",
        "build-name",
        "xmake-version",
        "configure",
    ):
        pack_parser.add_argument(f"--{option}", required=True)
    pack_parser.set_defaults(handler=pack)
    notes_parser = commands.add_parser("notes", help="Create release notes and checksums")
    for option in (
        "assets-dir",
        "commit-message",
        "output",
        "checksums",
        "run-url",
        "run-number",
        "run-attempt",
        "timestamp",
        "commit",
    ):
        notes_parser.add_argument(f"--{option}", required=True)
    notes_parser.add_argument("--expected-count", type=int)
    notes_parser.set_defaults(handler=notes)
    return root


def main():
    args = parser().parse_args()
    try:
        args.handler(args)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"release_artifacts: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
