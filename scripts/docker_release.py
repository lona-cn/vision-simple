#!/usr/bin/env python3
"""Plan and publish verified multi-platform CPU container releases."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


IMAGE = "ghcr.io/lona-cn/vision-simple"
VERSION = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
ARCHITECTURES = {
    "amd64": ("linux", "amd64"),
    "arm64": ("linux", "arm64"),
}


def release_plan(event, ref, sha, image=IMAGE, manual_publish=False):
    if event not in {"push", "workflow_dispatch"}:
        raise ValueError("Only tag pushes and manual dispatches are supported")
    if image != IMAGE:
        raise ValueError(f"Publication target is fixed to {IMAGE}")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Source commit must be a full 40-character SHA")
    publish = event == "push" or manual_publish
    result = {"publish": publish, "image": image, "sha": sha, "tags": [], "latest": False}
    if not publish:
        return result
    if not ref.startswith("refs/tags/"):
        raise ValueError("Publishing requires a version tag, not a branch")
    version = ref.removeprefix("refs/tags/")
    match = VERSION.fullmatch(version)
    if not match:
        raise ValueError("Publishing requires vX.Y.Z or vX.Y.Z-prerelease (no build metadata)")
    prerelease = match.group(4)
    if prerelease and any(part.isdigit() and len(part) > 1 and part[0] == "0" for part in prerelease.split(".")):
        raise ValueError("Numeric prerelease identifiers cannot have leading zeroes")
    version_tag = version[1:] + "-cpu"
    result["tags"] = [version_tag, f"sha-{sha}-cpu"]
    result["latest"] = prerelease is None
    if result["latest"]:
        result["tags"].append("latest")
    if any(len(tag) > 128 for tag in result["tags"]):
        raise ValueError("Version exceeds Docker's 128-character tag limit")
    if any(len(tag + "-arm64") > 128 for tag in result["tags"] if tag != "latest"):
        raise ValueError("Version exceeds Docker's 128-character platform tag limit")
    return result


def platform_tags(plan, architecture):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unsupported published architecture: {architecture}")
    return [f"{tag}-{architecture}" for tag in plan["tags"] if tag != "latest"]


def docker(*args):
    return subprocess.run(["docker", *args], check=True, capture_output=True, text=True, timeout=600).stdout


def registry_digest(reference, local):
    descriptor = json.loads(docker("buildx", "imagetools", "inspect", reference, "--format", "{{json .Manifest}}"))
    digest = descriptor.get("digest", "")
    if not DIGEST.fullmatch(digest):
        raise ValueError(f"Registry returned no valid digest for {reference}")
    if local_descriptor := local.get("Descriptor"):
        expected = local_descriptor.get("digest", "")
        if not DIGEST.fullmatch(expected) or digest != expected:
            raise ValueError(f"Registry manifest does not match the tested local image: {reference}")
    else:
        repository = reference.rsplit(":", 1)[0]
        manifest = json.loads(docker("buildx", "imagetools", "inspect", f"{repository}@{digest}", "--raw"))
        if manifest.get("config", {}).get("digest") != local["Id"]:
            raise ValueError(f"Registry image does not match the tested local image: {reference}")
    return digest


def publish_platform(plan, local_image, architecture):
    if not plan["publish"]:
        raise ValueError("Refusing publication of a build-only run")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unsupported published architecture: {architecture}")
    local = json.loads(docker("image", "inspect", local_image))[0]
    if not DIGEST.fullmatch(local.get("Id", "")) or (local.get("Os"), local.get("Architecture")) != ARCHITECTURES[architecture]:
        raise ValueError(f"The tested local image must be {ARCHITECTURES[architecture][0]}/{ARCHITECTURES[architecture][1]}")
    references = []
    expected_digest = None
    for tag in platform_tags(plan, architecture):
        reference = f"{plan['image']}:{tag}"
        docker("image", "tag", local["Id"], reference)
        docker("image", "push", reference)
        digest = registry_digest(reference, local)
        if expected_digest is not None and digest != expected_digest:
            raise ValueError(f"Registry digest mismatch for {reference}")
        expected_digest = digest
        references.append(reference)
    for reference in references:
        if registry_digest(reference, local) != expected_digest:
            raise ValueError(f"Registry digest changed during publication: {reference}")
    return {"architecture": architecture, "tags": references, "digest": expected_digest, "image_id": local["Id"]}


def registry_manifest(reference):
    manifest = json.loads(docker("buildx", "imagetools", "inspect", reference, "--raw"))
    entries = manifest.get("manifests")
    if not isinstance(entries, list):
        raise ValueError(f"Registry tag is not a multi-platform image index: {reference}")
    return manifest, entries


def manifest_source_tag(plan, tag):
    return plan["tags"][0] if tag == "latest" else tag


def manifest_source_refs(plan, tag):
    base_tag = manifest_source_tag(plan, tag)
    return [f"{plan['image']}:{base_tag}-{architecture}" for architecture in ARCHITECTURES]


def validate_manifest_entries(entries, digests, source_tag):
    platforms = {
        (entry.get("platform", {}).get("os"), entry.get("platform", {}).get("architecture"))
        for entry in entries
    }
    expected = set(ARCHITECTURES.values())
    if platforms != expected or len(entries) != len(expected):
        raise ValueError(f"Unexpected platforms in manifest: {sorted(platforms)}")
    for entry in entries:
        architecture = entry["platform"]["architecture"]
        if entry.get("digest") != digests[architecture][source_tag]:
            raise ValueError(f"Manifest entry does not match the tested {architecture} image")
    return sorted(f"{os}/{architecture}" for os, architecture in platforms)


def publish_manifest(plan):
    if not plan["publish"]:
        raise ValueError("Refusing publication of a build-only run")
    sources = {
        architecture: [
            f"{plan['image']}:{tag}" for tag in platform_tags(plan, architecture)
        ]
        for architecture in ARCHITECTURES
    }
    digests = {}
    for architecture, references in sources.items():
        digests[architecture] = {}
        for reference in references:
            descriptor = json.loads(docker("buildx", "imagetools", "inspect", reference, "--format", "{{json .Manifest}}"))
            digest = descriptor.get("digest", "")
            if not DIGEST.fullmatch(digest):
                raise ValueError(f"Registry returned no valid digest for {reference}")
            base_tag = reference.rsplit(":", 1)[-1].rsplit("-", 1)[0]
            digests[architecture][base_tag] = digest

    results = []
    expected_digest = None
    for tag in plan["tags"]:
        target = f"{plan['image']}:{tag}"
        source_tag_base = manifest_source_tag(plan, tag)
        docker("buildx", "imagetools", "create", "--tag", target, *manifest_source_refs(plan, tag))
        descriptor = json.loads(docker("buildx", "imagetools", "inspect", target, "--format", "{{json .Manifest}}"))
        digest = descriptor.get("digest", "")
        if not DIGEST.fullmatch(digest):
            raise ValueError(f"Registry returned no valid multi-platform digest for {target}")
        if expected_digest is not None and digest != expected_digest:
            raise ValueError(f"Multi-platform tags do not resolve to the same image: {target}")
        expected_digest = digest
        _, entries = registry_manifest(target)
        platforms = validate_manifest_entries(entries, digests, source_tag_base)
        results.append({"reference": target, "digest": digest, "platforms": platforms})
    return {"tags": results, "digest": expected_digest, "pull": f"{plan['image']}@{expected_digest}"}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "publish-platform", "publish-manifest"))
    parser.add_argument("--local-image", default="vision-simple:ci")
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURES))
    args = parser.parse_args()
    manual = os.environ.get("MANUAL_PUBLISH", "false")
    if manual not in {"true", "false"}:
        raise ValueError("MANUAL_PUBLISH must be true or false")
    plan = release_plan(
        os.environ["GITHUB_EVENT_NAME"], os.environ["GITHUB_REF"],
        os.environ["GITHUB_SHA"], os.environ.get("GHCR_IMAGE") or IMAGE,
        manual == "true",
    )
    if args.command == "plan":
        if output := os.environ.get("GITHUB_OUTPUT"):
            with Path(output).open("a", encoding="utf-8") as stream:
                stream.write(f"publish={str(plan['publish']).lower()}\n")
        print(json.dumps(plan))
        return
    if args.command == "publish-platform":
        if not args.architecture:
            raise ValueError("publish-platform requires --architecture")
        result = publish_platform(plan, args.local_image, args.architecture)
    else:
        result = publish_manifest(plan)
        if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
            text = "## Verified multi-platform Docker publication\n\n"
            text += "Both CPU images passed HTTP smoke tests; every manifest tag contains `linux/amd64` and `linux/arm64`.\n\n"
            text += "\n".join(f"- `{tag['reference']}` (`{tag['digest']}`)" for tag in result["tags"])
            text += f"\n\nImmutable pull reference: `{result['pull']}`\n"
            with Path(summary).open("a", encoding="utf-8") as stream:
                stream.write(text)
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(f"Docker release failed: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr.rstrip(), file=sys.stderr)
        sys.exit(1)
