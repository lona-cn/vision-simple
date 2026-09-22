#!/usr/bin/env python3
"""Decide CPU release tags and publish the already smoke-tested local image."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


VERSION = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?")
REPOSITORY = re.compile(r"[a-z0-9]+(?:[_-][a-z0-9]+)*/[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def release_plan(event, ref, sha, image, manual_publish=False):
    if event not in {"push", "workflow_dispatch"}:
        raise ValueError("Only tag pushes and manual dispatches are supported")
    if not REPOSITORY.fullmatch(image) or len(image) > 255 or image.split("/", 1)[0] == "localhost":
        raise ValueError("DOCKERHUB_IMAGE must be an untagged Docker Hub namespace/repository")
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
    version_tag = version[1:] + "-cpu-x86_64"
    if len(version_tag) > 128:
        raise ValueError("Version exceeds Docker's 128-character tag limit")
    result["tags"] = [version_tag, f"sha-{sha}-cpu-x86_64"]
    result["latest"] = prerelease is None
    if result["latest"]:
        result["tags"].append("latest")
    return result


def docker(*args):
    return subprocess.run(["docker", *args], check=True, capture_output=True, text=True, timeout=600).stdout


def registry_digest(reference, local):
    descriptor = json.loads(docker("buildx", "imagetools", "inspect", reference, "--format", "{{json .Manifest}}"))
    digest = descriptor.get("digest", "")
    if not DIGEST.fullmatch(digest):
        raise ValueError(f"Registry returned no valid digest for {reference}")
    if local_descriptor := local.get("Descriptor"):
        # The containerd image store can expose a manifest/index ID, not the
        # config ID used by the classic store. Match the complete local content.
        expected = local_descriptor.get("digest", "")
        if not DIGEST.fullmatch(expected) or digest != expected:
            raise ValueError(f"Registry manifest does not match the tested local image: {reference}")
    else:
        # Classic image stores expose the config digest as Id.
        repository = reference.rsplit(":", 1)[0]
        manifest = json.loads(docker("buildx", "imagetools", "inspect", f"{repository}@{digest}", "--raw"))
        if manifest.get("config", {}).get("digest") != local["Id"]:
            raise ValueError(f"Registry image does not match the tested local image: {reference}")
    return digest


def publish(plan, local_image):
    if not plan["publish"]:
        raise ValueError("Refusing publication of a build-only run")
    local = json.loads(docker("image", "inspect", local_image))[0]
    image_id = local["Id"]
    if not DIGEST.fullmatch(image_id) or (local.get("Os"), local.get("Architecture")) != ("linux", "amd64"):
        raise ValueError("The tested local image must be linux/amd64 with a valid image ID")
    expected_digest = None
    references = []
    for tag in plan["tags"]:
        reference = f"{plan['image']}:{tag}"
        # Pin every tag to the inspected local ID; never rebuild or pull for publication.
        docker("image", "tag", image_id, reference)
        docker("image", "push", reference)
        digest = registry_digest(reference, local)
        if expected_digest is not None and digest != expected_digest:
            raise ValueError(f"Registry digest mismatch for {reference}")
        expected_digest = digest
        references.append(reference)
    # Recheck all mutable references after the final (latest, for stable releases) push.
    for reference in references:
        if registry_digest(reference, local) != expected_digest:
            raise ValueError(f"Registry digest changed during publication: {reference}")
    return {"tags": references, "digest": expected_digest, "pull": f"{plan['image']}@{expected_digest}", "image_id": image_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "publish"))
    parser.add_argument("--local-image", default="vision-simple:ci")
    args = parser.parse_args()
    manual = os.environ.get("MANUAL_PUBLISH", "false")
    if manual not in {"true", "false"}:
        raise ValueError("MANUAL_PUBLISH must be true or false")
    plan = release_plan(
        os.environ["GITHUB_EVENT_NAME"], os.environ["GITHUB_REF"],
        os.environ["GITHUB_SHA"], os.environ.get("DOCKERHUB_IMAGE") or "lonacn/vision_simple",
        manual == "true",
    )
    if args.command == "plan":
        if output := os.environ.get("GITHUB_OUTPUT"):
            with Path(output).open("a", encoding="utf-8") as stream:
                stream.write(f"publish={str(plan['publish']).lower()}\n")
        print(json.dumps(plan))
        return
    result = publish(plan, args.local_image)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        text = "## Verified Docker publication\n\n"
        text += "Built and HTTP smoke-tested `linux/amd64` image; all registry tags verified.\n\n"
        text += "\n".join(f"- `{tag}`" for tag in result["tags"])
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
