#!/usr/bin/env python3
"""Regression tests for GHCR release tags and image identity checks."""

import json
import unittest
from unittest.mock import patch

from docker_release import (
    IMAGE,
    manifest_source_refs,
    manifest_source_tag,
    platform_tags,
    registry_digest,
    release_plan,
    validate_manifest_entries,
)


SHA = "1234567890abcdef1234567890abcdef12345678"


class DockerReleaseTest(unittest.TestCase):
    def test_stable_and_prerelease_tags_share_multi_platform_names(self):
        stable = release_plan("push", "refs/tags/v1.2.3", SHA)
        prerelease = release_plan("push", "refs/tags/v1.2.4-rc.1", SHA)
        self.assertEqual(stable["image"], "ghcr.io/lona-cn/vision-simple")
        self.assertEqual(stable["tags"], ["1.2.3-cpu", f"sha-{SHA}-cpu", "latest"])
        self.assertEqual(platform_tags(stable, "amd64"), ["1.2.3-cpu-amd64", f"sha-{SHA}-cpu-amd64"])
        self.assertEqual(platform_tags(stable, "arm64"), ["1.2.3-cpu-arm64", f"sha-{SHA}-cpu-arm64"])
        self.assertEqual(prerelease["tags"], ["1.2.4-rc.1-cpu", f"sha-{SHA}-cpu"])
        self.assertFalse(prerelease["latest"])

    def test_manual_dispatch_never_publishes_without_opt_in(self):
        for ref in ("refs/heads/main", "refs/tags/v1.2.3"):
            with self.subTest(ref=ref):
                plan = release_plan("workflow_dispatch", ref, SHA)
                self.assertFalse(plan["publish"])
                self.assertEqual(plan["tags"], [])
        opted_in = release_plan("workflow_dispatch", "refs/tags/v1.2.3-rc.1", SHA, manual_publish=True)
        self.assertTrue(opted_in["publish"])
        self.assertNotIn("latest", opted_in["tags"])

    def test_publish_rejects_branches_even_when_named_like_versions(self):
        for event in ("push", "workflow_dispatch"):
            with self.subTest(event=event), self.assertRaises(ValueError):
                release_plan(event, "refs/heads/v1.2.3", SHA, manual_publish=True)

    def test_publish_rejects_non_semver_and_unrepresentable_tags(self):
        for tag in ("v1.2", "v01.2.3", "v1.2.3-01", "v1.2.3-rc..1", "v1.2.3+build.1", "v1.2.3-" + "a" * 128):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                release_plan("push", "refs/tags/" + tag, SHA)

    def test_numeric_looking_alphanumeric_prerelease_is_valid(self):
        plan = release_plan("push", "refs/tags/v1.2.3-01a", SHA)
        self.assertEqual(plan["tags"][0], "1.2.3-01a-cpu")
        self.assertNotIn("latest", plan["tags"])

    def test_registry_target_cannot_override_ghcr_image(self):
        for image in (
            "example.com/org/image", "ghcr.io/other/image", "ghcr.io/lona-cn/vision-simple:latest",
            "ghcr.io/lona-cn/vision-simple@sha256:123", "ghcr.io/Lona-CN/vision-simple",
            "ghcr.io/lona-cn/vision-simple\ninjected=true",
        ):
            with self.subTest(image=image), self.assertRaises(ValueError):
                release_plan("push", "refs/tags/v1.2.3", SHA, image)

    def test_platform_tags_reject_unpublished_architectures(self):
        plan = release_plan("push", "refs/tags/v1.2.3", SHA)
        with self.assertRaises(ValueError):
            platform_tags(plan, "riscv64")

    def test_latest_manifest_uses_stable_architecture_tags(self):
        plan = release_plan("push", "refs/tags/v1.2.3", SHA)
        self.assertEqual(manifest_source_tag(plan, "latest"), "1.2.3-cpu")
        self.assertEqual(
            manifest_source_refs(plan, "latest"),
            [
                "ghcr.io/lona-cn/vision-simple:1.2.3-cpu-amd64",
                "ghcr.io/lona-cn/vision-simple:1.2.3-cpu-arm64",
            ],
        )

    def test_manifest_validation_requires_exact_tested_platform_digests(self):
        amd64, arm64 = "sha256:" + "a" * 64, "sha256:" + "b" * 64
        digests = {"amd64": {"1.2.3-cpu": amd64}, "arm64": {"1.2.3-cpu": arm64}}
        entries = [
            {"digest": amd64, "platform": {"os": "linux", "architecture": "amd64"}},
            {"digest": arm64, "platform": {"os": "linux", "architecture": "arm64"}},
        ]
        self.assertEqual(validate_manifest_entries(entries, digests, "1.2.3-cpu"), ["linux/amd64", "linux/arm64"])
        with self.assertRaises(ValueError):
            validate_manifest_entries(entries[:1], digests, "1.2.3-cpu")
        entries[1]["digest"] = "sha256:" + "c" * 64
        with self.assertRaises(ValueError):
            validate_manifest_entries(entries, digests, "1.2.3-cpu")

    def test_containerd_store_matches_manifest_not_config_identity(self):
        manifest_id = "sha256:" + "a" * 64
        local = {"Id": manifest_id, "Descriptor": {"digest": manifest_id}}
        with patch("docker_release.docker", return_value=json.dumps({"digest": manifest_id})):
            self.assertEqual(registry_digest(IMAGE + ":latest", local), manifest_id)
            local["Descriptor"]["digest"] = "sha256:" + "b" * 64
            with self.assertRaises(ValueError):
                registry_digest(IMAGE + ":latest", local)

    def test_classic_store_matches_config_identity(self):
        manifest_id, config_id = "sha256:" + "a" * 64, "sha256:" + "b" * 64
        local = {"Id": config_id}

        def response(*args):
            return json.dumps(
                {"config": {"digest": config_id}} if args[-1] == "--raw"
                else {"digest": manifest_id}
            )

        with patch("docker_release.docker", side_effect=response):
            self.assertEqual(registry_digest(IMAGE + ":latest", local), manifest_id)
            local["Id"] = "sha256:" + "c" * 64
            with self.assertRaises(ValueError):
                registry_digest(IMAGE + ":latest", local)


if __name__ == "__main__":
    unittest.main()
