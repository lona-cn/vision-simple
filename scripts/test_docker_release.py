#!/usr/bin/env python3
"""Regression tests for registry mutation boundaries and stable/prerelease policy."""

import json
import unittest
from unittest.mock import patch

from docker_release import registry_digest, release_plan


SHA = "1234567890abcdef1234567890abcdef12345678"
IMAGE = "lonacn/vision_simple"


class DockerReleaseTest(unittest.TestCase):
    def test_only_stable_release_advances_latest(self):
        stable = release_plan("push", "refs/tags/v1.2.3", SHA, IMAGE)
        prerelease = release_plan("push", "refs/tags/v1.2.4-rc.1", SHA, IMAGE)
        self.assertEqual(stable["tags"], ["1.2.3-cpu-x86_64", f"sha-{SHA}-cpu-x86_64", "latest"])
        self.assertEqual(prerelease["tags"], ["1.2.4-rc.1-cpu-x86_64", f"sha-{SHA}-cpu-x86_64"])
        self.assertFalse(prerelease["latest"])

    def test_manual_dispatch_never_publishes_without_opt_in(self):
        for ref in ("refs/heads/main", "refs/tags/v1.2.3"):
            with self.subTest(ref=ref):
                plan = release_plan("workflow_dispatch", ref, SHA, IMAGE)
                self.assertFalse(plan["publish"])
                self.assertEqual(plan["tags"], [])
        opted_in = release_plan("workflow_dispatch", "refs/tags/v1.2.3-rc.1", SHA, IMAGE, True)
        self.assertTrue(opted_in["publish"])
        self.assertNotIn("latest", opted_in["tags"])

    def test_publish_rejects_branches_even_when_named_like_versions(self):
        for event in ("push", "workflow_dispatch"):
            with self.subTest(event=event), self.assertRaises(ValueError):
                release_plan(event, "refs/heads/v1.2.3", SHA, IMAGE, True)

    def test_publish_rejects_non_semver_and_unrepresentable_tags(self):
        for tag in ("v1.2", "v01.2.3", "v1.2.3-01", "v1.2.3-rc..1", "v1.2.3+build.1", "v1.2.3-" + "a" * 128):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                release_plan("push", "refs/tags/" + tag, SHA, IMAGE)

    def test_numeric_looking_alphanumeric_prerelease_is_valid(self):
        plan = release_plan("push", "refs/tags/v1.2.3-01a", SHA, IMAGE)
        self.assertEqual(plan["tags"][0], "1.2.3-01a-cpu-x86_64")
        self.assertNotIn("latest", plan["tags"])

    def test_registry_target_cannot_override_tag_or_registry(self):
        for image in (
            "example.com/org/image", "example.com/image", "localhost/image",
            "lonacn/vision_simple:latest", "lonacn/vision_simple@sha256:123",
            "lonacn/vision_simple\ninjected=true",
        ):
            with self.subTest(image=image), self.assertRaises(ValueError):
                release_plan("push", "refs/tags/v1.2.3", SHA, image)

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
