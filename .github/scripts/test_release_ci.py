"""Exercise package selection and distribution validation before release tags."""

from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
import tempfile
import textwrap
import unittest
import zipfile
from pathlib import Path

from release_ci import check_dist, detect


class ReleaseCITest(unittest.TestCase):
    """Use disposable metadata and archives, without creating Git tags."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = self.root / "packages" / "barograph-met"
        self.package.mkdir(parents=True)
        self.metadata = self.package / "pyproject.toml"
        self.metadata.write_text('[project]\nname = "barograph-met"\n')
        self.dist = self.root / "dist"
        self.dist.mkdir()

    def distributions(self, wheel_version: str, sdist_version: str | None = None) -> None:
        """Create tiny archives containing independently versioned metadata."""
        with zipfile.ZipFile(self.dist / "package-py3-none-any.whl", "w") as wheel:
            wheel.writestr(
                "package.dist-info/METADATA",
                f"Metadata-Version: 2.4\nName: barograph-met\nVersion: {wheel_version}\n",
            )
        content = (
            f"Metadata-Version: 2.4\nName: barograph-met\n"
            f"Version: {sdist_version or wheel_version}\n"
        ).encode()
        with tarfile.open(self.dist / "package.tar.gz", "w:gz") as sdist:
            member = tarfile.TarInfo("package/PKG-INFO")
            member.size = len(content)
            sdist.addfile(member, io.BytesIO(content))

    def test_release_tag_resolves_package_and_version(self) -> None:
        result = detect(self.root, "barograph-met/v0.1.0", "")
        self.assertEqual(result["package-dir"], "packages/barograph-met")
        self.assertEqual(result["version"], "0.1.0")
        self.assertEqual(result["publishable"], "true")

    def test_prerelease_and_development_tags_are_supported(self) -> None:
        for version in ("0.1.0rc1", "1.2.3.dev4", "1.2.3.post1"):
            with self.subTest(version=version):
                result = detect(self.root, f"barograph-met/v{version}", "")
                self.assertEqual(result["version"], version)

    def test_manual_build_can_resolve_a_private_package_without_a_tag(self) -> None:
        self.metadata.write_text(
            self.metadata.read_text() + 'classifiers = ["Private :: Do Not Upload"]\n'
        )
        result = detect(self.root, "", "barograph-met")
        self.assertEqual(result["version"], "")
        self.assertEqual(result["publishable"], "false")

    def test_classifier_in_a_comment_does_not_block_publication(self) -> None:
        self.metadata.write_text(
            self.metadata.read_text() + "# Formerly Private :: Do Not Upload\n"
        )
        self.assertEqual(detect(self.root, "", "barograph-met")["publishable"], "true")

    def test_unrecognized_private_classifier_still_blocks_publication(self) -> None:
        self.metadata.write_text(
            self.metadata.read_text() + 'classifiers = ["Private :: Internal"]\n'
        )
        self.assertEqual(detect(self.root, "", "barograph-met")["publishable"], "false")

    def test_invalid_tags_cannot_select_a_package_or_inject_outputs(self) -> None:
        for tag in (
            "refs/heads/main",
            "../v0.1.0",
            "barograph-met/vbad",
            "barograph-met/v0.1.0\nx=y",
        ):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                detect(self.root, tag, "")

    def test_manual_package_cannot_escape_the_workspace(self) -> None:
        for package in ("../barograph-met", "/absolute", "barograph-met\nx=y", ""):
            with self.subTest(package=package), self.assertRaises(ValueError):
                detect(self.root, "", package)

    def test_package_name_must_match_tag_and_metadata(self) -> None:
        with self.assertRaises(ValueError):
            detect(self.root, "barograph-met/v0.1.0", "other-package")
        self.metadata.write_text('[project]\nname = "other-package"\n')
        with self.assertRaises(ValueError):
            detect(self.root, "barograph-met/v0.1.0", "")

    def test_correct_distributions_match_the_tag(self) -> None:
        self.distributions("0.1.0")
        self.assertEqual(check_dist(self.dist, "barograph-met", "0.1.0")["version"], "0.1.0")

    def test_untagged_build_accepts_matching_development_versions(self) -> None:
        self.distributions("0.1.0.dev63+g123abc")
        self.assertEqual(
            check_dist(self.dist, "barograph-met", "")["version"], "0.1.0.dev63+g123abc"
        )

    def test_wrong_version_or_package_rejects_the_release(self) -> None:
        self.distributions("0.1.1")
        with self.assertRaises(ValueError):
            check_dist(self.dist, "barograph-met", "0.1.0")
        with self.assertRaises(ValueError):
            check_dist(self.dist, "other-package", "0.1.1")

    def test_wheel_and_sdist_must_have_the_same_version(self) -> None:
        self.distributions("0.1.0", "0.1.1")
        with self.assertRaises(ValueError):
            check_dist(self.dist, "barograph-met", "")

    def test_missing_or_stale_distributions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            check_dist(self.dist, "barograph-met", "")
        self.distributions("0.1.0")
        (self.dist / "old.whl").touch()
        with self.assertRaises(ValueError):
            check_dist(self.dist, "barograph-met", "")


class CompositeActionTest(unittest.TestCase):
    """Execute the actual action's shell with manual and tag event contexts."""

    def action(self, ref: str, package: str = "", tag: str = "") -> dict[str, str]:
        root = Path(__file__).resolve().parents[2]
        action_file = root / ".github/actions/detect-package/action.yml"
        shell = textwrap.dedent(action_file.read_text().split("      run: |\n", 1)[1])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            process = subprocess.run(
                ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", shell],
                cwd=root,
                env={
                    **os.environ,
                    "GITHUB_REF": ref,
                    "INPUT_TAG": tag,
                    "INPUT_PACKAGE": package,
                    "GITHUB_OUTPUT": str(output),
                },
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(process.stdout)
            self.assertIn(f"package={result['package']}\n", output.read_text())
            return result

    def test_manual_branch_build_does_not_treat_the_branch_as_a_package(self) -> None:
        result = self.action("refs/heads/main", package="barograph-met")
        self.assertEqual(result["package"], "barograph-met")
        self.assertEqual(result["version"], "")

    def test_tag_push_derives_the_package_and_version(self) -> None:
        result = self.action("refs/tags/barograph-met/v0.1.0")
        self.assertEqual(result["package"], "barograph-met")
        self.assertEqual(result["version"], "0.1.0")

    def test_manual_package_build_ignores_the_workflow_ref_tag(self) -> None:
        result = self.action("refs/tags/other-package/v9.0.0", package="barograph-met")
        self.assertEqual(result["package"], "barograph-met")
        self.assertEqual(result["version"], "")

    def test_explicit_tag_input_overrides_a_branch_ref(self) -> None:
        result = self.action("refs/heads/main", tag="barograph-met/v0.1.0rc1")
        self.assertEqual(result["version"], "0.1.0rc1")


if __name__ == "__main__":
    unittest.main()
