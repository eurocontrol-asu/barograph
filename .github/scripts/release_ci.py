"""Resolve release packages and verify the distributions selected for upload."""

from __future__ import annotations

import argparse
import json
import os
import re
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path

_NAME = r"[A-Za-z0-9][A-Za-z0-9_-]*"
_VERSION = (
    r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?"
)


def detect(root: Path, tag: str, package: str) -> dict[str, str]:
    """Read package policy from TOML, resolving a tag or a manual package name."""
    version = ""
    if tag:
        match = re.fullmatch(rf"({_NAME})/v({_VERSION})", tag)
        if match is None:
            raise ValueError("Expected a package/vX.Y.Z tag (optionally pre/dev/post release)")
        tagged_package, version = match.groups()
        if package and package != tagged_package:
            raise ValueError("Manual package does not match the tag prefix")
        package = tagged_package
    if not re.fullmatch(_NAME, package):
        raise ValueError("Expected a package directory name without path separators")
    directory = root / "packages" / package
    directory.resolve().relative_to((root / "packages").resolve())
    project = tomllib.loads((directory / "pyproject.toml").read_text())["project"]
    if project["name"] != package:
        raise ValueError("Package directory and distribution name must match")
    private = any(c.startswith("Private ::") for c in project.get("classifiers", []))
    return {
        "package": package,
        "package-dir": f"packages/{package}",
        "publishable": str(not private).lower(),
        "version": version,
        "prerelease": str(bool(re.search(r"(?:a|b|rc)\d+|\.dev\d+", version))).lower(),
    }


def _metadata(path: Path) -> bytes:
    """Read distribution metadata without extracting archive paths."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as wheel:
            names = [n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                raise ValueError("Expected exactly one wheel METADATA")
            return wheel.read(names[0])
    with tarfile.open(path) as sdist:
        names = [
            m for m in sdist.getmembers() if m.name.count("/") == 1 and m.name.endswith("/PKG-INFO")
        ]
        if len(names) != 1:
            raise ValueError("Expected exactly one top-level sdist PKG-INFO")
        stream = sdist.extractfile(names[0])
        if stream is None:
            raise ValueError("Missing sdist PKG-INFO content")
        return stream.read()


def check_dist(directory: Path, package: str, version: str) -> dict[str, str]:
    """Require one matching wheel and sdist, with the tagged version on releases."""
    wheels, sdists = list(directory.glob("*.whl")), list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution")
    metadata = [BytesParser().parsebytes(_metadata(path)) for path in [*wheels, *sdists]]
    for item in metadata:
        if item["Name"] != package:
            raise ValueError("Built distribution name does not match requested package")
        if not item["Version"] or (version and item["Version"] != version):
            raise ValueError("Built distribution version does not match the release tag")
    if metadata[0]["Version"] != metadata[1]["Version"]:
        raise ValueError("Wheel and sdist versions differ")
    return {"package": package, "version": metadata[0]["Version"]}


def main() -> None:
    """Run release validation and write detector outputs for GitHub Actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    detection = commands.add_parser("detect")
    detection.add_argument("--tag", default="")
    detection.add_argument("--package", default="")
    check = commands.add_parser("check-dist")
    check.add_argument("--package", required=True)
    check.add_argument("--version", default="")
    check.add_argument("--directory", type=Path, default=Path("dist"))
    args = parser.parse_args()
    if args.command == "detect":
        result = detect(Path.cwd(), args.tag, args.package)
        if output := os.getenv("GITHUB_OUTPUT"):
            with Path(output).open("a") as stream:
                stream.writelines(f"{key}={value}\n" for key, value in result.items())
    else:
        result = check_dist(args.directory, args.package, args.version)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
