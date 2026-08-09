#!/usr/bin/env python3
"""Fail-closed operator helper for publishing a tagged tenferro release."""

from __future__ import annotations

import argparse
import io
import json
import re
import runpy
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NamedTuple


ROOT = Path(__file__).resolve().parents[1]
CRATES_IO_API = "https://crates.io/api/v1/crates"
EXACT_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
EXACT_REV = re.compile(r"^[0-9a-fA-F]{40}$")


class ReleaseError(RuntimeError):
    """A release invariant failed."""


class GitDependency(NamedTuple):
    name: str
    package: str
    version: str
    git: str
    rev: str


class ArchiveInspection(NamedTuple):
    metadata: dict
    files: tuple[str, ...]


def run(
    command: list[str], *, cwd: Path = ROOT, capture: bool = True
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        raise ReleaseError(f"{' '.join(command)} failed: {detail}") from error


def parse_workspace_git_dependencies(manifest_text: str) -> list[GitDependency]:
    try:
        manifest = tomllib.loads(manifest_text)
    except tomllib.TOMLDecodeError as error:
        raise ReleaseError(f"Cargo.toml is invalid TOML: {error}") from error
    dependencies = manifest.get("workspace", {}).get("dependencies", {})
    if not isinstance(dependencies, dict):
        raise ReleaseError("[workspace.dependencies] must be a table")

    result: list[GitDependency] = []
    for name, value in dependencies.items():
        if not isinstance(value, dict) or "git" not in value:
            continue
        package = value.get("package", name)
        version = value.get("version")
        git = value.get("git")
        rev = value.get("rev")
        if not all(isinstance(item, str) and item for item in (package, version, git, rev)):
            raise ReleaseError(
                f"workspace git dependency {name!r} must declare package identity, "
                "an exact registry version, git, and rev"
            )
        registry_version = version.removeprefix("=")
        if EXACT_VERSION.fullmatch(registry_version) is None:
            raise ReleaseError(
                f"workspace git dependency {name!r} version must be exact, found {version!r}"
            )
        if EXACT_REV.fullmatch(rev) is None:
            raise ReleaseError(
                f"workspace git dependency {name!r} rev must be a 40-digit commit, "
                f"found {rev!r}"
            )
        result.append(
            GitDependency(name, package, registry_version, git, rev.lower())
        )
    return result


def validate_revision_manifest(
    dependency: GitDependency, manifests: Mapping[str, str]
) -> None:
    parsed: dict[str, dict] = {}
    workspace_versions: list[str] = []
    for path, text in manifests.items():
        try:
            manifest = tomllib.loads(text)
        except tomllib.TOMLDecodeError as error:
            raise ReleaseError(
                f"{dependency.git}@{dependency.rev}:{path} is invalid TOML: {error}"
            ) from error
        parsed[path] = manifest
        workspace_version = manifest.get("workspace", {}).get("package", {}).get("version")
        if isinstance(workspace_version, str):
            workspace_versions.append(workspace_version)

    matches: list[tuple[str, object]] = []
    for path, manifest in parsed.items():
        package = manifest.get("package", {})
        if package.get("name") == dependency.package:
            matches.append((path, package.get("version")))
    if len(matches) != 1:
        raise ReleaseError(
            f"{dependency.git}@{dependency.rev} must contain exactly one package "
            f"named {dependency.package!r}, found {len(matches)}"
        )

    path, declared = matches[0]
    if declared == {"workspace": True}:
        versions = set(workspace_versions)
        if len(versions) != 1:
            raise ReleaseError(
                f"{dependency.git}@{dependency.rev}:{path} inherits an ambiguous "
                "workspace package version"
            )
        declared = versions.pop()
    if declared != dependency.version:
        raise ReleaseError(
            f"{dependency.git}@{dependency.rev}:{path} declares version {declared}, "
            f"expected registry version {dependency.version}"
        )


def fetch_revision_manifests(dependency: GitDependency) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="tenferro-release-git-") as directory:
        checkout = Path(directory)
        run(["git", "init", "--quiet"], cwd=checkout)
        run(
            ["git", "fetch", "--quiet", "--depth=1", dependency.git, dependency.rev],
            cwd=checkout,
        )
        paths = run(
            ["git", "ls-tree", "-r", "--name-only", "FETCH_HEAD"], cwd=checkout
        ).stdout.splitlines()
        manifests: dict[str, str] = {}
        for path in paths:
            if Path(path).name == "Cargo.toml":
                manifests[path] = run(
                    ["git", "show", f"FETCH_HEAD:{path}"], cwd=checkout
                ).stdout
        return manifests


Transport = Callable[[str], tuple[int, bytes]]


def crates_io_transport(url: str) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "tenferro-release-publish/0.4 (+https://github.com/tensor4all/tenferro-rs)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
    except OSError as error:
        raise ReleaseError(f"crates.io request failed for {url}: {error}") from error


class CratesIoClient:
    def __init__(self, transport: Transport = crates_io_transport):
        self.transport = transport

    @staticmethod
    def _url(*parts: str) -> str:
        return "/".join(
            [CRATES_IO_API, *(urllib.parse.quote(part, safe="") for part in parts)]
        )

    def _exists(self, url: str) -> bool:
        status, _ = self.transport(url)
        if status == 200:
            return True
        if status == 404:
            return False
        raise ReleaseError(f"crates.io query returned HTTP {status}: {url}")

    def package_exists(self, package: str) -> bool:
        return self._exists(self._url(package))

    def version_exists(self, package: str, version: str) -> bool:
        return self._exists(self._url(package, version))

    def download(self, package: str, version: str) -> bytes:
        url = self._url(package, version, "download")
        status, body = self.transport(url)
        if status != 200:
            raise ReleaseError(
                f"crates.io archive download returned HTTP {status}: {package} {version}"
            )
        return body


def validate_new_package_approvals(
    package_exists: Mapping[str, bool], approvals: set[str]
) -> None:
    new_packages = {package for package, exists in package_exists.items() if not exists}
    missing = sorted(new_packages - approvals)
    if missing:
        raise ReleaseError(
            "new crates.io packages require exact user approval before any publication: "
            f"{missing}; rerun with one --approve-new-package PACKAGE per approved package"
        )
    stale = sorted(approvals - new_packages)
    if stale:
        raise ReleaseError(f"approval named packages that are not new: {stale}")


def _read_archive_member(archive: tarfile.TarFile, name: str) -> bytes:
    member = archive.extractfile(name)
    if member is None:
        raise ReleaseError(f"crate archive member is not a regular file: {name}")
    return member.read()


def inspect_crate_archive(
    archive_data: bytes, expected_name: str, expected_version: str, expected_commit: str
) -> ArchiveInspection:
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_data), mode="r:gz") as archive:
            files = tuple(sorted(member.name for member in archive.getmembers() if member.isfile()))
            prefix = f"{expected_name}-{expected_version}/"
            manifest_name = prefix + "Cargo.toml"
            vcs_name = prefix + ".cargo_vcs_info.json"
            if manifest_name not in files or vcs_name not in files:
                raise ReleaseError(
                    f"crate archive must contain {manifest_name} and {vcs_name}"
                )
            manifest = tomllib.loads(
                _read_archive_member(archive, manifest_name).decode("utf-8")
            )
            vcs = json.loads(_read_archive_member(archive, vcs_name))
    except (tarfile.TarError, UnicodeDecodeError, tomllib.TOMLDecodeError, json.JSONDecodeError) as error:
        raise ReleaseError(f"invalid .crate archive: {error}") from error

    package = manifest.get("package", {})
    required_strings = (
        "name",
        "version",
        "description",
        "license",
        "repository",
        "homepage",
        "documentation",
        "readme",
        "rust-version",
    )
    for key in required_strings:
        if not isinstance(package.get(key), str) or not package[key].strip():
            raise ReleaseError(f"packaged Cargo.toml package.{key} must be a non-empty string")
    if package["name"] != expected_name or package["version"] != expected_version:
        raise ReleaseError(
            f"crate archive identity is {package['name']} {package['version']}, "
            f"expected {expected_name} {expected_version}"
        )
    for key in ("keywords", "categories"):
        values = package.get(key)
        if not isinstance(values, list) or not values or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise ReleaseError(f"packaged Cargo.toml package.{key} must be non-empty strings")
    for key in ("include", "exclude"):
        values = package.get(key)
        if values is not None and (
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise ReleaseError(f"packaged Cargo.toml package.{key} must be a string list")
    readme = prefix + package["readme"].lstrip("./")
    if readme not in files:
        raise ReleaseError(f"crate archive is missing declared README {readme}")

    git = vcs.get("git", {})
    if git.get("sha1") != expected_commit or git.get("dirty", False) is not False:
        raise ReleaseError(
            "crate archive provenance does not equal the clean tagged commit: "
            f"{git!r}, expected sha1 {expected_commit} and dirty false"
        )
    return ArchiveInspection(package, files)


def print_inspection(crate: str, inspection: ArchiveInspection) -> None:
    keys = (
        "name",
        "version",
        "description",
        "license",
        "repository",
        "homepage",
        "documentation",
        "readme",
        "rust-version",
        "keywords",
        "categories",
        "include",
        "exclude",
    )
    print(f"{crate}: packaged metadata")
    print(json.dumps({key: inspection.metadata.get(key) for key in keys}, indent=2))
    print(f"{crate}: archive files ({len(inspection.files)})")
    for path in inspection.files:
        print(f"  {path}")


def verify_release_checkout(version: str) -> str:
    if EXACT_VERSION.fullmatch(version) is None:
        raise ReleaseError(f"release version must be exact, found {version!r}")
    run(["git", "fetch", "origin", "main"], capture=False)
    if run(["git", "status", "--porcelain"]).stdout:
        raise ReleaseError("release worktree must be clean, including untracked files")
    symbolic = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"], cwd=ROOT, stdout=subprocess.PIPE
    )
    if symbolic.returncode == 0:
        raise ReleaseError("release worktree must be detached at the tag")

    tag = f"v{version}"
    lines = run(
        ["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"]
    ).stdout.splitlines()
    refs = {line.split()[1]: line.split()[0] for line in lines if len(line.split()) == 2}
    remote_commit = refs.get(f"refs/tags/{tag}^{{}}") or refs.get(f"refs/tags/{tag}")
    if remote_commit is None:
        raise ReleaseError(f"origin does not have tag {tag}")
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    if head != remote_commit:
        raise ReleaseError(
            f"HEAD {head} is not exact pushed remote tag {tag} commit {remote_commit}"
        )
    run(["git", "merge-base", "--is-ancestor", head, "origin/main"])

    root_manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    declared = root_manifest.get("workspace", {}).get("package", {}).get("version")
    if declared != version:
        raise ReleaseError(
            f"tag version {version} does not match workspace package version {declared!r}"
        )
    return head


def cargo_metadata() -> dict:
    return json.loads(
        run(["cargo", "metadata", "--no-deps", "--format-version", "1"]).stdout
    )


def publication_order(release_text: str) -> list[str]:
    checker = runpy.run_path(str(ROOT / "scripts/check-publish-layout.py"))
    errors: list[str] = []
    order = checker["phase3_publication_order"](release_text, errors)
    if order is None or errors:
        raise ReleaseError("; ".join(errors))
    return order


def inspect_registry_archive(
    client: CratesIoClient, crate: str, version: str, commit: str
) -> ArchiveInspection:
    return inspect_crate_archive(client.download(crate, version), crate, version, commit)


def await_registry_archive(
    client: CratesIoClient,
    crate: str,
    version: str,
    commit: str,
    *,
    attempts: int = 40,
    delay: float = 30.0,
) -> ArchiveInspection:
    for attempt in range(attempts):
        if client.version_exists(crate, version):
            try:
                return inspect_registry_archive(client, crate, version, commit)
            except ReleaseError:
                if attempt + 1 == attempts:
                    raise
        if attempt + 1 < attempts:
            time.sleep(delay)
    raise ReleaseError(
        f"{crate} {version} did not become visible with matching provenance on crates.io"
    )


def archive_path(metadata: dict, crate: str, version: str) -> Path:
    return Path(metadata["target_directory"]) / "package" / f"{crate}-{version}.crate"


def build_and_inspect_archive(
    metadata: dict, crate: str, version: str, commit: str, command: list[str]
) -> ArchiveInspection:
    path = archive_path(metadata, crate, version)
    path.unlink(missing_ok=True)
    run(command, capture=False)
    if not path.is_file():
        raise ReleaseError(f"Cargo did not create expected archive {path}")
    inspection = inspect_crate_archive(path.read_bytes(), crate, version, commit)
    print_inspection(crate, inspection)
    return inspection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="exact release version without the v prefix")
    parser.add_argument(
        "--approve-new-package",
        action="append",
        default=[],
        metavar="PACKAGE",
        help="assert the user explicitly approved this exact new crates.io package",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform irreversible cargo publish commands after all preflights",
    )
    args = parser.parse_args(argv)

    try:
        commit = verify_release_checkout(args.version)
        run([sys.executable, "scripts/check-publish-layout.py"], capture=False)
        root_text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
        git_dependencies = parse_workspace_git_dependencies(root_text)
        client = CratesIoClient()
        revision_cache: dict[tuple[str, str], dict[str, str]] = {}
        for dependency in git_dependencies:
            cache_key = (dependency.git, dependency.rev)
            if cache_key not in revision_cache:
                revision_cache[cache_key] = fetch_revision_manifests(dependency)
            validate_revision_manifest(dependency, revision_cache[cache_key])
            if not client.version_exists(dependency.package, dependency.version):
                raise ReleaseError(
                    f"workspace git dependency {dependency.name!r} declares missing "
                    f"crates.io package/version {dependency.package} {dependency.version}"
                )
            print(
                f"git dependency verified: {dependency.name} -> {dependency.package} "
                f"{dependency.version} at {dependency.rev}"
            )

        metadata = cargo_metadata()
        packages = {package["name"]: package for package in metadata["packages"]}
        release_text = (ROOT / "ai/contribution-workflows/release-publish.md").read_text(
            encoding="utf-8"
        )
        order = publication_order(release_text)
        package_exists = {crate: client.package_exists(crate) for crate in order}
        validate_new_package_approvals(package_exists, set(args.approve_new_package))

        already_published: set[str] = set()
        for crate in order:
            package = packages[crate]
            if package["version"] != args.version:
                raise ReleaseError(
                    f"{crate} version {package['version']} does not match {args.version}"
                )
            if client.version_exists(crate, args.version):
                inspection = inspect_registry_archive(
                    client, crate, args.version, commit
                )
                print_inspection(crate, inspection)
                already_published.add(crate)
                print(f"{crate} {args.version}: already published from this tag; skip")

        if not args.execute:
            print("preflight passed; no packages published (rerun with --execute)")
            return 0

        for crate in order:
            if crate in already_published:
                continue
            package = packages[crate]
            prerequisites = sorted(
                dependency["name"]
                for dependency in package["dependencies"]
                if dependency.get("kind") != "dev" and dependency["name"] in packages
            )
            for prerequisite in prerequisites:
                await_registry_archive(
                    client, prerequisite, args.version, commit
                )

            build_and_inspect_archive(
                metadata,
                crate,
                args.version,
                commit,
                ["cargo", "package", "-p", crate, "--locked"],
            )
            build_and_inspect_archive(
                metadata,
                crate,
                args.version,
                commit,
                [
                    "cargo",
                    "publish",
                    "--dry-run",
                    "-p",
                    crate,
                    "--locked",
                    "--registry",
                    "crates-io",
                ],
            )
            run(
                [
                    "cargo",
                    "publish",
                    "-p",
                    crate,
                    "--locked",
                    "--registry",
                    "crates-io",
                ],
                capture=False,
            )
            inspection = await_registry_archive(
                client, crate, args.version, commit
            )
            print_inspection(crate, inspection)
        return 0
    except ReleaseError as error:
        print(f"release-publish: abort: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
