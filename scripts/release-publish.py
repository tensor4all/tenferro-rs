#!/usr/bin/env python3
"""Fail-closed operator helper for publishing a tagged tenferro release."""

from __future__ import annotations

import argparse
import http.client
import io
import json
import re
import runpy
import stat
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
from pathlib import Path, PurePosixPath
from typing import NamedTuple


ROOT = Path(__file__).resolve().parents[1]
CRATES_IO_API = "https://crates.io/api/v1/crates"
COMMAND_TIMEOUT = 30 * 60.0
NETWORK_TIMEOUT = 30.0
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
    contents: dict[str, bytes]


def run(
    command: list[str],
    *,
    cwd: Path = ROOT,
    capture: bool = True,
    check: bool = True,
    text: bool = True,
    timeout: float = COMMAND_TIMEOUT,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=check,
            text=text,
            timeout=timeout,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except FileNotFoundError as error:
        raise ReleaseError(f"command not found: {command[0]}") from error
    except subprocess.TimeoutExpired as error:
        raise ReleaseError(
            f"{' '.join(command)} timed out after {timeout:g} seconds"
        ) from error
    except UnicodeDecodeError as error:
        raise ReleaseError(
            f"{' '.join(command)} returned output that is not valid UTF-8: {error}"
        ) from error
    except subprocess.CalledProcessError as error:
        output = error.stderr or error.stdout or b""
        detail = (
            output.decode("utf-8", errors="replace")
            if isinstance(output, bytes)
            else output
        ).strip()
        suffix = f": {detail}" if detail else ""
        raise ReleaseError(f"{' '.join(command)} failed{suffix}") from error
    except OSError as error:
        raise ReleaseError(f"could not run {' '.join(command)}: {error}") from error


def parse_workspace_git_dependencies(manifest_text: str) -> list[GitDependency]:
    try:
        manifest = tomllib.loads(manifest_text)
    except tomllib.TOMLDecodeError as error:
        raise ReleaseError(f"Cargo.toml is invalid TOML: {error}") from error
    workspace = manifest.get("workspace", {})
    if not isinstance(workspace, dict):
        raise ReleaseError("[workspace] must be a table")
    dependencies = workspace.get("dependencies", {})
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
        workspace = manifest.get("workspace", {})
        if not isinstance(workspace, dict):
            raise ReleaseError(
                f"{dependency.git}@{dependency.rev}:{path} [workspace] must be a table"
            )
        workspace_package = workspace.get("package", {})
        if not isinstance(workspace_package, dict):
            raise ReleaseError(
                f"{dependency.git}@{dependency.rev}:{path} [workspace.package] must be a table"
            )
        workspace_version = workspace_package.get("version")
        if isinstance(workspace_version, str):
            workspace_versions.append(workspace_version)

    matches: list[tuple[str, object]] = []
    for path, manifest in parsed.items():
        package = manifest.get("package", {})
        if not isinstance(package, dict):
            raise ReleaseError(
                f"{dependency.git}@{dependency.rev}:{path} [package] must be a table"
            )
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


def fetch_revision_manifests(
    dependency: GitDependency, *, runner: Callable = run
) -> dict[str, str]:
    try:
        temporary = tempfile.TemporaryDirectory(prefix="tenferro-release-git-")
    except OSError as error:
        raise ReleaseError(f"could not create temporary git checkout: {error}") from error
    with temporary as directory:
        checkout = Path(directory)
        runner(["git", "init", "--quiet"], cwd=checkout)
        runner(
            ["git", "fetch", "--quiet", "--depth=1", dependency.git, dependency.rev],
            cwd=checkout,
        )
        paths = runner(
            ["git", "ls-tree", "-r", "--name-only", "FETCH_HEAD"], cwd=checkout
        ).stdout.splitlines()
        manifests: dict[str, str] = {}
        for path in paths:
            if Path(path).name == "Cargo.toml":
                manifests[path] = runner(
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
        with urllib.request.urlopen(request, timeout=NETWORK_TIMEOUT) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        try:
            return error.code, error.read()
        except (OSError, EOFError, http.client.HTTPException) as read_error:
            raise ReleaseError(
                f"crates.io error response could not be read for {url}: {read_error}"
            ) from read_error
    except (OSError, TimeoutError, EOFError, http.client.HTTPException) as error:
        raise ReleaseError(f"crates.io request failed for {url}: {error}") from error


class CratesIoClient:
    def __init__(self, transport: Transport = crates_io_transport):
        self.transport = transport

    @staticmethod
    def _url(*parts: str) -> str:
        return "/".join(
            [CRATES_IO_API, *(urllib.parse.quote(part, safe="") for part in parts)]
        )

    def _request(self, url: str) -> tuple[int, bytes]:
        try:
            result = self.transport(url)
        except ReleaseError:
            raise
        except (OSError, TimeoutError, EOFError, http.client.HTTPException) as error:
            raise ReleaseError(f"crates.io request failed for {url}: {error}") from error
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or not isinstance(result[0], int)
            or not isinstance(result[1], bytes)
        ):
            raise ReleaseError(f"crates.io transport returned an invalid response for {url}")
        return result

    def _exists(self, url: str) -> bool:
        status, _ = self._request(url)
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
        status, body = self._request(url)
        if status != 200:
            raise ReleaseError(
                f"crates.io archive download returned HTTP {status}: {package} {version}"
            )
        return body


def validate_new_package_approvals(
    package_exists: Mapping[str, bool],
    version_exists: Mapping[str, bool],
    approvals: set[str],
) -> None:
    if package_exists.keys() != version_exists.keys():
        raise ReleaseError("package and target-version registry results do not match")
    inconsistent = sorted(
        package
        for package, exists in package_exists.items()
        if not exists and version_exists[package]
    )
    if inconsistent:
        raise ReleaseError(
            f"crates.io reported target versions for missing packages: {inconsistent}"
        )
    new_packages = {package for package, exists in package_exists.items() if not exists}
    missing = sorted(new_packages - approvals)
    if missing:
        raise ReleaseError(
            "new crates.io packages require exact user approval before any publication: "
            f"{missing}; rerun with one --approve-new-package PACKAGE per approved package"
        )
    unknown = sorted(approvals - package_exists.keys())
    if unknown:
        raise ReleaseError(f"approval named packages outside the publication order: {unknown}")
    stale = sorted(
        package
        for package in approvals
        if package_exists[package] and not version_exists[package]
    )
    if stale:
        raise ReleaseError(
            "approval named existing packages whose target version does not exist: "
            f"{stale}"
        )


TaggedSourceReader = Callable[[str], bytes | None]


def _read_archive_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ReleaseError(f"crate archive member is not a regular file: {member.name}")
    return extracted.read()


def inspect_crate_archive(
    archive_data: bytes,
    expected_name: str,
    expected_version: str,
    expected_commit: str,
    package_root: str,
    source_reader: TaggedSourceReader,
    expected_files: set[str],
    expected_contents: Mapping[str, bytes] | None = None,
) -> ArchiveInspection:
    prefix = f"{expected_name}-{expected_version}/"
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_data), mode="r:gz") as archive:
            contents: dict[str, bytes] = {}
            full_names: list[str] = []
            member_paths: set[str] = set()
            archive_root = prefix.removesuffix("/")
            for member in archive.getmembers():
                if member.name == archive_root and member.isdir():
                    relative = ""
                elif member.name.startswith(prefix):
                    relative = member.name[len(prefix) :]
                else:
                    raise ReleaseError(
                        f"crate archive member is outside expected prefix {prefix!r}: "
                        f"{member.name!r}"
                    )
                parts = relative.split("/") if relative else []
                path = PurePosixPath(relative)
                if (
                    (not relative and not member.isdir())
                    or path.is_absolute()
                    or any(part in {"", ".", ".."} for part in parts)
                ):
                    raise ReleaseError(f"invalid crate archive member path: {member.name!r}")
                canonical = path.as_posix() if relative else ""
                if canonical in member_paths:
                    raise ReleaseError(f"duplicate crate archive member path: {member.name!r}")
                member_paths.add(canonical)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ReleaseError(
                        f"crate archive has unsupported non-regular member {member.name!r}"
                    )
                contents[canonical] = _read_archive_member(archive, member)
                full_names.append(member.name)

            actual_files = set(contents)
            missing_expected = sorted(expected_files - actual_files)
            unexpected = sorted(actual_files - expected_files)
            if missing_expected or unexpected:
                raise ReleaseError(
                    "crate archive file list differs from Cargo's tagged package selection: "
                    f"missing {missing_expected}, unexpected {unexpected}"
                )
            required = {"Cargo.toml", "Cargo.toml.orig", ".cargo_vcs_info.json"}
            missing = sorted(required - actual_files)
            if missing:
                raise ReleaseError(f"crate archive is missing required files: {missing}")
            if expected_contents is not None:
                differing = sorted(
                    relative
                    for relative in actual_files
                    if expected_contents.get(relative) != contents[relative]
                )
                if set(expected_contents) != actual_files:
                    raise ReleaseError(
                        "crate archive file set differs from exact local tagged archive"
                    )
                if differing:
                    raise ReleaseError(
                        "crate archive members differ from exact local tagged archive: "
                        f"{differing}"
                    )
            manifest = tomllib.loads(contents["Cargo.toml"].decode("utf-8"))
            vcs = json.loads(contents[".cargo_vcs_info.json"].decode("utf-8"))
    except ReleaseError:
        raise
    except (
        OSError,
        EOFError,
        tarfile.TarError,
        UnicodeDecodeError,
        tomllib.TOMLDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise ReleaseError(f"invalid .crate archive: {error}") from error

    if not isinstance(manifest, dict) or not isinstance(vcs, dict):
        raise ReleaseError("invalid .crate archive: metadata roots must be objects")
    package = manifest.get("package")
    if not isinstance(package, dict):
        raise ReleaseError("packaged Cargo.toml must contain a [package] table")
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

    readme = package["readme"].lstrip("./")
    if readme not in contents:
        raise ReleaseError(f"crate archive is missing declared README {prefix + readme}")
    license_file = package.get("license-file")
    if license_file is not None and (
        not isinstance(license_file, str) or not license_file.strip()
    ):
        raise ReleaseError("packaged Cargo.toml package.license-file must be a string")
    if isinstance(license_file, str) and license_file not in contents:
        raise ReleaseError(
            f"crate archive is missing declared license file {prefix + license_file}"
        )

    git = vcs.get("git")
    if not isinstance(git, dict):
        raise ReleaseError("crate archive .cargo_vcs_info.json must contain a git object")
    if git.get("sha1") != expected_commit or git.get("dirty", False) is not False:
        raise ReleaseError(
            "crate archive provenance does not equal the clean tagged commit: "
            f"{git!r}, expected sha1 {expected_commit} and dirty false"
        )

    generated = {".cargo_vcs_info.json", "Cargo.toml", "Cargo.lock"}
    external = {readme}
    if isinstance(license_file, str):
        external.add(license_file)
    package_root = package_root.strip("/")
    for relative, archived in contents.items():
        if relative in generated:
            continue
        source_path = (
            f"{package_root}/Cargo.toml"
            if relative == "Cargo.toml.orig"
            else f"{package_root}/{relative}"
        )
        try:
            source = source_reader(source_path)
            if source is None and relative in external:
                source_path = relative
                source = source_reader(source_path)
        except ReleaseError:
            raise
        except (OSError, TimeoutError) as error:
            raise ReleaseError(
                f"could not read tagged source for archive member {relative}: {error}"
            ) from error
        if source is None:
            raise ReleaseError(
                f"crate archive member {relative!r} has no mapped file in tagged package tree"
            )
        if archived != source:
            raise ReleaseError(
                f"crate archive member {relative!r} differs from tagged source {source_path!r}"
            )
    return ArchiveInspection(package, tuple(sorted(full_names)), contents)


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


def verify_release_checkout(
    version: str, *, runner: Callable = run, root: Path = ROOT
) -> str:
    if EXACT_VERSION.fullmatch(version) is None:
        raise ReleaseError(f"release version must be exact, found {version!r}")
    runner(["git", "fetch", "origin", "main"], cwd=root, capture=False)
    if runner(["git", "status", "--porcelain"], cwd=root).stdout:
        raise ReleaseError("release worktree must be clean, including untracked files")
    symbolic = runner(
        ["git", "symbolic-ref", "-q", "HEAD"], cwd=root, check=False
    )
    if symbolic.returncode == 0:
        raise ReleaseError("release worktree must be detached at the tag")
    if symbolic.returncode != 1:
        raise ReleaseError("git could not determine whether the release checkout is detached")

    tag = f"v{version}"
    lines = runner(
        [
            "git",
            "ls-remote",
            "--tags",
            "origin",
            f"refs/tags/{tag}",
            f"refs/tags/{tag}^{{}}",
        ],
        cwd=root,
    ).stdout.splitlines()
    refs = {line.split()[1]: line.split()[0] for line in lines if len(line.split()) == 2}
    remote_commit = refs.get(f"refs/tags/{tag}^{{}}") or refs.get(f"refs/tags/{tag}")
    if remote_commit is None:
        raise ReleaseError(f"origin does not have tag {tag}")
    head = runner(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()
    if head != remote_commit:
        raise ReleaseError(
            f"HEAD {head} is not exact pushed remote tag {tag} commit {remote_commit}"
        )
    runner(
        ["git", "merge-base", "--is-ancestor", head, "origin/main"], cwd=root
    )

    try:
        root_manifest = tomllib.loads(
            (root / "Cargo.toml").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ReleaseError(f"could not parse release Cargo.toml: {error}") from error
    workspace = root_manifest.get("workspace")
    declared = (
        workspace.get("package", {}).get("version")
        if isinstance(workspace, dict)
        else None
    )
    if declared != version:
        raise ReleaseError(
            f"tag version {version} does not match workspace package version {declared!r}"
        )
    return head


def cargo_metadata(*, runner: Callable = run, root: Path = ROOT) -> dict:
    try:
        metadata = json.loads(
            runner(
                ["cargo", "metadata", "--no-deps", "--format-version", "1"],
                cwd=root,
            ).stdout
        )
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseError(f"cargo metadata returned invalid JSON: {error}") from error
    if not isinstance(metadata, dict):
        raise ReleaseError("cargo metadata JSON root must be an object")
    return metadata


def tagged_source_reader(
    commit: str, *, runner: Callable = run, root: Path = ROOT
) -> TaggedSourceReader:
    paths = set(
        runner(
            ["git", "ls-tree", "-r", "--name-only", commit], cwd=root
        ).stdout.splitlines()
    )

    def read_source(path: str) -> bytes | None:
        if path not in paths:
            return None
        output = runner(
            ["git", "show", f"{commit}:{path}"], cwd=root, text=False
        ).stdout
        if not isinstance(output, bytes):
            raise ReleaseError(f"git show returned non-binary output for {path}")
        return output

    return read_source


def publication_order(release_text: str, *, root: Path = ROOT) -> list[str]:
    try:
        checker = runpy.run_path(str(root / "scripts/check-publish-layout.py"))
        parser = checker["phase3_publication_order"]
        errors: list[str] = []
        order = parser(release_text, errors)
    except (OSError, SyntaxError, KeyError, TypeError) as error:
        raise ReleaseError(f"could not parse release publication order: {error}") from error
    if order is None or errors:
        raise ReleaseError("; ".join(errors) or "release publication order is missing")
    if not isinstance(order, list) or any(not isinstance(crate, str) for crate in order):
        raise ReleaseError("release publication order parser returned invalid data")
    return order


def cargo_package_files(
    crate: str, *, runner: Callable = run, root: Path = ROOT
) -> set[str]:
    lines = runner(
        ["cargo", "package", "--list", "-p", crate, "--locked"], cwd=root
    ).stdout.splitlines()
    files: set[str] = set()
    for line in lines:
        path = PurePosixPath(line)
        if (
            not line
            or path.is_absolute()
            or ".." in path.parts
            or line in files
        ):
            raise ReleaseError(
                f"cargo package --list returned an invalid path for {crate}: {line!r}"
            )
        files.add(line)
    if not files:
        raise ReleaseError(f"cargo package --list returned no files for {crate}")
    return files


def package_root(package: Mapping[str, object], *, root: Path = ROOT) -> str:
    manifest = package.get("manifest_path")
    if not isinstance(manifest, str):
        raise ReleaseError("cargo metadata package is missing manifest_path")
    try:
        return Path(manifest).resolve().parent.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ReleaseError(f"package manifest is outside release checkout: {manifest}") from error


def inspect_registry_archive(
    client: CratesIoClient,
    crate: str,
    version: str,
    commit: str,
    package_dir: str,
    source_reader: TaggedSourceReader,
    expected_files: set[str],
    expected_contents: Mapping[str, bytes],
    *,
    archive_inspector: Callable = inspect_crate_archive,
) -> ArchiveInspection:
    return archive_inspector(
        client.download(crate, version),
        crate,
        version,
        commit,
        package_dir,
        source_reader,
        expected_files,
        expected_contents,
    )


def await_registry_archive(
    client: CratesIoClient,
    crate: str,
    version: str,
    commit: str,
    package_dir: str,
    source_reader: TaggedSourceReader,
    expected_files: set[str],
    expected_contents: Mapping[str, bytes],
    *,
    archive_inspector: Callable = inspect_crate_archive,
    attempts: int = 40,
    delay: float = 30.0,
    sleeper: Callable[[float], None] = time.sleep,
) -> ArchiveInspection:
    if attempts < 1 or delay < 0:
        raise ReleaseError("registry propagation attempts must be positive and delay non-negative")
    last_mismatch: ReleaseError | None = None
    for attempt in range(attempts):
        if client.version_exists(crate, version):
            try:
                return inspect_registry_archive(
                    client,
                    crate,
                    version,
                    commit,
                    package_dir,
                    source_reader,
                    expected_files,
                    expected_contents,
                    archive_inspector=archive_inspector,
                )
            except ReleaseError as error:
                last_mismatch = error
        if attempt + 1 < attempts:
            sleeper(delay)
    detail = f"; last archive error: {last_mismatch}" if last_mismatch else ""
    raise ReleaseError(
        f"{crate} {version} did not become visible with matching provenance on crates.io"
        f" after {attempts} attempts{detail}"
    )


def archive_path(metadata: Mapping[str, object], crate: str, version: str) -> Path:
    target = metadata.get("target_directory")
    if not isinstance(target, str):
        raise ReleaseError("cargo metadata is missing target_directory")
    return Path(target) / "package" / f"{crate}-{version}.crate"


def build_and_inspect_archive(
    metadata: Mapping[str, object],
    crate: str,
    version: str,
    commit: str,
    package_dir: str,
    source_reader: TaggedSourceReader,
    expected_files: set[str],
    command: list[str],
    *,
    expected_contents: Mapping[str, bytes] | None = None,
    runner: Callable = run,
    root: Path = ROOT,
    archive_inspector: Callable = inspect_crate_archive,
) -> ArchiveInspection:
    path = archive_path(metadata, crate, version)
    candidates = (
        path,
        path.parent / "tmp-crate" / path.name,
        path.parent / "tmp-registry" / path.name,
    )
    try:
        for candidate in candidates:
            candidate.unlink(missing_ok=True)
    except OSError as error:
        raise ReleaseError(
            f"could not remove stale Cargo archive {candidate}: {error}"
        ) from error
    runner(command, cwd=root, capture=False)
    archives = []
    for candidate in candidates:
        try:
            status = candidate.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ReleaseError(
                f"could not inspect Cargo archive {candidate}: {error}"
            ) from error
        if not stat.S_ISREG(status.st_mode):
            raise ReleaseError(
                f"Cargo archive candidate is not a regular file: {candidate}"
            )
        archives.append(candidate)
    if not archives:
        raise ReleaseError(
            f"Cargo did not create expected archive at any of: {', '.join(map(str, candidates))}"
        )
    archive_data = None
    for candidate in archives:
        try:
            candidate_data = candidate.read_bytes()
        except OSError as error:
            raise ReleaseError(f"could not read Cargo archive {candidate}: {error}") from error
        if archive_data is None:
            archive_data = candidate_data
        elif candidate_data != archive_data:
            raise ReleaseError(f"Cargo created differing archives: {archives}")
    inspection = archive_inspector(
        archive_data,
        crate,
        version,
        commit,
        package_dir,
        source_reader,
        expected_files,
        expected_contents,
    )
    print_inspection(crate, inspection)
    return inspection


def _read_text(path: Path, description: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ReleaseError(f"could not read {description} {path}: {error}") from error


def publish_release(
    version: str,
    approvals: set[str],
    execute: bool,
    *,
    root: Path = ROOT,
    runner: Callable = run,
    client: CratesIoClient | None = None,
    checkout_verifier: Callable = verify_release_checkout,
    metadata_loader: Callable = cargo_metadata,
    order_loader: Callable = publication_order,
    package_files_loader: Callable = cargo_package_files,
    revision_fetcher: Callable = fetch_revision_manifests,
    archive_inspector: Callable = inspect_crate_archive,
    source_reader_factory: Callable = tagged_source_reader,
    attempts: int = 40,
    delay: float = 30.0,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    commit = checkout_verifier(version, runner=runner, root=root)
    runner(
        [sys.executable, "scripts/check-publish-layout.py"],
        cwd=root,
        capture=False,
    )
    root_text = _read_text(root / "Cargo.toml", "workspace manifest")
    git_dependencies = parse_workspace_git_dependencies(root_text)
    registry = client or CratesIoClient()
    revision_cache: dict[tuple[str, str], dict[str, str]] = {}
    for dependency in git_dependencies:
        cache_key = (dependency.git, dependency.rev)
        if cache_key not in revision_cache:
            revision_cache[cache_key] = revision_fetcher(
                dependency, runner=runner
            )
        validate_revision_manifest(dependency, revision_cache[cache_key])
        if not registry.version_exists(dependency.package, dependency.version):
            raise ReleaseError(
                f"workspace git dependency {dependency.name!r} declares missing "
                f"crates.io package/version {dependency.package} {dependency.version}"
            )
        print(
            f"git dependency verified: {dependency.name} -> {dependency.package} "
            f"{dependency.version} at {dependency.rev}"
        )

    metadata = metadata_loader(runner=runner, root=root)
    package_data = metadata.get("packages")
    if not isinstance(package_data, list) or any(
        not isinstance(package, dict) for package in package_data
    ):
        raise ReleaseError("cargo metadata packages must be a list of objects")
    packages: dict[str, dict] = {}
    for package in package_data:
        name = package.get("name")
        if not isinstance(name, str) or name in packages:
            raise ReleaseError("cargo metadata package names must be unique strings")
        packages[name] = package

    release_text = _read_text(
        root / "ai/contribution-workflows/release-publish.md", "release workflow"
    )
    order = order_loader(release_text)
    if not isinstance(order, list) or any(not isinstance(crate, str) for crate in order):
        raise ReleaseError("release publication order must be a list of package names")
    if len(order) != len(set(order)):
        raise ReleaseError("release publication order must not contain duplicates")
    missing_packages = sorted(set(order) - packages.keys())
    if missing_packages:
        raise ReleaseError(f"publication order names missing metadata packages: {missing_packages}")

    package_exists = {crate: registry.package_exists(crate) for crate in order}
    version_exists = {
        crate: registry.version_exists(crate, version) for crate in order
    }
    validate_new_package_approvals(package_exists, version_exists, approvals)
    source_reader = source_reader_factory(commit, runner=runner, root=root)

    positions = {crate: index for index, crate in enumerate(order)}
    prerequisites_by_crate: dict[str, list[str]] = {}
    for crate in order:
        package = packages[crate]
        package_version = package.get("version")
        if package_version != version:
            raise ReleaseError(
                f"{crate} version {package_version!r} does not match {version}"
            )
        dependencies = package.get("dependencies")
        if not isinstance(dependencies, list) or any(
            not isinstance(dependency, dict) for dependency in dependencies
        ):
            raise ReleaseError(f"cargo metadata dependencies are invalid for {crate}")
        prerequisites = sorted(
            dependency["name"]
            for dependency in dependencies
            if dependency.get("kind") != "dev"
            and isinstance(dependency.get("name"), str)
            and dependency["name"] in positions
        )
        misplaced = [
            prerequisite
            for prerequisite in prerequisites
            if positions[prerequisite] >= positions[crate]
        ]
        if misplaced:
            raise ReleaseError(
                f"publication order places {crate} before prerequisites {misplaced}"
            )
        prerequisites_by_crate[crate] = prerequisites

    expected_files_by_crate: dict[str, set[str]] = {}
    expected_contents_by_crate: dict[str, dict[str, bytes]] = {}
    for crate in order:
        if not execute and not version_exists[crate]:
            continue
        package = packages[crate]
        for prerequisite in prerequisites_by_crate[crate]:
            if prerequisite not in expected_contents_by_crate:
                raise ReleaseError(
                    f"cannot attest {crate}: prerequisite {prerequisite} {version} "
                    "has no exact local tagged archive"
                )
            await_registry_archive(
                registry,
                prerequisite,
                version,
                commit,
                package_root(packages[prerequisite], root=root),
                source_reader,
                expected_files_by_crate[prerequisite],
                expected_contents_by_crate[prerequisite],
                archive_inspector=archive_inspector,
                attempts=attempts,
                delay=delay,
                sleeper=sleeper,
            )

        package_dir = package_root(package, root=root)
        expected_files = package_files_loader(crate, runner=runner, root=root)
        expected_files_by_crate[crate] = expected_files
        local_inspection = build_and_inspect_archive(
            metadata,
            crate,
            version,
            commit,
            package_dir,
            source_reader,
            expected_files,
            ["cargo", "package", "-p", crate, "--locked"],
            runner=runner,
            root=root,
            archive_inspector=archive_inspector,
        )
        expected_contents_by_crate[crate] = local_inspection.contents

        if version_exists[crate]:
            inspection = inspect_registry_archive(
                registry,
                crate,
                version,
                commit,
                package_dir,
                source_reader,
                expected_files,
                local_inspection.contents,
                archive_inspector=archive_inspector,
            )
            print_inspection(crate, inspection)
            print(f"{crate} {version}: already published from this tag; skip")
            continue

        build_and_inspect_archive(
            metadata,
            crate,
            version,
            commit,
            package_dir,
            source_reader,
            expected_files,
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
            expected_contents=local_inspection.contents,
            runner=runner,
            root=root,
            archive_inspector=archive_inspector,
        )
        runner(
            [
                "cargo",
                "publish",
                "-p",
                crate,
                "--locked",
                "--registry",
                "crates-io",
            ],
            cwd=root,
            capture=False,
        )
        inspection = await_registry_archive(
            registry,
            crate,
            version,
            commit,
            package_dir,
            source_reader,
            expected_files,
            local_inspection.contents,
            archive_inspector=archive_inspector,
            attempts=attempts,
            delay=delay,
            sleeper=sleeper,
        )
        print_inspection(crate, inspection)

    if not execute:
        print("preflight passed; no packages published (rerun with --execute)")


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
        publish_release(
            args.version,
            set(args.approve_new_package),
            args.execute,
        )
        return 0
    except ReleaseError as error:
        print(f"release-publish: abort: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
