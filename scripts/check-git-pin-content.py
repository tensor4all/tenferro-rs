#!/usr/bin/env python3
"""Verify that git-pinned workspace dependencies match their crates.io release.

`cargo publish` strips the `git` source from a dependency and keeps only its
`version`, so a published crate resolves the registry package, not the pinned
revision. A registry package that exists at the required version but holds
different contents is invisible to a version-existence check: `strided-kernel
0.4.0` on crates.io was the pre-refactor crate while the pinned revision had the
post-refactor facade, so `cargo publish` failed ten crates into the release
order, and `strided-perm 0.4.0` drifted silently in `src/hptt/plan.rs`.

For every git-pinned `[workspace.dependencies]` entry, this check compares the
pinned revision against the crate archive of the version Cargo would actually
resolve from crates.io:

* `src/**` and `build.rs`: file set and content hashes, and
* normal and build dependency wiring (`package` rename, version requirement,
  optionality, features) with `[workspace.dependencies]` inheritance resolved.

Only files Cargo always packages are compared, so `include`/`exclude` selection
and cargo-generated members (`Cargo.toml`, `Cargo.toml.orig`,
`.cargo_vcs_info.json`) do not have to be re-implemented here.

The check fails closed on any difference. It warns when the requirement is not
an exact `= X.Y.Z` pin, because a caret range can resolve to content the pinned
revision never contained, and when an entry in
`scripts/git-pin-content-exceptions.toml` is no longer needed.

Examples:
    $ python3 scripts/check-git-pin-content.py
    $ python3 scripts/check-git-pin-content.py --root-dir .
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import runpy
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
EXCEPTIONS_PATH = "scripts/git-pin-content-exceptions.toml"
RELEASE_HELPER = ROOT / "scripts" / "release-publish.py"

# Files Cargo packages unconditionally for a library crate. Everything else is
# subject to include/exclude selection, which this check does not re-implement.
COMPARED_ROOT_FILES = ("build.rs",)
COMPARED_PREFIXES = ("src/",)

DIFF_LIMIT = 6
NUMBER = re.compile(r"^([0-9]+)(?:\.([0-9]+))?(?:\.([0-9]+))?")
PRERELEASE = re.compile(r"[0-9]+(?:[-+][0-9A-Za-z.-]+)$")
VCS_INFO = ".cargo_vcs_info.json"


class PinContentError(RuntimeError):
    """A pinned git dependency does not match its crates.io release."""


class Finding(NamedTuple):
    severity: str
    message: str


class PinnedPackage(NamedTuple):
    package: str
    declared_version: str
    manifest: Mapping
    workspace_manifest: Mapping
    files: Mapping[str, bytes]


class Requirement(NamedTuple):
    name: str
    raw: str
    dependency: object


class ExceptionEntry(NamedTuple):
    package: str
    reason: str
    issue: str


class Report(NamedTuple):
    package: str
    requirement: str
    declared_version: str
    resolved_version: str
    findings: tuple[Finding, ...]


def load_release_helper() -> dict:
    """Load the release helper for its shared pin, git, and registry plumbing."""

    return runpy.run_path(str(RELEASE_HELPER), run_name="tenferro_release_helper")


def parse_requirements(root: Path, helper: Mapping) -> list[Requirement]:
    """Parse git pins, keeping the raw version requirement from the manifest."""

    manifest_text = (root / "Cargo.toml").read_text(encoding="utf-8")
    dependencies = helper["parse_workspace_git_dependencies"](manifest_text)
    raw: dict[str, str] = {}
    workspace = tomllib.loads(manifest_text).get("workspace") or {}
    for name, value in (workspace.get("dependencies") or {}).items():
        if isinstance(value, dict) and isinstance(value.get("version"), str):
            raw[name] = value["version"]
    return [Requirement(dependency.name, raw[dependency.name], dependency) for dependency in dependencies]


def parse_version(text: str) -> tuple[int, int, int]:
    """Return the numeric release components of a version string."""

    match = NUMBER.match(text.strip())
    if match is None:
        raise PinContentError(f"unparsable version {text!r}")
    return tuple(int(part or 0) for part in match.groups())  # type: ignore[return-value]


def is_prerelease(text: str) -> bool:
    """Return True when a version carries a pre-release or build suffix."""

    stripped = text.strip()
    match = NUMBER.match(stripped)
    return match is not None and bool(stripped[match.end() :])


def requirement_bounds(requirement: str) -> tuple[tuple[int, int, int], tuple[int, int, int] | None]:
    """Return the lower bound and exclusive upper bound of a requirement."""

    text = requirement.strip()
    exact = text.startswith("=")
    if exact:
        text = text[1:].strip()
    lower = parse_version(text)
    if exact:
        return lower, None
    major, minor, patch = lower
    if major > 0:
        return lower, (major + 1, 0, 0)
    if minor > 0:
        return lower, (0, minor + 1, 0)
    return lower, (0, 0, patch + 1)


def effective_version(requirement: str, versions: Iterable[str]) -> str | None:
    """Return the highest version Cargo would resolve for ``requirement``."""

    text = requirement.strip()
    exact = text.startswith("=")
    lower, upper = requirement_bounds(text)
    candidates: list[tuple[tuple[int, int, int], str]] = []
    for candidate in versions:
        if is_prerelease(candidate):
            continue
        parsed = parse_version(candidate)
        if exact:
            if parsed != lower:
                continue
        elif parsed < lower or (upper is not None and not parsed < upper):
            continue
        candidates.append((parsed, candidate))
    if not candidates:
        return None
    return max(candidates)[1]


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compared_files(files: Mapping[str, bytes]) -> dict[str, str]:
    """Return ``posix path -> content hash`` for the files this check compares."""

    return {
        path: content_hash(data)
        for path, data in files.items()
        if path in COMPARED_ROOT_FILES or path.startswith(COMPARED_PREFIXES)
    }


def _dependency_spec(name: str, value: object, workspace: Mapping) -> dict:
    if isinstance(value, str):
        return {"package": name, "version": value, "optional": False, "default_features": True, "features": []}
    if not isinstance(value, dict):
        raise PinContentError(f"dependency {name!r} has an unsupported specification")
    if value.get("workspace") is True:
        inherited = workspace.get(name)
        if isinstance(inherited, str):
            inherited = {"version": inherited}
        if not isinstance(inherited, dict):
            raise PinContentError(f"dependency {name!r} inherits a missing workspace entry")
        merged = dict(inherited)
        # Cargo unions a member's `features` with the inherited workspace
        # entry; every other key comes from the inherited specification.
        member_features = value.get("features", [])
        inherited_features = inherited.get("features", [])
        if member_features or inherited_features:
            merged["features"] = [*inherited_features, *member_features]
        merged.update(
            {
                key: item
                for key, item in value.items()
                if key not in ("workspace", "features")
            }
        )
        value = merged
    features = value.get("features", [])
    return {
        "package": value.get("package", name),
        "version": value.get("version"),
        "optional": bool(value.get("optional", False)),
        "default_features": value.get("default-features", True),
        "features": sorted(set(features)) if isinstance(features, list) else [],
    }


def normalized_dependencies(manifest: Mapping, workspace_manifest: Mapping) -> dict[str, dict]:
    """Normalize normal and build dependency wiring, resolving inheritance."""

    workspace = ((workspace_manifest.get("workspace") or {}).get("dependencies")) or {}
    result: dict[str, dict] = {}
    for table in ("dependencies", "build-dependencies"):
        entries = manifest.get(table) or {}
        if not isinstance(entries, dict):
            raise PinContentError(f"[{table}] must be a table")
        for name, value in entries.items():
            result[f"{table}:{name}"] = _dependency_spec(name, value, workspace)
    return result


def published_commit(archive_files: Mapping[str, bytes]) -> str:
    """Return the commit recorded in a crate archive, when present."""

    raw = archive_files.get(VCS_INFO)
    if raw is None:
        return "unrecorded"
    try:
        info = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "unparsable"
    commit = info.get("git", {}).get("sha1") if isinstance(info, dict) else None
    return commit if isinstance(commit, str) else "unrecorded"


def compare_pin(
    *,
    pinned: PinnedPackage,
    version: str,
    archive_files: Mapping[str, bytes],
    archive_manifest: Mapping,
) -> list[Finding]:
    """Compare a pinned package tree against its registry archive."""

    findings: list[Finding] = []
    provenance = f".cargo_vcs_info.json commit={published_commit(archive_files)}"
    expected = compared_files(pinned.files)
    actual = compared_files(archive_files)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing:
        findings.append(
            Finding(
                "error",
                f"{pinned.package} {version}: registry release lacks pinned files "
                f"{_summarize(missing)} ({provenance})",
            )
        )
    if unexpected:
        findings.append(
            Finding(
                "error",
                f"{pinned.package} {version}: registry release ships files absent from the "
                f"pinned revision: {_summarize(unexpected)} ({provenance})",
            )
        )
    differing = sorted(
        path for path in set(expected) & set(actual) if expected[path] != actual[path]
    )
    if differing:
        findings.append(
            Finding(
                "error",
                f"{pinned.package} {version}: registry release content differs from the "
                f"pinned revision: {_summarize(differing)} ({provenance})",
            )
        )

    expected_dependencies = normalized_dependencies(pinned.manifest, pinned.workspace_manifest)
    actual_dependencies = normalized_dependencies(archive_manifest, {})
    if expected_dependencies != actual_dependencies:
        changed = sorted(
            key
            for key in set(expected_dependencies) | set(actual_dependencies)
            if expected_dependencies.get(key) != actual_dependencies.get(key)
        )
        details = "; ".join(
            f"{key}: pinned {expected_dependencies.get(key)!r} vs registry "
            f"{actual_dependencies.get(key)!r}"
            for key in changed[:DIFF_LIMIT]
        )
        findings.append(
            Finding(
                "error",
                f"{pinned.package} {version}: dependency wiring differs "
                f"({len(changed)} entr{'y' if len(changed) == 1 else 'ies'}): {details} "
                f"({provenance})",
            )
        )
    return findings


def _summarize(paths: Sequence[str]) -> str:
    shown = ", ".join(paths[:DIFF_LIMIT])
    if len(paths) > DIFF_LIMIT:
        return f"{shown}, ... (+{len(paths) - DIFF_LIMIT} more)"
    return shown


def package_manifest(
    dependency, manifests: Mapping[str, str]
) -> tuple[str, Mapping, Mapping]:
    """Return the package manifest path, parsed manifest, and workspace manifest."""

    parsed: dict[str, Mapping] = {}
    for path, text in manifests.items():
        try:
            parsed[path] = tomllib.loads(text)
        except tomllib.TOMLDecodeError as error:
            raise PinContentError(
                f"{dependency.git}@{dependency.rev}:{path} is invalid TOML: {error}"
            ) from error
    matches = [
        path
        for path, manifest in parsed.items()
        if (manifest.get("package") or {}).get("name") == dependency.package
    ]
    if len(matches) != 1:
        raise PinContentError(
            f"{dependency.git}@{dependency.rev} must contain exactly one package named "
            f"{dependency.package!r}, found {len(matches)}"
        )
    workspace_path = next(
        (
            path
            for path, manifest in parsed.items()
            if isinstance((manifest.get("workspace") or {}).get("package"), dict)
        ),
        None,
    )
    # A single-crate repository may declare no [workspace.package]; inheritance
    # then fails later, where it is actually required.
    workspace_manifest = parsed[workspace_path] if workspace_path is not None else {}
    return matches[0], parsed[matches[0]], workspace_manifest


def declared_version(manifest: Mapping, workspace_manifest: Mapping) -> str:
    """Return the registry version a pinned package manifest declares."""

    version = (manifest.get("package") or {}).get("version")
    if version == {"workspace": True}:
        version = ((workspace_manifest.get("workspace") or {}).get("package") or {}).get("version")
    if not isinstance(version, str):
        raise PinContentError("pinned package manifest does not declare a version")
    return version


def _archive_prefix(package_dir: PurePosixPath) -> str:
    """Return the archive member prefix of a package directory."""

    text = package_dir.as_posix()
    return "" if text in ("", ".") else text.rstrip("/") + "/"


def _relative_member(name: str, prefix: str) -> str | None:
    """Return a member path relative to the package directory, or None."""

    candidate = name[2:] if name.startswith("./") else name
    if prefix:
        if not candidate.startswith(prefix):
            return None
        candidate = candidate[len(prefix) :]
    return candidate or None


def fetch_pinned_package(dependency, helper: Mapping) -> PinnedPackage:
    """Fetch the pinned revision and read the package manifest and files."""

    run = helper["run"]
    with tempfile.TemporaryDirectory(prefix="tenferro-pin-content-") as directory:
        checkout = Path(directory)
        run(["git", "init", "--quiet"], cwd=checkout)
        run(
            ["git", "fetch", "--quiet", "--depth=1", dependency.git, dependency.rev],
            cwd=checkout,
        )
        paths = run(
            ["git", "ls-tree", "-r", "--name-only", "FETCH_HEAD"], cwd=checkout
        ).stdout.splitlines()
        manifests = {
            path: run(["git", "show", f"FETCH_HEAD:{path}"], cwd=checkout).stdout
            for path in paths
            if PurePosixPath(path).name == "Cargo.toml"
        }
        manifest_path, manifest, workspace_manifest = package_manifest(dependency, manifests)
        package_dir = PurePosixPath(manifest_path).parent
        archive_paths = [] if package_dir.as_posix() == "." else [package_dir.as_posix()]
        archive = subprocess.run(
            ["git", "archive", "--format=tar", "FETCH_HEAD", *archive_paths],
            cwd=checkout,
            check=True,
            capture_output=True,
        ).stdout
    prefix = _archive_prefix(package_dir)
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            relative = _relative_member(member.name, prefix)
            if relative is None:
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                raise PinContentError(f"pinned revision member is not a file: {member.name}")
            files[relative] = extracted.read()
    return PinnedPackage(
        dependency.package,
        declared_version(manifest, workspace_manifest),
        manifest,
        workspace_manifest,
        files,
    )


def registry_archive(client, package: str, version: str) -> tuple[dict[str, bytes], Mapping]:
    """Download a crate archive and return its files and parsed manifest."""

    data = client.download(package, version)
    prefix = f"{package}-{version}/"
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                raise PinContentError(f"crate archive member is not a file: {member.name}")
            files[member.name[len(prefix) :]] = extracted.read()
    raw_manifest = files.get("Cargo.toml")
    if raw_manifest is None:
        raise PinContentError(f"crate archive {package} {version} has no Cargo.toml")
    return files, tomllib.loads(raw_manifest.decode("utf-8"))


def load_exceptions(root: Path) -> dict[str, ExceptionEntry]:
    """Load reviewed, justified exceptions from the repository."""

    path = root / EXCEPTIONS_PATH
    if not path.exists():
        return {}
    entries = tomllib.loads(path.read_text(encoding="utf-8")).get("exception", [])
    exceptions: dict[str, ExceptionEntry] = {}
    for entry in entries:
        package = entry.get("package", "")
        reason = entry.get("reason", "")
        issue = entry.get("issue", "")
        if not (package and reason and issue):
            raise PinContentError(
                f"{EXCEPTIONS_PATH}: every exception needs package, reason, and issue"
            )
        exceptions[package] = ExceptionEntry(package, reason, issue)
    return exceptions


def apply_exceptions(
    reports: Sequence[Report], exceptions: Mapping[str, ExceptionEntry]
) -> tuple[list[Finding], list[Finding]]:
    """Split findings into failures and warnings, honoring reviewed exceptions."""

    errors: list[Finding] = []
    warnings: list[Finding] = []
    excepted: set[str] = set()
    for report in reports:
        failures = [finding for finding in report.findings if finding.severity == "error"]
        warnings.extend(finding for finding in report.findings if finding.severity != "error")
        if not failures:
            continue
        exception = exceptions.get(report.package)
        if exception is None:
            errors.extend(failures)
            continue
        excepted.add(report.package)
        for finding in failures:
            warnings.append(
                Finding(
                    "warning",
                    f"excepted by {EXCEPTIONS_PATH} ({exception.issue}): {finding.message}",
                )
            )
    for package, exception in sorted(exceptions.items()):
        if package not in {report.package for report in reports}:
            errors.append(
                Finding("error", f"{EXCEPTIONS_PATH} names unknown package {package!r}")
            )
        elif package not in excepted:
            warnings.append(
                Finding(
                    "warning",
                    f"{EXCEPTIONS_PATH} exception for {package!r} is stale; remove it "
                    f"({exception.issue})",
                )
            )
    return errors, warnings


def check(root: Path = ROOT, *, helper: Mapping | None = None, client=None) -> tuple[list[Finding], list[Finding]]:
    """Run the check for every git pin and return failures and warnings."""

    helper = helper or load_release_helper()
    client = client or helper["CratesIoClient"]()
    reports: list[Report] = []
    for requirement in parse_requirements(root, helper):
        dependency = requirement.dependency
        pinned = fetch_pinned_package(dependency, helper)
        versions = client.versions(dependency.package)
        resolved = effective_version(requirement.raw, versions)
        if resolved is None:
            raise PinContentError(
                f"no crates.io version of {dependency.package} satisfies "
                f"{requirement.raw!r} (available: {sorted(versions)})"
            )
        archive_files, archive_manifest = registry_archive(client, dependency.package, resolved)
        findings = compare_pin(
            pinned=pinned,
            version=resolved,
            archive_files=archive_files,
            archive_manifest=archive_manifest,
        )
        if not requirement.raw.strip().startswith("="):
            findings.append(
                Finding(
                    "warning",
                    f"{dependency.name} is not an exact pin ({requirement.raw!r}); a caret "
                    "range can resolve to content the pinned revision never contained",
                )
            )
        if resolved != pinned.declared_version:
            findings.append(
                Finding(
                    "warning",
                    f"{dependency.name} {requirement.raw!r} resolves to {resolved} while the "
                    f"pinned revision declares {pinned.declared_version}; pin the resolved "
                    "version to make publication reproducible",
                )
            )
        reports.append(
            Report(
                dependency.package,
                requirement.raw,
                pinned.declared_version,
                resolved,
                tuple(findings),
            )
        )
    return apply_exceptions(reports, load_exceptions(root))


def format_report(findings: Sequence[Finding], *, stream) -> None:
    for finding in findings:
        print(f"git-pin-content: {finding.severity}: {finding.message}", file=stream)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=".", help="Repository root")
    args = parser.parse_args(argv)
    root = Path(args.root_dir).resolve()
    try:
        errors, warnings = check(root)
    except PinContentError as error:
        print(f"git-pin-content: {error}", file=sys.stderr)
        return 1
    format_report(warnings, stream=sys.stderr)
    format_report(errors, stream=sys.stderr)
    if errors:
        return 1
    print(f"git-pin-content: {len(warnings)} warning(s), no content drift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
