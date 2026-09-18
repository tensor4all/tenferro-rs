#!/usr/bin/env python3
"""Collect object-level evidence that scalar-set composition shares kernels.

Builds the external proof crate's composition test with `--emit=asm`, then counts
the *defined* instantiations of the shared elementwise kernel and the call sites
that reach them. The claim under test is that one set-specific member does not add
a numerical specialization: the preset instantiation appears once even though two
sets use it, and the external instantiation is the same shared kernel parameterized
by the contribution's own operation type.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_TARGET = "composition"
PACKAGE = "tenferro-df64-proof"
KERNEL = "zip_map2"
# The kernel entry points themselves, as opposed to the threading, dispatch, and
# vectorization wrappers that also carry the kernel name in their symbol.
KERNEL_FUNCTIONS = ("map_view13zip_map2_into", "map_view29zip_map2_parts_into_validated")
WRAPPERS = (
    "rayon_core",
    "threading",
    "ThreadedMapReduce",
    "for_each_inner_block_preordered",
    "join_context",
    "3job",
    "vectorize",
)
EXTERNAL_MEMBER = "tenferro_df64_proof4Df64"
PRESET_OP = "op5AddOp"
EXTERNAL_OP = "7Df64Add"
SHARED_KERNEL_CRATE = "tenferro_internal_cpu_kernels10scalar_ops18scalar_binary_into"

DEFINITION = re.compile(r"^(_R[A-Za-z0-9_]+):$")


def target_root() -> Path:
    path = Path(os.environ.get("CARGO_TARGET_DIR", "target"))
    return path if path.is_absolute() else ROOT / path


def run(*args: str) -> str:
    return subprocess.run(
        args, cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def definitions(assembly: str) -> list[str]:
    """Symbols this assembly file defines."""
    return [match.group(1) for line in assembly.splitlines() if (match := DEFINITION.match(line))]


def references(assembly: str, symbol: str) -> int:
    """How many places use a symbol, excluding its own definition block."""
    pattern = re.compile(rf"(?:\b|,){re.escape(symbol)}(?:\b|,)")
    return sum(
        1
        for line in assembly.splitlines()
        if pattern.search(line)
        and not line.startswith(symbol)
        and "\t.type\t" not in line
        and "\t.size\t" not in line
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="inspect the debug profile, which is what a test lane has already built",
    )
    parser.add_argument(
        "--package",
        default=PACKAGE,
        help="package holding the test target to inspect",
    )
    parser.add_argument(
        "--target",
        default=TEST_TARGET,
        help="test target to build and inspect",
    )
    parser.add_argument(
        "--containing-sets",
        type=int,
        default=1,
        help="how many ScalarSets containing the contribution the target uses",
    )
    parser.add_argument(
        "--against-package",
        default="",
        help="package of a second target that uses one more containing set",
    )
    parser.add_argument(
        "--against-target",
        default="",
        help="target whose instantiations must equal the primary target's",
    )
    parser.add_argument(
        "--against-sets",
        type=int,
        default=2,
        help="how many ScalarSets containing the contribution the comparison target uses",
    )
    parser.add_argument(
        "--set-type-names",
        default="",
        help="comma-separated ScalarSet type names that must not appear in an instantiation",
    )
    parser.add_argument(
        "--require-all-kernel-functions",
        default="true",
        help="whether every shared kernel entry point must be instantiated by the target",
    )
    parser.add_argument(
        "--expect-preset",
        default="true",
        help="whether the primary target is expected to use a preset scalar",
    )
    args = parser.parse_args()
    package = args.package
    test_target = args.target

    command = [
        "cargo",
        "rustc",
        "-j",
        "16",
        "-p",
        package,
        "--test",
        test_target,
        *([] if args.debug else ["--release"]),
        "--",
        "--emit=asm",
    ]
    record: dict[str, object] = {
        "schema": "tenferro.scalar-composition-kernel-sharing.v1",
        "candidate_commit": run("git", "rev-parse", "HEAD"),
        "package": package,
        "test_target": test_target,
        "containing_sets": args.containing_sets,
        "command": " ".join(command),
        "rustc": run("rustc", "-Vv").splitlines()[0],
        "target": run("rustc", "-Vv").split("host: ", 1)[-1].splitlines()[0],
        "profile": "debug" if args.debug else "release",
        "status": "inconclusive",
        "observations": [],
    }
    try:
        subprocess.run(command, cwd=ROOT, check=True, text=True)
    except subprocess.CalledProcessError as error:
        record["observations"].append(f"cargo rustc failed with exit code {error.returncode}")
        write_report(args.report, record)
        return 1

    profile_dir = "debug" if args.debug else "release"
    assemblies = sorted(
        (target_root() / profile_dir / "deps").glob(f"{test_target}-*.s"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not assemblies:
        record["observations"].append("cargo emitted no assembly file")
        write_report(args.report, record)
        return 1

    assembly_path = assemblies[0]
    assembly = assembly_path.read_text(encoding="utf-8", errors="replace")
    defined = definitions(assembly)

    def kernel_entries(member_op: str, external: bool) -> list[str]:
        return sorted(
            {
                name
                for name in defined
                if KERNEL in name
                and any(function in name for function in KERNEL_FUNCTIONS)
                and not any(wrapper in name for wrapper in WRAPPERS)
                and member_op in name
                and ((EXTERNAL_MEMBER in name) == external)
            }
        )

    preset = kernel_entries(PRESET_OP, external=False)
    external = kernel_entries(EXTERNAL_OP, external=True)
    preset_references = sum(references(assembly, symbol) for symbol in preset)
    external_references = sum(references(assembly, symbol) for symbol in external)

    record["assembly"] = str(assembly_path.relative_to(ROOT))
    record["assembly_bytes"] = assembly_path.stat().st_size
    record["defined_symbols"] = len(defined)
    record["kernel_entries"] = {
        "preset": {
            "count": len(preset),
            "references": preset_references,
            "by_function": {
                function: [symbol for symbol in preset if function in symbol]
                for function in KERNEL_FUNCTIONS
            },
            "all_from_shared_kernel_crate": bool(preset)
            and all(SHARED_KERNEL_CRATE in symbol for symbol in preset),
        },
        "external": {
            "count": len(external),
            "references": external_references,
            "by_function": {
                function: [symbol for symbol in external if function in symbol]
                for function in KERNEL_FUNCTIONS
            },
            "all_from_shared_kernel_crate": bool(external)
            and all(SHARED_KERNEL_CRATE in symbol for symbol in external),
            "all_parameterized_by_contribution_op": bool(external)
            and all(EXTERNAL_OP in symbol for symbol in external),
        },
    }

    failures: list[str] = []
    expect_preset = args.expect_preset.lower() not in ("false", "0", "no")
    all_functions = args.require_all_kernel_functions.lower() not in ("false", "0", "no")
    external_by_function = record["kernel_entries"]["external"]["by_function"]
    if not any(external_by_function[function] for function in KERNEL_FUNCTIONS):
        failures.append("the target instantiates no contribution kernel")
    for function in KERNEL_FUNCTIONS:
        if expect_preset and not record["kernel_entries"]["preset"]["by_function"][function]:
            failures.append(f"the preset path instantiates no {function}")
        if all_functions and not external_by_function[function]:
            failures.append(f"the external path instantiates no {function}")
    if expect_preset and not record["kernel_entries"]["preset"]["all_from_shared_kernel_crate"]:
        failures.append("a preset instantiation does not come from the shared kernel crate")
    if not record["kernel_entries"]["external"]["all_from_shared_kernel_crate"]:
        failures.append("an external instantiation does not come from the shared kernel crate")
    if not record["kernel_entries"]["external"]["all_parameterized_by_contribution_op"]:
        failures.append(
            "an external instantiation is not parameterized by the contribution's operation"
        )
    if expect_preset and preset_references < 2:
        failures.append(
            f"the preset instantiations are reached from {preset_references} places, "
            "so they are not visibly shared between call sites"
        )

    if args.against_target:
        record["comparison"] = compare_with(args, failures)

    record["failures"] = failures
    record["status"] = "fail" if failures else "pass"
    write_report(args.report, record)
    print(json.dumps(record, indent=2))
    return 1 if failures else 0


def compare_with(
    args: argparse.Namespace,
    failures: list[str],
) -> dict[str, object]:
    """Inspect a program that uses a second ScalarSet containing the contribution.

    The claim is that the numerical bodies depend on the actual scalar and implementation
    and never on a ScalarSet, so a containing set adds membership and dispatch rather than a
    numerical specialization. The test is therefore not a count: the shared kernels are
    generic over layout as well as over the element and operation types, and a test that
    uses a second set also has its own call sites, so counts differ for reasons that have
    nothing to do with the set. What must hold is that every instantiation is parameterized
    by the contribution's scalar and operation and that no set type appears in the
    parameters at all.
    """
    command = [
        "cargo",
        "rustc",
        "-j",
        "16",
        "-p",
        args.against_package,
        "--test",
        args.against_target,
        *([] if args.debug else ["--release"]),
        "--",
        "--emit=asm",
    ]
    comparison: dict[str, object] = {
        "package": args.against_package,
        "test_target": args.against_target,
        "containing_sets": args.against_sets,
        "set_type_names": set_names(args),
        "command": " ".join(command),
    }
    try:
        subprocess.run(command, cwd=ROOT, check=True, text=True)
    except subprocess.CalledProcessError as error:
        failures.append(
            f"building the comparison target failed with exit code {error.returncode}"
        )
        comparison["status"] = "inconclusive"
        return comparison

    profile_dir = "debug" if args.debug else "release"
    assemblies = sorted(
        (target_root() / profile_dir / "deps").glob(f"{args.against_target}-*.s"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not assemblies:
        failures.append("the comparison target emitted no assembly file")
        comparison["status"] = "inconclusive"
        return comparison

    assembly_path = assemblies[0]
    assembly = assembly_path.read_text(encoding="utf-8", errors="replace")
    external = sorted(
        {
            name
            for name in definitions(assembly)
            if KERNEL in name
            and any(function in name for function in KERNEL_FUNCTIONS)
            and not any(wrapper in name for wrapper in WRAPPERS)
            and EXTERNAL_OP in name
            and EXTERNAL_MEMBER in name
        }
    )
    comparison["assembly"] = str(assembly_path.relative_to(ROOT))
    comparison["assembly_bytes"] = assembly_path.stat().st_size
    comparison["external_instantiations"] = len(external)
    named_after_a_set = [name for name in external if any(entry in name for entry in set_names(args))]
    comparison["instantiations_naming_a_set"] = named_after_a_set
    if not external:
        failures.append(
            "the comparison target instantiates no contribution kernel, so it does not "
            "exercise a containing set"
        )
    if named_after_a_set:
        failures.append(
            f"{len(named_after_a_set)} contribution instantiations name a ScalarSet, so the "
            "numerical body depends on the set rather than on the scalar"
        )
    comparison["status"] = "pass" if external and not named_after_a_set else "fail"
    return comparison


def set_names(args: argparse.Namespace) -> list[str]:
    """Type names that must not appear in a numerical instantiation."""
    return [name for name in args.set_type_names.split(",") if name]


def write_report(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Scalar composition kernel sharing\n\n"
        "Generated by `scripts/check-scalar-composition-kernel-sharing.py`. The record\n"
        "below is the object-level evidence that composing scalar sets shares compiled\n"
        "numerical kernels rather than adding a set-specific specialization, with the profile,\n"
        "target, compiler, and assembly size recorded as build and code-size evidence. Two facts are\n"
        "recorded. First, the shared elementwise entry points are instantiated once per\n"
        "element type, operation, and layout, and both the preset and the contribution\n"
        "paths reach instantiations that come from the same kernel crate. Second, a\n"
        "program that uses two ScalarSets containing the contribution instantiates the\n"
        "contribution's kernel without any set type appearing in the parameters, so the\n"
        "numerical body is parameterized by the scalar and the operation rather than by\n"
        "the set that contains it.\n\n"
        "```json\n" + json.dumps(record, indent=2) + "\n```\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
