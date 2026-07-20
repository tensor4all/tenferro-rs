#!/usr/bin/env python3
"""Classify a three-pair Criterion non-inferiority campaign."""

import argparse
import hashlib
import json
import math
import pathlib
import re
from collections import Counter


THRESHOLD = 0.05
PROTOCOL_VERSION = 1
PAIR_ORDERS = ("A/B", "B/A", "A/B")
RUN_ROLES = (
    "sentinel_before",
    "target_first",
    "target_second",
    "sentinel_after",
)
THREAD_ENVIRONMENT = {
    "RAYON_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
CRITERION_SETTINGS = {
    "warm_up_seconds": 2,
    "measurement_seconds": 5,
    "sample_size": 100,
    "confidence_level": 0.95,
}


def canonical_cases():
    cases = {}
    operations = (
        ("neg", "neg_f64"),
        ("add", "add_f64"),
        ("reduce", "reduce_sum_f64"),
        ("slice", "slice_f64"),
    )
    for mode in ("lazy", "materialized"):
        for tag, benchmark in operations:
            for size in (1, 8, 64):
                cases[f"{mode}_{tag}_{size}"] = (
                    f"eager_dispatch_baseline/{mode}/{benchmark}/{size}"
                )
        for size in (1, 2):
            cases[f"{mode}_dot_{size}"] = (
                f"eager_dispatch_baseline/{mode}/dot_general_f64/{size}"
            )
    return cases


CANONICAL_CASES = canonical_cases()


def parse_cpu_inventory(value):
    if not isinstance(value, str):
        raise ValueError("campaign.json: invalid allowed CPU inventory")
    cpus = set()
    try:
        for component in value.split(","):
            if "-" in component:
                first, last = (int(item) for item in component.split("-", 1))
                if last < first:
                    raise ValueError
                cpus.update(range(first, last + 1))
            else:
                cpus.add(int(component))
    except ValueError as error:
        raise ValueError("campaign.json: invalid allowed CPU inventory") from error
    return cpus


def invert_interval(lower, upper, point):
    """Invert a B/A relative-change estimate to A/B orientation."""
    if lower <= -1.0 or upper <= -1.0 or point <= -1.0:
        raise ValueError("relative-change values must be greater than -1")
    return (
        1.0 / (1.0 + upper) - 1.0,
        1.0 / (1.0 + lower) - 1.0,
        1.0 / (1.0 + point) - 1.0,
    )


def classify(intervals, threshold=THRESHOLD):
    """Apply the predeclared PASS/FAIL/INCONCLUSIVE rule."""
    if len(intervals) != 3:
        raise ValueError("classification requires exactly three intervals")
    if all(upper <= threshold for _, upper in intervals):
        return "PASS"
    if sum(lower > threshold for lower, _ in intervals) >= 2:
        return "FAIL"
    return "INCONCLUSIVE"


def sentinel_breached(lower, upper, threshold=THRESHOLD):
    """Return whether an A/A interval lies wholly outside the drift band."""
    return lower > threshold or upper < -threshold


def read_change(path):
    with path.open(encoding="utf-8") as source:
        estimate = json.load(source)["mean"]
    confidence = estimate["confidence_interval"]
    return (
        float(confidence["lower_bound"]),
        float(confidence["upper_bound"]),
        float(estimate["point_estimate"]),
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def require_sha256(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"missing campaign artifact: {path}")


def validate_run(
    run,
    role,
    selected_cpu,
    load_limit,
    context,
    expected_binary=None,
    expected_binary_sha=None,
):
    if run.get("role") != role:
        raise ValueError(f"{context}: expected run role {role!r}")
    if run.get("completed") is not True or run.get("exit_status") != 0:
        raise ValueError(f"{context} {role}: run did not complete successfully")
    if expected_binary is not None and run.get("binary") != expected_binary:
        raise ValueError(f"{context} {role}: binary identity mismatch")
    if expected_binary_sha is not None and run.get("binary_sha256") != expected_binary_sha:
        raise ValueError(f"{context} {role}: binary SHA-256 mismatch")
    violations = run.get("monitor_violations")
    if violations != []:
        raise ValueError(f"{context} {role}: monitor violation: {violations!r}")
    if run.get("observed_affinity") != str(selected_cpu):
        raise ValueError(f"{context} {role}: benchmark affinity mismatch")
    for endpoint in ("normalized_load_start", "normalized_load_end"):
        value = run.get(endpoint)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{context} {role}: missing finite normalized load")
        if value < 0.0 or value > load_limit:
            raise ValueError(
                f"{context} {role}: endpoint normalized load {value} exceeds "
                f"limit {load_limit}"
            )


def validate_pair(
    root,
    case,
    pair,
    order,
    entry,
    selected_cpu,
    allowed_count,
    load_limit,
    binary_shas,
):
    context = f"{case}/pair{pair}"
    expected_relative = pathlib.Path(case) / f"pair{pair}" / "validity.json"
    if entry.get("order") != order:
        raise ValueError(f"{context}: campaign order does not match {order}")
    if entry.get("validity") != expected_relative.as_posix():
        raise ValueError(f"{context}: campaign validity path is inconsistent")
    validity_path = root / expected_relative
    require_file(validity_path)
    expected_validity_sha = entry.get("validity_sha256")
    require_sha256(expected_validity_sha, f"{context} validity SHA-256")
    if file_sha256(validity_path) != expected_validity_sha:
        raise ValueError(f"{context}: validity SHA-256 mismatch")

    validity = read_json(validity_path)
    expected_identity = {
        "protocol_version": PROTOCOL_VERSION,
        "case": case,
        "pair": pair,
        "order": order,
        "selected_cpu": selected_cpu,
        "allowed_cpu_count": allowed_count,
        "valid": True,
    }
    for key, expected in expected_identity.items():
        if validity.get(key) != expected:
            raise ValueError(f"{context}: validity field {key!r} is inconsistent")

    runs = validity.get("runs")
    if not isinstance(runs, list) or len(runs) != 4:
        raise ValueError(f"{context}: validity must contain exactly four runs")
    target_identities = (
        ("baseline", "candidate") if order == "A/B" else ("candidate", "baseline")
    )
    run_identities = ("candidate", *target_identities, "candidate")
    for run, role, identity in zip(runs, RUN_ROLES, run_identities):
        validate_run(
            run,
            role,
            selected_cpu,
            load_limit,
            context,
            identity,
            binary_shas[identity],
        )

    pair_dir = validity_path.parent
    artifacts = validity.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"{context}: missing artifact inventory")
    artifact_paths = {}
    for name in ("change-estimates.json", "sentinel-change-estimates.json"):
        path = pair_dir / name
        require_file(path)
        artifact = artifacts.get(name)
        if not isinstance(artifact, dict):
            raise ValueError(f"{context}: missing artifact record for {name}")
        expected_sha = artifact.get("sha256")
        require_sha256(expected_sha, f"{context} {name} SHA-256")
        if file_sha256(path) != expected_sha:
            raise ValueError(f"{context}: {name} SHA-256 mismatch")
        artifact_paths[name] = path

    sentinel = read_change(artifact_paths["sentinel-change-estimates.json"])
    if sentinel_breached(sentinel[0], sentinel[1]):
        raise ValueError(f"{context}: sentinel interval breaches drift band")
    estimate = read_change(artifact_paths["change-estimates.json"])
    return invert_interval(*estimate) if order == "B/A" else estimate


def load_validated_campaign(root):
    manifest_path = root / "campaign.json"
    require_file(manifest_path)
    campaign = read_json(manifest_path)
    if campaign.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("campaign.json: unsupported protocol version")
    if not isinstance(campaign.get("completed_at"), str) or not campaign["completed_at"]:
        raise ValueError("campaign.json: campaign is not marked complete")
    require_sha256(campaign.get("lock_sha256"), "campaign lock SHA-256")
    binaries = campaign.get("binaries")
    if not isinstance(binaries, dict):
        raise ValueError("campaign.json: missing binary inventory")
    binary_shas = {}
    for binary in ("baseline", "candidate"):
        record = binaries.get(binary)
        if not isinstance(record, dict):
            raise ValueError(f"campaign.json: missing {binary} binary record")
        require_sha256(record.get("sha256"), f"campaign {binary} binary SHA-256")
        binary_shas[binary] = record["sha256"]

    selected_cpu = campaign.get("selected_cpu")
    allowed_count = campaign.get("allowed_cpu_count")
    if not isinstance(selected_cpu, int) or selected_cpu < 0:
        raise ValueError("campaign.json: invalid selected CPU")
    if not isinstance(allowed_count, int) or allowed_count <= 0:
        raise ValueError("campaign.json: invalid allowed CPU count")
    allowed_cpus = parse_cpu_inventory(campaign.get("allowed_cpus"))
    if len(allowed_cpus) != allowed_count or selected_cpu not in allowed_cpus:
        raise ValueError("campaign.json: inconsistent allowed CPU inventory")
    load_limit = campaign.get("normalized_load_limit")
    if load_limit != 0.25:
        raise ValueError("campaign.json: normalized load limit must be 0.25")
    if campaign.get("thread_environment") != THREAD_ENVIRONMENT:
        raise ValueError("campaign.json: single-thread environment is inconsistent")
    if campaign.get("orders") != list(PAIR_ORDERS):
        raise ValueError("campaign.json: pair order is inconsistent")
    if campaign.get("criterion") != CRITERION_SETTINGS:
        raise ValueError("campaign.json: Criterion settings are inconsistent")

    case_records = campaign.get("cases")
    if not isinstance(case_records, dict):
        raise ValueError("campaign.json: missing case inventory")
    if set(case_records) != set(CANONICAL_CASES):
        missing = sorted(set(CANONICAL_CASES) - set(case_records))
        extra = sorted(set(case_records) - set(CANONICAL_CASES))
        raise ValueError(
            f"campaign.json: incomplete canonical case inventory; "
            f"missing={missing}, extra={extra}"
        )

    cases = []
    for case in sorted(CANONICAL_CASES):
        record = case_records[case]
        if not isinstance(record, dict) or record.get("benchmark") != CANONICAL_CASES[case]:
            raise ValueError(f"campaign.json: benchmark mismatch for {case}")
        pair_records = record.get("pairs")
        if not isinstance(pair_records, dict) or set(pair_records) != {"1", "2", "3"}:
            raise ValueError(f"campaign.json: incomplete pair inventory for {case}")
        estimates = []
        for pair, order in enumerate(PAIR_ORDERS, start=1):
            entry = pair_records[str(pair)]
            if not isinstance(entry, dict):
                raise ValueError(f"campaign.json: invalid pair record for {case}/pair{pair}")
            estimates.append(
                validate_pair(
                    root,
                    case,
                    pair,
                    order,
                    entry,
                    selected_cpu,
                    allowed_count,
                    load_limit,
                    binary_shas,
                )
            )
        intervals = [(lower, upper) for lower, upper, _ in estimates]
        cases.append((case, estimates, classify(intervals)))
    return cases


def load_campaign(root, invert_pair2):
    cases = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        estimates = []
        for pair in (1, 2, 3):
            path = case_dir / f"pair{pair}" / "change-estimates.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing pair estimate: {path}")
            estimate = read_change(path)
            if pair == 2 and invert_pair2:
                estimate = invert_interval(*estimate)
            estimates.append(estimate)
        intervals = [(lower, upper) for lower, upper, _ in estimates]
        cases.append((case_dir.name, estimates, classify(intervals)))
    if not cases:
        raise ValueError(f"no case directories found below {root}")
    return cases


def format_interval(estimate):
    lower, upper, point = estimate
    return f"{100.0 * lower:+.2f}..{100.0 * upper:+.2f} ({100.0 * point:+.2f})"


def render_markdown(cases):
    lines = [
        "| Case | Pair 1 | Pair 2 | Pair 3 | Class |",
        "|---|---:|---:|---:|---|",
    ]
    for case, estimates, result in cases:
        rendered = [format_interval(estimate) for estimate in estimates]
        lines.append(
            f"| {case} | {rendered[0]} | {rendered[1]} | {rendered[2]} | {result} |"
        )
    counts = Counter(result for _, _, result in cases)
    if counts["FAIL"]:
        campaign = "FAIL"
    elif counts["PASS"] == len(cases):
        campaign = "PASS"
    else:
        campaign = "INCONCLUSIVE"
    lines.extend(
        [
            "",
            (
                f"Summary: {counts['PASS']} PASS / {counts['FAIL']} FAIL / "
                f"{counts['INCONCLUSIVE']} INCONCLUSIVE; campaign={campaign}"
            ),
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path, nargs="?")
    parser.add_argument(
        "--no-invert-pair2",
        action="store_true",
        help="legacy artifacts only: pair 2 already has the A/B orientation",
    )
    parser.add_argument(
        "--legacy-artifacts",
        action="store_true",
        help="classify historical raw estimates without a validity manifest",
    )
    parser.add_argument(
        "--sentinel-change",
        type=pathlib.Path,
        help="validate one A/A Criterion change estimate; exit 2 on drift breach",
    )
    args = parser.parse_args()
    if args.sentinel_change is not None:
        lower, upper, point = read_change(args.sentinel_change)
        status = "INVALID" if sentinel_breached(lower, upper) else "VALID"
        print(f"{status} {format_interval((lower, upper, point))}")
        raise SystemExit(2 if status == "INVALID" else 0)
    if args.root is None:
        parser.error("root is required unless --sentinel-change is used")
    if args.legacy_artifacts:
        cases = load_campaign(args.root, invert_pair2=not args.no_invert_pair2)
    else:
        if args.no_invert_pair2:
            parser.error("--no-invert-pair2 requires --legacy-artifacts")
        cases = load_validated_campaign(args.root)
    print(render_markdown(cases))


if __name__ == "__main__":
    main()
