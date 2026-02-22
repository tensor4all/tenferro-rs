#!/usr/bin/env python3
"""Check per-file line coverage against thresholds defined in coverage-thresholds.json."""

import json
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    thresholds_path = root / "coverage-thresholds.json"

    with open(thresholds_path) as f:
        config = json.load(f)
    default_threshold = config.get("default", 80)
    file_thresholds = config.get("files", {})
    excluded_files = set(config.get("exclude", []))

    report_only = "--report-only" in sys.argv[1:]
    args = [a for a in sys.argv[1:] if a != "--report-only"]

    if len(args) > 0:
        with open(args[0]) as f:
            cov_data = json.load(f)
    else:
        cov_data = json.load(sys.stdin)

    files = cov_data["data"][0]["files"]
    root_str = str(root) + "/"

    failures = []
    passed = 0
    skipped = 0

    for entry in files:
        abs_path = entry["filename"]
        if abs_path.startswith(root_str):
            rel_path = abs_path[len(root_str) :]
        else:
            rel_path = abs_path

        if rel_path in excluded_files:
            skipped += 1
            continue

        lines = entry["summary"]["lines"]
        percent = lines["percent"]
        threshold = file_thresholds.get(rel_path, default_threshold)

        if percent < threshold:
            failures.append((rel_path, percent, threshold))
        else:
            passed += 1

    total = passed + len(failures)
    print(f"Coverage check: {passed}/{total} files passed (excluded: {skipped})\n")

    if failures:
        print("FAILED files:")
        for path, actual, required in sorted(failures):
            print(f"  {path}: {actual:.1f}% < {required}%")
        print()
        if report_only:
            print("Report-only mode: not failing.")
            sys.exit(0)
        sys.exit(1)
    else:
        print("All files meet their coverage thresholds.")


if __name__ == "__main__":
    main()
