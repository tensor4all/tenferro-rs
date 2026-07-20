#!/usr/bin/env python3
"""Classify a three-pair Criterion non-inferiority campaign."""

import argparse
import json
import pathlib
from collections import Counter


THRESHOLD = 0.05


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


def read_change(path):
    with path.open(encoding="utf-8") as source:
        estimate = json.load(source)["mean"]
    confidence = estimate["confidence_interval"]
    return (
        float(confidence["lower_bound"]),
        float(confidence["upper_bound"]),
        float(estimate["point_estimate"]),
    )


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
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument(
        "--no-invert-pair2",
        action="store_true",
        help="pair 2 already has the same orientation as pairs 1 and 3",
    )
    args = parser.parse_args()
    cases = load_campaign(args.root, invert_pair2=not args.no_invert_pair2)
    print(render_markdown(cases))


if __name__ == "__main__":
    main()
