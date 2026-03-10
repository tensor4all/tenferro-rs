"""Audit published family tolerances against stored cross-oracle residuals."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_ROOT = REPO_ROOT / "cases"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generators.tolerance_audit import audit_case_tree


def main() -> int:
    audits = audit_case_tree(CASES_ROOT)
    flagged = [
        audit
        for audit in audits
        if audit.tighten_rtol
        or audit.tighten_atol
        or audit.tighten_second_order_rtol
        or audit.tighten_second_order_atol
    ]
    if flagged:
        lines = []
        for audit in flagged:
            first_order = (
                f"first_order rtol {audit.current_rtol:g}->{audit.proposed_rtol:g}, "
                f"atol {audit.current_atol:g}->{audit.proposed_atol:g}"
            )
            line = f"{audit.op}/{audit.family}/{audit.dtype}: {first_order}"
            if audit.current_second_order_rtol is not None and audit.current_second_order_atol is not None:
                line += (
                    ", second_order "
                    f"rtol {audit.current_second_order_rtol:g}->{audit.proposed_second_order_rtol:g}, "
                    f"atol {audit.current_second_order_atol:g}->{audit.proposed_second_order_atol:g}"
                )
            lines.append(line)
        raise SystemExit("tolerance audit failed:\n" + "\n".join(lines))
    print(f"audited_family_tolerances={len(audits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
