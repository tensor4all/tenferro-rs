---
description: Statically audit tenferro-rs code against PERFORMANCE_TIPS.md and report rule violations with file and line, without claiming measured speedups or slowdowns.
---

Use `$ARGUMENTS` as the audit scope: `full` for the whole repository, paths to
restrict the audit, or nothing to audit the current diff against `origin/main`.

Read the performance rules in full and follow their `Audit Procedure` section.
Report each finding as `file:line`, section title, evidence, and remediation
direction. Findings are static rule violations, not measured regressions.

@PERFORMANCE_TIPS.md
