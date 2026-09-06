---
name: audit-performance
description: Use when reviewing or auditing tenferro-rs code for performance rule violations: before merging changes that touch tensor kernels, graph/compiler planning, caches, GPU kernels, Faer integration, benchmarks, or examples, when a workload is unexpectedly slow, or when scanning the repository for latent performance anti-patterns. Static audit that reports violations of PERFORMANCE_TIPS.md with file and line and never claims a speedup or slowdown without measurement.
---

# Audit Performance

This is a thin launcher; it carries no rule content.

1. Read `PERFORMANCE_TIPS.md` in full. Its `Audit Procedure` section defines
   the scope handling, the `Detect`/`Fix` hints, and the report format.
2. Take the argument as the scope: `full` for the whole repository, otherwise
   the given paths. Without an argument, audit the current diff against
   `origin/main`.
3. Follow the audit procedure and report each finding as `file:line`, the
   `PERFORMANCE_TIPS.md` section title, the evidence, and the remediation
   direction.
4. Findings are static rule violations, not measured regressions. Do not claim
   a speedup or slowdown, and route any proposed optimization through the
   Performance-Gated Experiment Protocol in `PERFORMANCE_TIPS.md`.
