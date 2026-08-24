# macOS CI restoration

Date: 2026-08-24

## Summary

macOS is a supported tenferro platform, but pull requests previously only
cross-compiled one Apple-gated GPU integration target from Linux. Issue #1721
showed that this did not detect macOS-only Apple/Metal runtime failures.

The PR workspace workflow now runs the existing `workspace-faer` profile on
the GitHub-hosted Apple Silicon `macos-15` runner. The macOS job depends on the
existing Linux workspace aggregate gate, so failed Linux revisions do not
consume macOS runner time. RunPod GPU validation may proceed from the same
Linux gate in parallel.

## Decisions

- Reuse `workspace-faer`; do not add a macOS-specific command profile.
- Run full workspace tests and doctests rather than a Metal-only subset because
  macOS is a supported host platform.
- Run the native lane for code and macOS-control-plane changes; use
  `ubuntu-latest` for other CI-only and docs-only no-ops, avoiding unnecessary
  macOS allocation while preserving a stable required check name.
- Remove the Linux cross-target Apple type-check because the native macOS run
  supersedes it.
- Keep the stable Linux aggregate check unchanged so RunPod ordering and
  existing branch protection continue to work.

## Verification

- `python3 -m unittest scripts.ci.tests.test_change_policy scripts.ci.tests.test_workflow_contracts`
- `python3 scripts/ci/run_profile.py ci-config`
- `actionlint .github/workflows/ci.yml .github/workflows/ci-pr-workspace-tests.yml`

The first hosted `macOS workspace tests` run is the native execution
verification and may expose the currently reported macOS failures; those must
be fixed rather than skipped or allowed to fail before the check becomes a
merge requirement.
