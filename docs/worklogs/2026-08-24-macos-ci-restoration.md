# macOS CI restoration

Date: 2026-08-24

## Summary

macOS is a supported tenferro platform, but pull requests previously only
cross-compiled one Apple-gated GPU integration target from Linux. Issue #1721
showed that this did not detect macOS-only Apple/Metal runtime failures.

The PR workspace workflow now runs the existing `workspace-faer` profile on
the GitHub-hosted Apple Silicon `macos-15` runner. The macOS job depends on the
existing Linux workspace aggregate gate, so failed Linux revisions do not
consume macOS runner time. RunPod GPU validation remains further downstream,
so failed macOS revisions do not consume paid GPU time.

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
- `cargo test -p tenferro-fft --features webgpu --lib`
- `cargo test -p tenferro-linalg --features webgpu managed_cholesky -- --nocapture`
- `cargo test -p tenferro-gpu --features webgpu --lib webgpu::runtime::identity_tests::webgpu_backend_identity_tracks_the_exact_runtime_when_hardware_is_available -- --exact`
- `cargo test -p tenferro-tutorial-code tutorial_binaries_run_successfully -- --exact`
- Initial hosted evidence: [macOS workspace tests](https://github.com/tensor4all/tenferro-rs/actions/runs/32709423761/job/97377574384)
- Independent read-only review: `reviewer-gpt`, verdict `Correct-to-merge` after tracing the CI policy/order and all three blocker-fix paths.
- Deterministic repository-rules review: pass. It warned that the pre-existing 34-line WebGPU identity test module is inline; it remains in the 253-line leaf runtime module because this change only corrects one constructor call and moving the existing tests would be unrelated cleanup.

## Hosted baseline and remediation

The first hosted run executed 3,120 tests on GitHub's Apple Silicon runner:
3,115 passed and five failed. No failure was skipped or allowed to fail.

| Failure | Root cause | Fix |
| --- | --- | --- |
| Managed CPU/Metal FFT panics | Managed provider roots are scalar-independent, while the FFT path and test helpers still called the legacy scalar-typed `buffer()` accessor. | Validate placement/domain from tensor metadata and use guarded host mapping, retaining the legacy buffer path only for existing mock providers. |
| FFT rejection assertions | The CPU rejection test passed a backend handle instead of entering its FFT-capable session; after the panic was fixed, the Metal test also exposed its stale expectation for a foreign device-local domain. | Enter the CPU backend session and assert the typed `HostAccessError::ForeignDomain` reported by both domain boundaries. |
| WebGPU runtime identity test | Availability probed CubeCL's default adapter, but the test then requested discrete GPU ordinal 0; Apple Silicon exposes the supported integrated Metal adapter through default selection. | Construct both identity-test backends with `new_default()`. |
| `core_tensor_snippets` tutorial | The tutorial unconditionally requested explicit managed `AllAllowed` affinity, which is intentionally unsupported on macOS. | Query `supports_placement` and retain the default compatibility context when explicit affinity is unavailable. |

The same scalar-independent managed-root correction was applied to managed CPU
Cholesky and the Apple FFT/Cholesky tutorials because they shared the exact
storage contract and verification path.
