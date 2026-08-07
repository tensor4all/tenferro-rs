# 2026-08-07 FFT input-storage dispatch (#1579)

## Scope

Fix `CpuExecSession::execute_fft` so a domain-bound CPU session selects the
execution path from the input's actual storage. Host-owned inputs use the
existing host RustFFT path and produce host-owned outputs; backend inputs use
the managed path only when a shared allocation domain is bound. No transfer,
public API, dependency, or fallback behavior was added.

## Context reviewed

- Accepted implementation plan and comments for #1579, plus umbrella #1619.
- `AGENTS.md`, `REPOSITORY_RULES.md`, and the shared common/Rust,
  performance, numerical, and docs/tests rules.
- `docs/design/fft-backend-execution.md`.
- Existing FFT managed tests and the matching Cholesky dispatch/tests from
  #1428.

## Decision

Keep the existing managed executor and `with_managed_read` validation intact.
At the `execute_fft` boundary, check `input.is_backend_buffer()` before using a
session allocation domain. A host input therefore stays on the host path even
when the backend is domain-bound. A foreign or unsupported backend buffer still
enters `with_managed_read`, preserving its typed domain/backend error.

## Verification

- Added RED coverage for host-owned direct FFT and host read FFT on a
  domain-bound backend; both failed with `HostAccessError::Unsupported {
  backend: "host" }` before the dispatch fix.
- Added foreign-domain rejection coverage and retained matching-domain managed
  coverage.
- Targeted host tests and the full `cpu::managed_tests` suite pass after the
  fix; `cargo fmt --all -- --check` passes.

## Residual risks

The Apple/Metal integration tests remain hardware-gated and were not run on
this host. The change is limited to CPU FFT dispatch; no implicit transfer is
introduced.
