# Nested Cargo Debug Profile Propagation

## Session summary

Independent Cargo workspaces created by the guide dependency checker and trybuild did not inherit tenferro's root `[profile.dev] debug = 0`. Fresh measurements showed 19.07 GiB of allocated output for one guide check plus one focused trybuild contract, dominated by DWARF. A repository-local Cargo environment default and a direct Python subprocess default reduced the same workloads to 6.25 GiB while preserving an explicit `CARGO_PROFILE_DEV_DEBUG=2` override.

## Context reviewed

- Root `Cargo.toml` local and CI profiles.
- `scripts/check-guide-dependency-snippets.py` fixture creation and Cargo invocation.
- `scripts/ci/trybuild-cargo.py` and `scripts/ci/run_profile.py` nested-Cargo handling.
- `scripts/ci/tests/test_build_artifact_contracts.py` existing artifact contracts.
- trybuild entry points in `tenferro-tensor`, `tenferro-ad`, and `tenferro-gpu`.
- Issues #1397, #1399, #1401, and #1402 and the build-artifact worklog from #1397.
- `AGENTS.md`, `CONTRIBUTING.md`, `REPOSITORY_RULES.md`, and the shared tensor4all rules.

## Measurement contract

- Source basis: `origin/main` `036bd9f0833596c4f61e419e779318254028ead3`; the measurement branch differed only by planning documents before implementation.
- Host: Linux 6.8, `x86_64-unknown-linux-gnu`.
- Rust: `rustc 1.97.1 (8bab26f4f 2026-07-14)`; Cargo 1.97.1.
- Four Cargo jobs, empty `RUSTC_WRAPPER`, fresh target directories.
- Sizes are allocated bytes from `du -s --block-size=1`.
- Times are one cold sample per configuration and should be treated as indicative; disk and ELF-section results are deterministic evidence.

Commands:

```bash
rm -rf target/guide-snippet-check
/usr/bin/time -v env CARGO_BUILD_JOBS=4 RUSTC_WRAPPER= \
  python3 scripts/check-guide-dependency-snippets.py

rm -rf /tmp/tenferro-trybuild-{baseline,candidate}
/usr/bin/time -v env CARGO_TARGET_DIR=/tmp/tenferro-trybuild-<stage> \
  CARGO_BUILD_JOBS=4 RUSTC_WRAPPER= \
  cargo test -p tenferro-tensor --test storage_compile_contract \
  storage_ui_compile_contracts -- --exact
```

## Results

| Workload | Baseline allocated | Candidate allocated | Reduction | Baseline time | Candidate time |
|---|---:|---:|---:|---:|---:|
| Four guide dependency fixtures | 14,802,468,864 B (13.79 GiB) | 5,141,012,480 B (4.79 GiB) | 9,661,456,384 B (65.27%) | 146.17 s | 89.83 s |
| Focused storage trybuild | 5,670,899,712 B (5.28 GiB) | 1,573,789,696 B (1.47 GiB) | 4,097,110,016 B (72.25%) | 105.53 s | 85.18 s |
| Combined | 20,473,368,576 B (19.07 GiB) | 6,714,802,176 B (6.25 GiB) | 13,758,566,400 B (67.20%) | — | — |

Maximum RSS decreased from 5,697,524 to 2,759,288 KiB for the guide check and from 2,683,580 to 1,788,636 KiB for focused trybuild.

The baseline guide FFT executable was 1,840,560,280 bytes, including 1,342,075,096 direct `.debug_*` bytes. The baseline `trybuild011` executable contained 345,685,620 direct `.debug_*` bytes. Representative candidate guide and `trybuild011` ELF files contained no `.debug_*` sections.

An additional fresh guide build with `CARGO_PROFILE_DEV_DEBUG=2` produced `with debug_info` artifacts containing `.debug_info`, `.debug_line`, and the other expected DWARF sections. The explicit debugger override therefore remains effective.

Raw local evidence is under `/tmp/tenferro-build-artifact-measurements/`; it is not committed.

## Design decision

Cargo profile tables do not cross workspace roots, but Cargo-configured environment variables are passed to test processes and inherited by nested Cargo. The repository now supplies:

```toml
[env]
CARGO_PROFILE_DEV_DEBUG = { value = "0", force = false }
```

This covers trybuild without mutating process-global environment from Rust test threads. The guide checker may run directly from Python rather than under Cargo, so its fixture subprocess separately uses `setdefault("CARGO_PROFILE_DEV_DEBUG", "0")`. Both paths preserve caller overrides.

## Alternatives rejected

- Extending the BLAS-specific Cargo wrapper: it is not installed for direct local tests or the default faer profile and would enlarge the command-routing surface.
- Mutating the Rust test process environment in each trybuild harness: process-global environment changes can race other test threads and duplicate policy across three entry points.
- Forking or rewriting trybuild-generated manifests: disproportionate ownership and maintenance cost.
- Shared worktree target directories, sccache, or automatic artifact GC: unrelated to the root cause and outside this change.

## Verification

- Build-artifact contract tests specify the exact repository environment default and guide set-if-absent behavior.
- All nine focused Python contracts pass.
- All four guide dependency fixtures compile and run.
- The focused `tenferro-tensor` trybuild contract passes with snapshots unchanged.
- Candidate ELF inspection confirms DWARF removal; explicit `debug=2` inspection confirms debugger opt-in.

Repository-wide formatting, local PR gate, rule review, independent PR review, and hosted CI are completed in the delivery phase.

## Residual risks

- Timings are single cold samples and include warm registry/source and OS caches; the size and section deltas, not timing, are the primary evidence.
- Cargo retains artifacts from older profile/dependency combinations. Existing target directories require explicit cleanup to reclaim historical files; this change only reduces newly generated fixture artifacts.
