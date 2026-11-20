# Public API surface

The public API surface of every published tenferro crate is **generated, never
committed** ([#2663](https://github.com/tensor4all/tenferro-rs/issues/2663)):
`scripts/api-snapshot.sh` compiles it with `cargo public-api -ss` (blanket and
auto-trait impls omitted) into a temporary directory and runs
`scripts/check-api-registry-conformance.py --surface-dir <tmp>` on it
([#2602](https://github.com/tensor4all/tenferro-rs/issues/2602)). The
`api_ops!` registries decide whether the compiled surface is allowed, so an
undeclared eager entry point, a duplicate alias or an engine argument outside
the `EagerSession` boundary fails; the generated directory is kept and its path
printed when the check fails. The same script and check run in the
`api-surface` job of `.github/workflows/ci-pr-workspace-tests.yml`, which the
required `CI gate (PR workspace tests)` depends on.

**The surface is review evidence, not a proof of behaviour.** It shows which
names and signatures exist. Layout handling across adapters, aliasing,
ownership, destination updates and numerical results are checked by tests.

## Crate and feature matrix

The combinations are fixed in `scripts/api-snapshot.sh`; change the script and
this table together. Each combination is one generated profile file in the
temporary directory.

| Profile | Crate | Features |
|---|---|---|
| `<crate>.txt` | every crate below | default features |
| `tenferro-internal-ops+all.txt` | `tenferro-internal-ops` | `autodiff,cuda` |
| `tenferro-gpu+cuda.txt` | `tenferro-gpu` | `cuda` |
| `tenferro-gpu+metal.txt` | `tenferro-gpu` | `metal` |
| `tenferro-xla+pjrt.txt` | `tenferro-xla` | `pjrt` |
| `tenferro-einsum+autodiff-cuda.txt` | `tenferro-einsum` | `autodiff,cuda` |
| `tenferro-ad+cuda.txt` | `tenferro-ad` | `cuda` |
| `tenferro-linalg+autodiff-cuda.txt` | `tenferro-linalg` | `autodiff,cuda` |
| `tenferro-fft+autodiff-cuda-fftw.txt` | `tenferro-fft` | `autodiff,cuda,fftw` |

The second combination of a crate is its docs.rs feature set. `tenferro-runtime`
declares no features at all, and `tenferro-cpu` only `ledger` and `route-report` —
off by default, gating `#[doc(hidden)]` host-check hooks that `cargo public-api`
does not document — so both have one profile and their docs.rs build documents
the same surface as the default one. Add a row here when a crate gains a feature
that changes a *documented* item.

`allowlist.txt` is not a generated profile: it lists, symbol by symbol, the
approved external types and owner-scoped bridges of exception E06 (see the core
registry, `crates/tenferro-runtime/src/api/registry.rs`).

Not compiled:

- `publish = false` packages (`docs/tutorial-code`, the `ext/*` proof and
  consumer crates, and the standalone `ext/sparse` and `ext/tropical`
  manifests).
- `tenferro-internal-extension-macros`: a proc-macro crate, for which rustdoc
  emits no JSON.

## Running locally

```bash
scripts/api-snapshot.sh   # compile the matrix into a temporary directory and
                           # run the registry conformance check on it
```

The tool needs `cargo-public-api` and a nightly toolchain for rustdoc JSON. CI
uses the nightly that the `api-surface` job installs (pinned in
`.github/workflows/ci-pr-workspace-tests.yml`); if a nightly update changes
the rendering without an API change, re-pin both places at once
(`API_SNAPSHOT_TOOLCHAIN` overrides the pin for a local try).
