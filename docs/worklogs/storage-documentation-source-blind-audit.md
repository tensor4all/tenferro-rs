# Source-blind ownership documentation audit

The audit was performed against rendered artifacts, not Rust source links. The
frozen product candidate was `385a04db9a8cf5547784f0d756e9a7065b3d4efc`.

## Rendered inputs

- `target/docs-site/storage-ownership.html` (60,017 bytes)
- `target/docs-site/guides/views-and-slicing.html` (62,583 bytes)
- `target/docs-site/getting-started/core-concepts.html`
- `target/docs-site/guides/devices-and-gpu.html`
- `target/doc/tenferro_runtime/struct.TypedTensor.html`
- `target/doc/tenferro_runtime/struct.TypedTensorView.html`
- `target/doc/tenferro_runtime/index.html`

## Reconstructed user journey

1. The storage page identifies the owner/view/mutable-view capability triad and
   explains that `as_view()`/`as_view_mut()` are metadata-only borrows.
2. The views page demonstrates static-rank shape/stride preservation, exclusive
   mutation, slicing/transpose as descriptor operations, and `duplicate()` as a
   fresh-owner boundary with a pointer-identity assertion.
3. The ownership page names the CUDA, WebGPU, and Apple namespaces and separates
   upload/download from mapping and synchronization.
4. The prepared-access section explains one validation/setup boundary and the
   absence of provider lookup, allocation, synchronization, or full layout
   validation in the element loop.
5. The detached/scoped section explains group retention, pre-admission rejection,
   and completion-unproven diagnostics without promising owner recovery.

A minimal CPU example reconstructed from the rendered snippets was compiled and
run by the release tutorial acceptance command. It created a mutable view,
changed one element, duplicated the read view, compared values, and checked that
`duplicate()` did not alias the original host pointer.

## Commands and results

```text
bash scripts/build_docs_site.sh                         pass
python3 scripts/check-storage-docs.py --include-rendered pass
python3 scripts/check-storage-element-access-docs.py \
  docs/guides/views-and-slicing.md                      pass
python3 scripts/check-docs-site.py                      pass
cargo test -p tenferro-tutorial-code --release \
  tutorial_binaries_run_successfully -- --exact       pass
```

## Findings

- Critical usability gaps: 0
- Important usability gaps: 0
- Minor usability gaps: 0
- Structured hardware limitation: Apple/Metal is unavailable on this Linux
  runner; the hardware matrix records that lane as a skip with its exact
  command and evidence owner. This does not make the documentation claim that
  Metal was locally exercised.
