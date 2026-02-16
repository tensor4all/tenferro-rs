# tenferro: Unified Tensor Backend Overview

This page is a concise overview and stable entry point.
The canonical design source is:

- [tenferro Design](./tenferro_design.md)

Use `tenferro_design.md` for detailed architecture, crate APIs, and future plans.

## Why This Exists

Historically, this repository had both:

- a high-level "unified backend" document
- a detailed per-crate API design document

Those two documents drifted over time.
To keep the design KISS and avoid re-describing the same decisions in two places,
the detailed document is now the single source of truth.

## Current Workspace Snapshot (POC)

- Core crates: `tenferro-device`, `tenferro-algebra`, `tenferro-prims`,
  `tenferro-tensor`, `tenferro-einsum`, `tenferro-linalg`, `tenferro-capi`
- Extension crates: `extension/tenferro-tropical`,
  `extension/tenferro-tropical-capi`
- Extern crates: `extern/chainrules-core`, `extern/chainrules`

For exact APIs and dependencies, see:

- `docs/design/tenferro_design.md`
- workspace `Cargo.toml`

## Cross-References

- [Einsum Internal Design](./tenferro_einsum_internal_design.md)
- [Einsum Algorithm Comparison](./einsum_algorithm_comparison.md)
- [chainrules-core Design](./chainrules_core_design.md)
- [ITensor Ecosystem Analysis](./itensor_ecosystem_analysis.md)
- [libtorch Reference](./libtorch_reference.md)
