# Design: Restructure docs/design/ into topic-based documents

**Date**: 2026-02-21
**Status**: Approved

## Motivation

The current `docs/design/` has two monolith files (`tenferro_design.md` at 47KB,
`tenferro_einsum_internal_design.md` at 39KB) that overlap heavily on
`TensorPrims<A>`, `PrimDescriptor`, and the crate dependency graph.
`contract-as-core-op.md` contains a newer design (MakeContiguous alternative)
that contradicts the canonical docs. `tenferro_unified_tensor_backend.md` is
a 40-line stub with no real content.

## Target Structure

```
docs/design/
├── README.md                            # Index with one-line descriptions
├── architecture.md                      # Workspace layers, crate graph, device layer
├── tensor-prims.md                      # TensorPrims<A>, PrimDescriptor, plan-based execution
├── einsum.md                            # Public API, N-ary tree, algebra dispatch
├── contraction-pipeline.md              # Binary contraction pipeline, copy elision, MakeContiguous
├── autodiff.md                          # chainrules-core/chainrules split, linalg AD
├── tensor.md                            # Tensor<T>, TensorView, ownership, async
├── algebra.md                           # HasAlgebra, Semiring, tropical extensibility
├── reference/
│   ├── libtorch.md                      # PyTorch/libtorch reference
│   ├── itensor-ecosystem.md             # ITensor Julia ecosystem analysis
│   └── einsum-algorithm-comparison.md   # strided-rs vs omeinsum-rs comparison
└── integrations/
    └── burn.md                          # Burn framework integration
```

## Content Mapping

### README.md (new)
- One-paragraph project overview
- File listing with one-line descriptions
- Cross-references to `docs/plans/`

### architecture.md
Source → target:
- `tenferro_design.md` lines 17-33: Scope section
- `tenferro_design.md` lines 35-116: tenferro-device (LogicalMemorySpace, ComputeDevice, Error)
- `tenferro_einsum_internal_design.md` lines 14-46: Layered Architecture diagram
- `tenferro_einsum_internal_design.md` lines 47-94: Design Rationale
- `tenferro_einsum_internal_design.md` lines 969-1049: Compile-Time vs Runtime, Crate Dependency Graph
- `tenferro_design.md` lines 1147-1181: ITensor insights table, mdarray relationship
- `tenferro_design.md` lines 386-389: No Metal note

### tensor-prims.md
Source → target:
- `tenferro_design.md` lines 153-389: tenferro-prims section (overview, adjoint pairs, key types, trait, CpuBackend, backend matrix, usage examples)
- `tenferro_einsum_internal_design.md` lines 98-294: Operation Categories table, ReduceOp, Plan-Based Execution, Extended Operations, Custom Closures
- `tenferro_einsum_internal_design.md` lines 298-461: Device Layer, CPU Backend, GEMM Backend Selection, CPU Contraction Plan
- Deduplicate: both monoliths have TensorPrims<A> trait and PrimDescriptor — merge into one

### einsum.md
Source → target:
- `tenferro_design.md` lines 565-713: tenferro-einsum public API (Subscripts, ContractionTree, three levels, consuming variants, user examples)
- `tenferro_einsum_internal_design.md` lines 642-808: Public API (with _into variants, Subscripts, ContractionTree, String Notation)
- `tenferro_einsum_internal_design.md` lines 810-914: N-ary Contraction, Binary Contraction Decomposition, Single-Tensor Decomposition
- `tenferro_einsum_internal_design.md` lines 916-965: Algebra Dispatch, Backward Pass
- `tenferro_design.md` lines 1134-1145: einsum variants summary

### contraction-pipeline.md (most significant change)
Source → target:
- `tenferro_einsum_internal_design.md` lines 462-498: CPU Contraction Pipeline (6-step)
- `contract-as-core-op.md`: Full content — problem statement, copy strategies, benchmark evidence
- NEW: Incorporate the `permute_view + MakeContiguous + BatchedGemm` alternative with recommendation (CPU: decomposed; GPU: Contract)

### autodiff.md
Source → target:
- `chainrules_core_design.md`: Full content (two-crate split, design decisions, API entries)
- `tenferro_design.md` lines 726-793: Minimal feature extensions for linalg AD, SVD rrule algorithm structure, dependency change

### tensor.md
Source → target:
- `tenferro_design.md` lines 391-563: tenferro-tensor (Tensor<T>, DataBuffer, constructors, metadata, view ops, data ops)
- `tenferro_design.md` lines 796-918: Async/CompletionEvent (chosen approach, alternatives, CPU applicability)
- `tenferro_design.md` lines 921-1132: Tensor/TensorView split (motivation, API design, ownership examples, Arc comparison, async coherence)

### algebra.md
Source → target:
- `tenferro_design.md` lines 118-151: tenferro-algebra (HasAlgebra, Semiring, Standard)
- `tenferro_design.md` lines 309-358: Tropical backend, User-defined algebra, Backend implementation matrix

### reference/ (move existing files)
- `libtorch_reference.md` → `reference/libtorch.md`
- `itensor_ecosystem_analysis.md` → `reference/itensor-ecosystem.md`
- `einsum_algorithm_comparison.md` → `reference/einsum-algorithm-comparison.md`

### integrations/ (move existing file)
- `burn_integration.md` → `integrations/burn.md`

## Files to Delete
- `tenferro_unified_tensor_backend.md` (dead stub)
- `tenferro_design.md` (split into topic files)
- `tenferro_einsum_internal_design.md` (split into topic files)
- `contract-as-core-op.md` (merged into contraction-pipeline.md)

## Principles
- No content loss — every section from the old files maps to a new file
- Deduplicate overlapping content (TensorPrims trait, PrimDescriptor, crate graph)
- Update stale content (Contract extended op → MakeContiguous alternative for CPU)
- Reference docs kept as-is (only renamed/moved)
