# Internals

The source code is the source of truth for internal design. Development assumes AI agentic coding. [AGENTS.md](https://github.com/tensor4all/tenferro-rs/blob/main/AGENTS.md) is the entry point.

## Architecture

Design rationale for each subsystem — *what* each layer does and *why*.

- [Crate structure](../architecture/tenferro-crates.md)
- [Computation graph](../architecture/computegraph.md)
- [Primitive AD traits](../architecture/primitive-ad.md)
- [Semantic AD ownership](../architecture/semantic-ad.md)
- [End-to-end AD pipeline](../architecture/ad-pipeline.md)

## Specification

Normative specs — trait signatures, op semantics, backend contracts.

- [Primitive catalog](../spec/primitive-catalog.md)
- [Backend contract](../spec/backend-contract.md)
- [AD contract](../spec/ad-contract.md)
- [Optimizer passes](../spec/optimizer-passes.md)
- [Tensor semantics](../spec/tensor-semantics.md)
- [Public-boundary overhead inventory](public-boundary-overhead-inventory.md)

The overhead inventory records source-path responsibilities, not measured CPU
performance. CUDA/GPU, XLA, and multi-device execution each remain separately
unmeasured follow-ups under [#1758](https://github.com/tensor4all/tenferro-rs/issues/1758).
CPU source inspection or future CPU timings do not establish their support,
synchronization, transfer costs, or performance. Applicable device-specific evidence
is required before the parent program's final acceptance.

## Source Pointers

| Topic | Location |
|---|---|
| Op vocabulary | `tenferro-internal-ops/src/std_tensor_op.rs` |
| Backend contract | `tenferro-tensor/src/backend.rs` |
| Execution session | `tenferro-tensor/src/backend.rs` |
| AD rules | `tenferro-internal-ops/src/ad/` |
| Compilation pipeline | `tenferro-runtime/src/compiler/` |
| Buffer pool | `tenferro-tensor/src/buffer_pool.rs` |
| CPU context | `tenferro-tensor/src/cpu/context.rs` |
| GPU design | `docs/design/gpu-backend-design.md` |
| Public-boundary inventory | `scripts/check-public-boundary-inventory.py`, `docs/internals/public-boundary-overhead-inventory.json`, and benchmark export |
