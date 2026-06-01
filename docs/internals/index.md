# Internals

The source code is the source of truth for internal design. Development assumes AI agentic coding. [AGENTS.md](https://github.com/tensor4all/tenferro-rs/blob/main/AGENTS.md) is the entry point.

## Architecture

Design rationale for each subsystem — *what* each layer does and *why*.

- [Crate structure](../architecture/tenferro-crates.md)
- [Computation graph](../architecture/computegraph.md)
- [Primitive AD traits](../architecture/primitive-ad.md)
- [tidu AD engine](../architecture/tidu.md)
- [End-to-end AD pipeline](../architecture/ad-pipeline.md)

## Specification

Normative specs — trait signatures, op semantics, backend contracts.

- [Primitive catalog](../spec/primitive-catalog.md)
- [Backend contract](../spec/backend-contract.md)
- [AD contract](../spec/ad-contract.md)
- [Optimizer passes](../spec/optimizer-passes.md)
- [Tensor semantics](../spec/tensor-semantics.md)

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
