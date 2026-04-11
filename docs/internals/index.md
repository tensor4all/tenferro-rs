# Internals

The source code is the source of truth for internal design. Development assumes AI agentic coding. [AGENTS.md](https://github.com/tensor4all/tenferro-rs/blob/main/AGENTS.md) is the entry point.

| Topic | Location |
|---|---|
| Op vocabulary | `tenferro-ops/src/std_tensor_op.rs` |
| Backend contract | `tenferro-tensor/src/backend.rs` |
| Execution session | `tenferro-tensor/src/backend.rs` |
| AD rules | `tenferro-ops/src/ad/` |
| Compilation pipeline | `tenferro/src/compiler.rs` |
| Buffer pool | `tenferro-tensor/src/buffer_pool.rs` |
| CPU context | `tenferro-tensor/src/cpu/context.rs` |
| GPU design (future) | `docs/design/exec-session.md` |
