# CPU execution entry cost

**Superseded by [#2206](https://github.com/tensor4all/tenferro-rs/issues/2206).**
This guide described the retired scoped-versus-held session entry. Execution now
uses a caller-held engine: the caller builds one Rayon pool, hands it to
`CpuEngine::new` (or uses `CpuEngine::serial()`), and passes `&mut` to each
operation. There is no per-operation session entry to amortize.

- [Execution engine ownership](../design/execution-engine-ownership.md): the
  engine/prepare/run model, pool ownership and the removed mechanisms.
- [Ordinary and prepared execution](./ordinary-and-prepared-execution.md):
  `Runtime::prepare` and `Runtime::run` with caller-supplied engines.
- [CPU backend execution](../design/cpu-backend-execution.md): provider routes
  and the engine's parallelism token.
