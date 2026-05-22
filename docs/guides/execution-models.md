# Execution Models

tenferro supports direct execution, PyTorch-like eager AD, and JAX-like traced
execution on the same dense tensor stack. The key distinction is when work is
submitted and when the host waits for results.

![Eager CPU, Eager GPU, and Traced execution timelines](../assets/execution-models.svg)

## Eager CPU

Direct CPU tensor operations and eager CPU AD run immediately. A call enters
the CPU backend, the CPU work completes, and the returned `Tensor` is
host-readable.

This is the easiest model for debugging and for ordinary no-AD numeric code.
Use `TypedTensor<T>` or `Tensor` when no gradient state is needed. Use
`EagerTensor` when you want PyTorch-style scalar-loss `backward()`.

## Eager GPU

Eager GPU work is immediate at the Rust API boundary, but it is not a ready
flag API and it does not imply host synchronization after every kernel. An op
submits work to the CUDA backend and returns a CUDA-resident `Tensor` handle.
Subsequent CUDA ops can consume that handle on the same backend stream.

The host waits when values are downloaded or otherwise need host inspection.
Some library-backed operations also synchronize internally when they must read
device-side status.

## Traced Mode

Traced mode records operations into a graph first. It is similar to JAX's
tracing and `jit` workflow: build the expression, compile it, then run the
compiled program through a `GraphExecutor<B>`.

Use traced mode for transform AD (`grad`, `vjp`, `jvp`, HVP), symbolic inputs,
graph optimization, and repeated execution. The executor backend decides
whether the compiled program runs on CPU or CUDA for supported operations.

## Why Support Both?

Eager and traced serve different workflows on the same tensor stack.

| Need | Better fit |
| --- | --- |
| Inspect intermediate values while developing | Eager CPU or eager GPU with explicit download |
| Scalar-loss reverse-mode AD with gradient accumulation | `EagerTensor` |
| Transform AD and higher-order AD | `TracedTensor` |
| Reuse the same computation many times | `GraphCompiler` + `GraphExecutor<B>` |
| Keep no-AD code simple | `TypedTensor<T>` or `Tensor` |
