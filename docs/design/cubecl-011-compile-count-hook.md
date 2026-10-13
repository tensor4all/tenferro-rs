# Disposition: the compile-count test hook (`compiled_kernel_ids`)

Status: open, blocking one observation in `kernel_specialization_tests.rs`.

## What is needed

`crates/tenferro-gpu/src/cubecl/tests/kernel_specialization_tests.rs` verifies
issue #2010 / #1885 §3.4: layout values (extents, strides, offsets, lengths,
window sizes) reach the CUDA kernels as run-time arguments, so launching one
kernel over several layouts must not compile a second module for that kernel.
The observation was `CudaServer::compiled_kernel_ids()` — the fork's accessor
over `self.ctx.module_names`:

```rust
pub fn compiled_kernel_ids(&self) -> Vec<KernelId> {
    self.ctx.module_names.keys().cloned().collect()
}
```

## What 0.11 has

No public equivalent. Searched the port tree for an enumeration of compiled
kernels or loaded modules (`compiled_kernel`, `kernel_ids`, `module_names`,
`loaded_module`) in `cubecl-runtime`, `cubecl-cuda`, `cubecl-server`: none. The
CUDA context has `is_loaded(&KernelId) -> bool`
(`cubecl-cuda/src/compute/context.rs:150`), but it answers for one id and is not
reachable through the server's public surface, which is what a test needs to
notice a *new* id appearing.

So the migration deletes the counter assertion and keeps the numeric
observation: every layout still launches and its result is compared with the
exact host reference, and the helper documents the missing counter. The
assertion is restored when the hook exists upstream.

## Options

- **Upstream proposal (test-only introspection):** expose an enumeration of the
  compiled kernel ids (or "is this kernel type loaded") on the server, as the
  fork did. Small and test-only; needs the user's approval like every other
  `tracel-ai/*` PR, and is independent of the #27/#28/#29 soundness set.
- **Rework the assertion onto a public proxy.** Not available: the property is
  "the server compiled nothing new", and no public 0.11 API observes
  compilation; tenferro-side `KernelId` equality would not catch a layout value
  becoming comptime, because the id is built from the plan, not the layouts.
