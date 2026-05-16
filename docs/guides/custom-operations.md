# Custom Tensor Operations

tenferro covers common dense tensor workflows in the `tenferro` crate. When a
project needs an operation that is not part of that built-in surface, an
extension crate can define a custom tensor operation and expose it as a normal
Rust API.

Use an extension operation when the implementation needs a specialized kernel,
an external library, or a domain-specific operation that would be awkward or
too slow to express only with existing tensor methods. Prefer ordinary tensor
composition when the operation is small and the built-in ops already describe
it clearly.

## How Extensions Fit

An extension operation is a tensor operation supplied by another crate. The
extension crate owns the public function names, validates arguments, and
registers the operation with tenferro through `tenferro::extension`.

An extension can participate in the same eager and traced workflows as built-in
tensor operations when it provides the required metadata and execution hooks.
If the extension also registers automatic-differentiation rules, gradients can
flow through it. If it does not, AD reports the operation as unsupported rather
than silently dropping the gradient.

For most users, the expected workflow is to depend on an extension crate and
call its public functions. Directly implementing the lower-level extension
traits is for authors of those crates.

## What An Extension Crate Provides

An extension crate is responsible for:

- a stable operation family and payload, so graphs can compare and cache it,
- output dtype and shape inference,
- concrete execution for the supported backend and device combinations,
- optional JVP/VJP rules for automatic differentiation,
- clear errors when a dtype, shape, backend, or AD path is not supported.

The detailed trait contract is documented in the internal
[ExtensionOp specification](../spec/extension-op.md). User-facing extension
crates should wrap that machinery in small APIs that look like the equivalent
PyTorch or JAX operation.

## Example: FFT

[`tenferro-fft`](tenferro-fft.md) is the first extension package following this
pattern. It provides Fourier transforms as tensor extension operations while
keeping the core tenferro crate focused on the common dense operation set.
