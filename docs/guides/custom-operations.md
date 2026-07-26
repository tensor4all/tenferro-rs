# Custom Tensor Operations

tenferro covers common dense tensor workflows through explicit runtime, AD, and
extension crates. When a project needs an operation that is not part of the
standard operation set, an extension crate can define a custom tensor operation
and expose it as a normal Rust API.

Use an extension operation when the implementation needs a specialized kernel,
an external library, or a domain-specific operation that would be awkward or
too slow to express only with existing tensor methods. Prefer ordinary tensor
composition when the operation is small and the built-in ops already describe
it clearly.

## How Extensions Fit

An extension operation is a tensor operation supplied by another crate. The
extension crate owns the public API, validates arguments, and
applies the lower-level extension operation through `tenferro_runtime::extension`.
Extension-family helpers that target foreign tensor types should be exposed as
crate-root extension traits or ordinary crate-root functions, depending on the
operation shape. Do not add public `eager_tensor` or `traced_tensor` module
namespaces. Core AD operations stay on `EagerTensor` and `TracedTensor` as
methods or associated functions.

An extension can participate in the same eager and traced workflows as built-in
tensor operations when it provides the required metadata and execution hooks.
If the extension also registers automatic-differentiation rules, gradients can
flow through it. If it does not, AD reports the operation as unsupported rather
than silently dropping the gradient.

For most users, the expected workflow is to depend on an extension crate, import
its public extension traits when needed, and call its public APIs. Directly
implementing the lower-level extension traits is for authors of those crates.

## What An Extension Crate Provides

An extension crate is responsible for:

- a stable operation descriptor, so graphs can compare and cache it,
- output dtype and shape inference,
- concrete execution for the supported backend and device combinations,
- optional host/reference execution for families that can run without a
  backend-specific kernel,
- optional linearize, linear-transpose, and direct primal-VJP rules for
  automatic differentiation,
- clear errors when a dtype, shape, backend, or AD path is not supported.

## Cache Ownership

Keep operation identity and runtime caches separate.

The `ExtensionOp` value is the operation's graph identity. It participates
in hashing, equality, graph comparison, and AD rule lookup. Put parameters that
change the meaning of the operation there: axes, normalization mode, algorithm
choice, and similar values. Do not hide unbounded plan caches, vendor handles,
or mutable global state inside operation identity.

There is no single process-global owner for arbitrary extension state.
`EagerRuntime`, `GraphCompiler`, and `Runtime` own explicit bounded cache
stores. Extension modules register cache owners with `Runtime`, so retained
plans are tied to the compiler, eager context, or runtime that uses them.

An operation descriptor may hold an `Arc` to such an extension-owned cache object only
when the cache is a performance detail and is not part of semantic equality.
Two extension ops that compare equal must remain interchangeable even if their
caches are empty, warm, or independently owned.

For einsum, `GraphCompiler` owns parse and static-plan caches, while installed
runtime cache owners and `EagerRuntime` own runtime contraction-plan caches.
These caches are bounded and expose clear, entry count, and retained-byte stats
through their owning runtime surfaces.

Avoid hidden process-global or thread-local caches in extension crates. If a
cache lives longer than one call, make the owner explicit and bounded, and give
users a way to clear it and inspect retained entries.

## Implementing An Extension Op

Implement `tenferro_runtime::extension::ExtensionOp` for the operation
descriptor. It carries operation parameters such as axes, modes, constants, or
kernel configuration. Tensor-valued parameters should usually be normal inputs,
not descriptor fields.

Extension operation descriptors do not need process-global registration. Construct
`Arc<dyn ExtensionOp>` and pass it to `tenferro_runtime::extension::apply` for
traced tensors or `apply_eager` for eager tensors.

Forward execution goes through a registered `ExtensionModule`. If the operation
has a simple host/reference implementation, expose it with
`ExtensionOp::host_reference` and provide a host-reference module; otherwise
provide a backend-specific module owned by the extension crate.

`ExtensionModule` registers one or more extension preparation/execution engines
and their planning configs against an immutable runtime snapshot. Use it for
runtime-owned preparation metadata and bounded cache owners, not for hidden
process-global provider state.

For AD, implement the role-specific traits the operation needs:
`tenferro_ad::semantic_extension::SemanticLinearizeRule`,
`SemanticLinearTransposeRule`, and optionally `SemanticPrimalVjpRule`. Put
those rules in a
`tenferro_ad::semantic_extension::SemanticExtensionRuleSet`, and attach that
set with
`tenferro_ad::AdContext::builder().with_semantic_extension_rules(...)`.
Requests expose ordered primal values, tangents/cotangents, active masks, and
provenance; rules emit validated values through `SemanticProgramBuilder`.
The extension crate should expose a small `semantic_ad_rules()` helper that
constructs the fresh rule set its operations need.

When porting Julia `frule` / `rrule` code:

- map `NoTangent` / `ZeroTangent` to `None` when the tangent slot is inactive,
- represent scalar parameters as tensor inputs when users need to vary them,
- use `reduce_sum_all` for broadcasted scalar inputs,
- add only built-in tensor operations or extension ops whose AD rules are
  registered before a later AD pass reaches them.

The lower-level adapter API remains available for specialized extension
authors who need direct graph-builder control. New extension authors should
start with role-specific AD rule traits plus an explicit `SemanticExtensionRuleSet`.
The old `ExtensionFactory` / `register_extension` op-registration API has been
removed; operation descriptors are carried directly in the graph.

The detailed trait contract is documented in the internal
[ExtensionOp specification](../spec/extension-op.md). User-facing extension
crates should wrap that machinery in small APIs that look like the equivalent
PyTorch or JAX operation.

## Examples

[FFT (extension)](tenferro-fft.md) provides Fourier transforms as tensor
extension operations while keeping the runtime and tensor crates focused on the
common dense operation set.

The nested `ext/tropical` crate is a worked numeric-extension example for
non-standard arithmetic. It exposes scalar tropical newtypes, traced
composition helpers, fused binary tropical einsum extension ops, optional
`tropical-gemm` CPU dispatch for matmul-shaped contractions, and optional
traced AD rules for unique-winner tropical einsum paths. See the
[tropical extension tutorial](../tutorials/tropical-extension.md) for the
walkthrough.

The nested `ext/sparse` crate is a worked sparse-representation example. It
stores COO metadata and nonzero values with dense tenferro tensors, emits a
sparse-sparse matmul extension op for traced execution, and registers JVP/VJP
rules for differentiation with respect to the nonzero values. See the
[sparse tensor extension tutorial](../tutorials/sparse-extension.md) for the
walkthrough.
