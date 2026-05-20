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

## Implementing An Extension Op

Implement `tenferro::extension::ExtensionOpTrait` for the op payload. The
payload carries operation parameters such as axes, modes, constants, or kernel
configuration. Tensor-valued parameters should usually be normal inputs, not
payload fields.

Extension op payloads do not need process-global registration. Construct
`Arc<dyn ExtensionOpTrait>` and pass it to `tenferro::extension::apply` for
traced tensors or `apply_eager` for eager tensors.

For AD, register a rule separately:

```rust
use std::sync::Arc;
use chainrules_core::ADRuleResult;
use tenferro::extension::{
    register_extension_chain_rule, ExtensionChainRuleTrait, ExtensionOpTrait,
    FruleBuilder, RRuleBuilder,
};

#[derive(Debug)]
struct AddScalarRule;

impl ExtensionChainRuleTrait for AddScalarRule {
    fn family_id(&self) -> &'static str { "my-crate.add_scalar.v1" }

    fn frule(&self, _op: &dyn ExtensionOpTrait, cx: &mut FruleBuilder<'_>) -> ADRuleResult<()> {
        cx.set_output_tangent(0, cx.tangent(0)?)
    }

    fn rrule(&self, _op: &dyn ExtensionOpTrait, cx: &mut RRuleBuilder<'_>) -> ADRuleResult<()> {
        let Some(cot_y) = cx.cotangent(0)? else { return Ok(()); };
        let cot_x = cot_y.clone();
        let cot_a = cx.reduce_sum_all(cot_y, cx.input_rank(0)?)?;
        cx.set_input_cotangent(0, Some(cot_x))?;
        cx.set_input_cotangent(1, Some(cot_a))
    }
}

register_extension_chain_rule(Arc::new(AddScalarRule)).expect("register AD rule");
```

The builder methods intentionally hide `LocalValId`. Use `tangent(i)` and
`cotangent(i)` to read incoming AD values, use helpers such as `add`, `mul`,
`reduce_sum_all`, `emit`, and `apply_extension` to create new values, then set
the desired output or input cotangent.

When porting Julia `frule` / `rrule` code:

- map `NoTangent` / `ZeroTangent` to `None` when the tangent slot is inactive,
- represent scalar parameters as tensor inputs when users need to vary them,
- use `reduce_sum_all` for broadcasted scalar inputs,
- emit only core `StdTensorOp` operations or extension ops whose AD rules are
  registered before a later AD pass reaches them.

The lower-level `ExtensionAdRuleTrait` and `register_extension_rule` remain as
an adapter surface for code that needs direct `FragmentBuilder` / `OpEmitter`
control. New extension authors should start with `ExtensionChainRuleTrait`.
The old `ExtensionFactory` / `register_extension` op-registration API has been
removed; operation payloads are carried directly in the graph.

The detailed trait contract is documented in the internal
[ExtensionOp specification](../spec/extension-op.md). User-facing extension
crates should wrap that machinery in small APIs that look like the equivalent
PyTorch or JAX operation.

## Example: FFT

[`tenferro-fft`](tenferro-fft.md) is the first extension package following this
pattern. It provides Fourier transforms as tensor extension operations while
keeping the core tenferro crate focused on the common dense operation set.
