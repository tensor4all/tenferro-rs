//! The eager AD, value-extraction and leaf-construction registry (#2602): the
//! `EagerSession` entry points that need an engine, declared once with their
//! canonical compiled signatures, and the exceptions of that scope (E11–E13).
//!
//! The layer is described, not generated: the methods are inherent
//! `EagerSession` methods in `crate::eager`. `scripts/check-api-registry-conformance.py`
//! checks the compiled public surface (`api/tenferro-ad*.txt`) against the
//! `signature eager:` lines and the exception signatures below.

use tenferro_extension_macros::api_ops;

api_ops! {
    family: eager_ad;

    layer eager { describe_only; }

    op grad {
        summary: "Gradient of a scalar eager output with respect to an eager tensor; returns a composable eager tensor and touches no gradient slot.";
        operands: (output, wrt);
        options: none;
        output: EagerTensor;
        dtypes: "See the method documentation.";
        layers: [eager];
        into: [];
        layout: "Reads the retained values of the operands; the derivative program is prepared and run on the session's engine. Every eager operand and seed is checked against the session's runtime before any value is materialized or any gradient slot is written.";
        errors: [ContextMismatch, NonScalarGrad, Validation, UnsupportedAdRule];
        omissions: ["concrete: not available (no AD); traced: `AdContext::grad`/`vjp`/`jvp` build a graph and take no engine"];
        signature eager: "pub fn tenferro_ad::EagerSession<'_>::grad(&mut self, &tenferro_ad::EagerTensor, &tenferro_ad::EagerTensor) -> tenferro_runtime::error::Result<tenferro_ad::EagerTensor>";
    }
    op vjp {
        summary: "Reverse-mode vector-Jacobian product of eager tensors; touches no gradient slot.";
        operands: (output, wrt, cotangent);
        options: none;
        output: EagerTensor;
        dtypes: "See the method documentation.";
        layers: [eager];
        into: [];
        layout: "Reads the retained values of the operands; the derivative program is prepared and run on the session's engine. Every eager operand and seed is checked against the session's runtime before any value is materialized or any gradient slot is written.";
        errors: [ContextMismatch, Validation, UnsupportedAdRule];
        omissions: ["concrete: not available (no AD); traced: `AdContext::grad`/`vjp`/`jvp` build a graph and take no engine"];
        signature eager: "pub fn tenferro_ad::EagerSession<'_>::vjp(&mut self, &tenferro_ad::EagerTensor, &tenferro_ad::EagerTensor, &tenferro_ad::EagerTensor) -> tenferro_runtime::error::Result<tenferro_ad::EagerTensor>";
    }
    op jvp {
        summary: "Forward-mode Jacobian-vector product of eager tensors.";
        operands: (output, wrt, tangent);
        options: none;
        output: EagerTensor;
        dtypes: "See the method documentation.";
        layers: [eager];
        into: [];
        layout: "Reads the retained values of the operands; the derivative program is prepared and run on the session's engine. Every eager operand and seed is checked against the session's runtime before any value is materialized or any gradient slot is written.";
        errors: [ContextMismatch, Validation, UnsupportedAdRule];
        omissions: ["concrete: not available (no AD); traced: `AdContext::grad`/`vjp`/`jvp` build a graph and take no engine"];
        signature eager: "pub fn tenferro_ad::EagerSession<'_>::jvp(&mut self, &tenferro_ad::EagerTensor, &tenferro_ad::EagerTensor, &tenferro_ad::EagerTensor) -> tenferro_runtime::error::Result<tenferro_ad::EagerTensor>";
    }
    op backward {
        summary: "Reverse-mode AD from a scalar eager output: returns the cotangent map and accumulates into the `grad()` slots of the reachable tracked leaves.";
        operands: (output);
        options: none;
        output: Gradients;
        dtypes: "See the method documentation.";
        layers: [eager];
        into: [];
        layout: "Reads the retained values of the operands; the derivative program is prepared and run on the session's engine. Every eager operand and seed is checked against the session's runtime before any value is materialized or any gradient slot is written.";
        errors: [ContextMismatch, NonScalarGrad, Validation, RuntimeState, UnsupportedAdRule];
        omissions: ["concrete and traced: not available; traced code differentiates with `AdContext::grad` and has no gradient slots"];
        signature eager: "pub fn tenferro_ad::EagerSession<'_>::backward(&mut self, &tenferro_ad::EagerTensor) -> tenferro_runtime::error::Result<tenferro_ad::Gradients>";
    }
    op detach {
        summary: "Detach an eager tensor from the reverse graph: same value and runtime, no gradient flow.";
        operands: (a);
        options: none;
        output: EagerTensor;
        dtypes: "See the method documentation.";
        layers: [eager];
        into: [];
        layout: "Copies the retained value once through the session's engine (`to_contiguous_read`), at the same placement and dtype.";
        errors: [ContextMismatch, TensorRuntime, RuntimeState];
        omissions: ["concrete and traced: not available; cross-runtime import is `duplicate_value` in the source session plus `constant_from` in the target session"];
        signature eager: "pub fn tenferro_ad::EagerSession<'_>::detach(&mut self, &tenferro_ad::EagerTensor) -> tenferro_runtime::error::Result<tenferro_ad::EagerTensor>";
    }
    op duplicate_value {
        summary: "Copy an eager value into an independent owned `Tensor` (the one owned-copy spelling; `EagerTensor::into_value` consumes without copying).";
        operands: (a);
        options: none;
        output: Tensor;
        dtypes: "See the method documentation.";
        layers: [eager];
        into: [];
        layout: "Copies the retained value once through the session's engine (`to_contiguous_read`), at the same placement and dtype.";
        errors: [ContextMismatch, TensorRuntime, RuntimeState];
        omissions: ["concrete and traced: not available"];
        signature eager: "pub fn tenferro_ad::EagerSession<'_>::duplicate_value(&mut self, &tenferro_ad::EagerTensor) -> tenferro_runtime::error::Result<tenferro_tensor::types::Tensor>";
    }
    op constant_from {
        summary: "Import a tensor the engine can read as an untracked eager leaf.";
        operands: (tensor: Tensor);
        options: none;
        output: EagerTensor;
        dtypes: "See the method documentation.";
        layers: [eager];
        into: [];
        layout: "The input is read through the session's engine, so a device-resident or strided tensor the engine can read becomes a leaf without a host round trip.";
        errors: [TensorRuntime, RuntimeState];
        omissions: ["concrete and traced: not available; `EagerRuntime::constant_from` is the engine-free form for owned host tensors (E12)"];
        signature eager: "pub fn tenferro_ad::EagerSession<'_>::constant_from(&mut self, tenferro_tensor::types::Tensor) -> tenferro_runtime::error::Result<tenferro_ad::EagerTensor>";
    }
    op constant_from_host {
        summary: "Upload a host tensor to the session's backend and import it as an untracked eager leaf.";
        operands: (tensor: Tensor);
        options: none;
        output: EagerTensor;
        dtypes: "See the method documentation.";
        layers: [eager];
        into: [];
        layout: "Uploads the host tensor once through the session's engine (`upload_host_tensor`); a host copy on a CPU runtime.";
        errors: [TensorRuntime, RuntimeState];
        omissions: ["concrete and traced: not available"];
        signature eager: "pub fn tenferro_ad::EagerSession<'_>::constant_from_host(&mut self, tenferro_tensor::types::Tensor) -> tenferro_runtime::error::Result<tenferro_ad::EagerTensor>";
    }
    op variable_from {
        summary: "Import a tensor the engine can read as a trainable eager leaf.";
        operands: (tensor: Tensor);
        options: none;
        output: EagerTensor;
        dtypes: "See the method documentation.";
        layers: [eager];
        into: [];
        layout: "The input is read through the session's engine, so a device-resident or strided tensor the engine can read becomes a leaf without a host round trip.";
        errors: [TensorRuntime, RuntimeState];
        omissions: ["concrete and traced: not available; `EagerRuntime::variable_from` is the engine-free form for owned host tensors (E12)"];
        signature eager: "pub fn tenferro_ad::EagerSession<'_>::variable_from(&mut self, tenferro_tensor::types::Tensor) -> tenferro_runtime::error::Result<tenferro_ad::EagerTensor>";
    }

    exceptions {
        E11 {
            symbols: [
                "pub fn tenferro_ad::EagerSession<'_>::grad_optional(&mut self, &tenferro_ad::EagerTensor, &tenferro_ad::EagerTensor) -> tenferro_runtime::error::Result<core::option::Option<tenferro_ad::EagerTensor>>",
                "pub fn tenferro_ad::EagerSession<'_>::vjp_optional(&mut self, &tenferro_ad::EagerTensor, &tenferro_ad::EagerTensor, &tenferro_ad::EagerTensor) -> tenferro_runtime::error::Result<core::option::Option<tenferro_ad::EagerTensor>>",
                "pub fn tenferro_ad::EagerSession<'_>::jvp_optional(&mut self, &tenferro_ad::EagerTensor, &tenferro_ad::EagerTensor, &tenferro_ad::EagerTensor) -> tenferro_runtime::error::Result<core::option::Option<tenferro_ad::EagerTensor>>",
                "pub fn tenferro_ad::EagerSession<'_>::backward_with(&mut self, &tenferro_ad::EagerTensor, &tenferro_ad::EagerTensor) -> tenferro_runtime::error::Result<tenferro_ad::Gradients>"
            ];
            scope: "the inactive-input result form of the eager functional transforms, and the seeded form of eager backward";
            category: "eager AD | result form, seed operand";
            owner_module: "tenferro_ad::eager";
            reason: "`_optional` returns `None` for an inactive input instead of an error, an output mode the grammar does not otherwise spell as a suffix; `backward_with(&output, &cotangent)` takes the seed as an operand, not an options value, so it is not the `_with` options form (#2602)";
            verification: "scripts/check-api-registry-conformance.py; tenferro-ad `eager_session_ad_contract` tests (inactive inputs, seeded backward, foreign seeds)";
            exit_condition: "kept while eager AD distinguishes inactive inputs and seeded backward";
            status: retained;
        }
        E12 {
            symbols: [
                "pub fn tenferro_ad::EagerRuntime::constant_from(&alloc::sync::Arc<Self>, tenferro_tensor::types::Tensor) -> tenferro_runtime::error::Result<tenferro_ad::EagerTensor>",
                "pub fn tenferro_ad::EagerRuntime::variable_from(&alloc::sync::Arc<Self>, tenferro_tensor::types::Tensor) -> tenferro_runtime::error::Result<tenferro_ad::EagerTensor>",
                "pub fn tenferro_ad::EagerTensor::grad(&self) -> tenferro_runtime::error::Result<core::option::Option<tenferro_ad::GradientValue>>",
                "pub fn tenferro_ad::EagerTensor::into_value(self) -> core::result::Result<tenferro_tensor::types::Tensor, tenferro_ad::IntoValueError<Self>>",
                "pub fn tenferro_ad::Gradients::grad(&self, &computegraph::types::ValueKey<tenferro_ops::std_tensor_op::StdTensorOp>) -> core::option::Option<tenferro_tensor::types::TensorView<'_>>",
                "pub fn tenferro_ad::GradientValue::to_tensor(&self, &mut dyn tenferro_tensor::backend::Engine) -> tenferro_runtime::error::Result<tenferro_tensor::types::Tensor>"
            ];
            scope: "engine-free eager leaf import, gradient-slot lookup, consuming value extraction, and the stored-gradient copy";
            category: "eager value | engine-free, concrete record";
            owner_module: "tenferro_ad::eager";
            reason: "the runtime constructors import an owned host tensor without an engine and reject device or view inputs (use the session constructors); `grad()` reads a slot and `into_value` moves the allocation out, so neither needs an engine; a `GradientValue` carries no eager-runtime identity and follows the concrete grammar `recv.op(exec)` (#2602)";
            verification: "scripts/check-api-registry-conformance.py; tenferro-ad leaf-construction and into_value tests";
            exit_condition: "kept";
            status: retained;
        }
        E13 {
            symbols: [
                "pub fn tenferro_ad::EagerRuntime::with_eager_session<T, E: core::convert::From<tenferro_runtime::error::Error>>(&alloc::sync::Arc<Self>, &mut dyn tenferro_runtime::runtime::engine::ExecutionEngine, impl core::ops::function::FnOnce(&mut tenferro_ad::EagerSession<'_>) -> core::result::Result<T, E>) -> core::result::Result<T, E>",
                "pub fn tenferro_ad::EagerSession<'_>::engine(&mut self) -> &mut dyn tenferro_runtime::runtime::engine::ExecutionEngine",
                "pub fn tenferro_ad::extension::apply_eager(&mut dyn tenferro_runtime::runtime::engine::ExecutionEngine, alloc::sync::Arc<dyn tenferro_ops::ext_op::ExtensionOp>, &[&tenferro_ad::EagerTensor]) -> tenferro_runtime::error::Result<alloc::vec::Vec<tenferro_ad::EagerTensor>>"
            ];
            scope: "the eager session boundary, its one engine accessor, and the extension-authoring entry";
            category: "eager boundary | lifecycle, accessor, bridge";
            owner_module: "tenferro_ad::eager, tenferro_ad::extension";
            reason: "`with_eager_session` is where the caller's engine enters; `engine()` lends that engine as `&mut dyn ExecutionEngine` (which is also an `Engine`) for AD-free tensor work inside the boundary; `apply_eager` is the extension-authoring entry that opens its own session for an extension op";
            verification: "scripts/check-api-registry-conformance.py; tenferro-ad session-boundary and extension tests";
            exit_condition: "apply_eager moves to an EagerSession method when the extension-authoring facade is revised";
            status: retained;
        }
    }
}
