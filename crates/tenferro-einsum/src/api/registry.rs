//! The einsum operation registry: one `api_ops!` invocation, the single
//! source of the public einsum grammar (`einsum`, `einsum_with`,
//! `einsum_into`, `tensordot`).
//!
//! The macro generates the traits, the forwarding impls, the rustdoc and the
//! metadata constants; the bodies are the hand-written functions of
//! `crate::api::{concrete, traced, eager}`.

use tenferro_extension_macros::api_ops;

use crate::EinsumOptions;

api_ops! {
    family: einsum;

    layer concrete {
        trait_doc: "Einsum on a list of concrete inputs: a slice, array or `Vec` of anything that implements `AsView` (owned, borrowed and typed tensors, strided views) with one representation. The result is a `Tensor` for erased inputs and a typed tensor for typed inputs. The spec comes first, then the options, then the engine.";
        trait_def: pub trait EinsumOps: crate::EinsumInputs;
        impl_def: impl<I: crate::EinsumInputs + ?Sized> EinsumOps for I;
        method_where: [];
        receiver: self;
        tensor_arg: &impl tenferro_tensor::AsView<Repr = Self::Repr>;
        exec: &mut dyn tenferro_tensor::Engine;
        dest: tenferro_tensor::Dest<<Self::Repr as tenferro_tensor::Repr>::Write<'_>>;
        result: tenferro_tensor::Result;
        types { V = <Self::Repr as tenferro_tensor::Repr>::Value }
        forward: crate::api::concrete;
    }
    layer concrete_tensordot {
        trait_doc: "Tensordot of two concrete inputs that implement `AsView` with one representation; the result has that representation.";
        trait_def: pub trait TensordotOps: tenferro_tensor::AsView;
        impl_def: impl<X: tenferro_tensor::AsView + ?Sized> TensordotOps for X;
        method_where: [];
        receiver: self;
        tensor_arg: &impl tenferro_tensor::AsView<Repr = Self::Repr>;
        exec: &mut dyn tenferro_tensor::Engine;
        result: tenferro_tensor::Result;
        types { V = <Self::Repr as tenferro_tensor::Repr>::Value }
        forward: crate::api::concrete;
    }
    layer traced {
        trait_doc: "Einsum recorded on a trace: the trace is the receiver, the inputs are traced tensors of that trace, and the result is a traced tensor. Execution happens when the graph is prepared and run.";
        trait_def: pub trait TracedEinsumOps;
        impl_def: impl TracedEinsumOps for tenferro_runtime::TraceContext;
        method_where: [];
        receiver: context;
        tensor_arg: &tenferro_runtime::TracedTensor;
        result: tenferro_runtime::Result;
        types { V = tenferro_runtime::TracedTensor, S = &[&tenferro_runtime::TracedTensor] }
        forward: crate::api::traced;
    }
    layer traced_tensordot {
        trait_doc: "Tensordot recorded on traced tensors.";
        trait_def: pub trait TracedTensordotOps;
        impl_def: impl TracedTensordotOps for tenferro_runtime::TracedTensor;
        method_where: [];
        receiver: self;
        tensor_arg: &tenferro_runtime::TracedTensor;
        result: tenferro_runtime::Result;
        types { V = tenferro_runtime::TracedTensor }
        forward: crate::api::traced;
    }
    layer eager {
        trait_doc: "Einsum and tensordot on eager tensors, with the session as receiver. Eager operations execute immediately and record the tape for reverse-mode differentiation.";
        trait_def: pub trait EagerEinsumOps;
        impl_def: impl EagerEinsumOps for tenferro_ad::EagerSession<'_>;
        method_where: [];
        receiver: session;
        first_arg: &tenferro_ad::EagerTensor;
        tensor_arg: &tenferro_ad::EagerTensor;
        result: tenferro_ad::Result;
        types { V = tenferro_ad::EagerTensor, S = &[&tenferro_ad::EagerTensor] }
        forward: crate::api::eager;
        cfg: feature = "autodiff";
    }

    op einsum {
        summary: "Einstein summation over N inputs: contract, trace, transpose, reduce and broadcast as the spec says.";
        operands: (inputs: S, spec: &(impl crate::EinsumSpec + ?Sized));
        options: EinsumOptions;
        output: V;
        dtypes: "Every input has the same dtype (any numeric dtype the backend supports); the result has it too.";
        layers: [concrete, traced, eager];
        into: [overwrite, accumulate, scaled];
        layout: "Strided inputs are read in place by the contraction kernels (any strides, including transposed and negative-stride views); an operand that a kernel cannot read is packed once into a compact buffer of the same placement and dtype. A two-input contraction that maps to one dot_general writes straight into the destination. Inputs stay on their device. `_into` checks the destination's dtype and shape before any write; `Accumulate` and `Scaled` read the destination, `Overwrite` and `Scaled` with `beta = 0` never read it.";
        errors: [Validation, Unsupported, RuntimeState];
        omissions: ["`_into` is concrete only: traced and eager results are new values", "lanes (`CpuEngine::with_lanes`): einsum is unsupported in a lane", "`Scaled` needs `alpha` and `beta` of the input dtype"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_einsum::{ConjMask, EinsumOps, EinsumOptions};
use tenferro_tensor::{Conj, Dest, Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let b = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0])?;
let c = [&a, &b].einsum("ij,jk->ik", &mut engine)?;
assert_eq!(c.as_slice::<f64>()?, &[1.0, 2.0, 3.0, 4.0]);

// Options: conjugate the first operand (a no-op for real data).
let opts = EinsumOptions::default().conj(ConjMask::new([Conj::Yes]));
let d = [&a, &b].einsum_with("ij,jk->ik", &opts, &mut engine)?;
assert_eq!(d.as_slice::<f64>()?, c.as_slice::<f64>()?);

// Accumulate into an existing result: out = a b + out.
let mut out = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4])?;
[&a, &b].einsum_into("ij,jk->ik", &EinsumOptions::default(), &mut engine, Dest::accumulate(&mut out))?;
assert_eq!(out.as_slice::<f64>()?, &[2.0, 3.0, 4.0, 5.0]);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use tenferro_cpu::CpuEngine;
use tenferro_einsum::{EinsumOptimize, EinsumOptions, TracedEinsumOps};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let b = trace.from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0])?;
let c = trace.einsum(&[&a, &b], "ij,jk->ik")?;
let d = trace.einsum_with(&[&a, &b, &b], "ij,jk,kl->il", &EinsumOptions::default().optimize(EinsumOptimize::False))?;

let program = GraphCompiler::new().compile_many(&[&c, &d])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_einsum::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
for output in outputs {
    let t = output.into_tensor(&mut engine)?;
    assert_eq!(t.as_slice::<f64>()?, &[1.0, 2.0, 3.0, 4.0]);
}
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{EagerEinsumOps, EinsumOptimize, EinsumOptions};
use tenferro_tensor::Tensor;

let runtime = EagerRuntime::new()?;
let (c, d) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?)?;
    let b = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0])?)?;
    let c = s.einsum(&[&a, &b], "ij,jk->ik")?;
    let opts = EinsumOptions::default().optimize(EinsumOptimize::Path(vec![(0, 1)]));
    let d = s.einsum_with(&[&a, &b], "ij,jk->ik", &opts)?;
    Ok::<_, tenferro_ad::Error>((c, d))
})?;
assert_eq!(c.value()?.as_slice::<f64>()?, &[1.0, 2.0, 3.0, 4.0]);
assert_eq!(d.value()?.as_slice::<f64>()?, &[1.0, 2.0, 3.0, 4.0]);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op tensordot {
        summary: "Contract `a` with `b` over the given axis pairs (or the last/first `n` axes); free axes of `a` come first.";
        operands: (a, b, axes: crate::TensorDotAxes<'_>);
        options: none;
        output: V;
        dtypes: "`a` and `b` have the same dtype; the result has it too.";
        layers: [concrete_tensordot, traced_tensordot, eager];
        into: [];
        layout: "Lowered to one dot_general, which reads strided operands in place; an operand that the provider cannot read is packed once into a compact buffer of the same placement and dtype.";
        errors: [Validation, Unsupported];
        omissions: ["lanes (`CpuEngine::with_lanes`): tensordot is unsupported in a lane"];
        example concrete_tensordot: r#"use tenferro_cpu::CpuEngine;
use tenferro_einsum::{TensorDotAxes, TensordotOps};
use tenferro_tensor::Tensor;

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])?;
let b = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12])?;
let c = a.tensordot(&b, TensorDotAxes::Count(1), &mut engine)?;
assert_eq!(c.shape(), &[2, 4]);
assert_eq!(c.as_slice::<f64>()?, &[3.0; 8]);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced_tensordot: r#"use tenferro_einsum::{TensorDotAxes, TracedTensordotOps};
use tenferro_runtime::TraceContext;

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])?;
let b = trace.from_vec_col_major(vec![3, 4], vec![1.0_f64; 12])?;
let c = a.tensordot(&b, TensorDotAxes::Count(1))?;
assert_eq!(c.rank(), 2);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{EagerEinsumOps, TensorDotAxes};
use tenferro_tensor::Tensor;

let runtime = EagerRuntime::new()?;
let c = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])?)?;
    let b = s.constant_from(Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12])?)?;
    s.tensordot(&a, &b, TensorDotAxes::Count(1))
})?;
assert_eq!(c.shape(), &[2, 4]);
assert_eq!(c.value()?.as_slice::<f64>()?, &[3.0; 8]);
# Ok::<(), tenferro_ad::Error>(())"#;
    }

    exceptions {
        E07 {
            symbols: [
                "tenferro_einsum::TraceContextEinsumExt::einsum",
                "tenferro_einsum::TraceContextEinsumExt::einsum_notation",
                "tenferro_einsum::TraceContextEinsumExt::einsum_subscripts",
                "tenferro_einsum::TraceContextEinsumExt::einsum_with",
                "tenferro_einsum::TraceContextEinsumExt::einsum_notation_with",
                "tenferro_einsum::TraceContextEinsumExt::einsum_subscripts_with"
            ];
            scope: "einsum entries that took and returned TraceValue on a `&mut TraceContext`";
            category: "legacy | trait";
            owner_module: "tenferro_einsum::traced";
            reason: "retired: replaced by TracedEinsumOps for TraceContext (`trace.einsum(&[&t1, &t2], spec)`), which takes and returns TracedTensor (#2552 decision 4)";
            verification: "retired: a temporary old-vs-new test recorded the same values before removal (#2552 S4)";
            exit_condition: "met: the trait was removed in #2552 S4";
            status: retired;
        }
    }
}
