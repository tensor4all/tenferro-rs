//! The linalg operation registry: one `api_ops!` invocation, the single source
//! of the public linalg grammar (`op`, `op_with`, `op_into` on three layers).
//!
//! The macro generates the traits, the forwarding impls, the rustdoc and the
//! metadata constants; the numerical bodies are the hand-written functions of
//! `crate::api::{concrete, traced, eager}`.

use tenferro_extension_macros::api_ops;

use crate::{
    CholeskyOptions, Eig, Eigh, EighOptions, EigvalshOptions, FullPivLu, HouseholderQr, Lu,
    NormOptions, PinvOptions, Qr, QrOptions, RankRevealingQrOptions, RankRevealingQrResult,
    Slogdet, Svd, SvdOptions, SvdvalsOptions, TriangularSolveOptions,
};

api_ops! {
    family: linalg;

    layer concrete {
        trait_doc: "Linear algebra on anything that implements `AsView`: owned and borrowed tensors, typed tensors and strided views. Results are `Tensor`s for erased inputs and typed tensors for typed inputs. Every operation takes the options last-but-one and the engine last.";
        trait_def: pub trait LinalgOps: tenferro_tensor::AsView;
        impl_def: impl<X: tenferro_tensor::AsView + ?Sized> LinalgOps for X;
        method_where: [Self::Repr: crate::LinalgRepr];
        receiver: self;
        tensor_arg: &impl tenferro_tensor::AsView<Repr = Self::Repr>;
        exec: &mut dyn tenferro_tensor::Engine;
        dest: tenferro_tensor::Dest<<Self::Repr as tenferro_tensor::Repr>::Write<'_>>;
        result: tenferro_tensor::Result;
        types {
            V = <Self::Repr as tenferro_tensor::Repr>::Value,
            R = <Self::Repr as crate::LinalgRepr>::Real,
            C = <Self::Repr as crate::LinalgRepr>::Complex,
            I = <Self::Repr as crate::LinalgRepr>::Index
        }
        forward: crate::api::concrete;
    }
    layer traced {
        trait_doc: "Linear algebra on traced tensors: each call records graph nodes and returns traced results; execution happens when the graph is prepared and run. Shape and dtype problems that are known at record time are reported by the call; the rest surface when the graph is compiled or run.";
        trait_def: pub trait TracedLinalgOps;
        impl_def: impl TracedLinalgOps for tenferro_runtime::TracedTensor;
        method_where: [];
        receiver: self;
        tensor_arg: &tenferro_runtime::TracedTensor;
        result: tenferro_runtime::Result;
        types {
            V = tenferro_runtime::TracedTensor,
            R = tenferro_runtime::TracedTensor,
            C = tenferro_runtime::TracedTensor,
            I = tenferro_runtime::TracedTensor
        }
        forward: crate::api::traced;
    }
    layer eager {
        trait_doc: "Linear algebra on eager tensors, with the session as receiver. Eager operations execute immediately and record the tape for reverse-mode differentiation.";
        trait_def: pub trait EagerLinalgOps;
        impl_def: impl EagerLinalgOps for tenferro_ad::EagerSession<'_>;
        method_where: [];
        receiver: session;
        first_arg: &tenferro_ad::EagerTensor;
        tensor_arg: &tenferro_ad::EagerTensor;
        result: tenferro_ad::Result;
        types {
            V = tenferro_ad::EagerTensor,
            R = tenferro_ad::EagerTensor,
            C = tenferro_ad::EagerTensor,
            I = tenferro_ad::EagerTensor
        }
        forward: crate::api::eager;
        cfg: feature = "autodiff";
    }

    op svd {
        summary: "Singular value decomposition `a = u diag(s) vh`; `SvdShape::Full` returns the full-matrices factors.";
        operands: (a);
        options: SvdOptions;
        output: Svd<V, R>;
        dtypes: "`a`: f32, f64, Complex32 or Complex64. `u` and `vh` have the dtype of `a`; `s` is real.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "A rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. Operands stay on their device; nothing is transferred. In a lane (`CpuEngine::with_lanes`), a negative-stride input is gathered once into the lane's pool; zero steady-state allocation is promised only for non-negative strides. A batched operand is read in place in a lane.";
        errors: [Validation, Unsupported, NumericalFailure(NonConvergence)];
        omissions: ["`SvdShape::Full` combined with a non-default `gauge`, `derivative_eps` or `driver` returns `Unsupported`: the full-matrices route has no gauge or derivative regularization yet", "`SvdShape::Full` has no derivative: differentiating through it is unsupported", "lanes (`CpuEngine::with_lanes`): `SvdShape::Thin`, `SvdShape::Full` and the gauge are available"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps, SvdOptions, SvdShape};
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0])?;
let r = a.svd(&mut engine)?;
let w = a.svd_with(&SvdOptions::default().shape(SvdShape::Full), &mut engine)?;
let rt = &r.s;
let wt = &w.u;
assert_eq!(rt.shape(), &[2]);
let s = rt.as_slice::<f64>()?;
assert!((s[0] - 2.0).abs() < 1e-12 && (s[1] - 1.0).abs() < 1e-12);
// `SvdShape::Full` returns the full-matrices `u`.
assert_eq!(wt.shape(), &[3, 3]);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{TracedLinalgOps, SvdOptions, SvdShape};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![3, 2], vec![1.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0])?;
let r = a.svd()?;
let w = a.svd_with(&SvdOptions::default().shape(SvdShape::Full))?;
let rt = r.s.clone();
let wt = w.u.clone();

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&rt, &wt])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let rt = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
let wt = outputs.next().ok_or("missing second output")?.into_tensor(&mut engine)?;
assert_eq!(rt.shape(), &[2]);
let s = rt.as_slice::<f64>()?;
assert!((s[0] - 2.0).abs() < 1e-12 && (s[1] - 1.0).abs() < 1e-12);
// `SvdShape::Full` returns the full-matrices `u`.
assert_eq!(wt.shape(), &[3, 3]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{EagerLinalgOps, SvdOptions, SvdShape};
use tenferro_tensor::{Tensor};

let runtime = EagerRuntime::new()?;
let (r, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0])?)?;
    let r = s.svd(&a)?;
    let w = s.svd_with(&a, &SvdOptions::default().shape(SvdShape::Full))?;
    Ok::<_, tenferro_ad::Error>((r.s, w.u))
})?;
let rt = r.value()?;
let wt = w.value()?;
assert_eq!(rt.shape(), &[2]);
let s = rt.as_slice::<f64>()?;
assert!((s[0] - 2.0).abs() < 1e-12 && (s[1] - 1.0).abs() < 1e-12);
// `SvdShape::Full` returns the full-matrices `u`.
assert_eq!(wt.shape(), &[3, 3]);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op svdvals {
        summary: "Singular values only, in descending order.";
        operands: (a);
        options: SvdvalsOptions;
        output: R;
        dtypes: "`a`: f32, f64, Complex32 or Complex64. The result is real.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "A rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. Operands stay on their device; nothing is transferred. In a lane (`CpuEngine::with_lanes`), a negative-stride input is gathered once into the lane's pool; zero steady-state allocation is promised only for non-negative strides. A batched operand is read in place in a lane.";
        errors: [Validation, Unsupported, NumericalFailure(NonConvergence)];
        omissions: ["`driver` is a hint for providers that can choose a driver; the current CPU provider ignores it"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps, SvdvalsOptions, SvdDriver};
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.svdvals(&mut engine)?;
let w = a.svdvals_with(&SvdvalsOptions::default().driver(SvdDriver::QrBased), &mut engine)?;
let rt = &r;
let wt = &w;
assert_eq!(rt.shape(), &[2]);
let s = rt.as_slice::<f64>()?;
assert!(s[0] > s[1] && s[1] > 0.0);
// The CPU provider ignores the driver hint.
assert_eq!(wt.as_slice::<f64>()?, s);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{TracedLinalgOps, SvdvalsOptions, SvdDriver};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.svdvals()?;
let w = a.svdvals_with(&SvdvalsOptions::default().driver(SvdDriver::QrBased))?;
let rt = r;
let wt = w;

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&rt, &wt])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let rt = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
let wt = outputs.next().ok_or("missing second output")?.into_tensor(&mut engine)?;
assert_eq!(rt.shape(), &[2]);
let s = rt.as_slice::<f64>()?;
assert!(s[0] > s[1] && s[1] > 0.0);
// The CPU provider ignores the driver hint.
assert_eq!(wt.as_slice::<f64>()?, s);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{EagerLinalgOps, SvdvalsOptions, SvdDriver};
use tenferro_tensor::{Tensor};

let runtime = EagerRuntime::new()?;
let (r, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?)?;
    let r = s.svdvals(&a)?;
    let w = s.svdvals_with(&a, &SvdvalsOptions::default().driver(SvdDriver::QrBased))?;
    Ok::<_, tenferro_ad::Error>((r, w))
})?;
let rt = r.value()?;
let wt = w.value()?;
assert_eq!(rt.shape(), &[2]);
let s = rt.as_slice::<f64>()?;
assert!(s[0] > s[1] && s[1] > 0.0);
// The CPU provider ignores the driver hint.
assert_eq!(wt.as_slice::<f64>()?, s);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op qr {
        summary: "Thin QR factorization `a = q r`.";
        operands: (a);
        options: QrOptions;
        output: Qr<V>;
        dtypes: "`a`: f32, f64, Complex32 or Complex64. `q` and `r` have the dtype of `a`.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "A rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. Operands stay on their device; nothing is transferred. In a lane (`CpuEngine::with_lanes`), a negative-stride input is gathered once into the lane's pool; zero steady-state allocation is promised only for non-negative strides. A batched operand is read in place in a lane.";
        errors: [Validation, Unsupported];
        omissions: ["lanes (`CpuEngine::with_lanes`): only the default `QrOptions` run in a lane; a non-default gauge returns `Unsupported`"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps, QrOptions, QrGauge};
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.qr(&mut engine)?;
let w = a.qr_with(&QrOptions::default().gauge(QrGauge::PositiveDiagonal), &mut engine)?;
let rt = &r.r;
let wt = &w.r;
assert_eq!(rt.shape(), &[2, 2]);
// `PositiveDiagonal` makes the diagonal of `r` positive.
let r = wt.as_slice::<f64>()?;
assert!(r[0] > 0.0 && r[3] > 0.0);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{TracedLinalgOps, QrOptions, QrGauge};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.qr()?;
let w = a.qr_with(&QrOptions::default().gauge(QrGauge::PositiveDiagonal))?;
let rt = r.r.clone();
let wt = w.r.clone();

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&rt, &wt])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let rt = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
let wt = outputs.next().ok_or("missing second output")?.into_tensor(&mut engine)?;
assert_eq!(rt.shape(), &[2, 2]);
// `PositiveDiagonal` makes the diagonal of `r` positive.
let r = wt.as_slice::<f64>()?;
assert!(r[0] > 0.0 && r[3] > 0.0);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{EagerLinalgOps, QrOptions, QrGauge};
use tenferro_tensor::{Tensor};

let runtime = EagerRuntime::new()?;
let (r, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?)?;
    let r = s.qr(&a)?;
    let w = s.qr_with(&a, &QrOptions::default().gauge(QrGauge::PositiveDiagonal))?;
    Ok::<_, tenferro_ad::Error>((r.r, w.r))
})?;
let rt = r.value()?;
let wt = w.value()?;
assert_eq!(rt.shape(), &[2, 2]);
// `PositiveDiagonal` makes the diagonal of `r` positive.
let r = wt.as_slice::<f64>()?;
assert!(r[0] > 0.0 && r[3] > 0.0);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op householder_qr {
        summary: "Compact Householder QR: an opaque factorization state from which `r` and `q_columns` are extracted.";
        operands: (a);
        options: none;
        output: HouseholderQr<V>;
        dtypes: "`a`: f32, f64, Complex32 or Complex64.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "The compact factorization takes a compact operand: an owned tensor is read in place, any other operand (a view or a typed tensor) is first made compact by the backend (same placement and dtype). Operands stay on their device; nothing is transferred.";
        errors: [Validation, Unsupported];
        omissions: ["lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps, QrOptions};
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.householder_qr(&mut engine)?;
let rt = r.r(QrOptions::default(), &mut engine)?;
assert_eq!(rt.shape(), &[2, 2]);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{TracedLinalgOps, QrOptions};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.householder_qr()?;
let rt = r.r(QrOptions::default())?;

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&rt])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let rt = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
assert_eq!(rt.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{EagerLinalgOps, QrOptions};
use tenferro_tensor::{Tensor};

let runtime = EagerRuntime::new()?;
let r = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?)?;
    let r = s.householder_qr(&a)?;
    Ok::<_, tenferro_ad::Error>(r.r(QrOptions::default(), s)?)
})?;
let rt = r.value()?;
assert_eq!(rt.shape(), &[2, 2]);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op rank_revealing_qr {
        summary: "Column-pivoted QR with a numerical rank: `q`, `r`, `column_permutation` and `rank`.";
        operands: (a);
        options: RankRevealingQrOptions;
        output: RankRevealingQrResult<V, I>;
        dtypes: "`a`: f32, f64, Complex32 or Complex64. `q` and `r` have the dtype of `a`; `column_permutation` and `rank` are i64.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "A rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. Operands stay on their device; nothing is transferred.";
        errors: [Validation, Unsupported];
        omissions: ["lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps, RankRevealingQrOptions};
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.rank_revealing_qr(&mut engine)?;
let w = a.rank_revealing_qr_with(&RankRevealingQrOptions::default().rtol(1.0e-10), &mut engine)?;
let rt = &r.rank;
let wt = &w.rank;
// A full-rank matrix has numerical rank 2.
assert_eq!(rt.as_slice::<i64>()?, &[2]);
assert_eq!(wt.as_slice::<i64>()?, &[2]);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{TracedLinalgOps, RankRevealingQrOptions};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.rank_revealing_qr()?;
let w = a.rank_revealing_qr_with(&RankRevealingQrOptions::default().rtol(1.0e-10))?;
let rt = r.rank.clone();
let wt = w.rank.clone();

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&rt, &wt])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let rt = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
let wt = outputs.next().ok_or("missing second output")?.into_tensor(&mut engine)?;
// A full-rank matrix has numerical rank 2.
assert_eq!(rt.as_slice::<i64>()?, &[2]);
assert_eq!(wt.as_slice::<i64>()?, &[2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{EagerLinalgOps, RankRevealingQrOptions};
use tenferro_tensor::{Tensor};

let runtime = EagerRuntime::new()?;
let (r, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?)?;
    let r = s.rank_revealing_qr(&a)?;
    let w = s.rank_revealing_qr_with(&a, &RankRevealingQrOptions::default().rtol(1.0e-10))?;
    Ok::<_, tenferro_ad::Error>((r.rank, w.rank))
})?;
let rt = r.value()?;
let wt = w.value()?;
// A full-rank matrix has numerical rank 2.
assert_eq!(rt.as_slice::<i64>()?, &[2]);
assert_eq!(wt.as_slice::<i64>()?, &[2]);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op lu {
        summary: "Partial-pivot LU `a = p l u`; `p` is a permutation matrix and `parity` is the real sign of the permutation.";
        operands: (a);
        options: none;
        output: Lu<V>;
        dtypes: "`a`: f32, f64, Complex32 or Complex64. `p`, `l`, `u` and `parity` (+1 or -1) have the dtype of `a`.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "A rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. Operands stay on their device; nothing is transferred.";
        errors: [Validation, Unsupported];
        omissions: ["lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps};
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.lu(&mut engine)?;
let rt = &r.l;
assert_eq!(rt.shape(), &[2, 2]);
// `l` is unit lower triangular.
assert_eq!(rt.as_slice::<f64>()?[0], 1.0);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{TracedLinalgOps};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.lu()?;
let rt = r.l.clone();

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&rt])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let rt = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
assert_eq!(rt.shape(), &[2, 2]);
// `l` is unit lower triangular.
assert_eq!(rt.as_slice::<f64>()?[0], 1.0);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{EagerLinalgOps};
use tenferro_tensor::{Tensor};

let runtime = EagerRuntime::new()?;
let r = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?)?;
    let r = s.lu(&a)?;
    Ok::<_, tenferro_ad::Error>(r.l)
})?;
let rt = r.value()?;
assert_eq!(rt.shape(), &[2, 2]);
// `l` is unit lower triangular.
assert_eq!(rt.as_slice::<f64>()?[0], 1.0);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op full_piv_lu {
        summary: "Complete-pivot LU `a = p l u q`; `p` and `q` are permutation matrices and `parity` is the real sign of the permutations.";
        operands: (a);
        options: none;
        output: FullPivLu<V, R>;
        dtypes: "`a`: f32, f64, Complex32 or Complex64. `p`, `l`, `u` and `q` have the dtype of `a`; `parity` (+1 or -1) is real for an owned `Tensor` and has the dtype of `a` on the borrowed route (#2552 Q21).";
        layers: [concrete, traced, eager];
        into: [];
        layout: "A rank-2 matrix with non-negative strides is read in place; a negative stride or a batched input (rank >= 3) is gathered once into a compact buffer of the same placement and dtype. Operands stay on their device; nothing is transferred.";
        errors: [Validation, Unsupported];
        omissions: ["lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`", "typed Complex32/Complex64 input: returns `Unsupported`; the borrowed route returns `parity` in the input dtype while the owned route and the typed result type have it real (#2552 Q21); erased inputs keep the dtype their route returns"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps};
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.full_piv_lu(&mut engine)?;
let rt = &r.u;
assert_eq!(rt.shape(), &[2, 2]);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{TracedLinalgOps};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.full_piv_lu()?;
let rt = r.u.clone();

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&rt])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let rt = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
assert_eq!(rt.shape(), &[2, 2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{EagerLinalgOps};
use tenferro_tensor::{Tensor};

let runtime = EagerRuntime::new()?;
let r = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?)?;
    let r = s.full_piv_lu(&a)?;
    Ok::<_, tenferro_ad::Error>(r.u)
})?;
let rt = r.value()?;
assert_eq!(rt.shape(), &[2, 2]);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op full_piv_lu_solve {
        summary: "Solves `a x = b` through the complete-pivot LU of `a`; a rank-deficient system is rejected.";
        operands: (a, b);
        options: none;
        output: V;
        dtypes: "`a` and `b`: the same dtype, one of f32, f64, Complex32 or Complex64. `x` has that dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "`a` and `b` that are not owned compact tensors (views, typed tensors) are first made compact (same placement and dtype); owned tensors are read in place. Operands stay on their device; nothing is transferred.";
        errors: [Validation, Unsupported, NumericalFailure(Singular)];
        omissions: ["lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps};
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let b = Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0])?;
let r = a.full_piv_lu_solve(&b, &mut engine)?;
let rt = &r;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2, 1]);
// [[4, 2], [2, 3]] x = [1, 2] has the solution [-1/8, 3/4].
let x = rt.as_slice::<f64>()?;
assert!(close(x[0], -0.125) && close(x[1], 0.75));
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{TracedLinalgOps};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let b = trace.from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0])?;
let r = a.full_piv_lu_solve(&b)?;
let rt = r;

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&rt])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let rt = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2, 1]);
// [[4, 2], [2, 3]] x = [1, 2] has the solution [-1/8, 3/4].
let x = rt.as_slice::<f64>()?;
assert!(close(x[0], -0.125) && close(x[1], 0.75));
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{EagerLinalgOps};
use tenferro_tensor::{Tensor};

let runtime = EagerRuntime::new()?;
let r = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?)?;
    let b = s.constant_from(Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0])?)?;
    let r = s.full_piv_lu_solve(&a, &b)?;
    Ok::<_, tenferro_ad::Error>(r)
})?;
let rt = r.value()?;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2, 1]);
// [[4, 2], [2, 3]] x = [1, 2] has the solution [-1/8, 3/4].
let x = rt.as_slice::<f64>()?;
assert!(close(x[0], -0.125) && close(x[1], 0.75));
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op solve {
        summary: "Solves `a x = b` with a partial-pivot LU of `a`; `solve_into` overwrites a destination.";
        operands: (a, b);
        options: none;
        output: V;
        dtypes: "`a` and `b`: the same dtype, one of f32, f64, Complex32 or Complex64. `x` has that dtype.";
        layers: [concrete, traced, eager];
        into: [overwrite];
        layout: "A rank-2 `a` with a rank-1 or rank-2 non-empty `b` is read in place (`b` is gathered straight into the right-hand side); any other shape or layout is gathered once into compact buffers of the same placement and dtype. `solve_into` writes the solution into `dest` directly when `dest` is a column-major matrix of the shape of `b`; otherwise it computes the solution and copies it in. Operands and `dest` must be on the same device; nothing is transferred.";
        errors: [Validation, Unsupported, NumericalFailure(Singular)];
        omissions: ["`solve_into` accepts only `Dest::Overwrite`; `Accumulate` and `Scaled` return `Unsupported` before `dest` is touched", "lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps};
use tenferro_tensor::{Tensor, Dest};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let b = Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0])?;
let r = a.solve(&b, &mut engine)?;
let rt = &r;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2, 1]);
// [[4, 2], [2, 3]] x = [1, 2] has the solution [-1/8, 3/4].
let x = rt.as_slice::<f64>()?;
assert!(close(x[0], -0.125) && close(x[1], 0.75));

// `solve_into` overwrites the destination, whatever it held before.
let mut out = Tensor::from_vec_col_major(vec![2, 1], vec![f64::NAN; 2])?;
a.solve_into(&b, &mut engine, Dest::overwrite(&mut out))?;
assert_eq!(out.as_slice::<f64>()?, rt.as_slice::<f64>()?);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{TracedLinalgOps};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let b = trace.from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0])?;
let r = a.solve(&b)?;
let rt = r;

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&rt])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let rt = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2, 1]);
// [[4, 2], [2, 3]] x = [1, 2] has the solution [-1/8, 3/4].
let x = rt.as_slice::<f64>()?;
assert!(close(x[0], -0.125) && close(x[1], 0.75));
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{EagerLinalgOps};
use tenferro_tensor::{Tensor};

let runtime = EagerRuntime::new()?;
let r = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?)?;
    let b = s.constant_from(Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0])?)?;
    let r = s.solve(&a, &b)?;
    Ok::<_, tenferro_ad::Error>(r)
})?;
let rt = r.value()?;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2, 1]);
// [[4, 2], [2, 3]] x = [1, 2] has the solution [-1/8, 3/4].
let x = rt.as_slice::<f64>()?;
assert!(close(x[0], -0.125) && close(x[1], 0.75));
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op lstsq {
        summary: "Least-squares solution of `a x = b` for a tall or square, full-column-rank `a` (thin QR, then a triangular solve).";
        operands: (a, b);
        options: none;
        output: V;
        dtypes: "`a` and `b`: the same dtype, one of f32, f64, Complex32 or Complex64. `x` has that dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "`a` is read like `qr` reads it (rank-2 with non-negative strides in place, otherwise gathered once into a compact buffer of the same placement and dtype); `b` is read in place by the contraction. Operands stay on their device; nothing is transferred.";
        errors: [Validation, Unsupported];
        omissions: ["an underdetermined (wide, `rows < cols`) system returns a `Validation` error", "lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps};
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0])?;
let b = Tensor::from_vec_col_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0])?;
let r = a.lstsq(&b, &mut engine)?;
let rt = &r;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2, 1]);
// The columns of `a` are the first two unit vectors, so x = [1, 2].
let x = rt.as_slice::<f64>()?;
assert!(close(x[0], 1.0) && close(x[1], 2.0));
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{TracedLinalgOps};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![3, 2], vec![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0])?;
let b = trace.from_vec_col_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0])?;
let r = a.lstsq(&b)?;
let rt = r;

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&rt])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let rt = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2, 1]);
// The columns of `a` are the first two unit vectors, so x = [1, 2].
let x = rt.as_slice::<f64>()?;
assert!(close(x[0], 1.0) && close(x[1], 2.0));
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{EagerLinalgOps};
use tenferro_tensor::{Tensor};

let runtime = EagerRuntime::new()?;
let r = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0])?)?;
    let b = s.constant_from(Tensor::from_vec_col_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0])?)?;
    let r = s.lstsq(&a, &b)?;
    Ok::<_, tenferro_ad::Error>(r)
})?;
let rt = r.value()?;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2, 1]);
// The columns of `a` are the first two unit vectors, so x = [1, 2].
let x = rt.as_slice::<f64>()?;
assert!(close(x[0], 1.0) && close(x[1], 2.0));
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op triangular_solve {
        summary: "Solves a triangular system with `a` (or its transpose) on the chosen side; only the selected triangle of `a` is read.";
        operands: (a, b);
        options: TriangularSolveOptions;
        output: V;
        dtypes: "`a` and `b`: the same dtype, one of f32, f64, Complex32 or Complex64. `x` has that dtype.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "A rank-2 `a` with non-negative strides is read in place and a rank-2 `b` is gathered straight into the right-hand side; any other shape or layout is gathered once into compact buffers of the same placement and dtype. Operands stay on their device; nothing is transferred.";
        errors: [Validation, Unsupported, NumericalFailure(Singular)];
        omissions: ["`MatOp::Adjoint` on a complex matrix returns `Unsupported`; for a real matrix it equals `MatOp::Transpose`", "lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps, TriangularSolveOptions, MatOp};
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 1.0, 0.0, 3.0])?;
let b = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 5.0])?;
let r = a.triangular_solve(&b, &mut engine)?;
let w = a.triangular_solve_with(&b, &TriangularSolveOptions::default().op(MatOp::Transpose), &mut engine)?;
let rt = &r;
let wt = &w;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
// The lower triangle of `a` is [[2, 0], [1, 3]]: solving `l x = b` gives [1, 4/3].
let x = rt.as_slice::<f64>()?;
assert!(close(x[0], 1.0) && close(x[1], 4.0 / 3.0));
// With `MatOp::Transpose` the system is `l^T x = b`: x = [1/6, 5/3].
let y = wt.as_slice::<f64>()?;
assert!(close(y[0], 1.0 / 6.0) && close(y[1], 5.0 / 3.0));
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{TracedLinalgOps, TriangularSolveOptions, MatOp};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![2, 2], vec![2.0_f64, 1.0, 0.0, 3.0])?;
let b = trace.from_vec_col_major(vec![2, 1], vec![2.0_f64, 5.0])?;
let r = a.triangular_solve(&b)?;
let w = a.triangular_solve_with(&b, &TriangularSolveOptions::default().op(MatOp::Transpose))?;
let rt = r;
let wt = w;

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&rt, &wt])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let rt = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
let wt = outputs.next().ok_or("missing second output")?.into_tensor(&mut engine)?;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
// The lower triangle of `a` is [[2, 0], [1, 3]]: solving `l x = b` gives [1, 4/3].
let x = rt.as_slice::<f64>()?;
assert!(close(x[0], 1.0) && close(x[1], 4.0 / 3.0));
// With `MatOp::Transpose` the system is `l^T x = b`: x = [1/6, 5/3].
let y = wt.as_slice::<f64>()?;
assert!(close(y[0], 1.0 / 6.0) && close(y[1], 5.0 / 3.0));
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{EagerLinalgOps, TriangularSolveOptions, MatOp};
use tenferro_tensor::{Tensor};

let runtime = EagerRuntime::new()?;
let (r, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 1.0, 0.0, 3.0])?)?;
    let b = s.constant_from(Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 5.0])?)?;
    let r = s.triangular_solve(&a, &b)?;
    let w = s.triangular_solve_with(&a, &b, &TriangularSolveOptions::default().op(MatOp::Transpose))?;
    Ok::<_, tenferro_ad::Error>((r, w))
})?;
let rt = r.value()?;
let wt = w.value()?;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
// The lower triangle of `a` is [[2, 0], [1, 3]]: solving `l x = b` gives [1, 4/3].
let x = rt.as_slice::<f64>()?;
assert!(close(x[0], 1.0) && close(x[1], 4.0 / 3.0));
// With `MatOp::Transpose` the system is `l^T x = b`: x = [1/6, 5/3].
let y = wt.as_slice::<f64>()?;
assert!(close(y[0], 1.0 / 6.0) && close(y[1], 5.0 / 3.0));
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op cholesky {
        summary: "Lower Cholesky factor `l` of a Hermitian positive-definite matrix, `a = l l^H`.";
        operands: (a);
        options: CholeskyOptions;
        output: V;
        dtypes: "`a`: f32, f64, Complex32 or Complex64. `l` has the dtype of `a`.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "A rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. Operands stay on their device; nothing is transferred.";
        errors: [Validation, Unsupported, NumericalFailure(NonConvergence)];
        omissions: ["`Uplo::Upper` returns `Unsupported`: only the lower factor is computed", "lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps, CholeskyOptions, Uplo};
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.cholesky(&mut engine)?;
let w = a.cholesky_with(&CholeskyOptions::default().uplo(Uplo::Lower), &mut engine)?;
let rt = &r;
let wt = &w;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2, 2]);
// l = [[2, 0], [1, sqrt(2)]].
let l = rt.as_slice::<f64>()?;
assert!(close(l[0], 2.0) && close(l[1], 1.0) && close(l[3], 2.0_f64.sqrt()));
assert_eq!(wt.as_slice::<f64>()?, l);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{TracedLinalgOps, CholeskyOptions, Uplo};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.cholesky()?;
let w = a.cholesky_with(&CholeskyOptions::default().uplo(Uplo::Lower))?;
let rt = r;
let wt = w;

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&rt, &wt])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let rt = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
let wt = outputs.next().ok_or("missing second output")?.into_tensor(&mut engine)?;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2, 2]);
// l = [[2, 0], [1, sqrt(2)]].
let l = rt.as_slice::<f64>()?;
assert!(close(l[0], 2.0) && close(l[1], 1.0) && close(l[3], 2.0_f64.sqrt()));
assert_eq!(wt.as_slice::<f64>()?, l);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{EagerLinalgOps, CholeskyOptions, Uplo};
use tenferro_tensor::{Tensor};

let runtime = EagerRuntime::new()?;
let (r, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?)?;
    let r = s.cholesky(&a)?;
    let w = s.cholesky_with(&a, &CholeskyOptions::default().uplo(Uplo::Lower))?;
    Ok::<_, tenferro_ad::Error>((r, w))
})?;
let rt = r.value()?;
let wt = w.value()?;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2, 2]);
// l = [[2, 0], [1, sqrt(2)]].
let l = rt.as_slice::<f64>()?;
assert!(close(l[0], 2.0) && close(l[1], 1.0) && close(l[3], 2.0_f64.sqrt()));
assert_eq!(wt.as_slice::<f64>()?, l);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op eig {
        summary: "General eigendecomposition `a v = v diag(values)`; eigenvectors have unit 2-norm.";
        operands: (a);
        options: none;
        output: Eig<C>;
        dtypes: "`a`: f32, f64, Complex32 or Complex64. `values` and `vectors` are complex (Complex32 for f32 and Complex32 input, Complex64 otherwise).";
        layers: [concrete, traced, eager];
        into: [];
        layout: "A rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. Operands stay on their device; nothing is transferred.";
        errors: [Validation, Unsupported, NumericalFailure(NonConvergence), NumericalFailure(NonFinite)];
        omissions: ["lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps};
use tenferro_tensor::{Tensor, Complex64};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.eig(&mut engine)?;
let rt = &r.values;
assert_eq!(rt.shape(), &[2]);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{TracedLinalgOps};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};
use tenferro_tensor::{Complex64};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.eig()?;
let rt = r.values.clone();

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&rt])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let rt = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
assert_eq!(rt.shape(), &[2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{EagerLinalgOps};
use tenferro_tensor::{Tensor, Complex64};

let runtime = EagerRuntime::new()?;
let r = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?)?;
    let r = s.eig(&a)?;
    Ok::<_, tenferro_ad::Error>(r.values)
})?;
let rt = r.value()?;
assert_eq!(rt.shape(), &[2]);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op eigvals {
        summary: "Eigenvalues of a general matrix, as complex numbers.";
        operands: (a);
        options: none;
        output: C;
        dtypes: "`a`: f32, f64, Complex32 or Complex64. The result is complex.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "A rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. Operands stay on their device; nothing is transferred. In a lane (`CpuEngine::with_lanes`), a negative-stride input is gathered once into the lane's pool; zero steady-state allocation is promised only for non-negative strides. A batched operand is read in place in a lane.";
        errors: [Validation, Unsupported, NumericalFailure(NonConvergence), NumericalFailure(NonFinite)];
        omissions: [];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps};
use tenferro_tensor::{Tensor, Complex64};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.eigvals(&mut engine)?;
let rt = &r;
assert_eq!(rt.shape(), &[2]);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{TracedLinalgOps};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};
use tenferro_tensor::{Complex64};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.eigvals()?;
let rt = r;

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&rt])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let rt = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
assert_eq!(rt.shape(), &[2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{EagerLinalgOps};
use tenferro_tensor::{Tensor, Complex64};

let runtime = EagerRuntime::new()?;
let r = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?)?;
    let r = s.eigvals(&a)?;
    Ok::<_, tenferro_ad::Error>(r)
})?;
let rt = r.value()?;
assert_eq!(rt.shape(), &[2]);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op eigh {
        summary: "Eigendecomposition of a Hermitian matrix (the lower triangle is read); `values` are real and ascending.";
        operands: (a);
        options: EighOptions;
        output: Eigh<V, R>;
        dtypes: "`a`: f32, f64, Complex32 or Complex64. `vectors` has the dtype of `a`; `values` is real.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "A rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. Operands stay on their device; nothing is transferred. In a lane (`CpuEngine::with_lanes`), a negative-stride input is gathered once into the lane's pool; zero steady-state allocation is promised only for non-negative strides. A batched operand is read in place in a lane.";
        errors: [Validation, Unsupported, NumericalFailure(NonConvergence)];
        omissions: ["`driver` is a hint for providers that can choose a driver; the current CPU provider ignores it"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps, EighOptions, EighGauge};
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.eigh(&mut engine)?;
let w = a.eigh_with(&EighOptions::default().gauge(EighGauge::CanonicalPivot), &mut engine)?;
let rt = &r.values;
let wt = &w.values;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2]);
// The eigenvalues of [[4, 2], [2, 3]] are (7 -+ sqrt(17)) / 2, ascending.
let v = rt.as_slice::<f64>()?;
assert!(close(v[0], (7.0 - 17.0_f64.sqrt()) / 2.0) && close(v[1], (7.0 + 17.0_f64.sqrt()) / 2.0));
assert_eq!(wt.shape(), &[2]);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{TracedLinalgOps, EighOptions, EighGauge};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.eigh()?;
let w = a.eigh_with(&EighOptions::default().gauge(EighGauge::CanonicalPivot))?;
let rt = r.values.clone();
let wt = w.values.clone();

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&rt, &wt])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let rt = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
let wt = outputs.next().ok_or("missing second output")?.into_tensor(&mut engine)?;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2]);
// The eigenvalues of [[4, 2], [2, 3]] are (7 -+ sqrt(17)) / 2, ascending.
let v = rt.as_slice::<f64>()?;
assert!(close(v[0], (7.0 - 17.0_f64.sqrt()) / 2.0) && close(v[1], (7.0 + 17.0_f64.sqrt()) / 2.0));
assert_eq!(wt.shape(), &[2]);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{EagerLinalgOps, EighOptions, EighGauge};
use tenferro_tensor::{Tensor};

let runtime = EagerRuntime::new()?;
let (r, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?)?;
    let r = s.eigh(&a)?;
    let w = s.eigh_with(&a, &EighOptions::default().gauge(EighGauge::CanonicalPivot))?;
    Ok::<_, tenferro_ad::Error>((r.values, w.values))
})?;
let rt = r.value()?;
let wt = w.value()?;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2]);
// The eigenvalues of [[4, 2], [2, 3]] are (7 -+ sqrt(17)) / 2, ascending.
let v = rt.as_slice::<f64>()?;
assert!(close(v[0], (7.0 - 17.0_f64.sqrt()) / 2.0) && close(v[1], (7.0 + 17.0_f64.sqrt()) / 2.0));
assert_eq!(wt.shape(), &[2]);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op eigvalsh {
        summary: "Eigenvalues of a Hermitian matrix (the lower triangle is read), real and ascending.";
        operands: (a);
        options: EigvalshOptions;
        output: R;
        dtypes: "`a`: f32, f64, Complex32 or Complex64. The result is real.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "A rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. Operands stay on their device; nothing is transferred. In a lane (`CpuEngine::with_lanes`), a negative-stride input is gathered once into the lane's pool; zero steady-state allocation is promised only for non-negative strides. A batched operand is read in place in a lane.";
        errors: [Validation, Unsupported, NumericalFailure(NonConvergence)];
        omissions: ["`driver` is a hint for providers that can choose a driver; the current CPU provider ignores it"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps, EigvalshOptions, EighDriver};
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.eigvalsh(&mut engine)?;
let w = a.eigvalsh_with(&EigvalshOptions::default().driver(EighDriver::Jacobi), &mut engine)?;
let rt = &r;
let wt = &w;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2]);
let v = rt.as_slice::<f64>()?;
assert!(close(v[0], (7.0 - 17.0_f64.sqrt()) / 2.0) && close(v[1], (7.0 + 17.0_f64.sqrt()) / 2.0));
// The CPU provider ignores the driver hint.
assert_eq!(wt.as_slice::<f64>()?, v);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{TracedLinalgOps, EigvalshOptions, EighDriver};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.eigvalsh()?;
let w = a.eigvalsh_with(&EigvalshOptions::default().driver(EighDriver::Jacobi))?;
let rt = r;
let wt = w;

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&rt, &wt])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let rt = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
let wt = outputs.next().ok_or("missing second output")?.into_tensor(&mut engine)?;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2]);
let v = rt.as_slice::<f64>()?;
assert!(close(v[0], (7.0 - 17.0_f64.sqrt()) / 2.0) && close(v[1], (7.0 + 17.0_f64.sqrt()) / 2.0));
// The CPU provider ignores the driver hint.
assert_eq!(wt.as_slice::<f64>()?, v);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{EagerLinalgOps, EigvalshOptions, EighDriver};
use tenferro_tensor::{Tensor};

let runtime = EagerRuntime::new()?;
let (r, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?)?;
    let r = s.eigvalsh(&a)?;
    let w = s.eigvalsh_with(&a, &EigvalshOptions::default().driver(EighDriver::Jacobi))?;
    Ok::<_, tenferro_ad::Error>((r, w))
})?;
let rt = r.value()?;
let wt = w.value()?;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2]);
let v = rt.as_slice::<f64>()?;
assert!(close(v[0], (7.0 - 17.0_f64.sqrt()) / 2.0) && close(v[1], (7.0 + 17.0_f64.sqrt()) / 2.0));
// The CPU provider ignores the driver hint.
assert_eq!(wt.as_slice::<f64>()?, v);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op slogdet {
        summary: "Sign and natural log of the absolute determinant of a square matrix.";
        operands: (a);
        options: none;
        output: Slogdet<V, R>;
        dtypes: "`a`: f32, f64, Complex32 or Complex64. `sign` has the dtype of `a`; `log_abs_det` is real.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "A rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. Operands stay on their device; nothing is transferred.";
        errors: [Validation, Unsupported];
        omissions: ["lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps};
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.slogdet(&mut engine)?;
let rt = &r.log_abs_det;
assert!(rt.shape().is_empty());
// det = 4 * 3 - 2 * 2 = 8.
assert!((rt.as_slice::<f64>()?[0] - 8.0_f64.ln()).abs() < 1e-12);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{TracedLinalgOps};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.slogdet()?;
let rt = r.log_abs_det.clone();

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&rt])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let rt = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
assert!(rt.shape().is_empty());
// det = 4 * 3 - 2 * 2 = 8.
assert!((rt.as_slice::<f64>()?[0] - 8.0_f64.ln()).abs() < 1e-12);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{EagerLinalgOps};
use tenferro_tensor::{Tensor};

let runtime = EagerRuntime::new()?;
let r = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?)?;
    let r = s.slogdet(&a)?;
    Ok::<_, tenferro_ad::Error>(r.log_abs_det)
})?;
let rt = r.value()?;
assert!(rt.shape().is_empty());
// det = 4 * 3 - 2 * 2 = 8.
assert!((rt.as_slice::<f64>()?[0] - 8.0_f64.ln()).abs() < 1e-12);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op det {
        summary: "Determinant of a square matrix, computed as `sign * exp(log_abs_det)`.";
        operands: (a);
        options: none;
        output: V;
        dtypes: "`a`: f32, f64, Complex32 or Complex64. The result has the dtype of `a`.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "A rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the computation. Operands stay on their device; nothing is transferred.";
        errors: [Validation, Unsupported];
        omissions: ["lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps};
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.det(&mut engine)?;
let rt = &r;
assert!(rt.shape().is_empty());
assert!((rt.as_slice::<f64>()?[0] - 8.0).abs() < 1e-10);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{TracedLinalgOps};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.det()?;
let rt = r;

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&rt])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let rt = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
assert!(rt.shape().is_empty());
assert!((rt.as_slice::<f64>()?[0] - 8.0).abs() < 1e-10);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{EagerLinalgOps};
use tenferro_tensor::{Tensor};

let runtime = EagerRuntime::new()?;
let r = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?)?;
    let r = s.det(&a)?;
    Ok::<_, tenferro_ad::Error>(r)
})?;
let rt = r.value()?;
assert!(rt.shape().is_empty());
assert!((rt.as_slice::<f64>()?[0] - 8.0).abs() < 1e-10);
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op inv {
        summary: "Inverse of a square matrix (a solve against the identity).";
        operands: (a);
        options: none;
        output: V;
        dtypes: "`a`: f32, f64, Complex32 or Complex64. The result has the dtype of `a`.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "A rank-2 `a` is read in place by the solve (the identity right-hand side is allocated compact); any other shape or layout is gathered once into a compact buffer of the same placement and dtype. Operands stay on their device; nothing is transferred.";
        errors: [Validation, Unsupported, NumericalFailure(Singular)];
        omissions: ["lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps};
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.inv(&mut engine)?;
let rt = &r;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2, 2]);
// inv([[4, 2], [2, 3]]) = [[3, -2], [-2, 4]] / 8.
let m = rt.as_slice::<f64>()?;
assert!(close(m[0], 0.375) && close(m[1], -0.25) && close(m[3], 0.5));
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{TracedLinalgOps};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.inv()?;
let rt = r;

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&rt])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let rt = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2, 2]);
// inv([[4, 2], [2, 3]]) = [[3, -2], [-2, 4]] / 8.
let m = rt.as_slice::<f64>()?;
assert!(close(m[0], 0.375) && close(m[1], -0.25) && close(m[3], 0.5));
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{EagerLinalgOps};
use tenferro_tensor::{Tensor};

let runtime = EagerRuntime::new()?;
let r = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?)?;
    let r = s.inv(&a)?;
    Ok::<_, tenferro_ad::Error>(r)
})?;
let rt = r.value()?;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2, 2]);
// inv([[4, 2], [2, 3]]) = [[3, -2], [-2, 4]] / 8.
let m = rt.as_slice::<f64>()?;
assert!(close(m[0], 0.375) && close(m[1], -0.25) && close(m[3], 0.5));
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op pinv {
        summary: "Moore-Penrose pseudoinverse through the SVD; singular values below `rtol * max(s)` are dropped.";
        operands: (a);
        options: PinvOptions;
        output: V;
        dtypes: "`a`: f32, f64, Complex32 or Complex64. The result has the dtype of `a`.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "A rank-2 matrix with non-negative strides (compact, transposed or any strided view) is read in place; a negative stride, or a batched input (rank >= 3), is gathered once into a compact buffer of the same placement and dtype before the SVD. Operands stay on their device; nothing is transferred.";
        errors: [Validation, Unsupported, NumericalFailure(NonConvergence)];
        omissions: ["lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps, PinvOptions};
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.pinv(&mut engine)?;
let w = a.pinv_with(&PinvOptions::default().rtol(Some(1.0e-10)), &mut engine)?;
let rt = &r;
let wt = &w;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2, 2]);
// The matrix is invertible, so the pseudoinverse is the inverse.
let m = rt.as_slice::<f64>()?;
assert!(close(m[0], 0.375) && close(m[1], -0.25) && close(m[3], 0.5));
assert!(close(wt.as_slice::<f64>()?[0], 0.375));
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{TracedLinalgOps, PinvOptions};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.pinv()?;
let w = a.pinv_with(&PinvOptions::default().rtol(Some(1.0e-10)))?;
let rt = r;
let wt = w;

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&rt, &wt])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let rt = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
let wt = outputs.next().ok_or("missing second output")?.into_tensor(&mut engine)?;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2, 2]);
// The matrix is invertible, so the pseudoinverse is the inverse.
let m = rt.as_slice::<f64>()?;
assert!(close(m[0], 0.375) && close(m[1], -0.25) && close(m[3], 0.5));
assert!(close(wt.as_slice::<f64>()?[0], 0.375));
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{EagerLinalgOps, PinvOptions};
use tenferro_tensor::{Tensor};

let runtime = EagerRuntime::new()?;
let (r, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?)?;
    let r = s.pinv(&a)?;
    let w = s.pinv_with(&a, &PinvOptions::default().rtol(Some(1.0e-10)))?;
    Ok::<_, tenferro_ad::Error>((r, w))
})?;
let rt = r.value()?;
let wt = w.value()?;
let close = |x: f64, y: f64| (x - y).abs() < 1e-9;
assert_eq!(rt.shape(), &[2, 2]);
// The matrix is invertible, so the pseudoinverse is the inverse.
let m = rt.as_slice::<f64>()?;
assert!(close(m[0], 0.375) && close(m[1], -0.25) && close(m[3], 0.5));
assert!(close(wt.as_slice::<f64>()?[0], 0.375));
# Ok::<(), tenferro_ad::Error>(())"#;
    }
    op norm {
        summary: "Vector or matrix norm over the chosen axes, selected by `NormOrd`.";
        operands: (a);
        options: NormOptions;
        output: R;
        dtypes: "`a`: f32, f64, Complex32 or Complex64. The result is real.";
        layers: [concrete, traced, eager];
        into: [];
        layout: "`NormOrd::Frobenius` over all axes of a strided operand is one fused pass that reads it in place; the other orders read the operand through strided reductions, and a matrix norm over axes that are not the leading two first moves them to the front (a layout change of the same placement and dtype). Operands stay on their device; nothing is transferred.";
        errors: [Validation, Unsupported];
        omissions: ["lanes (`CpuEngine::with_lanes`): not available; returns `Unsupported`"];
        example concrete: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps, NormOptions, NormOrd};
use tenferro_tensor::{Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.norm(&mut engine)?;
let w = a.norm_with(&NormOptions::default().ord(NormOrd::Inf), &mut engine)?;
let rt = &r;
let wt = &w;
assert!(rt.shape().is_empty());
// Frobenius norm: sqrt(4^2 + 2^2 + 2^2 + 3^2).
assert!((rt.as_slice::<f64>()?[0] - 33.0_f64.sqrt()).abs() < 1e-12);
// The infinity norm is the largest absolute row sum.
assert!((wt.as_slice::<f64>()?[0] - 6.0).abs() < 1e-12);
# Ok::<(), tenferro_tensor::Error>(())"#;
        example traced: r#"use tenferro_cpu::CpuEngine;
use tenferro_linalg::{TracedLinalgOps, NormOptions, NormOrd};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
let r = a.norm()?;
let w = a.norm_with(&NormOptions::default().ord(NormOrd::Inf))?;
let rt = r;
let wt = w;

// Build the graph, then run it.
let program = GraphCompiler::new().compile_many(&[&rt, &wt])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
let mut outputs = outputs.into_iter();
let rt = outputs.next().ok_or("missing first output")?.into_tensor(&mut engine)?;
let wt = outputs.next().ok_or("missing second output")?.into_tensor(&mut engine)?;
assert!(rt.shape().is_empty());
// Frobenius norm: sqrt(4^2 + 2^2 + 2^2 + 3^2).
assert!((rt.as_slice::<f64>()?[0] - 33.0_f64.sqrt()).abs() < 1e-12);
// The infinity norm is the largest absolute row sum.
assert!((wt.as_slice::<f64>()?[0] - 6.0).abs() < 1e-12);
# Ok::<(), Box<dyn std::error::Error>>(())"#;
        example eager: r#"use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{EagerLinalgOps, NormOptions, NormOrd};
use tenferro_tensor::{Tensor};

let runtime = EagerRuntime::new()?;
let (r, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?)?;
    let r = s.norm(&a)?;
    let w = s.norm_with(&a, &NormOptions::default().ord(NormOrd::Inf))?;
    Ok::<_, tenferro_ad::Error>((r, w))
})?;
let rt = r.value()?;
let wt = w.value()?;
assert!(rt.shape().is_empty());
// Frobenius norm: sqrt(4^2 + 2^2 + 2^2 + 3^2).
assert!((rt.as_slice::<f64>()?[0] - 33.0_f64.sqrt()).abs() < 1e-12);
// The infinity norm is the largest absolute row sum.
assert!((wt.as_slice::<f64>()?[0] - 6.0).abs() < 1e-12);
# Ok::<(), tenferro_ad::Error>(())"#;
    }

    exceptions {
        E08 {
            symbols: [
                "tenferro_linalg::svd_with_exec",
                "tenferro_linalg::svd_values_with_exec",
                "tenferro_linalg::qr_with_exec",
                "tenferro_linalg::eigh_with_exec",
                "tenferro_linalg::eigh_values_with_exec",
                "tenferro_linalg::eig_values_with_exec"
            ];
            scope: "free functions that run a factorization on a borrowed execution bundle and buffer pool";
            category: "legacy | free function";
            owner_module: "tenferro_linalg::with_exec";
            reason: "retired: replaced by CpuEngine::with_lanes, which hands each lane an engine that runs the ordinary grammar (view.svd_with(&opts, lane)); the six entries were added for #2462 to split threads into lanes";
            verification: "retired: the P7 prototype verified lifetime, buffer reuse, nesting and concurrency, and no retention after the call; the lane route keeps the steady-state allocation counts of the former entries";
            exit_condition: "met: the six functions, their module and the Borrowed-Exec section of the crate docs were removed (#2552 S3); lanes and the grammar replace them";
            status: retired;
        }
    }
}
