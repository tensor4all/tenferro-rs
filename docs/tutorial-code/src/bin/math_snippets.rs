//! Compiled documentation snippets for issue #1609.

// INVARIANT: Independent documentation examples intentionally leave some imports,
// variables, and helper mains unused when compiled as one family binary.
#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    unused_mut,
    clippy::needless_borrow
)]

#[rustfmt::skip]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    snippet_linear_algebra_1()?;

    // snippet source: docs/guides/linear-algebra.md:101
    fn snippet_linear_algebra_1() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:linear_algebra_1
use tenferro_cpu::CpuBackend;
use tenferro_runtime::BackendSessionHost;
use tenferro_linalg::TensorLinalgExt;
use tenferro_runtime::Tensor;

let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0])?;
let b = Tensor::from_vec_col_major(vec![2, 1], vec![8.0_f64, 27.0])?;
let mut backend = CpuBackend::new();
let x = backend.with_backend_session(|session| a.solve(&b, session))?;

assert_eq!(x.shape(), &[2, 1]);
assert_eq!(x.as_slice::<f64>()?, &[2.0, 3.0]);
        // snippet-end:linear_algebra_1
        Ok(())
    }

    snippet_linear_algebra_2()?;

    // snippet source: docs/guides/linear-algebra.md:118
    fn snippet_linear_algebra_2() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:linear_algebra_2
use tenferro_cpu::CpuBackend;
use tenferro_runtime::BackendSessionHost;
use tenferro_linalg::TensorLinalgExt;
use tenferro_runtime::{Tensor, TensorSessionOpsExt};

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> Result<f64, tenferro_tensor::Error> {
    let lhs = lhs.as_slice::<f64>()?;
    let rhs = rhs.as_slice::<f64>()?;
    Ok(lhs
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max))
}
let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 1.0, 1.0, 3.0])?;
let mut backend = CpuBackend::new();
let (factor, reconstructed) = backend.with_backend_session(|session| -> tenferro_tensor::Result<(Tensor, Tensor)> {
    let factor = a.cholesky(session)?;
    let factor_t = factor.transpose(&[1, 0], session)?;
    let reconstructed = factor.matmul(&factor_t, session)?;
    Ok((factor, reconstructed))
})?;

assert_eq!(factor.shape(), &[2, 2]);
assert_eq!(a.shape(), &[2, 2]);
assert!(max_abs_diff(&reconstructed, &a)? < 1.0e-12);
        // snippet-end:linear_algebra_2
        Ok(())
    }

    snippet_linear_algebra_3()?;

    // snippet source: docs/guides/linear-algebra.md:157
    fn snippet_linear_algebra_3() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:linear_algebra_3
use tenferro_ad::AdContext;

let ad = AdContext::builder()
    .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules()?)?
    .build()?;
        // snippet-end:linear_algebra_3
        Ok(())
    }

    snippet_linear_algebra_4()?;

    // snippet source: docs/guides/linear-algebra.md:169
    fn snippet_linear_algebra_4() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:linear_algebra_4
use tenferro_cpu::CpuBackend;
use tenferro_runtime::BackendSessionHost;
use tenferro_linalg::TensorLinalgExt;
use tenferro_runtime::{Tensor, TensorSessionOpsExt};

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> Result<f64, tenferro_tensor::Error> {
    let lhs = lhs.as_slice::<f64>()?;
    let rhs = rhs.as_slice::<f64>()?;
    Ok(lhs
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max))
}
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])?;
let mut backend = CpuBackend::new();
let (u, s, vt) = backend.with_backend_session(|session| a.svd(session))?;

assert_eq!(u.shape(), &[2, 2]);
assert_eq!(vt.shape(), &[2, 2]);

let s_values = s.as_slice::<f64>()?;
let sigma = Tensor::from_vec_col_major(
    vec![2, 2],
    vec![s_values[0], 0.0, 0.0, s_values[1]],
)?;
let (reconstructed,) = backend.with_backend_session(|session| -> tenferro_tensor::Result<(Tensor,)> {
    let us = u.matmul(&sigma, session)?;
    let reconstructed = us.matmul(&vt, session)?;
    Ok((reconstructed,))
})?;

assert_eq!(a.shape(), &[2, 2]);
assert!(max_abs_diff(&reconstructed, &a)? < 1.0e-12);
        // snippet-end:linear_algebra_4
        Ok(())
    }

    snippet_linear_algebra_5()?;

    // snippet source: docs/guides/linear-algebra.md:210
    fn snippet_linear_algebra_5() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:linear_algebra_5
use tenferro_linalg::{SvdGauge, SvdOptions, TracedTensorLinalgExt};
use tenferro_runtime::TracedTensor;

let a = TracedTensor::from_vec_col_major(
    vec![3, 3],
    vec![
        3.0_f64, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 1.0,
    ],
)?;
let (u, s, vt) = a
    .svd_with_options(
        SvdOptions::default()
            .gauge(SvdGauge::CanonicalPivot)
            .derivative_eps(1.0e-10),
    )?;

let rank = 2;
let u_rank2 = u.slice_axis(1, 0..rank)?;
let s_rank2 = s.slice_axis(0, 0..rank)?;
let vt_rank2 = vt.slice_axis(0, 0..rank)?;

assert_eq!(u_rank2.concrete_shape()?, vec![3, 2]);
assert_eq!(s_rank2.concrete_shape()?, vec![2]);
assert_eq!(vt_rank2.concrete_shape()?, vec![2, 3]);
        // snippet-end:linear_algebra_5
        Ok(())
    }

    snippet_linear_algebra_6()?;

    // snippet source: docs/guides/linear-algebra.md:244
    fn snippet_linear_algebra_6() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:linear_algebra_6
use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_runtime::TracedTensor;

let a = TracedTensor::from_vec_col_major(
    vec![3, 3],
    vec![
        3.0_f64, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 1.0,
    ],
)?;
let (_u, s, _vt) = a.svd()?;
let repeated = s.take_axis(0, &[0, 1, 0])?;

assert_eq!(repeated.concrete_shape()?, vec![3]);
        // snippet-end:linear_algebra_6
        Ok(())
    }

    snippet_linear_algebra_7()?;

    // snippet source: docs/guides/linear-algebra.md:265
    fn snippet_linear_algebra_7() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:linear_algebra_7
use tenferro_cpu::CpuBackend;
use tenferro_runtime::BackendSessionHost;
use tenferro_linalg::{QrGauge, QrOptions, TensorLinalgExt};
use tenferro_runtime::{Tensor, TensorSessionOpsExt};

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> Result<f64, tenferro_tensor::Error> {
    let lhs = lhs.as_slice::<f64>()?;
    let rhs = rhs.as_slice::<f64>()?;
    Ok(lhs
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max))
}
let a = Tensor::from_vec_col_major(
    vec![4, 3],
    vec![
        1.0_f64, 4.0, 7.0, 2.0,
        2.0, 5.0, 8.0, 3.0,
        3.0, 6.0, 10.0, 5.0,
    ],
)?;
let mut backend = CpuBackend::new();
let (q, r) = backend.with_backend_session(|session| {
    a.qr_with_options(
        QrOptions::default().gauge(QrGauge::PositiveDiagonal),
        session,
    )
})?;

let identity = Tensor::from_vec_col_major(
    vec![3, 3],
    vec![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
)?;
let (reconstructed, qtq) = backend.with_backend_session(|session| -> tenferro_tensor::Result<(Tensor, Tensor)> {
    let reconstructed = q.matmul(&r, session)?;
    let qt = q.transpose(&[1, 0], session)?;
    let qtq = qt.matmul(&q, session)?;
    Ok((reconstructed, qtq))
})?;

assert_eq!(q.shape(), &[4, 3]);
assert_eq!(r.shape(), &[3, 3]);
assert_eq!(identity.shape(), &[3, 3]);
assert!(max_abs_diff(&reconstructed, &a)? < 1.0e-12);
assert!(max_abs_diff(&qtq, &identity)? < 1.0e-12);
        // snippet-end:linear_algebra_7
        Ok(())
    }

    snippet_linear_algebra_8()?;

    // snippet source: docs/guides/linear-algebra.md:312
    fn snippet_linear_algebra_8() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:linear_algebra_8
use tenferro_cpu::CpuBackend;
use tenferro_runtime::BackendSessionHost;
use tenferro_linalg::TensorLinalgExt;
use tenferro_runtime::{Tensor, TensorSessionOpsExt};

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> Result<f64, tenferro_tensor::Error> {
    let lhs = lhs.as_slice::<f64>()?;
    let rhs = rhs.as_slice::<f64>()?;
    Ok(lhs
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max))
}
let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 1.0, 1.0, 2.0])?;
let mut backend = CpuBackend::new();
let (values, vectors) = backend.with_backend_session(|session| a.eigh(session))?;

assert_eq!(values.shape(), &[2]);
assert_eq!(vectors.shape(), &[2, 2]);

let value_slice = values.as_slice::<f64>()?;
let diagonal = Tensor::from_vec_col_major(
    vec![2, 2],
    vec![value_slice[0], 0.0, 0.0, value_slice[1]],
)?;
let (reconstructed,) = backend.with_backend_session(|session| -> tenferro_tensor::Result<(Tensor,)> {
    let vd = vectors.matmul(&diagonal, session)?;
    let vt = vectors.transpose(&[1, 0], session)?;
    let reconstructed = vd.matmul(&vt, session)?;
    Ok((reconstructed,))
})?;

assert_eq!(a.shape(), &[2, 2]);
assert!(max_abs_diff(&reconstructed, &a)? < 1.0e-12);
        // snippet-end:linear_algebra_8
        Ok(())
    }

    snippet_linear_algebra_9()?;

    // snippet source: docs/guides/linear-algebra.md:347
    fn snippet_linear_algebra_9() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:linear_algebra_9
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend};
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
use tenferro_linalg::TracedTensorLinalgExt;

let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0])?;
let factor = a.cholesky()?;

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&factor)?;
let backend = CpuBackend::new();
let mut builder = Runtime::builder();
builder
    .register_engine(runtime_engine_registration(&backend)?)?;
builder
    .install_extension_module(
        tenferro_linalg::extension_module::<CpuBackend>(runtime_engine_id()?)?,
    )?;
let runtime = builder.build()?;
let mut outputs = runtime.run_compiled(&program, &[])?;
let result = outputs.remove(0);

assert_eq!(result.shape(), &[2, 2]);
assert_eq!(result.as_slice::<f64>()?, &[2.0, 0.0, 0.0, 3.0]);
        // snippet-end:linear_algebra_9
        Ok(())
    }

    snippet_linear_algebra_10()?;

    // snippet source: docs/guides/linear-algebra.md:377
    fn snippet_linear_algebra_10() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:linear_algebra_10
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend};
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
use tenferro_linalg::TracedTensorLinalgExt;

let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0])?;
let b = TracedTensor::from_vec_col_major(vec![2, 1], vec![8.0_f64, 27.0])?;
let x = a.solve(&b)?;

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&x)?;
let backend = CpuBackend::new();
let mut builder = Runtime::builder();
builder
    .register_engine(runtime_engine_registration(&backend)?)?;
builder
    .install_extension_module(
        tenferro_linalg::extension_module::<CpuBackend>(runtime_engine_id()?)?,
    )?;
let runtime = builder.build()?;
let mut outputs = runtime.run_compiled(&program, &[])?;
let result = outputs.remove(0);

assert_eq!(result.shape(), &[2, 1]);
assert_eq!(result.as_slice::<f64>()?, &[2.0, 3.0]);
        // snippet-end:linear_algebra_10
        Ok(())
    }

    snippet_linear_algebra_11()?;

    // snippet source: docs/guides/linear-algebra.md:415
    fn snippet_linear_algebra_11() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:linear_algebra_11
use tenferro_cpu::CpuBackend;
use tenferro_runtime::BackendSessionHost;
use tenferro_linalg::TensorLinalgExt;
use tenferro_runtime::{Tensor, TensorSessionOpsExt};

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> Result<f64, tenferro_tensor::Error> {
    let lhs = lhs.as_slice::<f64>()?;
    let rhs = rhs.as_slice::<f64>()?;
    Ok(lhs
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max))
}
let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(
    vec![4, 4],
    vec![
        0.0_f64, 4.0, 7.0, 1.0,
        2.0, 5.0, 8.0, 0.0,
        3.0, 6.0, 10.0, 2.0,
        1.0, 2.0, 3.0, 4.0,
    ],
)?;
let b = Tensor::from_vec_col_major(vec![4, 1], vec![1.0_f64, 2.0, 3.0, 4.0])?;

let (p, l, u, q, parity) = backend.with_backend_session(|session| a.full_piv_lu(session))?;
let (reconstructed,) = backend.with_backend_session(|session| -> tenferro_tensor::Result<(Tensor,)> {
    let pt = p.transpose(&[1, 0], session)?;
    let pt_l = pt.matmul(&l, session)?;
    let pt_lu = pt_l.matmul(&u, session)?;
    let reconstructed = pt_lu.matmul(&q, session)?;
    Ok((reconstructed,))
})?;
let x = backend.with_backend_session(|session| a.full_piv_lu_solve(&b, session))?;

assert_eq!(p.shape(), &[4, 4]);
assert_eq!(a.shape(), &[4, 4]);
assert!(max_abs_diff(&reconstructed, &a)? < 1.0e-12);
assert_eq!(parity.shape(), &[] as &[usize]);
let parity_value = parity
    .as_slice::<f64>()?
    .first()
    .copied()
    .ok_or_else(|| tenferro_tensor::Error::Internal("missing LU parity value".to_owned()))?;
assert!(parity_value == 1.0 || parity_value == -1.0);
assert_eq!(x.shape(), &[4, 1]);
        // snippet-end:linear_algebra_11
        Ok(())
    }

    snippet_einsum_12()?;

    // snippet source: docs/guides/einsum.md:64
    fn snippet_einsum_12() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:einsum_12
use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_einsum::{
    TensorEinsumExt, TensorEinsumIntoExt, TypedTensorEinsumExt,
    TypedTensorEinsumIntoExt, TypedTensorReadEinsumIntoExt,
};
use tenferro_tensor::{
    BackendSessionHost, Tensor, TensorWrite, TypedTensor, TypedTensorView,
    TypedTensorViewMut, TypedTensorWrite,
};

let lhs = Tensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
let rhs = Tensor::from_vec_col_major(
    vec![3, 2],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
let mut backend = CpuBackend::new();
let product = backend.with_backend_session(|session| {
    [&lhs, &rhs].einsum("ij,jk->ik", session)
})?;
assert_eq!(product.as_slice::<f64>()?, &[22.0, 28.0, 49.0, 64.0]);

let mut product_out = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64; 4])?;
backend.with_backend_session(|session| {
    [&lhs, &rhs].einsum_into(
        "ij,jk->ik",
        session,
        TensorWrite::from_tensor(&mut product_out),
    )
})?;
assert_eq!(product_out.as_slice::<f64>()?, &[22.0, 28.0, 49.0, 64.0]);

let complex_lhs = TypedTensor::<Complex64>::from_vec_col_major(
    vec![2, 2],
    vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -1.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 2.0),
    ],
)?;
let complex_rhs = TypedTensor::<Complex64>::from_vec_col_major(
    vec![2, 1],
    vec![Complex64::new(5.0, 0.0), Complex64::new(6.0, -1.0)],
)?;
let complex = backend.with_backend_session(|session| {
    [&complex_lhs, &complex_rhs].einsum("ij,jk->ik", session)
})?;
assert_eq!(
    complex.as_slice()?,
    &[Complex64::new(23.0, 2.0), Complex64::new(36.0, 3.0)],
);

let borrowed = TypedTensorView::from_slice([2, 2], [1, 2], 0, complex_lhs.as_slice()?)?;
let borrowed_rhs = complex_rhs.as_view();
let mut borrowed_storage = [Complex64::new(0.0, 0.0); 4];
let borrowed_out =
    TypedTensorViewMut::from_slice([2, 1], [2, 4], 1, &mut borrowed_storage)?;
backend.with_backend_session(|session| {
    [borrowed, borrowed_rhs].einsum_read_into(
        "ij,jk->ik",
        session,
        TypedTensorWrite::from_view(borrowed_out),
    )
})?;
assert_eq!(
    [borrowed_storage[1], borrowed_storage[3]],
    [Complex64::new(23.0, 2.0), Complex64::new(36.0, 3.0)],
);
        // snippet-end:einsum_12
        Ok(())
    }

    snippet_einsum_13()?;

    // snippet source: docs/guides/einsum.md:146
    fn snippet_einsum_13() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:einsum_13
use tenferro_cpu::CpuBackend;
use tenferro_einsum::{ConcreteEinsumPlan, TensorReadEinsumExt, TensorReadEinsumIntoExt};
use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead, TensorView, TensorWrite, TypedTensorView};

let matrix_data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
let matrix = TypedTensorView::from_slice([2, 3], [3, 1], 0, &matrix_data)?;
let vector = Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0])?;
let inputs = [
    TensorRead::from_view(TensorView::F64(matrix)),
    TensorRead::from_tensor(&vector),
];

let mut backend = CpuBackend::new();
let result = backend.with_backend_session(|session| {
    inputs.einsum_read("ij,j->i", session)
})?;
assert_eq!(result.as_slice::<f64>()?, &[140.0, 320.0]);

let plan = ConcreteEinsumPlan::prepare_read(inputs.clone(), "ij,j->i")?;
let planned = backend
    .with_backend_session(|session| plan.execute_read(inputs, session))?;
assert_eq!(planned.as_slice::<f64>()?, &[140.0, 320.0]);

let mut planned_out = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2])?;
let matrix = TypedTensorView::from_slice([2, 3], [3, 1], 0, &matrix_data)?;
let inputs = [
    TensorRead::from_view(TensorView::F64(matrix)),
    TensorRead::from_tensor(&vector),
];
backend.with_backend_session(|session| {
    plan.execute_read_into(
        inputs,
        session,
        TensorWrite::from_tensor(&mut planned_out),
    )
})?;
assert_eq!(planned_out.as_slice::<f64>()?, &[140.0, 320.0]);
        // snippet-end:einsum_13
        Ok(())
    }

    snippet_einsum_14()?;

    // snippet source: docs/guides/einsum.md:186
    fn snippet_einsum_14() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:einsum_14
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend};
use tenferro_einsum::TraceContextEinsumExt;
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{GraphCompiler, Runtime, Tensor, TraceContext};

let a = Tensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
let b = Tensor::from_vec_col_major(
    vec![3, 2],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;

let mut trace = TraceContext::new();
let a_value = trace.input(ProgramInputSpec::new(a.dtype(), [2.into(), 3.into()]))?;
let b_value = trace.input(ProgramInputSpec::new(b.dtype(), [3.into(), 2.into()]))?;
let c = trace.einsum(&[a_value, b_value], "ij,jk->ik")?;
let graph = trace.finish(&[c])?;
let program = GraphCompiler::new().compile_traced_graph(&graph)?;

let backend = CpuBackend::new();
let mut builder = Runtime::builder();
builder.register_engine(runtime_engine_registration(&backend)?)?;
builder.install_extension_module(tenferro_einsum::extension_module::<CpuBackend>(
    runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let mut outputs = runtime.run_compiled(&program, &[&a, &b])?;
let result = outputs.remove(0);

assert_eq!(result.shape(), &[2, 2]);
assert_eq!(result.as_slice::<f64>()?, &[22.0, 28.0, 49.0, 64.0]);
        // snippet-end:einsum_14
        Ok(())
    }

    snippet_einsum_15()?;

    // snippet source: docs/guides/einsum.md:230
    fn snippet_einsum_15() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:einsum_15
use tenferro_ad::{EagerRuntime, Tensor};
use tenferro_einsum::EagerEinsumExt;

let ctx = EagerRuntime::new()?;
let u = ctx.variable_from(Tensor::from_vec_col_major(
    vec![2],
    vec![1.0_f64, 2.0],
)?)?;
let v = ctx.variable_from(Tensor::from_vec_col_major(
    vec![3],
    vec![3.0_f64, 4.0, 5.0],
)?)?;

let outer = [&u, &v].einsum("i,j->ij")?;
let diag = [&v].einsum("i->ii")?;

assert_eq!(outer.shape(), &[2, 3]);
let outer_tensor = outer.to_tensor()?;
assert_eq!(
    outer_tensor.as_slice::<f64>()?,
    &[3.0, 6.0, 4.0, 8.0, 5.0, 10.0],
);
assert_eq!(diag.shape(), &[3, 3]);
let diag_tensor = diag.to_tensor()?;
assert_eq!(
    diag_tensor.as_slice::<f64>()?,
    &[3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 5.0],
);
        // snippet-end:einsum_15
        Ok(())
    }

    snippet_einsum_16()?;

    // snippet source: docs/guides/einsum.md:264
    fn snippet_einsum_16() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:einsum_16
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend};
use tenferro_einsum::{TensorDotAxes, TracedTensorEinsumExt};
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};

let lhs = TracedTensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
let rhs = TracedTensor::from_vec_col_major(
    vec![3, 4],
    vec![
        1.0_f64, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ],
)?;
let out = lhs.tensordot(&rhs, TensorDotAxes::Count(1))?;

assert_eq!(out.concrete_shape()?, vec![2, 4]);
let mut compiler = GraphCompiler::new();
let program = compiler.compile(&out)?;
let backend = CpuBackend::new();
let mut builder = Runtime::builder();
builder.register_engine(runtime_engine_registration(&backend)?)?;
builder.install_extension_module(tenferro_einsum::extension_module::<CpuBackend>(
    runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let mut outputs = runtime.run_compiled(&program, &[])?;
let result = outputs.remove(0);

assert_eq!(result.shape(), &[2, 4]);
assert_eq!(
    result.as_slice::<f64>()?,
    &[22.0, 28.0, 49.0, 64.0, 76.0, 100.0, 103.0, 136.0],
);
        // snippet-end:einsum_16
        Ok(())
    }

    snippet_einsum_17()?;

    // snippet source: docs/guides/einsum.md:325
    fn snippet_einsum_17() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:einsum_17
use tenferro_einsum::{EinsumOptimize, TraceContextEinsumExt};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{DType, TraceContext};

let mut trace = TraceContext::new();
let a = trace.input(ProgramInputSpec::new(
    DType::F64,
    DimExpr::from_concrete(&[2, 3]),
))?;
let b = trace.input(ProgramInputSpec::new(
    DType::F64,
    DimExpr::from_concrete(&[3, 2]),
))?;
let c = trace.einsum_with(
    &[a, b],
    "ij,jk->ik",
    EinsumOptimize::False,
)?;

let graph = trace.finish(&[c])?;
let metadata = graph
    .program()
    .value_metadata(graph.program().outputs()[0])?;
assert_eq!(metadata.shape().len(), 2);
        // snippet-end:einsum_17
        Ok(())
    }

    snippet_autodiff_18()?;

    // snippet source: docs/guides/autodiff.md:77
    fn snippet_autodiff_18() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:autodiff_18
use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};

let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?;
let loss = (&x * &x)?.reduce_sum(Some(&[0]))?;
let ad = AdContext::builder().build()?;
let grad = ad.grad(&loss, &x)?;

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&grad)?;
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&CpuBackend::new())?)?;
let runtime = builder.build()?;
let mut outputs = runtime.run_compiled(&program, &[])?;
let result = outputs.remove(0);

assert_eq!(result.shape(), &[3]);
assert_eq!(result.as_slice::<f64>()?, &[2.0, 4.0, 6.0]);
        // snippet-end:autodiff_18
        Ok(())
    }

    snippet_autodiff_19()?;

    // snippet source: docs/guides/autodiff.md:104
    fn snippet_autodiff_19() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:autodiff_19
use tenferro_ad::AdContext;
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend};
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
use tenferro_linalg::TracedTensorLinalgExt;

let mut compiler = GraphCompiler::new();
let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0])?;
let factor = a.cholesky()?;
let ad = AdContext::builder()
    .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules()?)?
    .build()?;
let loss = factor.reduce_sum(Some(&[0, 1]))?;
let grad_a = ad.grad(&loss, &a)?;
let program = compiler.compile(&grad_a)?;

let backend = CpuBackend::new();
let mut builder = Runtime::builder();
builder.register_engine(runtime_engine_registration(&backend)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuBackend>(
    runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let mut outputs = runtime.run_compiled(&program, &[])?;
let result = outputs.remove(0);
assert_eq!(result.shape(), &[2, 2]);
        // snippet-end:autodiff_19
        Ok(())
    }

    snippet_autodiff_20()?;

    // snippet source: docs/guides/autodiff.md:139
    fn snippet_autodiff_20() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:autodiff_20
use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};

let a = TracedTensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
let b = TracedTensor::from_vec_col_major(
    vec![3, 2],
    vec![0.5_f64, -1.0, 2.0, 1.5, -0.25, 3.0],
)?;
let cotangent = TracedTensor::from_vec_col_major(
    vec![2, 2],
    vec![1.0_f64, -0.5, 0.25, 2.0],
)?;

let mut compiler = GraphCompiler::new();
let y = a.matmul(&b)?;
let ad = AdContext::builder().build()?;
let ct_a = ad.vjp(&y, &a, &cotangent)?;
let program = compiler.compile(&ct_a)?;

let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&CpuBackend::new())?)?;
let runtime = builder.build()?;
let mut outputs = runtime.run_compiled(&program, &[])?;
let result = outputs.remove(0);
assert_eq!(result.shape(), &[2, 3]);
// For y = A * B, the cotangent with respect to A is cotangent * B^T.
assert_eq!(
    result.as_slice::<f64>()?,
    &[0.875, 2.75, -1.0625, 0.0, 2.75, 5.0],
);
        // snippet-end:autodiff_20
        Ok(())
    }

    snippet_autodiff_21()?;

    // snippet source: docs/guides/autodiff.md:180
    fn snippet_autodiff_21() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:autodiff_21
use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};

let a = TracedTensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
let b = TracedTensor::from_vec_col_major(
    vec![3, 2],
    vec![0.5_f64, -1.0, 2.0, 1.5, -0.25, 3.0],
)?;
let tangent = TracedTensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, -0.5, 0.25, 0.0, 2.0, -1.0],
)?;

let mut compiler = GraphCompiler::new();
let y = a.matmul(&b)?;
let ad = AdContext::builder().build()?;
let dy = ad.jvp(&y, &a, &tangent)?;
let program = compiler.compile(&dy)?;

let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&CpuBackend::new())?)?;
let runtime = builder.build()?;
let mut outputs = runtime.run_compiled(&program, &[])?;
let result = outputs.remove(0);
assert_eq!(result.shape(), &[2, 2]);
// For y = A * B, the directional derivative with respect to A is dA * B.
assert_eq!(
    result.as_slice::<f64>()?,
    &[4.25, -2.25, 7.4375, -3.75],
);
        // snippet-end:autodiff_21
        Ok(())
    }

    snippet_tenferro_fft_22()?;

    // snippet source: docs/guides/tenferro-fft.md:100
    fn snippet_tenferro_fft_22() -> Result<(), Box<dyn std::error::Error>> {
// snippet-start:tenferro_fft_22
use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_fft::{FftNorm, TensorFftExt, TensorReadFftExt};
use tenferro_runtime::BackendSessionHost;
use tenferro_tensor::{Tensor, TensorRead, TensorView, TypedTensorView};

let mut backend = CpuBackend::new();
backend.with_backend_session(|session| -> Result<(), tenferro_tensor::Error> {
    let x = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    let full = x.fft(None, -1, FftNorm::Backward, session)?;
    let one_sided = x.rfft(None, -1, FftNorm::Backward, session)?;
    assert_eq!(full.as_slice::<Complex64>()?[0], Complex64::new(10.0, 0.0));
    assert_eq!(one_sided.shape(), &[3]);

    let data = [1.0_f64, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0];
    let view = TypedTensorView::from_slice([4], [2], 0, &data)?;
    let read = TensorRead::from_view(TensorView::F64(view));
    let read_full = read.fft_read(None, -1, FftNorm::Backward, session)?;
    assert_eq!(
        read_full.as_slice::<Complex64>()?[0],
        Complex64::new(10.0, 0.0),
    );
    Ok(())
})?;
// snippet-end:tenferro_fft_22

        Ok(())
    }

    #[cfg(feature = "apple-shared")]
    snippet_tenferro_fft_23()?;

    // snippet source: docs/guides/tenferro-fft.md:132
    #[cfg(feature = "apple-shared")]
    fn snippet_tenferro_fft_23() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:tenferro_fft_23
use num_complex::Complex64;
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_fft::{EagerTensorFftExt, FftNorm};

let x = EagerTensor::from_tensor_in(
    Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0])?,
    EagerRuntime::new()?,
)?;
let spectrum = x.rfft(None, -1, FftNorm::Backward)?;
let restored = spectrum.irfft(Some(4), -1, FftNorm::Backward)?;

assert_eq!(spectrum.shape(), &[3]);
assert_eq!(restored.to_tensor()?.as_slice::<f64>()?, &[1.0, 2.0, 3.0, 4.0]);
        // snippet-end:tenferro_fft_23
        Ok(())
    }

    #[cfg(all(feature = "apple-shared", target_os = "macos"))]
    snippet_tenferro_fft_24()?;

    // snippet source: docs/guides/tenferro-fft.md:155
    #[cfg(all(feature = "apple-shared", target_os = "macos"))]
    fn snippet_tenferro_fft_24() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:tenferro_fft_24
use tenferro_cpu::CpuBackend;
use tenferro_fft::{FftNorm, TensorFftExt};
use tenferro_gpu::apple::AppleContext;
use tenferro_runtime::BackendSessionHost;
use tenferro_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = AppleContext::new()?;
    let host = Tensor::from_vec_col_major([8], vec![1.0_f32; 8])?;
    let managed = context.upload_tensor(&host)?;
    let after_creation = context.transfer_stats();

    let mut cpu = context.cpu_backend().clone();
    let cpu_spectrum = cpu
        .with_backend_session(|session| managed.rfft(None, 0, FftNorm::Backward, session))?;

    let mut metal = context.metal_backend().clone();
    let metal_spectrum = metal
        .with_backend_session(|session| managed.rfft(None, 0, FftNorm::Backward, session))?;
    metal.synchronize()?;

    assert_eq!(cpu_spectrum.shape(), metal_spectrum.shape());
    assert_eq!(context.transfer_stats(), after_creation);
    Ok(())
}
        // snippet-end:tenferro_fft_24
        Ok(())
    }

    Ok(())
}
