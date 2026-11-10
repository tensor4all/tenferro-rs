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
use tenferro_cpu::CpuEngine;
use tenferro_linalg::LinalgOps;
use tenferro_runtime::Tensor;

let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0])?;
let b = Tensor::from_vec_col_major(vec![2, 1], vec![8.0_f64, 27.0])?;
let mut engine = CpuEngine::serial();
let x = a.solve(&b, &mut engine)?;

assert_eq!(x.shape(), &[2, 1]);
assert_eq!(x.as_slice::<f64>()?, &[2.0, 3.0]);
        // snippet-end:linear_algebra_1
        Ok(())
    }

    snippet_linear_algebra_2()?;

    // snippet source: docs/guides/linear-algebra.md:118
    fn snippet_linear_algebra_2() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:linear_algebra_2
use tenferro_cpu::CpuEngine;
use tenferro_linalg::LinalgOps;
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
let mut engine = CpuEngine::serial();
let factor = a.cholesky(&mut engine)?;
let factor_t = factor.transpose(&[1, 0], &mut engine)?;
let reconstructed = factor.matmul(&factor_t, &mut engine)?;

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
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps, Svd};
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
let mut engine = CpuEngine::serial();
let Svd { u, s, vh: vt } = a.svd(&mut engine)?;

assert_eq!(u.shape(), &[2, 2]);
assert_eq!(vt.shape(), &[2, 2]);

let s_values = s.as_slice::<f64>()?;
let sigma = Tensor::from_vec_col_major(
    vec![2, 2],
    vec![s_values[0], 0.0, 0.0, s_values[1]],
)?;
let us = u.matmul(&sigma, &mut engine)?;
let reconstructed = us.matmul(&vt, &mut engine)?;

assert_eq!(a.shape(), &[2, 2]);
assert!(max_abs_diff(&reconstructed, &a)? < 1.0e-12);
        // snippet-end:linear_algebra_4
        Ok(())
    }

    snippet_linear_algebra_5()?;

    // snippet source: docs/guides/linear-algebra.md:210
    fn snippet_linear_algebra_5() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:linear_algebra_5
use tenferro_linalg::{Svd, SvdGauge, SvdOptions, TracedLinalgOps};
use tenferro_runtime::{TraceContext, TracedTensor};
let trace = TraceContext::new();

let a = trace.from_vec_col_major(
    vec![3, 3],
    vec![
        3.0_f64, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 1.0,
    ],
)?;
let Svd { u, s, vh: vt } = a
    .svd_with(&SvdOptions::default()
            .gauge(SvdGauge::CanonicalPivot)
            .derivative_eps(1.0e-10))?;

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

    snippet_linear_algebra_svd_driver()?;

    // snippet source: docs/guides/linear-algebra.md:276
    fn snippet_linear_algebra_svd_driver() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:linear_algebra_svd_driver
use tenferro_linalg::{Svd, SvdDriver, SvdOptions, TracedLinalgOps};
use tenferro_runtime::{TraceContext, TracedTensor};
let trace = TraceContext::new();

let a = trace.from_vec_col_major(
    vec![3, 2],
    vec![
        3.0_f64, 1.0, 0.5,
        -2.0, 0.25, 1.5,
    ],
)?;
// Forces cuSOLVER `gesvd` on CUDA; CPU providers ignore the driver.
let Svd { u, s, vh: vt } = a.svd_with(&SvdOptions::default().driver(SvdDriver::QrBased))?;

assert_eq!(u.concrete_shape()?, vec![3, 2]);
assert_eq!(s.concrete_shape()?, vec![2]);
assert_eq!(vt.concrete_shape()?, vec![2, 2]);
        // snippet-end:linear_algebra_svd_driver
        Ok(())
    }

    snippet_linear_algebra_eigh_driver()?;

    // snippet source: docs/guides/linear-algebra.md
    fn snippet_linear_algebra_eigh_driver() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:linear_algebra_eigh_driver
use tenferro_linalg::{Eigh, EighDriver, EighOptions, TracedLinalgOps};
use tenferro_runtime::{TraceContext, TracedTensor};
let trace = TraceContext::new();

// A batch of two 2x2 symmetric matrices, batch on the trailing axis.
let a = trace.from_vec_col_major(
    vec![2, 2, 2],
    vec![
        2.0_f64, 0.5, 0.5, 3.0,
        4.0, -0.25, -0.25, 1.0,
    ],
)?;
// On CUDA this reaches `syevjBatched`: one launch for the whole batch.
// CPU providers ignore the driver.
let Eigh { values, vectors } =
    a.eigh_with(&EighOptions::default().driver(EighDriver::Jacobi))?;

assert_eq!(values.concrete_shape()?, vec![2, 2]);
assert_eq!(vectors.concrete_shape()?, vec![2, 2, 2]);
        // snippet-end:linear_algebra_eigh_driver
        Ok(())
    }

    snippet_linear_algebra_6()?;

    // snippet source: docs/guides/linear-algebra.md:244
    fn snippet_linear_algebra_6() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:linear_algebra_6
use tenferro_linalg::{Svd, TracedLinalgOps};
use tenferro_runtime::{TraceContext, TracedTensor};
let trace = TraceContext::new();

let a = trace.from_vec_col_major(
    vec![3, 3],
    vec![
        3.0_f64, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 1.0,
    ],
)?;
let Svd { u: _u, s, vh: _vt } = a.svd()?;
let repeated = s.take_axis(0, &[0, 1, 0])?;

assert_eq!(repeated.concrete_shape()?, vec![3]);
        // snippet-end:linear_algebra_6
        Ok(())
    }

    snippet_linear_algebra_7()?;

    // snippet source: docs/guides/linear-algebra.md:265
    fn snippet_linear_algebra_7() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:linear_algebra_7
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps, Qr, QrGauge, QrOptions};
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
let mut engine = CpuEngine::serial();
let Qr { q, r } = a.qr_with(&QrOptions::default().gauge(QrGauge::PositiveDiagonal), &mut engine)?;

let identity = Tensor::from_vec_col_major(
    vec![3, 3],
    vec![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
)?;
let reconstructed = q.matmul(&r, &mut engine)?;
let qt = q.transpose(&[1, 0], &mut engine)?;
let qtq = qt.matmul(&q, &mut engine)?;

assert_eq!(q.shape(), &[4, 3]);
assert_eq!(r.shape(), &[3, 3]);
assert_eq!(identity.shape(), &[3, 3]);
assert!(max_abs_diff(&reconstructed, &a)? < 1.0e-12);
assert!(max_abs_diff(&qtq, &identity)? < 1.0e-12);
        // snippet-end:linear_algebra_7
        Ok(())
    }

    snippet_incremental_householder_qr()?;

    fn snippet_incremental_householder_qr() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:incremental_householder_qr
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{QrGauge, QrOptions, LinalgOps};
use tenferro_runtime::{Tensor, TensorSessionOpsExt};

let initial = Tensor::from_vec_col_major(
    vec![3, 2],
    vec![1.0_f64, 2.0, 0.5, -1.0, 0.2, 2.0],
)?;
let block = Tensor::from_vec_col_major(vec![3, 1], vec![0.5_f64, 1.0, -2.0])?;
let accumulated = Tensor::from_vec_col_major(
    vec![3, 3],
    vec![1.0_f64, 2.0, 0.5, -1.0, 0.2, 2.0, 0.5, 1.0, -2.0],
)?;
let options = QrOptions::default().gauge(QrGauge::PositiveDiagonal);
let mut engine = CpuEngine::serial();
let state = initial
    .householder_qr(&mut engine)?
    .append_columns(&block, &mut engine)?;
let q = state.q_columns(0..3, options, &mut engine)?;
let r = state.r(options, &mut engine)?;
let reconstructed = q.matmul(&r, &mut engine)?;

assert_eq!(q.shape(), &[3, 3]);
assert_eq!(r.shape(), &[3, 3]);
let error = reconstructed
    .as_slice::<f64>()?
    .iter()
    .zip(accumulated.as_slice::<f64>()?)
    .map(|(actual, expected)| (actual - expected).abs())
    .fold(0.0_f64, f64::max);
assert!(error < 1.0e-12);
        // snippet-end:incremental_householder_qr
        Ok(())
    }

    snippet_linear_algebra_8()?;

    // snippet source: docs/guides/linear-algebra.md:312
    fn snippet_linear_algebra_8() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:linear_algebra_8
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{Eigh, LinalgOps};
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
let mut engine = CpuEngine::serial();
let Eigh { values, vectors } = a.eigh(&mut engine)?;

assert_eq!(values.shape(), &[2]);
assert_eq!(vectors.shape(), &[2, 2]);

let value_slice = values.as_slice::<f64>()?;
let diagonal = Tensor::from_vec_col_major(
    vec![2, 2],
    vec![value_slice[0], 0.0, 0.0, value_slice[1]],
)?;
let vd = vectors.matmul(&diagonal, &mut engine)?;
let vt = vectors.transpose(&[1, 0], &mut engine)?;
let reconstructed = vd.matmul(&vt, &mut engine)?;

assert_eq!(a.shape(), &[2, 2]);
assert!(max_abs_diff(&reconstructed, &a)? < 1.0e-12);
        // snippet-end:linear_algebra_8
        Ok(())
    }

    snippet_linear_algebra_9()?;

    // snippet source: docs/guides/linear-algebra.md:347
    fn snippet_linear_algebra_9() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:linear_algebra_9
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuEngine};
use tenferro_runtime::{TraceContext, ExecutionEngine, GraphCompiler, Runtime, TracedTensor};
use tenferro_linalg::TracedLinalgOps;
let trace = TraceContext::new();

let a = trace.from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0])?;
let factor = a.cholesky()?;

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&factor)?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder
    .register_engine(runtime_engine_registration(&engine)?)?;
builder
    .install_extension_module(
        tenferro_linalg::extension_module::<CpuEngine>(runtime_engine_id()?)?,
    )?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let mut outputs = runtime.run(&mut [&mut engine], &prepared, &[])?;
let result = outputs.remove(0).into_tensor(&mut engine)?;

assert_eq!(result.shape(), &[2, 2]);
assert_eq!(result.as_slice::<f64>()?, &[2.0, 0.0, 0.0, 3.0]);
        // snippet-end:linear_algebra_9
        Ok(())
    }

    snippet_linear_algebra_10()?;

    // snippet source: docs/guides/linear-algebra.md:377
    fn snippet_linear_algebra_10() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:linear_algebra_10
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuEngine};
use tenferro_runtime::{TraceContext, ExecutionEngine, GraphCompiler, Runtime, TracedTensor};
use tenferro_linalg::TracedLinalgOps;
let trace = TraceContext::new();

let a = trace.from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0])?;
let b = trace.from_vec_col_major(vec![2, 1], vec![8.0_f64, 27.0])?;
let x = a.solve(&b)?;

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&x)?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder
    .register_engine(runtime_engine_registration(&engine)?)?;
builder
    .install_extension_module(
        tenferro_linalg::extension_module::<CpuEngine>(runtime_engine_id()?)?,
    )?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let mut outputs = runtime.run(&mut [&mut engine], &prepared, &[])?;
let result = outputs.remove(0).into_tensor(&mut engine)?;

assert_eq!(result.shape(), &[2, 1]);
assert_eq!(result.as_slice::<f64>()?, &[2.0, 3.0]);
        // snippet-end:linear_algebra_10
        Ok(())
    }

    snippet_linear_algebra_11()?;

    // snippet source: docs/guides/linear-algebra.md:415
    fn snippet_linear_algebra_11() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:linear_algebra_11
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{FullPivLu, LinalgOps};
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
let mut engine = CpuEngine::serial();
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

let FullPivLu { p, l, u, q, parity } = a.full_piv_lu(&mut engine)?;
let pt = p.transpose(&[1, 0], &mut engine)?;
let pt_l = pt.matmul(&l, &mut engine)?;
let pt_lu = pt_l.matmul(&u, &mut engine)?;
let reconstructed = pt_lu.matmul(&q, &mut engine)?;
let x = a.full_piv_lu_solve(&b, &mut engine)?;

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
use tenferro_cpu::CpuEngine;
use tenferro_einsum::EinsumOps;
use tenferro_tensor::{
    Tensor, TensorWrite, TypedTensor, TypedTensorView,
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
let mut engine = CpuEngine::serial();
let product = [&lhs, &rhs].einsum("ij,jk->ik", &mut engine)?;
assert_eq!(product.as_slice::<f64>()?, &[22.0, 28.0, 49.0, 64.0]);

let mut product_out = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64; 4])?;
[&lhs, &rhs].einsum_into("ij,jk->ik", &tenferro_einsum::EinsumOptions::default(), &mut engine, tenferro_tensor::Dest::overwrite(TensorWrite::from_tensor(&mut product_out)))?;
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
let complex = [&complex_lhs, &complex_rhs].einsum("ij,jk->ik", &mut engine)?;
assert_eq!(
    complex.as_slice()?,
    &[Complex64::new(23.0, 2.0), Complex64::new(36.0, 3.0)],
);

let borrowed = TypedTensorView::from_slice([2, 2], [1, 2], 0, complex_lhs.as_slice()?)?;
let borrowed_rhs = complex_rhs.as_view();
let mut borrowed_storage = [Complex64::new(0.0, 0.0); 4];
let borrowed_out =
    TypedTensorViewMut::from_slice([2, 1], [2, 4], 1, &mut borrowed_storage)?;
[borrowed, borrowed_rhs].einsum_into("ij,jk->ik", &tenferro_einsum::EinsumOptions::default(), &mut engine, tenferro_tensor::Dest::overwrite(TypedTensorWrite::from_view(borrowed_out)))?;
assert_eq!(
    [borrowed_storage[1], borrowed_storage[3]],
    [Complex64::new(23.0, 2.0), Complex64::new(36.0, 3.0)],
);
        // snippet-end:einsum_12
        Ok(())
    }

    snippet_einsum_20()?;

    // snippet source: docs/guides/einsum.md:146
    fn snippet_einsum_20() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:einsum_20
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{EinsumAxis, EinsumNotation, EinsumOps};
use tenferro_tensor::Tensor;

let lhs = Tensor::from_vec_col_major(vec![2, 2, 3], vec![1.0_f64; 12])?;
let rhs = Tensor::from_vec_col_major(vec![1, 3, 2], vec![1.0_f64; 6])?;
let notation = EinsumNotation::new(
    &[
        &[EinsumAxis::Ellipsis, EinsumAxis::Label(0), EinsumAxis::Label(1)],
        &[EinsumAxis::Ellipsis, EinsumAxis::Label(1), EinsumAxis::Label(2)],
    ],
    &[EinsumAxis::Ellipsis, EinsumAxis::Label(0), EinsumAxis::Label(2)],
);
let mut engine = CpuEngine::serial();
let string_result = [&lhs, &rhs].einsum("...ij,...jk->...ik", &mut engine)?;
let programmatic_result = [&lhs, &rhs].einsum(&notation, &mut engine)?;
assert_eq!(string_result.shape(), &[2, 2, 2]);
assert_eq!(string_result.as_slice::<f64>()?, &[3.0; 8]);
assert_eq!(programmatic_result.as_slice::<f64>()?, &[3.0; 8]);
        // snippet-end:einsum_20
        Ok(())
    }

    snippet_einsum_13()?;

    // snippet source: docs/guides/einsum.md:146
    fn snippet_einsum_13() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:einsum_13
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{ConcreteEinsumPlan, EinsumOps};
use tenferro_tensor::{Tensor, TensorRead, TensorView, TensorWrite, TypedTensorView};

let matrix_data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
let matrix = TypedTensorView::from_slice([2, 3], [3, 1], 0, &matrix_data)?;
let vector = Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0])?;
let inputs = [
    TensorRead::from_view(TensorView::F64(matrix)),
    TensorRead::from_tensor(&vector),
];

let mut engine = CpuEngine::serial();
let result = inputs.einsum("ij,j->i", &mut engine)?;
assert_eq!(result.as_slice::<f64>()?, &[140.0, 320.0]);

let plan = ConcreteEinsumPlan::prepare(&inputs.clone(), "ij,j->i", &tenferro_einsum::EinsumOptions::default())?;
let planned = plan.execute(&inputs, &mut engine)?;
assert_eq!(planned.as_slice::<f64>()?, &[140.0, 320.0]);

let mut planned_out = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2])?;
let matrix = TypedTensorView::from_slice([2, 3], [3, 1], 0, &matrix_data)?;
let inputs = [
    TensorRead::from_view(TensorView::F64(matrix)),
    TensorRead::from_tensor(&vector),
];
plan.execute_into(&inputs, &mut engine, tenferro_tensor::Dest::overwrite(TensorWrite::from_tensor(&mut planned_out)))?;
assert_eq!(planned_out.as_slice::<f64>()?, &[140.0, 320.0]);
        // snippet-end:einsum_13
        Ok(())
    }

    snippet_einsum_14()?;

    // snippet source: docs/guides/einsum.md:186
    fn snippet_einsum_14() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:einsum_14
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuEngine};
use tenferro_einsum::TracedEinsumOps;
use tenferro_runtime::{TraceContext, ExecutionEngine, GraphCompiler, Runtime, Tensor};
let trace = TraceContext::new();

let a = Tensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
let b = Tensor::from_vec_col_major(
    vec![3, 2],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;

let a_in = trace.input_concrete_shape(a.dtype(), &[2, 3])?;
let b_in = trace.input_concrete_shape(b.dtype(), &[3, 2])?;
let c = trace.einsum(&[&a_in, &b_in], "ij,jk->ik")?;
let program = GraphCompiler::new().compile(&c)?;

let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_einsum::extension_module::<CpuEngine>(
    runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let reads = [
    tenferro_runtime::TensorRead::from_tensor(&a),
    tenferro_runtime::TensorRead::from_tensor(&b),
];
let prepared = runtime.prepare(&program, &reads, &[engine.target()])?;
let mut outputs = runtime.run(&mut [&mut engine], &prepared, &reads)?;
let result = outputs.remove(0).into_tensor(&mut engine)?;

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
use tenferro_cpu::CpuEngine;
use tenferro_einsum::EagerEinsumOps;

let ctx = EagerRuntime::new()?;
let mut engine = CpuEngine::serial();
let u = ctx.variable_from(Tensor::from_vec_col_major(
    vec![2],
    vec![1.0_f64, 2.0],
)?)?;
let v = ctx.variable_from(Tensor::from_vec_col_major(
    vec![3],
    vec![3.0_f64, 4.0, 5.0],
)?)?;

let (outer, diag) = ctx.with_eager_session(&mut engine, |s| {
    let outer = s.einsum(&[&u, &v], "i,j->ij")?;
    let diag = s.einsum(&[&v], "i->ii")?;
    Ok::<_, tenferro_einsum::Error>((outer, diag))
})?;

assert_eq!(outer.shape(), &[2, 3]);
let outer_tensor = outer.to_tensor(&mut engine)?;
assert_eq!(
    outer_tensor.as_slice::<f64>()?,
    &[3.0, 6.0, 4.0, 8.0, 5.0, 10.0],
);
assert_eq!(diag.shape(), &[3, 3]);
let diag_tensor = diag.to_tensor(&mut engine)?;
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
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuEngine};
use tenferro_einsum::{TensorDotAxes, TracedTensordotOps};
use tenferro_runtime::{TraceContext, ExecutionEngine, GraphCompiler, Runtime, TracedTensor};
let trace = TraceContext::new();

let lhs = trace.from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
let rhs = trace.from_vec_col_major(
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
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_einsum::extension_module::<CpuEngine>(
    runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let mut outputs = runtime.run(&mut [&mut engine], &prepared, &[])?;
let result = outputs.remove(0).into_tensor(&mut engine)?;

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
use tenferro_einsum::{EinsumOptimize, EinsumOptions, TracedEinsumOps};
use tenferro_runtime::{TraceContext, DType};
let trace = TraceContext::new();

let a = trace.input_concrete_shape(DType::F64, &[2, 3])?;
let b = trace.input_concrete_shape(DType::F64, &[3, 2])?;
let c = trace.einsum_with(
    &[&a, &b],
    "ij,jk->ik",
    &EinsumOptions::default().optimize(EinsumOptimize::False),
)?;

assert_eq!(c.rank(), 2);
assert_eq!(c.try_concrete_shape(), Some(vec![2, 2]));
        // snippet-end:einsum_17
        Ok(())
    }

    snippet_autodiff_18()?;

    // snippet source: docs/guides/autodiff.md:77
    fn snippet_autodiff_18() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:autodiff_18
use tenferro_ad::AdContext;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::TracedLinalgOps;
use tenferro_runtime::{TraceContext, ExecutionEngine, GraphCompiler, Runtime, TracedTensor};
let trace = TraceContext::new();

let x = trace.from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?;
let loss = (&x * &x)?.reduce_sum(Some(&[0]))?;
let ad = AdContext::builder().build()?;
let grad = ad.grad(&loss, &x)?;

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&grad)?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let mut outputs = runtime.run(&mut [&mut engine], &prepared, &[])?;
let result = outputs.remove(0).into_tensor(&mut engine)?;

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
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuEngine};
use tenferro_runtime::{TraceContext, ExecutionEngine, GraphCompiler, Runtime, TracedTensor};
use tenferro_linalg::TracedLinalgOps;
let trace = TraceContext::new();

let mut compiler = GraphCompiler::new();
let a = trace.from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0])?;
let factor = a.cholesky()?;
let ad = AdContext::builder()
    .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules()?)?
    .build()?;
let loss = factor.reduce_sum(Some(&[0, 1]))?;
let grad_a = ad.grad(&loss, &a)?;
let program = compiler.compile(&grad_a)?;

let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
    runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let mut outputs = runtime.run(&mut [&mut engine], &prepared, &[])?;
let result = outputs.remove(0).into_tensor(&mut engine)?;
assert_eq!(result.shape(), &[2, 2]);
        // snippet-end:autodiff_19
        Ok(())
    }

    snippet_autodiff_20()?;

    // snippet source: docs/guides/autodiff.md:139
    fn snippet_autodiff_20() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:autodiff_20
use tenferro_ad::AdContext;
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{TraceContext, ExecutionEngine, GraphCompiler, Runtime, TracedTensor};
let trace = TraceContext::new();

let a = trace.from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
let b = trace.from_vec_col_major(
    vec![3, 2],
    vec![0.5_f64, -1.0, 2.0, 1.5, -0.25, 3.0],
)?;
let cotangent = trace.from_vec_col_major(
    vec![2, 2],
    vec![1.0_f64, -0.5, 0.25, 2.0],
)?;

let mut compiler = GraphCompiler::new();
let y = a.matmul(&b)?;
let ad = AdContext::builder().build()?;
let ct_a = ad.vjp(&y, &a, &cotangent)?;
let program = compiler.compile(&ct_a)?;

let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let mut outputs = runtime.run(&mut [&mut engine], &prepared, &[])?;
let result = outputs.remove(0).into_tensor(&mut engine)?;
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
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{TraceContext, ExecutionEngine, GraphCompiler, Runtime, TracedTensor};
let trace = TraceContext::new();

let a = trace.from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
let b = trace.from_vec_col_major(
    vec![3, 2],
    vec![0.5_f64, -1.0, 2.0, 1.5, -0.25, 3.0],
)?;
let tangent = trace.from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, -0.5, 0.25, 0.0, 2.0, -1.0],
)?;

let mut compiler = GraphCompiler::new();
let y = a.matmul(&b)?;
let ad = AdContext::builder().build()?;
let dy = ad.jvp(&y, &a, &tangent)?;
let program = compiler.compile(&dy)?;

let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let mut outputs = runtime.run(&mut [&mut engine], &prepared, &[])?;
let result = outputs.remove(0).into_tensor(&mut engine)?;
assert_eq!(result.shape(), &[2, 2]);
// For y = A * B, the directional derivative with respect to A is dA * B.
assert_eq!(
    result.as_slice::<f64>()?,
    &[4.25, -2.25, 7.4375, -3.75],
);
        // snippet-end:autodiff_21
        Ok(())
    }

    snippet_tenferro_fft_traced()?;

    // snippet source: docs/guides/tenferro-fft.md (traced graph)
    fn snippet_tenferro_fft_traced() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:tenferro_fft_traced
use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::TracedFftOps;
use tenferro_runtime::{TraceContext, ExecutionEngine, GraphCompiler, Runtime, TracedTensor};

fn cpu_runtime_with_fft(engine: &CpuEngine) -> Result<Runtime, Box<dyn std::error::Error>> {
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(engine)?)?;
    builder.install_extension_module(tenferro_fft::extension_module::<CpuEngine>(
        tenferro_cpu::runtime_engine_id()?,
    )?)?;
    Ok(builder.build()?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let trace = TraceContext::new();
    let x = trace.from_vec_col_major(
        vec![4],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    )
    .unwrap();
    let y = x.fft()?;

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y)?;
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime_with_fft(&engine)?;
    let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
    let mut outputs = runtime.run(&mut [&mut engine], &prepared, &[])?;
    assert_eq!(outputs.len(), 1);
    let out = outputs.remove(0).into_tensor(&mut engine)?;
    assert_eq!(out.shape(), &[4]);
    assert_eq!(
        out.as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-2.0, -2.0),
        ],
    );

    Ok(())
}
        // snippet-end:tenferro_fft_traced
        Ok(())
    }

    snippet_tenferro_fft_22()?;

    // snippet source: docs/guides/tenferro-fft.md:100
    fn snippet_tenferro_fft_22() -> Result<(), Box<dyn std::error::Error>> {
// snippet-start:tenferro_fft_22
use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::FftOps;
use tenferro_tensor::{Tensor, TensorRead, TensorView, TypedTensorView};

let mut engine = CpuEngine::serial();
let x = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let full = x.fft(&mut engine)?;
let one_sided = x.rfft(&mut engine)?;
assert_eq!(full.as_slice::<Complex64>()?[0], Complex64::new(10.0, 0.0));
assert_eq!(one_sided.shape(), &[3]);

let data = [1.0_f64, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0];
let view = TypedTensorView::from_slice([4], [2], 0, &data)?;
let read = TensorRead::from_view(TensorView::F64(view));
let read_full = read.fft(&mut engine)?;
assert_eq!(
    read_full.as_slice::<Complex64>()?[0],
    Complex64::new(10.0, 0.0),
);
// snippet-end:tenferro_fft_22

        Ok(())
    }

    // snippet source: docs/guides/tenferro-fft.md:132
    Ok(())
}
