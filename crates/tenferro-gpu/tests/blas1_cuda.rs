#![cfg(feature = "cuda")]

//! Hardware parity gate for the cuBLAS-backed BLAS1 session hooks.
//!
//! `vdot_read`, `norm_squared_read`, and `axpby_read_into_accum` on
//! [`CudaBackend`] bypass the generic `dot_general`/elementwise composition and
//! call cuBLAS directly. These tests pin the numerics against the CPU backend,
//! which keeps the composed reference semantics. They are no-ops on machines
//! without an available CUDA device.
//!
//! The conjugation convention in `vdot_read` is the reason this file exists: a
//! `dotu`-for-`dotc` slip is invisible on real inputs and silently wrong on the
//! complex ones a Krylov loop actually feeds it.

use num_complex::Complex64;

use tenferro_cpu::CpuBackend;
use tenferro_gpu::cuda::{download_tensor, gpu_available, upload_tensor, CudaBackend, CudaDeviceId};
use tenferro_tensor::backend::BackendSession;
use tenferro_tensor::{
    ContractionScalar, Tensor, TensorRead, TensorView, TensorWrite, TypedTensorView,
};

fn c64(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// Deterministic, non-degenerate complex data: distinct real and imaginary
/// parts so a conjugation or transposition error cannot cancel out.
fn sample(len: usize, seed: f64) -> Tensor {
    let data: Vec<Complex64> = (0..len)
        .map(|i| {
            let x = i as f64;
            c64(seed + 0.5 * x, 1.0 - 0.25 * x + seed)
        })
        .collect();
    Tensor::from_vec_col_major(vec![len], data).unwrap()
}

fn backends() -> Option<(CudaBackend, CpuBackend)> {
    if !gpu_available() {
        return None;
    }
    let cuda = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    Some((cuda, CpuBackend::default()))
}

fn assert_close_c64(actual: &Tensor, expected: &Tensor, what: &str) {
    let a = actual.as_slice::<Complex64>().unwrap();
    let e = expected.as_slice::<Complex64>().unwrap();
    assert_eq!(a.len(), e.len(), "{what}: length mismatch");
    for (i, (a, e)) in a.iter().zip(e.iter()).enumerate() {
        let tol = 1e-12 * (1.0 + e.norm());
        assert!(
            (a - e).norm() <= tol,
            "{what}: element {i} is {a} but CPU says {e}"
        );
    }
}

fn assert_close_f64(actual: &Tensor, expected: &Tensor, what: &str) {
    let a = actual.as_slice::<f64>().unwrap();
    let e = expected.as_slice::<f64>().unwrap();
    assert_eq!(a.len(), e.len(), "{what}: length mismatch");
    for (i, (a, e)) in a.iter().zip(e.iter()).enumerate() {
        let tol = 1e-12 * (1.0 + e.abs());
        assert!(
            (a - e).abs() <= tol,
            "{what}: element {i} is {a} but CPU says {e}"
        );
    }
}

#[test]
fn cuda_vdot_matches_cpu_and_conjugates_the_left_operand() {
    let Some((mut cuda, mut cpu)) = backends() else {
        return;
    };

    for len in [1_usize, 7, 1024] {
        let host_lhs = sample(len, 0.25);
        let host_rhs = sample(len, -1.5);

        let expected = cpu
            .vdot_read(
                TensorRead::from_tensor(&host_lhs),
                TensorRead::from_tensor(&host_rhs),
            )
            .unwrap();

        let lhs = upload_tensor(cuda.runtime(), &host_lhs).unwrap();
        let rhs = upload_tensor(cuda.runtime(), &host_rhs).unwrap();
        let got = cuda
            .vdot_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
            .unwrap();
        let got = download_tensor(cuda.runtime(), &got).unwrap();

        assert_close_c64(&got, &expected, &format!("vdot len={len}"));

        // Independent check of the convention itself, not just CPU agreement.
        let manual: Complex64 = host_lhs
            .as_slice::<Complex64>()
            .unwrap()
            .iter()
            .zip(host_rhs.as_slice::<Complex64>().unwrap())
            .map(|(l, r)| l.conj() * r)
            .sum();
        let got = got.as_slice::<Complex64>().unwrap()[0];
        assert!(
            (got - manual).norm() <= 1e-12 * (1.0 + manual.norm()),
            "vdot len={len} is {got} but sum(conj(l)*r) is {manual}"
        );
    }
}

/// A row-major-strided `[3, 4]` view over a column-major `[4, 3]` buffer, i.e.
/// the transpose, expressed the way the sweep code builds strided operands.
///
/// Device and host tensors take different view constructors: `backend_region_view`
/// requires a device allocation, and host storage goes through `from_slice`.
fn transposed_device_view(tensor: &Tensor) -> TensorRead<'_> {
    let Tensor::C64(typed) = tensor else {
        panic!("blas1 parity tests use C64 tensors")
    };
    let view = typed
        .backend_region_view(vec![3, 4], vec![4, 1], 0)
        .unwrap();
    TensorRead::from_view(TensorView::C64(view))
}

fn transposed_host_view(tensor: &Tensor) -> TensorRead<'_> {
    let view = TypedTensorView::from_slice(
        vec![3, 4],
        vec![4_isize, 1],
        0,
        tensor.as_slice::<Complex64>().unwrap(),
    )
    .unwrap();
    TensorRead::from_view(TensorView::C64(view))
}

#[test]
fn cuda_vdot_accepts_a_non_contiguous_read() {
    let Some((mut cuda, mut cpu)) = backends() else {
        return;
    };

    // The cuBLAS path must canonicalize a strided operand on the device rather
    // than reading the underlying storage order.
    let host = Tensor::from_vec_col_major(
        vec![4, 3],
        (0..12)
            .map(|i| c64(i as f64, 3.0 - 0.5 * i as f64))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let other = Tensor::from_vec_col_major(
        vec![3, 4],
        (0..12)
            .map(|i| c64(0.75 + 0.5 * i as f64, 1.0 - 0.25 * i as f64))
            .collect::<Vec<_>>(),
    )
    .unwrap();

    let expected = cpu
        .vdot_read(transposed_host_view(&host), TensorRead::from_tensor(&other))
        .unwrap();

    let device = upload_tensor(cuda.runtime(), &host).unwrap();
    let device_other = upload_tensor(cuda.runtime(), &other).unwrap();
    let got = cuda
        .vdot_read(
            transposed_device_view(&device),
            TensorRead::from_tensor(&device_other),
        )
        .unwrap();
    let got = download_tensor(cuda.runtime(), &got).unwrap();

    assert_close_c64(&got, &expected, "vdot transposed");
}

#[test]
fn cuda_norm_squared_matches_cpu() {
    let Some((mut cuda, mut cpu)) = backends() else {
        return;
    };

    for len in [1_usize, 5, 4096] {
        let host = sample(len, 0.125);
        let expected = cpu
            .norm_squared_read(TensorRead::from_tensor(&host))
            .unwrap();

        let device = upload_tensor(cuda.runtime(), &host).unwrap();
        let got = cuda
            .norm_squared_read(TensorRead::from_tensor(&device))
            .unwrap();
        let got = download_tensor(cuda.runtime(), &got).unwrap();

        assert_close_f64(&got, &expected, &format!("norm_squared len={len}"));

        let manual: f64 = host
            .as_slice::<Complex64>()
            .unwrap()
            .iter()
            .map(|z| z.norm_sqr())
            .sum();
        let got = got.as_slice::<f64>().unwrap()[0];
        assert!(
            (got - manual).abs() <= 1e-12 * (1.0 + manual),
            "norm_squared len={len} is {got} but sum(|z|^2) is {manual}"
        );
    }
}

#[test]
fn cuda_axpby_matches_cpu_with_complex_coefficients() {
    let Some((mut cuda, mut cpu)) = backends() else {
        return;
    };

    let alpha = ContractionScalar::C64(c64(0.75, -1.25));
    let beta = ContractionScalar::C64(c64(-0.5, 2.0));

    for len in [1_usize, 9, 2048] {
        let host_x = sample(len, 0.5);
        let host_y = sample(len, -0.25);

        let mut expected = sample(len, -0.25);
        cpu.axpby_read_into_accum(
            alpha,
            TensorRead::from_tensor(&host_x),
            beta,
            TensorWrite::from_tensor(&mut expected),
        )
        .unwrap();

        let x = upload_tensor(cuda.runtime(), &host_x).unwrap();
        let mut y = upload_tensor(cuda.runtime(), &host_y).unwrap();
        cuda.axpby_read_into_accum(
            alpha,
            TensorRead::from_tensor(&x),
            beta,
            TensorWrite::from_tensor(&mut y),
        )
        .unwrap();
        let got = download_tensor(cuda.runtime(), &y).unwrap();

        assert_close_c64(&got, &expected, &format!("axpby len={len}"));
    }
}

#[test]
fn cuda_blas1_handles_empty_inputs() {
    let Some((mut cuda, _)) = backends() else {
        return;
    };

    let host = Tensor::from_vec_col_major(vec![0], Vec::<Complex64>::new()).unwrap();
    let device = upload_tensor(cuda.runtime(), &host).unwrap();

    let dot = cuda
        .vdot_read(
            TensorRead::from_tensor(&device),
            TensorRead::from_tensor(&device),
        )
        .unwrap();
    let dot = download_tensor(cuda.runtime(), &dot).unwrap();
    assert_eq!(dot.as_slice::<Complex64>().unwrap(), &[c64(0.0, 0.0)]);

    let norm = cuda
        .norm_squared_read(TensorRead::from_tensor(&device))
        .unwrap();
    let norm = download_tensor(cuda.runtime(), &norm).unwrap();
    assert_eq!(norm.as_slice::<f64>().unwrap(), &[0.0]);
}
