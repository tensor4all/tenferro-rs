//! #2552 rule 5: multi-output operations return named structs that are generic over the
//! tensor type, with mixed dtypes where the operation has them.

use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{
    Eig, Eigh, FullPivLu, Lu, Qr, Slogdet, Svd, TensorLinalgExt, TracedTensorLinalgExt,
};
use tenferro_runtime::TraceContext;
use tenferro_tensor::{Tensor, TypedTensor};

fn matrix(data: Vec<f64>) -> Tensor {
    Tensor::from_vec_col_major(vec![2, 2], data).unwrap()
}

#[test]
fn concrete_results_are_named_structs() {
    let mut engine = CpuEngine::serial();
    let a = matrix(vec![4.0, 1.0, 1.0, 3.0]);

    let Svd { u, s, vh } = a.svd(&mut engine).unwrap();
    assert_eq!(
        (u.shape(), s.shape(), vh.shape()),
        (&[2, 2][..], &[2][..], &[2, 2][..])
    );
    let Qr { q, r } = a.qr(&mut engine).unwrap();
    assert_eq!((q.shape(), r.shape()), (&[2, 2][..], &[2, 2][..]));
    let Lu { p, l, u, parity } = a.lu(&mut engine).unwrap();
    assert_eq!(
        (p.shape(), l.shape(), u.shape()),
        (&[2, 2][..], &[2, 2][..], &[2, 2][..])
    );
    assert_eq!(parity.shape(), &[] as &[usize]);
    let FullPivLu { p, l, u, q, parity } = a.full_piv_lu(&mut engine).unwrap();
    assert_eq!(
        (p.shape(), l.shape(), u.shape(), q.shape()),
        (&[2, 2][..], &[2, 2][..], &[2, 2][..], &[2, 2][..])
    );
    assert_eq!(parity.shape(), &[] as &[usize]);
    let Eigh { values, vectors } = a.eigh(&mut engine).unwrap();
    assert_eq!((values.shape(), vectors.shape()), (&[2][..], &[2, 2][..]));
    let Eig { values, vectors } = a.eig(&mut engine).unwrap();
    assert_eq!((values.shape(), vectors.shape()), (&[2][..], &[2, 2][..]));
    let Slogdet { sign, logabsdet } = a.slogdet(&mut engine).unwrap();
    assert_eq!(sign.as_slice::<f64>().unwrap(), &[1.0]);
    assert!((logabsdet.as_slice::<f64>().unwrap()[0] - 11.0_f64.ln()).abs() < 1.0e-12);
}

#[test]
fn typed_results_keep_mixed_dtypes() {
    let mut engine = CpuEngine::serial();
    let c = TypedTensor::<Complex64>::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 3.0),
        ],
    )
    .unwrap();
    let svd: Svd<TypedTensor<Complex64>, TypedTensor<f64>> = c.as_view().svd(&mut engine).unwrap();
    assert_eq!(svd.s.host_data().unwrap(), &[3.0, 2.0]);
    let eigh: Eigh<TypedTensor<Complex64>, TypedTensor<f64>> =
        c.as_view().eigh(&mut engine).unwrap();
    assert_eq!(eigh.values.shape(), &[2]);
    let slogdet: Slogdet<TypedTensor<Complex64>, TypedTensor<f64>> =
        c.as_view().slogdet(&mut engine).unwrap();
    assert!((slogdet.logabsdet.host_data().unwrap()[0] - 6.0_f64.ln()).abs() < 1.0e-12);
}

#[test]
fn traced_results_are_named_structs() {
    let trace = TraceContext::new();
    let a = trace
        .from_vec_col_major(vec![2, 2], vec![4.0_f64, 1.0, 1.0, 3.0])
        .unwrap();
    let Svd { u, s, vh } = a.svd().unwrap();
    assert_eq!((u.rank(), s.rank(), vh.rank()), (2, 1, 2));
    let Qr { q, r } = a.qr().unwrap();
    assert_eq!((q.rank(), r.rank()), (2, 2));
    let Lu { parity, .. } = a.lu().unwrap();
    assert_eq!(parity.rank(), 0);
    let Eigh { values, .. } = a.eigh().unwrap();
    assert_eq!(values.rank(), 1);
}

#[cfg(feature = "autodiff")]
#[test]
fn eager_results_are_named_structs() {
    use tenferro_ad::{EagerRuntime, EagerTensor};
    use tenferro_linalg::EagerSessionLinalgExt;

    let runtime = EagerRuntime::new().unwrap();
    let a = EagerTensor::from_tensor_in(matrix(vec![4.0, 1.0, 1.0, 3.0]), runtime.clone()).unwrap();
    let (svd, qr) = runtime
        .with_eager_session(&mut CpuEngine::serial(), |session| {
            Ok::<_, tenferro_ad::Error>((session.svd(&a)?, session.qr(&a)?))
        })
        .unwrap();
    let Svd { s, .. } = svd;
    let Qr { r, .. } = qr;
    assert_eq!((s.shape(), r.shape()), (&[2][..], &[2, 2][..]));
}

#[test]
fn owned_typed_tensors_use_the_same_entry_as_views() {
    let mut engine = CpuEngine::serial();
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![4.0, 1.0, 1.0, 3.0]).unwrap();
    let owned: Svd<TypedTensor<f64>> = a.svd(&mut engine).unwrap();
    let borrowed: Svd<TypedTensor<f64>> = a.as_view().svd(&mut engine).unwrap();
    assert_eq!(
        owned.s.host_data().unwrap(),
        borrowed.s.host_data().unwrap()
    );
    let Lu { parity, .. }: Lu<TypedTensor<f64>> = a.lu(&mut engine).unwrap();
    assert_eq!(parity.shape(), &[] as &[usize]);
}
