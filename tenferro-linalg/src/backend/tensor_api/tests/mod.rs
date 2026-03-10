use super::*;
use crate::backend::CpuTensorLinalgBackend;
use tenferro_linalg_prims::{
    EigTensorResult as PrimEigTensorResult, EigenTensorResult as PrimEigenTensorResult,
    LuTensorResult as PrimLuTensorResult, QrTensorResult as PrimQrTensorResult,
    SvdTensorResult as PrimSvdTensorResult, TensorLinalgPrims,
};
use tenferro_tensor::MemoryOrder;
use tenferro_tensor::Tensor;

#[test]
fn tensor_result_structs_clone_and_preserve_shapes() {
    let q = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let r = q.clone();
    let s = Tensor::<f64>::from_slice(&[3.0, 1.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let complex_values = Tensor::<num_complex::Complex64>::from_slice(
        &[
            num_complex::Complex64::new(1.0, 0.5),
            num_complex::Complex64::new(2.0, -0.5),
        ],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let complex_vectors = Tensor::<num_complex::Complex64>::from_slice(
        &[
            num_complex::Complex64::new(1.0, 0.0),
            num_complex::Complex64::new(0.0, 0.0),
            num_complex::Complex64::new(0.0, 0.0),
            num_complex::Complex64::new(1.0, 0.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let qr = QrTensorResult {
        q: q.clone(),
        r: r.clone(),
    };
    let svd = SvdTensorResult {
        u: q.clone(),
        s: s.clone(),
        vt: r.clone(),
    };
    let lu = LuTensorResult {
        l: q.clone(),
        u: r.clone(),
        pivots: vec![1, 0],
    };
    let eigen = EigenTensorResult {
        values: s.clone(),
        vectors: q.clone(),
    };
    let eig = EigTensorResult::<f64> {
        values: complex_values.clone(),
        vectors: complex_vectors.clone(),
    };

    let qr_clone = qr.clone();
    let svd_clone = svd.clone();
    let lu_clone = lu.clone();
    let eigen_clone = eigen.clone();
    let eig_clone = eig.clone();

    assert_eq!(qr_clone.q.dims(), &[2, 2]);
    assert_eq!(qr_clone.r.dims(), &[2, 2]);
    assert_eq!(svd_clone.u.dims(), &[2, 2]);
    assert_eq!(svd_clone.s.dims(), &[2]);
    assert_eq!(svd_clone.vt.dims(), &[2, 2]);
    assert_eq!(lu_clone.l.dims(), &[2, 2]);
    assert_eq!(lu_clone.u.dims(), &[2, 2]);
    assert_eq!(lu_clone.pivots, vec![1, 0]);
    assert_eq!(eigen_clone.values.dims(), &[2]);
    assert_eq!(eigen_clone.vectors.dims(), &[2, 2]);
    assert_eq!(eig_clone.values.dims(), &[2]);
    assert_eq!(eig_clone.vectors.dims(), &[2, 2]);
}

#[test]
fn linalg_prims_contract_is_visible_from_backend_tests() {
    fn assert_backend<B: TensorLinalgPrims<f64>>() {}
    assert_backend::<CpuTensorLinalgBackend>();

    let q = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let r = q.clone();
    let s = Tensor::<f64>::from_slice(&[3.0, 1.0], &[2], MemoryOrder::ColumnMajor).unwrap();

    let qr = PrimQrTensorResult {
        q: q.clone(),
        r: r.clone(),
    };
    let svd = PrimSvdTensorResult {
        u: q.clone(),
        s: s.clone(),
        vt: r.clone(),
    };
    let lu = PrimLuTensorResult {
        l: q.clone(),
        u: r.clone(),
        pivots: vec![1, 0],
    };
    let eigen = PrimEigenTensorResult {
        values: s.clone(),
        vectors: q.clone(),
    };
    let eig = PrimEigTensorResult::<f64> {
        values: Tensor::from_slice(
            &[
                num_complex::Complex64::new(1.0, 0.5),
                num_complex::Complex64::new(2.0, -0.5),
            ],
            &[2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
        vectors: Tensor::from_slice(
            &[
                num_complex::Complex64::new(1.0, 0.0),
                num_complex::Complex64::new(0.0, 0.0),
                num_complex::Complex64::new(0.0, 0.0),
                num_complex::Complex64::new(1.0, 0.0),
            ],
            &[2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
    };

    assert_eq!(qr.q.dims(), &[2, 2]);
    assert_eq!(svd.s.dims(), &[2]);
    assert_eq!(lu.pivots, vec![1, 0]);
    assert_eq!(eigen.values.dims(), &[2]);
    assert_eq!(eig.vectors.dims(), &[2, 2]);
}

#[test]
fn backend_surface_reexports_linalg_prims_types_without_copy_adapters() {
    let q = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let r = q.clone();
    let s = Tensor::<f64>::from_slice(&[3.0, 1.0], &[2], MemoryOrder::ColumnMajor).unwrap();

    let qr: QrTensorResult<f64> = PrimQrTensorResult {
        q: q.clone(),
        r: r.clone(),
    };
    let svd: SvdTensorResult<f64> = PrimSvdTensorResult {
        u: q.clone(),
        s: s.clone(),
        vt: r.clone(),
    };
    let lu: LuTensorResult<f64> = PrimLuTensorResult {
        l: q.clone(),
        u: r.clone(),
        pivots: vec![1, 0],
    };
    let eigen: EigenTensorResult<f64> = PrimEigenTensorResult {
        values: s.clone(),
        vectors: q.clone(),
    };
    let eig: EigTensorResult<f64> = PrimEigTensorResult {
        values: Tensor::from_slice(
            &[
                num_complex::Complex64::new(1.0, 0.5),
                num_complex::Complex64::new(2.0, -0.5),
            ],
            &[2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
        vectors: Tensor::from_slice(
            &[
                num_complex::Complex64::new(1.0, 0.0),
                num_complex::Complex64::new(0.0, 0.0),
                num_complex::Complex64::new(0.0, 0.0),
                num_complex::Complex64::new(1.0, 0.0),
            ],
            &[2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
    };

    fn assert_backend_alias<B: TensorLinalgBackend<f64> + TensorLinalgPrims<f64>>() {}
    assert_backend_alias::<CpuTensorLinalgBackend>();

    assert_eq!(qr.q.dims(), &[2, 2]);
    assert_eq!(svd.s.dims(), &[2]);
    assert_eq!(lu.pivots, vec![1, 0]);
    assert_eq!(eigen.values.dims(), &[2]);
    assert_eq!(eig.vectors.dims(), &[2, 2]);
}
