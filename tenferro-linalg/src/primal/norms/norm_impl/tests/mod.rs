use super::*;
use num_complex::Complex64;
use tenferro_device::LogicalMemorySpace;
use tenferro_prims::CpuContext;
use tenferro_tensor::MemoryOrder;

const CPU: LogicalMemorySpace = LogicalMemorySpace::MainMemory;

fn tensor_data<T: tenferro_algebra::Scalar + Copy>(tensor: &Tensor<T>) -> Vec<T> {
    let contiguous = tensor.contiguous(MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len = contiguous.len();
    contiguous.buffer().as_slice().unwrap()[offset..offset + len].to_vec()
}

fn assert_close(value: f64, expected: f64) {
    assert!(
        (value - expected).abs() < 1.0e-12,
        "got {value}, expected {expected}"
    );
}

#[test]
fn norm_real_impl_covers_vector_and_matrix_branches() {
    let mut ctx = CpuContext::new(1);

    let empty_vec = Tensor::<f64>::zeros(&[0], CPU, MemoryOrder::ColumnMajor).unwrap();
    assert_eq!(
        tensor_data(&norm_real_impl(&mut ctx, &empty_vec, NormKind::Inf).unwrap()),
        vec![0.0]
    );

    let vector = Tensor::from_slice(&[3.0_f64, -4.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    assert_eq!(
        tensor_data(&norm_real_impl(&mut ctx, &vector, NormKind::L1).unwrap()),
        vec![7.0]
    );
    assert_close(
        tensor_data(&norm_real_impl(&mut ctx, &vector, NormKind::Fro).unwrap())[0],
        5.0,
    );
    assert_close(
        tensor_data(&norm_real_impl(&mut ctx, &vector, NormKind::Lp(3.0)).unwrap())[0],
        (3.0_f64.powi(3) + 4.0_f64.powi(3)).powf(1.0 / 3.0),
    );
    assert!(norm_real_impl(&mut ctx, &vector, NormKind::Nuclear).is_err());

    let empty_matrix = Tensor::<f64>::zeros(&[0, 3], CPU, MemoryOrder::ColumnMajor).unwrap();
    assert_eq!(
        tensor_data(&norm_real_impl(&mut ctx, &empty_matrix, NormKind::L1).unwrap()),
        vec![0.0]
    );
    assert_eq!(
        tensor_data(&norm_real_impl(&mut ctx, &empty_matrix, NormKind::Inf).unwrap()),
        vec![0.0]
    );

    let matrix =
        Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 2.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    assert_eq!(
        tensor_data(&norm_real_impl(&mut ctx, &matrix, NormKind::Nuclear).unwrap()),
        vec![3.0]
    );
    assert_eq!(
        tensor_data(&norm_real_impl(&mut ctx, &matrix, NormKind::Spectral).unwrap()),
        vec![2.0]
    );
    assert_eq!(
        tensor_data(&norm_real_impl(&mut ctx, &matrix, NormKind::L1).unwrap()),
        vec![2.0]
    );
    assert_eq!(
        tensor_data(&norm_real_impl(&mut ctx, &matrix, NormKind::Inf).unwrap()),
        vec![2.0]
    );
    assert_close(
        tensor_data(&norm_real_impl(&mut ctx, &matrix, NormKind::Fro).unwrap())[0],
        5.0_f64.sqrt(),
    );

    let err = norm_real_impl(&mut ctx, &matrix, NormKind::Lp(2.0)).unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("not yet implemented")));
}

#[test]
fn norm_complex_impl_covers_vector_and_matrix_branches() {
    let mut ctx = CpuContext::new(1);

    let empty_vec = Tensor::<Complex64>::zeros(&[0], CPU, MemoryOrder::ColumnMajor).unwrap();
    assert_eq!(
        tensor_data(&norm_complex_impl(&mut ctx, &empty_vec, NormKind::Inf).unwrap()),
        vec![0.0]
    );

    let vector = Tensor::from_slice(
        &[Complex64::new(3.0, 4.0), Complex64::new(0.0, 2.0)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    assert_eq!(
        tensor_data(&norm_complex_impl(&mut ctx, &vector, NormKind::L1).unwrap()),
        vec![7.0]
    );
    assert_close(
        tensor_data(&norm_complex_impl(&mut ctx, &vector, NormKind::Fro).unwrap())[0],
        29.0_f64.sqrt(),
    );
    assert_close(
        tensor_data(&norm_complex_impl(&mut ctx, &vector, NormKind::Lp(3.0)).unwrap())[0],
        (5.0_f64.powi(3) + 2.0_f64.powi(3)).powf(1.0 / 3.0),
    );
    assert!(norm_complex_impl(&mut ctx, &vector, NormKind::Nuclear).is_err());

    let empty_matrix = Tensor::<Complex64>::zeros(&[0, 3], CPU, MemoryOrder::ColumnMajor).unwrap();
    assert_eq!(
        tensor_data(&norm_complex_impl(&mut ctx, &empty_matrix, NormKind::L1).unwrap()),
        vec![0.0]
    );
    assert_eq!(
        tensor_data(&norm_complex_impl(&mut ctx, &empty_matrix, NormKind::Inf).unwrap()),
        vec![0.0]
    );

    let matrix = Tensor::from_slice(
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 2.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    assert_close(
        tensor_data(&norm_complex_impl(&mut ctx, &matrix, NormKind::Nuclear).unwrap())[0],
        2.0_f64.sqrt() + 2.0,
    );
    assert_eq!(
        tensor_data(&norm_complex_impl(&mut ctx, &matrix, NormKind::Spectral).unwrap()),
        vec![2.0]
    );
    assert_eq!(
        tensor_data(&norm_complex_impl(&mut ctx, &matrix, NormKind::L1).unwrap()),
        vec![2.0]
    );
    assert_eq!(
        tensor_data(&norm_complex_impl(&mut ctx, &matrix, NormKind::Inf).unwrap()),
        vec![2.0]
    );
    assert_close(
        tensor_data(&norm_complex_impl(&mut ctx, &matrix, NormKind::Fro).unwrap())[0],
        6.0_f64.sqrt(),
    );

    let err = norm_complex_impl(&mut ctx, &matrix, NormKind::Lp(2.0)).unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("not yet implemented")));
}
