use crate::support::{forward_rank0_f64, reverse_rank0_f64};
use num_complex::Complex64;
use tenferro::{AdMode, ScalarValue, Tensor};
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

#[test]
fn tensor_mode_reports_primal_forward_and_reverse_states() {
    let primal = Tensor::from_tensor(
        DenseTensor::<f64>::from_slice(&[2.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    );
    let forward = forward_rank0_f64(2.0, 1.0);
    let reverse = reverse_rank0_f64(2.0);

    assert_eq!(primal.mode(), AdMode::Primal);
    assert_eq!(forward.mode(), AdMode::Forward);
    assert_eq!(reverse.mode(), AdMode::Reverse);
}

#[test]
fn conj_is_identity_for_real_tensors() {
    let x = Tensor::from_tensor(
        DenseTensor::<f64>::from_slice(&[1.0, -2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    );

    let y = x.conj();
    assert_eq!(y.scalar_type(), x.scalar_type());
    assert_eq!(y.dims(), x.dims());
    assert_eq!(
        y.as_f64().unwrap().primal().buffer().as_slice().unwrap(),
        &[1.0, -2.0]
    );
}

#[test]
fn conj_conjugates_complex_payload_and_preserves_mode() {
    let x = Tensor::from_tensor(
        DenseTensor::<Complex64>::from_slice(
            &[Complex64::new(1.0, 2.0)],
            &[],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
    );
    let y = x.conj();

    assert_eq!(y.mode(), AdMode::Primal);
    assert_eq!(
        y.try_scalar_value().unwrap(),
        ScalarValue::C64(Complex64::new(1.0, -2.0))
    );

    let z = y.conj();
    assert_eq!(
        z.try_scalar_value().unwrap(),
        ScalarValue::C64(Complex64::new(1.0, 2.0))
    );
}
