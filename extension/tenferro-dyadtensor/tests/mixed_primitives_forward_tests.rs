use num_complex::Complex64;
use tenferro_dyadtensor::{forward_ad, ScalarType, Tensor};
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

mod support;

use support::{primal_rank0_c64, primal_rank0_f64};

#[test]
fn scale_preserves_forward_tensor_and_scalar_ad() {
    let (primal, tangent) = forward_ad::dual_level(|fw| {
        let x = fw.make_dual(
            &Tensor::from_tensor(
                DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor)
                    .unwrap(),
            ),
            &Tensor::from_tensor(
                DenseTensor::<f64>::from_slice(&[0.5, 0.25], &[2], MemoryOrder::ColumnMajor)
                    .unwrap(),
            ),
        )?;
        let a = fw.make_dual(
            &Tensor::from_tensor(
                DenseTensor::<f64>::from_slice(&[3.0], &[], MemoryOrder::ColumnMajor).unwrap(),
            ),
            &Tensor::from_tensor(
                DenseTensor::<f64>::from_slice(&[0.1], &[], MemoryOrder::ColumnMajor).unwrap(),
            ),
        )?;
        let y = x.scale(&a)?;
        fw.unpack_dual(&y)
    })
    .unwrap();

    let primal = primal.as_f64().unwrap();
    let tangent = tangent.unwrap();
    let tangent = tangent.as_f64().unwrap();

    assert_eq!(primal.primal().buffer().as_slice().unwrap(), &[3.0, 6.0]);
    assert_eq!(tangent.primal().buffer().as_slice().unwrap(), &[1.6, 0.95]);
}

#[test]
fn axpby_works_for_complex64_primal_values() {
    let x = Tensor::from_tensor(
        DenseTensor::<Complex64>::from_slice(
            &[Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
            &[2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
    );
    let y = x.clone();
    let a = primal_rank0_c64(Complex64::new(2.0, 0.0));
    let b = primal_rank0_c64(Complex64::new(-1.0, 0.5));

    let out = x.axpby(&a, &y, &b).unwrap();
    assert_eq!(out.scalar_type(), ScalarType::C64);

    let values = out.as_c64().unwrap().primal().buffer().as_slice().unwrap();
    assert_eq!(values[0], Complex64::new(0.5, 1.5));
    assert_eq!(values[1], Complex64::new(2.5, 0.0));
}

#[test]
fn scalar_mul_and_tensor_div_scalar_delegate_to_named_primitives() {
    let x = Tensor::from_tensor(
        DenseTensor::<f64>::from_slice(&[2.0, 4.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    );
    let a = primal_rank0_f64(3.0_f64);

    let scaled = x.scale(&a).unwrap();
    let divided = scaled.div_scalar(&a).unwrap();

    assert_eq!(
        scaled
            .as_f64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[6.0, 12.0]
    );
    assert_eq!(
        divided
            .as_f64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[2.0, 4.0]
    );
}

#[test]
fn scale_promotes_f64_tensor_with_c64_rank0_tensor() {
    let x = Tensor::from_tensor(
        DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    );
    let a = primal_rank0_c64(Complex64::new(0.0, 2.0));

    let out = x.scale(&a).unwrap();
    assert_eq!(out.scalar_type(), ScalarType::C64);
    assert_eq!(
        out.as_c64().unwrap().primal().buffer().as_slice().unwrap(),
        &[Complex64::new(0.0, 2.0), Complex64::new(0.0, 4.0)]
    );
}
