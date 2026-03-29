use num_complex::Complex64;
use tenferro::{set_default_runtime, AdMode, Error, RuntimeContext, Tensor};
use tenferro_internal_ad_core::{pullback, AdTensor};
use tenferro_internal_frontend_core::DynTensor;
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
use tidu::Tape;

#[test]
fn rank0_reverse_tensor_scale_allocates_fresh_output_node() {
    let tape = Tape::<DynTensor>::new();
    let x = AdTensor::new_reverse_leaf_with_tangent(
        DenseTensor::<Complex64>::from_slice(
            &[Complex64::new(1.0, 2.0)],
            &[],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
        DenseTensor::<Complex64>::from_slice(
            &[Complex64::new(-1.0, 0.5)],
            &[],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
        &tape,
    )
    .unwrap();
    let alpha: Tensor = AdTensor::new_reverse_leaf(
        DenseTensor::<Complex64>::from_slice(
            &[Complex64::new(0.0, 1.0)],
            &[],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
        &tape,
    )
    .unwrap()
    .into();
    let y = Tensor::from(x.clone()).scale(&alpha).unwrap();
    let y = y.as_c64().unwrap();
    assert_eq!(y.mode(), AdMode::Reverse);
    assert!(y
        .tape()
        .expect("reverse output should expose a tape")
        .same_tape(&tape));
    assert_ne!(y.node_id(), x.node_id());
    assert_eq!(
        y.primal().buffer().as_slice().unwrap()[0],
        Complex64::new(-2.0, 1.0)
    );
}

#[test]
fn rank0_reverse_tensor_sqrt_registers_pullback_chain() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let tape = Tape::<DynTensor>::new();
    let x = AdTensor::new_reverse_leaf(
        DenseTensor::<f64>::from_slice(&[4.0_f64], &[], MemoryOrder::ColumnMajor).unwrap(),
        &tape,
    )
    .unwrap();
    let y = Tensor::from(x.clone()).sqrt().unwrap();
    let cotangent =
        DenseTensor::<f64>::from_slice(&[3.0_f64], &[], MemoryOrder::ColumnMajor).unwrap();
    let grads = pullback(y.as_f64().unwrap(), &cotangent).unwrap();
    assert_eq!(
        grads
            .get(&x.node_id().unwrap())
            .unwrap()
            .buffer()
            .as_slice()
            .unwrap(),
        &[0.75]
    );
}

#[test]
fn rank0_reverse_tensor_scale_returns_error_on_mixed_reverse_tapes() {
    let tape_a = Tape::<DynTensor>::new();
    let tape_b = Tape::<DynTensor>::new();
    let x: Tensor = AdTensor::new_reverse_leaf(
        DenseTensor::<f64>::from_slice(&[2.0_f64], &[], MemoryOrder::ColumnMajor).unwrap(),
        &tape_a,
    )
    .unwrap()
    .into();
    let y: Tensor = AdTensor::new_reverse_leaf(
        DenseTensor::<f64>::from_slice(&[3.0_f64], &[], MemoryOrder::ColumnMajor).unwrap(),
        &tape_b,
    )
    .unwrap()
    .into();
    let err = match x.scale(&y) {
        Ok(_) => panic!("mixed reverse tapes should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::MixedReverseTape { .. }));
}

#[test]
fn rank0_reverse_tensor_div_scalar_returns_error_on_mixed_reverse_tapes() {
    let tape_a = Tape::<DynTensor>::new();
    let tape_b = Tape::<DynTensor>::new();
    let x: Tensor = AdTensor::new_reverse_leaf(
        DenseTensor::<f64>::from_slice(&[2.0_f64], &[], MemoryOrder::ColumnMajor).unwrap(),
        &tape_a,
    )
    .unwrap()
    .into();
    let y: Tensor = AdTensor::new_reverse_leaf(
        DenseTensor::<f64>::from_slice(&[3.0_f64], &[], MemoryOrder::ColumnMajor).unwrap(),
        &tape_b,
    )
    .unwrap()
    .into();
    let err = match x.div_scalar(&y) {
        Ok(_) => panic!("mixed reverse tapes should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::MixedReverseTape { .. }));
}
