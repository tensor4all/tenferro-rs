use tenferro_internal_ad_core::{AdMode, DynAdTensor, DynAdTensorRef};
use tenferro_internal_ad_ops::ad::{
    acos_dyn, acosh_dyn, add_dyn, asin_dyn, asinh_dyn, atan2_dyn, atan_dyn, atanh_dyn, cos_dyn,
    cosh_dyn, einsum_dyn, exp_dyn, expm1_dyn, hypot_dyn, log1p_dyn, log_dyn, mean_dyn, pow_dyn,
    sin_dyn, sinh_dyn, sqrt_dyn, std_dyn, sum_dyn, tanh_dyn, var_dyn,
};
use tenferro_internal_frontend_core::DynTensor;
use tenferro_internal_runtime::{set_default_runtime, RuntimeContext};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};
use tidu::expert::Tape;

fn vector_f64(values: &[f64]) -> Tensor<f64> {
    Tensor::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn matrix_f64(values: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(values, dims, MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn sum_dyn_preserves_reverse_metadata() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = Tape::<DynTensor>::new();
    let x = DynAdTensor::new_reverse_leaf(vector_f64(&[1.0, 2.0, 3.0]), &tape).unwrap();

    let out = sum_dyn(DynAdTensorRef::from(&x)).unwrap();
    assert_eq!(
        out.scalar_type(),
        tenferro_internal_frontend_core::ScalarType::F64
    );
    assert_eq!(out.mode(), AdMode::Reverse);
    assert!(out.node_id().is_some());
    assert!(out.dims().is_empty());
}

#[test]
fn einsum_dyn_preserves_reverse_metadata() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = Tape::<DynTensor>::new();
    let a =
        DynAdTensor::new_reverse_leaf(matrix_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), &tape).unwrap();
    let b =
        DynAdTensor::new_reverse_leaf(matrix_f64(&[5.0, 6.0, 7.0, 8.0], &[2, 2]), &tape).unwrap();

    let out = einsum_dyn(
        "ij,jk->ik",
        &[DynAdTensorRef::from(&a), DynAdTensorRef::from(&b)],
    )
    .unwrap();
    assert_eq!(
        out.scalar_type(),
        tenferro_internal_frontend_core::ScalarType::F64
    );
    assert_eq!(out.mode(), AdMode::Reverse);
    assert!(out.node_id().is_some());
    assert_eq!(out.dims(), &[2, 2]);
}

#[test]
fn exp_dyn_preserves_reverse_metadata() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = Tape::<DynTensor>::new();
    let x = DynAdTensor::new_reverse_leaf(vector_f64(&[1.0, 2.0, 3.0]), &tape).unwrap();

    let out = exp_dyn(DynAdTensorRef::from(&x)).unwrap();
    assert_eq!(
        out.scalar_type(),
        tenferro_internal_frontend_core::ScalarType::F64
    );
    assert_eq!(out.mode(), AdMode::Reverse);
    assert!(out.node_id().is_some());
    assert_eq!(out.dims(), &[3]);
}

#[test]
fn add_dyn_preserves_reverse_metadata() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = Tape::<DynTensor>::new();
    let lhs = DynAdTensor::new_reverse_leaf(vector_f64(&[1.0, 2.0, 3.0]), &tape).unwrap();
    let rhs = DynAdTensor::new_reverse_leaf(vector_f64(&[4.0, 5.0, 6.0]), &tape).unwrap();

    let out = add_dyn(DynAdTensorRef::from(&lhs), DynAdTensorRef::from(&rhs)).unwrap();
    assert_eq!(
        out.scalar_type(),
        tenferro_internal_frontend_core::ScalarType::F64
    );
    assert_eq!(out.mode(), AdMode::Reverse);
    assert!(out.node_id().is_some());
    assert_eq!(out.dims(), &[3]);
}

#[test]
fn mean_dyn_preserves_reverse_metadata() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = Tape::<DynTensor>::new();
    let x = DynAdTensor::new_reverse_leaf(vector_f64(&[1.0, 2.0, 3.0]), &tape).unwrap();

    let out = mean_dyn(DynAdTensorRef::from(&x)).unwrap();
    assert_eq!(
        out.scalar_type(),
        tenferro_internal_frontend_core::ScalarType::F64
    );
    assert_eq!(out.mode(), AdMode::Reverse);
    assert!(out.node_id().is_some());
    assert!(out.dims().is_empty());
}

#[test]
fn remaining_unary_dyn_wrappers_preserve_reverse_metadata() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = Tape::<DynTensor>::new();
    let x = DynAdTensor::new_reverse_leaf(vector_f64(&[0.25, 0.5, 0.75]), &tape).unwrap();
    let y = DynAdTensor::new_reverse_leaf(vector_f64(&[1.25, 1.5, 1.75]), &tape).unwrap();

    let outputs: Vec<DynAdTensor> = vec![
        sqrt_dyn(DynAdTensorRef::from(&x)).unwrap(),
        expm1_dyn(DynAdTensorRef::from(&x)).unwrap(),
        log_dyn(DynAdTensorRef::from(&x)).unwrap(),
        log1p_dyn(DynAdTensorRef::from(&x)).unwrap(),
        sin_dyn(DynAdTensorRef::from(&x)).unwrap(),
        cos_dyn(DynAdTensorRef::from(&x)).unwrap(),
        tanh_dyn(DynAdTensorRef::from(&x)).unwrap(),
        asin_dyn(DynAdTensorRef::from(&x)).unwrap(),
        acos_dyn(DynAdTensorRef::from(&x)).unwrap(),
        atan_dyn(DynAdTensorRef::from(&x)).unwrap(),
        sinh_dyn(DynAdTensorRef::from(&x)).unwrap(),
        cosh_dyn(DynAdTensorRef::from(&x)).unwrap(),
        asinh_dyn(DynAdTensorRef::from(&x)).unwrap(),
        atanh_dyn(DynAdTensorRef::from(&x)).unwrap(),
        acosh_dyn(DynAdTensorRef::from(&y)).unwrap(),
    ];

    for out in outputs {
        assert_eq!(
            out.scalar_type(),
            tenferro_internal_frontend_core::ScalarType::F64
        );
        assert_eq!(out.mode(), AdMode::Reverse);
        assert!(out.node_id().is_some());
        assert_eq!(out.dims(), &[3]);
    }
}

#[test]
fn remaining_reduction_and_binary_dyn_wrappers_preserve_reverse_metadata() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = Tape::<DynTensor>::new();
    let lhs = DynAdTensor::new_reverse_leaf(vector_f64(&[1.0, 2.0, 3.0]), &tape).unwrap();
    let rhs = DynAdTensor::new_reverse_leaf(vector_f64(&[4.0, 5.0, 6.0]), &tape).unwrap();

    let outputs: Vec<DynAdTensor> = vec![
        std_dyn(DynAdTensorRef::from(&lhs)).unwrap(),
        var_dyn(DynAdTensorRef::from(&lhs)).unwrap(),
        atan2_dyn(DynAdTensorRef::from(&lhs), DynAdTensorRef::from(&rhs)).unwrap(),
        hypot_dyn(DynAdTensorRef::from(&lhs), DynAdTensorRef::from(&rhs)).unwrap(),
        pow_dyn(DynAdTensorRef::from(&lhs), DynAdTensorRef::from(&rhs)).unwrap(),
    ];

    for out in outputs {
        assert_eq!(
            out.scalar_type(),
            tenferro_internal_frontend_core::ScalarType::F64
        );
        assert_eq!(out.mode(), AdMode::Reverse);
        assert!(out.node_id().is_some());
    }
}

#[test]
fn real_only_dyn_binary_entrypoints_distinguish_dtype_mismatch_from_non_real_inputs() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let lhs32 = DynAdTensor::new_primal(
        Tensor::from_slice(&[0.25_f32], &[], MemoryOrder::ColumnMajor).unwrap(),
    );
    let rhs64 = DynAdTensor::new_primal(
        Tensor::from_slice(&[0.5_f64], &[], MemoryOrder::ColumnMajor).unwrap(),
    );
    let complex = DynAdTensor::new_primal(
        Tensor::from_slice(
            &[num_complex::Complex64::new(0.5, 0.25)],
            &[],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
    );

    let mismatch = match atan2_dyn(DynAdTensorRef::from(&lhs32), DynAdTensorRef::from(&rhs64)) {
        Ok(_) => panic!("mixed real dtypes should hit same-dtype rejection"),
        Err(err) => err,
    };
    assert!(matches!(
        mismatch,
        tenferro_internal_ad_ops::Error::InvalidAdTensor { message }
            if message.contains("requires matching DynAdTensor inputs")
    ));

    let non_real = match hypot_dyn(DynAdTensorRef::from(&rhs64), DynAdTensorRef::from(&complex)) {
        Ok(_) => panic!("complex operands should hit real-only rejection"),
        Err(err) => err,
    };
    assert!(matches!(
        non_real,
        tenferro_internal_ad_ops::Error::InvalidAdTensor { message }
            if message.contains("requires real-valued operands")
    ));
}
