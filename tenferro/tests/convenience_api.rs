mod support;
use num_complex::{Complex32, Complex64};
use support::{
    einsum, einsum_subscripts, einsum_subscripts_with, einsum_with, run_many_traced_with, RunTraced,
};
use tenferro::{CpuBackend, GraphExecutor, Tensor, TensorScalar, TracedTensor};
use tenferro_tensor::DType;

#[test]
fn traced_tensor_new_and_tensor_as_slice_cover_common_f64_flow() {
    let a = TracedTensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = TracedTensor::from_vec(vec![2, 3], vec![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0]);

    let mut sum = &a + &b;
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = sum.run_with(&mut engine).unwrap();

    assert_eq!(
        result.as_slice::<f64>(),
        Some([7.0, 7.0, 7.0, 7.0, 7.0, 7.0].as_slice())
    );
    assert_eq!(result.as_slice::<f32>(), None);
}

#[test]
fn tensor_scalar_is_reexported_from_tenferro() {
    let tensor = <f64 as TensorScalar>::into_tensor(vec![2], vec![1.0, 2.0]);

    assert_eq!(tensor.as_slice::<f64>(), Some([1.0, 2.0].as_slice()));
}

#[test]
fn traced_tensor_shape_helpers_and_aliases_cover_public_surface() {
    let a = TracedTensor::from_vec(vec![2, 3], vec![1.0_f64; 6]);
    let b = TracedTensor::from_vec(vec![3, 4], vec![1.0_f64; 12]);

    assert_eq!(a.try_concrete_shape(), Some(vec![2, 3]));
    assert!(a.input_key().is_some());
    assert!((&a + &a).input_key().is_none());
    assert_eq!(a.axis_sym_dim(0).constant_value(), Some(2));
    assert_eq!(a.sym_shape().unwrap().len(), 2);

    let symbolic = TracedTensor::input_symbolic_shape(DType::F64, 2);
    assert_eq!(symbolic.try_concrete_shape(), None);
    assert!(symbolic.sym_shape().is_none());
    assert!(symbolic.axis_sym_dim(1).constant_value().is_none());

    let m = a.axis_sym_dim(0);
    let k = a.axis_sym_dim(1);
    let n = b.axis_sym_dim(1);
    let broad = a.broadcast_in_dim_sym(&[m, k, n], &[0, 1], &[&b]);
    assert_eq!(broad.rank, 3);

    assert_eq!(a.broadcast(&[2, 3], &[0, 1]).rank, 2);
    assert_eq!(a.reduce_max(&[0]).rank, 1);
    assert_eq!(a.reduce_min(&[1]).rank, 1);
    assert_eq!(a.reduce_prod(&[0, 1]).rank, 0);
}

#[test]
fn traced_tensor_scaling_covers_dtype_specific_constants() {
    let f32_tensor =
        TracedTensor::from_tensor_concrete_shape(Tensor::from_vec(vec![1], vec![1.0_f32]));
    assert_eq!(f32_tensor.scale_real(2.5).dtype, DType::F32);

    let i64_tensor =
        TracedTensor::from_tensor_concrete_shape(Tensor::from_vec(vec![1], vec![2_i64]));
    assert_eq!(i64_tensor.scale_real(2.5).dtype, DType::I64);

    let c64_tensor = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec(
        vec![1],
        vec![Complex64::new(1.0, 2.0)],
    ));
    assert_eq!(c64_tensor.scale_real(2.0).dtype, DType::C64);
    assert_eq!(
        c64_tensor.scale_complex(Complex64::new(0.0, 1.0)).dtype,
        DType::C64
    );

    let c32_tensor = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec(
        vec![1],
        vec![Complex32::new(1.0, 2.0)],
    ));
    assert_eq!(c32_tensor.scale_real(2.0).dtype, DType::C32);
    assert_eq!(
        c32_tensor.scale_complex(Complex64::new(0.0, 1.0)).dtype,
        DType::C32
    );
}
