use std::any::Any;
use std::sync::Arc;

use num_complex::Complex64;
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_ops::{ext_op::ExtensionOp, std_tensor_op::StdTensorOp, SymDim};
use tenferro_runtime::{
    CompareDir, DType, Error as RuntimeError, ErrorPhase, GraphCompiler, Tensor, TensorOpsExt,
    TracedTensor, TypedTensor, TypedTensorMaskOpsExt, TypedTensorOpsExt,
};
use tenferro_tensor::ValidationError;

use crate::support::{cpu_runtime, run_compiled_one};

#[derive(Clone, Debug)]
struct TestExtensionOp;

impl ExtensionOp for TestExtensionOp {
    fn family_id(&self) -> &'static str {
        "tenferro_ad_tests.identity.v1"
    }

    fn payload_hash(&self, _hasher: &mut dyn std::hash::Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
    }
}

#[test]
fn traced_add_uses_numpy_broadcasting_for_rank_padding_and_singletons() {
    let lhs = TracedTensor::from_vec_col_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let rhs =
        TracedTensor::from_vec_col_major(vec![1, 4], vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap();
    let y = lhs.add(&rhs).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let executor = cpu_runtime();
    let out = run_compiled_one(&executor, &program, &[]).unwrap();

    assert_eq!(out.shape(), &[3, 4]);
    assert_eq!(
        out.into_vec_col_major::<f64>().unwrap().1,
        vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 31.0, 32.0, 33.0, 41.0, 42.0, 43.0,]
    );
}

#[test]
fn traced_tensor_methods_cover_core_elementwise_surface() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
    let y = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();
    let cond = x.compare(&y, CompareDir::Gt).unwrap();

    let _ = x.sub(&y).unwrap();
    let _ = x.mul(&y).unwrap();
    let _ = x.div(&y).unwrap();
    let _ = x.pow(&y).unwrap();
    let _ = x.maximum(&y).unwrap();
    let _ = x.minimum(&y).unwrap();
    let _ = TracedTensor::where_select(&cond, &x, &y).unwrap();
    let _ = x.clamp(&y, &x).unwrap();
    let _ = x.neg();
    let _ = x.abs();
    let _ = x.sign();
    let _ = x.conj();
    let _ = x.exp();
    let _ = x.log();
    let _ = x.sin();
    let _ = x.cos();
    let _ = x.tanh();
    let _ = x.sqrt();
    let _ = x.rsqrt();
    let _ = x.expm1();
    let _ = x.log1p();
}

#[test]
fn traced_unary_methods_forward_to_distinct_primal_ops() {
    fn run(output: &TracedTensor) -> Tensor {
        let mut compiler = GraphCompiler::new();
        let program = compiler.compile(output).unwrap();
        let executor = cpu_runtime();
        run_compiled_one(&executor, &program, &[]).unwrap()
    }

    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 4.0]).unwrap();
    assert_eq!(
        run(&x.neg().unwrap()).as_slice::<f64>().unwrap(),
        &[-1.0, -4.0]
    );
    assert_eq!(
        run(&x.abs().unwrap()).as_slice::<f64>().unwrap(),
        &[1.0, 4.0]
    );
    assert_eq!(
        run(&x.sign().unwrap()).as_slice::<f64>().unwrap(),
        &[1.0, 1.0]
    );
    assert_eq!(
        run(&x.sqrt().unwrap()).as_slice::<f64>().unwrap(),
        &[1.0, 2.0]
    );
    assert_eq!(
        run(&x.rsqrt().unwrap()).as_slice::<f64>().unwrap(),
        &[1.0, 0.5]
    );

    let analytic = TracedTensor::from_vec_col_major(vec![2], vec![0.0_f64, 1.0]).unwrap();
    let exp = run(&analytic.exp().unwrap());
    let log = run(&x.log().unwrap());
    let sin = run(&analytic.sin().unwrap());
    let cos = run(&analytic.cos().unwrap());
    let tanh = run(&analytic.tanh().unwrap());
    let expm1 = run(&analytic.expm1().unwrap());
    let log1p = run(&analytic.log1p().unwrap());
    assert_eq!(exp.as_slice::<f64>().unwrap(), &[1.0, std::f64::consts::E]);
    assert_eq!(log.as_slice::<f64>().unwrap(), &[0.0, 4.0_f64.ln()]);
    assert_eq!(sin.as_slice::<f64>().unwrap(), &[0.0, 1.0_f64.sin()]);
    assert_eq!(cos.as_slice::<f64>().unwrap(), &[1.0, 1.0_f64.cos()]);
    assert_eq!(tanh.as_slice::<f64>().unwrap(), &[0.0, 1.0_f64.tanh()]);
    assert_eq!(expm1.as_slice::<f64>().unwrap(), &[0.0, 1.0_f64.exp_m1()]);
    assert_eq!(log1p.as_slice::<f64>().unwrap(), &[0.0, 1.0_f64.ln_1p()]);

    let z = TracedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)],
    )
    .unwrap();
    assert_eq!(
        run(&z.conj().unwrap()).as_slice::<Complex64>().unwrap(),
        &[Complex64::new(1.0, -2.0), Complex64::new(3.0, 4.0)]
    );
}

#[test]
fn traced_compare_returns_bool_and_where_select_accepts_bool_condition() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
    let y = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();
    let cond = x.compare(&y, CompareDir::Gt).unwrap();
    assert_eq!(cond.dtype, DType::Bool);

    let selected = TracedTensor::where_select(&cond, &x, &y).unwrap();

    let mut compiler = GraphCompiler::new();
    let cond_program = compiler.compile(&cond).unwrap();
    let selected_program = compiler.compile(&selected).unwrap();
    let executor = cpu_runtime();
    let cond_out = run_compiled_one(&executor, &cond_program, &[]).unwrap();
    let selected_out = run_compiled_one(&executor, &selected_program, &[]).unwrap();

    assert_eq!(cond_out.dtype(), DType::Bool);
    assert_eq!(cond_out.as_slice::<bool>().unwrap(), &[true, false]);
    assert_eq!(
        selected_out.into_vec_col_major::<f64>().unwrap().1,
        vec![2.0, 8.0]
    );
}

#[test]
fn eager_add_uses_numpy_broadcasting_for_rank_padding_and_singletons() {
    let ctx = EagerRuntime::new();
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1, 4], vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap(),
        ctx,
    )
    .unwrap();

    let out = lhs.add(&rhs).unwrap();

    assert_eq!(out.shape(), &[3, 4]);
    assert_eq!(
        out.to_tensor()
            .unwrap()
            .into_vec_col_major::<f64>()
            .unwrap()
            .1,
        vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 31.0, 32.0, 33.0, 41.0, 42.0, 43.0,]
    );
}

#[test]
fn eager_tensor_methods_cover_core_elementwise_surface() {
    let ctx = EagerRuntime::new();
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap(),
        ctx,
    )
    .unwrap();
    let cond = x.compare(&y, CompareDir::Gt).unwrap();

    let _ = x.sub(&y).unwrap();
    let _ = x.mul(&y).unwrap();
    let _ = x.div(&y).unwrap();
    let _ = x.pow(&y).unwrap();
    let _ = x.maximum(&y).unwrap();
    let _ = x.minimum(&y).unwrap();
    let _ = EagerTensor::where_select(&cond, &x, &y).unwrap();
    let _ = x.clamp(&y, &x).unwrap();
    let _ = x.neg().unwrap();
    let _ = x.abs().unwrap();
    let _ = x.sign().unwrap();
    let _ = x.conj().unwrap();
    let _ = x.exp().unwrap();
    let _ = x.log().unwrap();
    let _ = x.sin().unwrap();
    let _ = x.cos().unwrap();
    let _ = x.tanh().unwrap();
    let _ = x.sqrt().unwrap();
    let _ = x.rsqrt().unwrap();
    let _ = x.expm1().unwrap();
    let _ = x.log1p().unwrap();
}

#[test]
fn eager_tensor_methods_cover_conversion_matmul_and_extension_standard_op() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let converted = x.convert(DType::C64).unwrap();
    assert_eq!(converted.dtype(), DType::C64);
    assert_eq!(
        converted
            .materialized()
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap(),
        &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]
    );

    let convert_err = x.convert(DType::I32).unwrap_err();
    assert!(convert_err
        .to_string()
        .contains("unsupported dtype conversion"));

    let casted = x.cast(DType::I32).unwrap();
    assert_eq!(
        casted.materialized().unwrap().as_slice::<i32>().unwrap(),
        &[1, 2]
    );

    let negated = tenferro_ad::extension::apply_standard_op(StdTensorOp::Neg, &[&x]).unwrap();
    assert_eq!(
        negated.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[-1.0, -2.0]
    );

    let extension_err = tenferro_ad::extension::apply_standard_op(
        StdTensorOp::Extension(Arc::new(TestExtensionOp)),
        &[&x],
    )
    .err()
    .unwrap();
    assert!(matches!(
        extension_err,
        RuntimeError::Validation {
            phase: ErrorPhase::Execution,
            source: ValidationError::InvalidArgument { argument: "op", .. },
            ..
        }
    ));

    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let product = a.matmul(&b).unwrap();
    assert_eq!(product.shape(), &[2, 2]);
    assert_eq!(
        product.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[22.0, 28.0, 49.0, 64.0]
    );

    let other_ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let other = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        other_ctx,
    )
    .unwrap();
    let err = x.add(&other).err().unwrap();
    assert!(matches!(err, tenferro_ad::Error::ContextMismatch { .. }));
}

#[test]
fn eager_compare_returns_bool_and_where_select_accepts_bool_condition() {
    let ctx = EagerRuntime::new();
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap(),
        ctx,
    )
    .unwrap();

    let cond = x.compare(&y, CompareDir::Gt).unwrap();
    let selected = EagerTensor::where_select(&cond, &x, &y).unwrap();

    assert_eq!(cond.dtype(), DType::Bool);
    assert_eq!(
        cond.materialized().unwrap().as_slice::<bool>().unwrap(),
        &[true, false]
    );
    assert_eq!(
        selected
            .to_tensor()
            .unwrap()
            .into_vec_col_major::<f64>()
            .unwrap()
            .1,
        vec![2.0, 8.0]
    );
}

#[test]
fn tensor_add_uses_numpy_broadcasting_with_explicit_backend() {
    let mut backend = CpuBackend::new();
    let lhs = Tensor::from_vec_col_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1, 4], vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap();

    let out = lhs.add(&rhs, &mut backend).unwrap();

    assert_eq!(out.shape(), &[3, 4]);
    assert_eq!(
        out.into_vec_col_major::<f64>().unwrap().1,
        vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 31.0, 32.0, 33.0, 41.0, 42.0, 43.0,]
    );
}

#[test]
fn tensor_extension_trait_exposes_initial_elementwise_methods() {
    let mut backend = CpuBackend::new();
    let x = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
    let y = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();
    let cond = x.compare(&y, CompareDir::Gt, &mut backend).unwrap();

    let _ = x.sub(&y, &mut backend).unwrap();
    let _ = x.mul(&y, &mut backend).unwrap();
    let _ = x.div(&y, &mut backend).unwrap();
    let _ = x.pow(&y, &mut backend).unwrap();
    let _ = x.maximum(&y, &mut backend).unwrap();
    let _ = x.minimum(&y, &mut backend).unwrap();
    let _ = cond.where_select(&x, &y, &mut backend).unwrap();
    let _ = x.clamp(&y, &x, &mut backend).unwrap();
    let _ = x.neg(&mut backend).unwrap();
    let _ = x.abs(&mut backend).unwrap();
    let _ = x.sign(&mut backend).unwrap();
    let _ = x.conj(&mut backend).unwrap();
    let _ = x.exp(&mut backend).unwrap();
    let _ = x.log(&mut backend).unwrap();
    let _ = x.sin(&mut backend).unwrap();
    let _ = x.cos(&mut backend).unwrap();
    let _ = x.tanh(&mut backend).unwrap();
    let _ = x.sqrt(&mut backend).unwrap();
    let _ = x.rsqrt(&mut backend).unwrap();
    let _ = x.expm1(&mut backend).unwrap();
    let _ = x.log1p(&mut backend).unwrap();
}

#[test]
fn tensor_compare_returns_bool_and_where_select_accepts_bool_condition() {
    let mut backend = CpuBackend::new();
    let x = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
    let y = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();

    let cond = x.compare(&y, CompareDir::Gt, &mut backend).unwrap();
    let selected = cond.where_select(&x, &y, &mut backend).unwrap();

    assert_eq!(cond.dtype(), DType::Bool);
    assert_eq!(cond.as_slice::<bool>().unwrap(), &[true, false]);
    assert_eq!(
        selected.into_vec_col_major::<f64>().unwrap().1,
        vec![2.0, 8.0]
    );
}

#[test]
fn typed_tensor_add_uses_numpy_broadcasting_with_explicit_backend() {
    let mut backend = CpuBackend::new();
    let lhs = TypedTensor::<f64>::from_vec_col_major(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap();
    let rhs =
        TypedTensor::<f64>::from_vec_col_major(vec![1, 4], vec![10.0, 20.0, 30.0, 40.0]).unwrap();

    let out = lhs.add(&rhs, &mut backend).unwrap();

    assert_eq!(out.shape(), &[3, 4]);
    assert_eq!(
        out.into_vec_col_major().unwrap().1,
        vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 31.0, 32.0, 33.0, 41.0, 42.0, 43.0,]
    );
}

#[test]
fn typed_tensor_extension_trait_exposes_initial_elementwise_methods() {
    let mut backend = CpuBackend::new();
    let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
    let y = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]).unwrap();
    let cond = x.compare(&y, CompareDir::Gt, &mut backend).unwrap();

    let _ = x.sub(&y, &mut backend).unwrap();
    let _ = x.mul(&y, &mut backend).unwrap();
    let _ = x.div(&y, &mut backend).unwrap();
    let _ = x.pow(&y, &mut backend).unwrap();
    let _ = x.maximum(&y, &mut backend).unwrap();
    let _ = x.minimum(&y, &mut backend).unwrap();
    let _ = cond.where_select(&x, &y, &mut backend).unwrap();
    let _ = x.clamp(&y, &x, &mut backend).unwrap();
    let _ = x.neg(&mut backend).unwrap();
    let _ = x.abs(&mut backend).unwrap();
    let _ = x.sign(&mut backend).unwrap();
    let _ = x.conj(&mut backend).unwrap();
    let _ = x.exp(&mut backend).unwrap();
    let _ = x.log(&mut backend).unwrap();
    let _ = x.sin(&mut backend).unwrap();
    let _ = x.cos(&mut backend).unwrap();
    let _ = x.tanh(&mut backend).unwrap();
    let _ = x.sqrt(&mut backend).unwrap();
    let _ = x.rsqrt(&mut backend).unwrap();
    let _ = x.expm1(&mut backend).unwrap();
    let _ = x.log1p(&mut backend).unwrap();
}

#[test]
fn typed_tensor_compare_returns_bool_and_where_select_accepts_bool_condition() {
    let mut backend = CpuBackend::new();
    let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
    let y = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]).unwrap();

    let cond: TypedTensor<bool> = x.compare(&y, CompareDir::Gt, &mut backend).unwrap();
    let selected = cond.where_select(&x, &y, &mut backend).unwrap();

    assert_eq!(cond.host_data().unwrap(), &[true, false]);
    assert_eq!(selected.into_vec_col_major().unwrap().1, vec![2.0, 8.0]);
}
