use std::any::Any;
use std::sync::Arc;

use num_complex::Complex64;
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_ops::{ext_op::ExtensionOp, std_tensor_op::StdTensorOp, SymDim};
use tenferro_runtime::{
    traced_tensor, CompareDir, DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor,
    TypedTensor,
};

use tenferro_ad::eager_tensor;

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
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> Vec<(DType, Vec<SymDim>)> {
        vec![(input_dtypes[0], input_shapes[0].to_vec())]
    }

    fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        Ok(vec![inputs[0].clone()])
    }
}

#[test]
fn traced_add_uses_numpy_broadcasting_for_rank_padding_and_singletons() {
    let lhs = TracedTensor::from_vec_row_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let rhs =
        TracedTensor::from_vec_row_major(vec![1, 4], vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap();
    let y = traced_tensor::add(&lhs, &rhs).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let out = executor.run(&program).unwrap();

    assert_eq!(out.shape(), &[3, 4]);
    assert_eq!(
        out.into_vec_row_major::<f64>().unwrap().1,
        vec![11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0,]
    );
}

#[test]
fn traced_tensor_module_exposes_initial_elementwise_free_functions() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
    let y = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();
    let cond = traced_tensor::compare(&x, &y, CompareDir::Gt).unwrap();

    let _ = traced_tensor::sub(&x, &y).unwrap();
    let _ = traced_tensor::mul(&x, &y).unwrap();
    let _ = traced_tensor::div(&x, &y).unwrap();
    let _ = traced_tensor::pow(&x, &y).unwrap();
    let _ = traced_tensor::maximum(&x, &y).unwrap();
    let _ = traced_tensor::minimum(&x, &y).unwrap();
    let _ = traced_tensor::where_select(&cond, &x, &y).unwrap();
    let _ = traced_tensor::clamp(&x, &y, &x).unwrap();
    let _ = traced_tensor::neg(&x);
    let _ = traced_tensor::abs(&x);
    let _ = traced_tensor::sign(&x);
    let _ = traced_tensor::conj(&x);
    let _ = traced_tensor::exp(&x);
    let _ = traced_tensor::log(&x);
    let _ = traced_tensor::sin(&x);
    let _ = traced_tensor::cos(&x);
    let _ = traced_tensor::tanh(&x);
    let _ = traced_tensor::sqrt(&x);
    let _ = traced_tensor::rsqrt(&x);
    let _ = traced_tensor::expm1(&x);
    let _ = traced_tensor::log1p(&x);
}

#[test]
fn traced_unary_methods_forward_to_distinct_primal_ops() {
    fn run(output: &TracedTensor) -> Tensor {
        let mut compiler = GraphCompiler::new();
        let program = compiler.compile(output).unwrap();
        let mut executor = GraphExecutor::new(CpuBackend::new());
        executor.run(&program).unwrap()
    }

    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 4.0]).unwrap();
    assert_eq!(run(&x.neg()).as_slice::<f64>().unwrap(), &[-1.0, -4.0]);
    assert_eq!(run(&x.abs()).as_slice::<f64>().unwrap(), &[1.0, 4.0]);
    assert_eq!(run(&x.sign()).as_slice::<f64>().unwrap(), &[1.0, 1.0]);
    assert_eq!(run(&x.sqrt()).as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    assert_eq!(run(&x.rsqrt()).as_slice::<f64>().unwrap(), &[1.0, 0.5]);

    let analytic = TracedTensor::from_vec_col_major(vec![2], vec![0.0_f64, 1.0]).unwrap();
    let exp = run(&analytic.exp());
    let log = run(&x.log());
    let sin = run(&analytic.sin());
    let cos = run(&analytic.cos());
    let tanh = run(&analytic.tanh());
    let expm1 = run(&analytic.expm1());
    let log1p = run(&analytic.log1p());
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
        run(&z.conj()).as_slice::<Complex64>().unwrap(),
        &[Complex64::new(1.0, -2.0), Complex64::new(3.0, 4.0)]
    );
}

#[test]
fn traced_compare_returns_bool_and_where_select_accepts_bool_condition() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
    let y = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();
    let cond = traced_tensor::compare(&x, &y, CompareDir::Gt).unwrap();
    assert_eq!(cond.dtype, DType::Bool);

    let selected = traced_tensor::where_select(&cond, &x, &y).unwrap();

    let mut compiler = GraphCompiler::new();
    let cond_program = compiler.compile(&cond).unwrap();
    let selected_program = compiler.compile(&selected).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let cond_out = executor.run(&cond_program).unwrap();
    let selected_out = executor.run(&selected_program).unwrap();

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
        Tensor::from_vec_row_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0]).unwrap(),
        ctx.clone(),
    );
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_row_major(vec![1, 4], vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap(),
        ctx,
    );

    let out = eager_tensor::add(&lhs, &rhs).unwrap();

    assert_eq!(out.data().shape(), &[3, 4]);
    assert_eq!(
        out.data().clone().into_vec_row_major::<f64>().unwrap().1,
        vec![11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0,]
    );
}

#[test]
fn eager_tensor_module_exposes_initial_elementwise_free_functions() {
    let ctx = EagerRuntime::new();
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap(),
        ctx.clone(),
    );
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap(),
        ctx,
    );
    let cond = eager_tensor::compare(&x, &y, CompareDir::Gt).unwrap();

    let _ = eager_tensor::sub(&x, &y).unwrap();
    let _ = eager_tensor::mul(&x, &y).unwrap();
    let _ = eager_tensor::div(&x, &y).unwrap();
    let _ = eager_tensor::pow(&x, &y).unwrap();
    let _ = eager_tensor::maximum(&x, &y).unwrap();
    let _ = eager_tensor::minimum(&x, &y).unwrap();
    let _ = eager_tensor::where_select(&cond, &x, &y).unwrap();
    let _ = eager_tensor::clamp(&x, &y, &x).unwrap();
    let _ = eager_tensor::neg(&x).unwrap();
    let _ = eager_tensor::abs(&x).unwrap();
    let _ = eager_tensor::sign(&x).unwrap();
    let _ = eager_tensor::conj(&x).unwrap();
    let _ = eager_tensor::exp(&x).unwrap();
    let _ = eager_tensor::log(&x).unwrap();
    let _ = eager_tensor::sin(&x).unwrap();
    let _ = eager_tensor::cos(&x).unwrap();
    let _ = eager_tensor::tanh(&x).unwrap();
    let _ = eager_tensor::sqrt(&x).unwrap();
    let _ = eager_tensor::rsqrt(&x).unwrap();
    let _ = eager_tensor::expm1(&x).unwrap();
    let _ = eager_tensor::log1p(&x).unwrap();
}

#[test]
fn eager_tensor_module_covers_conversion_matmul_standard_op_and_fusion() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        ctx.clone(),
    );

    let converted = eager_tensor::convert(&x, DType::F32).unwrap();
    assert_eq!(converted.data().dtype(), DType::F32);
    assert_eq!(converted.data().as_slice::<f32>().unwrap(), &[1.0, 2.0]);

    let negated = eager_tensor::apply_standard_op(StdTensorOp::Neg, &[&x]).unwrap();
    assert_eq!(negated.data().as_slice::<f64>().unwrap(), &[-1.0, -2.0]);

    let extension_err =
        eager_tensor::apply_standard_op(StdTensorOp::Extension(Arc::new(TestExtensionOp)), &[&x])
            .err()
            .unwrap();
    assert!(
        extension_err
            .to_string()
            .contains("does not accept Extension ops"),
        "got: {extension_err}"
    );

    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    );
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    );
    let product = eager_tensor::matmul(&a, &b).unwrap();
    assert_eq!(product.data().shape(), &[2, 2]);
    assert_eq!(
        product.data().as_slice::<f64>().unwrap(),
        &[22.0, 28.0, 49.0, 64.0]
    );

    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
        ctx.clone(),
    );
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![5.0_f64, 7.0, 11.0]).unwrap(),
        ctx.clone(),
    );
    let fused = eager_tensor::backend_broadcast_multiply_untracked(
        &lhs,
        &[2, 3],
        &[0],
        &rhs,
        &[2, 3],
        &[1],
    )
    .unwrap()
    .expect("CPU backend should fuse broadcast multiply for untracked tensors");
    assert_eq!(fused.data().shape(), &[2, 3]);
    assert_eq!(
        fused.data().as_slice::<f64>().unwrap(),
        &[10.0, 15.0, 14.0, 21.0, 22.0, 33.0]
    );

    let tracked = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
        ctx.clone(),
    );
    let skipped = eager_tensor::backend_broadcast_multiply_untracked(
        &tracked,
        &[2, 3],
        &[0],
        &rhs,
        &[2, 3],
        &[1],
    )
    .unwrap();
    assert!(skipped.is_none());

    let other_ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let other = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        other_ctx,
    );
    let err = eager_tensor::add(&x, &other).err().unwrap();
    assert!(matches!(err, tenferro_ad::Error::ContextMismatch { .. }));
}

#[test]
fn eager_compare_returns_bool_and_where_select_accepts_bool_condition() {
    let ctx = EagerRuntime::new();
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap(),
        ctx.clone(),
    );
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap(),
        ctx,
    );

    let cond = eager_tensor::compare(&x, &y, CompareDir::Gt).unwrap();
    let selected = eager_tensor::where_select(&cond, &x, &y).unwrap();

    assert_eq!(cond.data().dtype(), DType::Bool);
    assert_eq!(cond.data().as_slice::<bool>().unwrap(), &[true, false]);
    assert_eq!(
        selected
            .data()
            .clone()
            .into_vec_col_major::<f64>()
            .unwrap()
            .1,
        vec![2.0, 8.0]
    );
}

#[test]
fn tensor_add_uses_numpy_broadcasting_with_explicit_backend() {
    let mut backend = CpuBackend::new();
    let lhs = Tensor::from_vec_row_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let rhs = Tensor::from_vec_row_major(vec![1, 4], vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap();

    let out = tenferro_runtime::tensor::add(&lhs, &rhs, &mut backend).unwrap();

    assert_eq!(out.shape(), &[3, 4]);
    assert_eq!(
        out.into_vec_row_major::<f64>().unwrap().1,
        vec![11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0,]
    );
}

#[test]
fn tensor_module_exposes_initial_elementwise_free_functions() {
    let mut backend = CpuBackend::new();
    let x = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
    let y = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();
    let cond = tenferro_runtime::tensor::compare(&x, &y, CompareDir::Gt, &mut backend).unwrap();

    let _ = tenferro_runtime::tensor::sub(&x, &y, &mut backend).unwrap();
    let _ = tenferro_runtime::tensor::mul(&x, &y, &mut backend).unwrap();
    let _ = tenferro_runtime::tensor::div(&x, &y, &mut backend).unwrap();
    let _ = tenferro_runtime::tensor::pow(&x, &y, &mut backend).unwrap();
    let _ = tenferro_runtime::tensor::maximum(&x, &y, &mut backend).unwrap();
    let _ = tenferro_runtime::tensor::minimum(&x, &y, &mut backend).unwrap();
    let _ = tenferro_runtime::tensor::where_select(&cond, &x, &y, &mut backend).unwrap();
    let _ = tenferro_runtime::tensor::clamp(&x, &y, &x, &mut backend).unwrap();
    let _ = tenferro_runtime::tensor::neg(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::tensor::abs(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::tensor::sign(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::tensor::conj(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::tensor::exp(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::tensor::log(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::tensor::sin(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::tensor::cos(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::tensor::tanh(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::tensor::sqrt(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::tensor::rsqrt(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::tensor::expm1(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::tensor::log1p(&x, &mut backend).unwrap();
}

#[test]
fn tensor_compare_returns_bool_and_where_select_accepts_bool_condition() {
    let mut backend = CpuBackend::new();
    let x = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
    let y = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();

    let cond = tenferro_runtime::tensor::compare(&x, &y, CompareDir::Gt, &mut backend).unwrap();
    let selected = tenferro_runtime::tensor::where_select(&cond, &x, &y, &mut backend).unwrap();

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
    let lhs = TypedTensor::<f64>::from_vec_row_major(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap();
    let rhs =
        TypedTensor::<f64>::from_vec_row_major(vec![1, 4], vec![10.0, 20.0, 30.0, 40.0]).unwrap();

    let out = tenferro_runtime::typed_tensor::add(&lhs, &rhs, &mut backend).unwrap();

    assert_eq!(out.shape(), &[3, 4]);
    assert_eq!(
        out.into_vec_row_major().unwrap().1,
        vec![11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0,]
    );
}

#[test]
fn typed_tensor_module_exposes_initial_elementwise_free_functions() {
    let mut backend = CpuBackend::new();
    let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
    let y = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]).unwrap();
    let cond =
        tenferro_runtime::typed_tensor::compare(&x, &y, CompareDir::Gt, &mut backend).unwrap();

    let _ = tenferro_runtime::typed_tensor::sub(&x, &y, &mut backend).unwrap();
    let _ = tenferro_runtime::typed_tensor::mul(&x, &y, &mut backend).unwrap();
    let _ = tenferro_runtime::typed_tensor::div(&x, &y, &mut backend).unwrap();
    let _ = tenferro_runtime::typed_tensor::pow(&x, &y, &mut backend).unwrap();
    let _ = tenferro_runtime::typed_tensor::maximum(&x, &y, &mut backend).unwrap();
    let _ = tenferro_runtime::typed_tensor::minimum(&x, &y, &mut backend).unwrap();
    let _ = tenferro_runtime::typed_tensor::where_select(&cond, &x, &y, &mut backend).unwrap();
    let _ = tenferro_runtime::typed_tensor::clamp(&x, &y, &x, &mut backend).unwrap();
    let _ = tenferro_runtime::typed_tensor::neg(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::typed_tensor::abs(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::typed_tensor::sign(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::typed_tensor::conj(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::typed_tensor::exp(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::typed_tensor::log(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::typed_tensor::sin(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::typed_tensor::cos(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::typed_tensor::tanh(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::typed_tensor::sqrt(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::typed_tensor::rsqrt(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::typed_tensor::expm1(&x, &mut backend).unwrap();
    let _ = tenferro_runtime::typed_tensor::log1p(&x, &mut backend).unwrap();
}

#[test]
fn typed_tensor_compare_returns_bool_and_where_select_accepts_bool_condition() {
    let mut backend = CpuBackend::new();
    let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
    let y = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]).unwrap();

    let cond: TypedTensor<bool> =
        tenferro_runtime::typed_tensor::compare(&x, &y, CompareDir::Gt, &mut backend).unwrap();
    let selected =
        tenferro_runtime::typed_tensor::where_select(&cond, &x, &y, &mut backend).unwrap();

    assert_eq!(cond.host_data().unwrap(), &[true, false]);
    assert_eq!(selected.into_vec_col_major().unwrap().1, vec![2.0, 8.0]);
}
