use tenferro::{
    traced_tensor, CompareDir, CpuBackend, DType, GraphCompiler, GraphExecutor, Tensor,
    TracedTensor, TypedTensor,
};

#[cfg(feature = "autodiff")]
use tenferro::{eager_tensor, EagerRuntime, EagerTensor};

#[test]
fn traced_add_uses_numpy_broadcasting_for_rank_padding_and_singletons() {
    let lhs = TracedTensor::from_vec_row_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0]);
    let rhs = TracedTensor::from_vec_row_major(vec![1, 4], vec![10.0_f64, 20.0, 30.0, 40.0]);
    let y = traced_tensor::add(&lhs, &rhs);

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let out = executor.run(&program).unwrap();

    assert_eq!(out.shape(), &[3, 4]);
    assert_eq!(
        out.try_into_vec_row_major::<f64>().unwrap().1,
        vec![11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0,]
    );
}

#[test]
fn traced_tensor_module_exposes_initial_elementwise_free_functions() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]);
    let y = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]);
    let cond = traced_tensor::compare(&x, &y, CompareDir::Gt);

    let _ = traced_tensor::sub(&x, &y);
    let _ = traced_tensor::mul(&x, &y);
    let _ = traced_tensor::div(&x, &y);
    let _ = traced_tensor::pow(&x, &y);
    let _ = traced_tensor::maximum(&x, &y);
    let _ = traced_tensor::minimum(&x, &y);
    let _ = traced_tensor::where_select(&cond, &x, &y);
    let _ = traced_tensor::clamp(&x, &y, &x);
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
fn traced_compare_returns_bool_and_where_select_accepts_bool_condition() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]);
    let y = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]);
    let cond = traced_tensor::compare(&x, &y, CompareDir::Gt);
    assert_eq!(cond.dtype, DType::Bool);

    let selected = traced_tensor::where_select(&cond, &x, &y);

    let mut compiler = GraphCompiler::new();
    let cond_program = compiler.compile(&cond).unwrap();
    let selected_program = compiler.compile(&selected).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let cond_out = executor.run(&cond_program).unwrap();
    let selected_out = executor.run(&selected_program).unwrap();

    assert_eq!(cond_out.dtype(), DType::Bool);
    assert_eq!(cond_out.as_slice::<bool>().unwrap(), &[true, false]);
    assert_eq!(
        selected_out.try_into_vec_col_major::<f64>().unwrap().1,
        vec![2.0, 8.0]
    );
}

#[cfg(feature = "autodiff")]
#[test]
fn eager_add_uses_numpy_broadcasting_for_rank_padding_and_singletons() {
    let ctx = EagerRuntime::new();
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_row_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0]),
        ctx.clone(),
    );
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_row_major(vec![1, 4], vec![10.0_f64, 20.0, 30.0, 40.0]),
        ctx,
    );

    let out = eager_tensor::add(&lhs, &rhs).unwrap();

    assert_eq!(out.data().shape(), &[3, 4]);
    assert_eq!(
        out.data()
            .clone()
            .try_into_vec_row_major::<f64>()
            .unwrap()
            .1,
        vec![11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0,]
    );
}

#[cfg(feature = "autodiff")]
#[test]
fn eager_tensor_module_exposes_initial_elementwise_free_functions() {
    let ctx = EagerRuntime::new();
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]),
        ctx.clone(),
    );
    let y =
        EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]), ctx);
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

#[cfg(feature = "autodiff")]
#[test]
fn eager_compare_returns_bool_and_where_select_accepts_bool_condition() {
    let ctx = EagerRuntime::new();
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]),
        ctx.clone(),
    );
    let y =
        EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]), ctx);

    let cond = eager_tensor::compare(&x, &y, CompareDir::Gt).unwrap();
    let selected = eager_tensor::where_select(&cond, &x, &y).unwrap();

    assert_eq!(cond.data().dtype(), DType::Bool);
    assert_eq!(cond.data().as_slice::<bool>().unwrap(), &[true, false]);
    assert_eq!(
        selected
            .data()
            .clone()
            .try_into_vec_col_major::<f64>()
            .unwrap()
            .1,
        vec![2.0, 8.0]
    );
}

#[test]
fn tensor_add_uses_numpy_broadcasting_with_explicit_backend() {
    let mut backend = CpuBackend::new();
    let lhs = Tensor::from_vec_row_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0]);
    let rhs = Tensor::from_vec_row_major(vec![1, 4], vec![10.0_f64, 20.0, 30.0, 40.0]);

    let out = tenferro::tensor::add(&lhs, &rhs, &mut backend).unwrap();

    assert_eq!(out.shape(), &[3, 4]);
    assert_eq!(
        out.try_into_vec_row_major::<f64>().unwrap().1,
        vec![11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0,]
    );
}

#[test]
fn tensor_module_exposes_initial_elementwise_free_functions() {
    let mut backend = CpuBackend::new();
    let x = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]);
    let y = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]);
    let cond = tenferro::tensor::compare(&x, &y, CompareDir::Gt, &mut backend).unwrap();

    let _ = tenferro::tensor::sub(&x, &y, &mut backend).unwrap();
    let _ = tenferro::tensor::mul(&x, &y, &mut backend).unwrap();
    let _ = tenferro::tensor::div(&x, &y, &mut backend).unwrap();
    let _ = tenferro::tensor::pow(&x, &y, &mut backend).unwrap();
    let _ = tenferro::tensor::maximum(&x, &y, &mut backend).unwrap();
    let _ = tenferro::tensor::minimum(&x, &y, &mut backend).unwrap();
    let _ = tenferro::tensor::where_select(&cond, &x, &y, &mut backend).unwrap();
    let _ = tenferro::tensor::clamp(&x, &y, &x, &mut backend).unwrap();
    let _ = tenferro::tensor::neg(&x, &mut backend).unwrap();
    let _ = tenferro::tensor::abs(&x, &mut backend).unwrap();
    let _ = tenferro::tensor::sign(&x, &mut backend).unwrap();
    let _ = tenferro::tensor::conj(&x, &mut backend).unwrap();
    let _ = tenferro::tensor::exp(&x, &mut backend).unwrap();
    let _ = tenferro::tensor::log(&x, &mut backend).unwrap();
    let _ = tenferro::tensor::sin(&x, &mut backend).unwrap();
    let _ = tenferro::tensor::cos(&x, &mut backend).unwrap();
    let _ = tenferro::tensor::tanh(&x, &mut backend).unwrap();
    let _ = tenferro::tensor::sqrt(&x, &mut backend).unwrap();
    let _ = tenferro::tensor::rsqrt(&x, &mut backend).unwrap();
    let _ = tenferro::tensor::expm1(&x, &mut backend).unwrap();
    let _ = tenferro::tensor::log1p(&x, &mut backend).unwrap();
}

#[test]
fn tensor_compare_returns_bool_and_where_select_accepts_bool_condition() {
    let mut backend = CpuBackend::new();
    let x = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]);
    let y = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]);

    let cond = tenferro::tensor::compare(&x, &y, CompareDir::Gt, &mut backend).unwrap();
    let selected = tenferro::tensor::where_select(&cond, &x, &y, &mut backend).unwrap();

    assert_eq!(cond.dtype(), DType::Bool);
    assert_eq!(cond.as_slice::<bool>().unwrap(), &[true, false]);
    assert_eq!(
        selected.try_into_vec_col_major::<f64>().unwrap().1,
        vec![2.0, 8.0]
    );
}

#[test]
fn typed_tensor_add_uses_numpy_broadcasting_with_explicit_backend() {
    let mut backend = CpuBackend::new();
    let lhs = TypedTensor::<f64>::from_vec_row_major(vec![3, 1], vec![1.0, 2.0, 3.0]);
    let rhs = TypedTensor::<f64>::from_vec_row_major(vec![1, 4], vec![10.0, 20.0, 30.0, 40.0]);

    let out = tenferro::typed_tensor::add(&lhs, &rhs, &mut backend).unwrap();

    assert_eq!(out.shape, vec![3, 4]);
    assert_eq!(
        out.try_into_vec_row_major().unwrap().1,
        vec![11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0,]
    );
}

#[test]
fn typed_tensor_module_exposes_initial_elementwise_free_functions() {
    let mut backend = CpuBackend::new();
    let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]);
    let y = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]);
    let cond = tenferro::typed_tensor::compare(&x, &y, CompareDir::Gt, &mut backend).unwrap();

    let _ = tenferro::typed_tensor::sub(&x, &y, &mut backend).unwrap();
    let _ = tenferro::typed_tensor::mul(&x, &y, &mut backend).unwrap();
    let _ = tenferro::typed_tensor::div(&x, &y, &mut backend).unwrap();
    let _ = tenferro::typed_tensor::pow(&x, &y, &mut backend).unwrap();
    let _ = tenferro::typed_tensor::maximum(&x, &y, &mut backend).unwrap();
    let _ = tenferro::typed_tensor::minimum(&x, &y, &mut backend).unwrap();
    let _ = tenferro::typed_tensor::where_select(&cond, &x, &y, &mut backend).unwrap();
    let _ = tenferro::typed_tensor::clamp(&x, &y, &x, &mut backend).unwrap();
    let _ = tenferro::typed_tensor::neg(&x, &mut backend).unwrap();
    let _ = tenferro::typed_tensor::abs(&x, &mut backend).unwrap();
    let _ = tenferro::typed_tensor::sign(&x, &mut backend).unwrap();
    let _ = tenferro::typed_tensor::conj(&x, &mut backend).unwrap();
    let _ = tenferro::typed_tensor::exp(&x, &mut backend).unwrap();
    let _ = tenferro::typed_tensor::log(&x, &mut backend).unwrap();
    let _ = tenferro::typed_tensor::sin(&x, &mut backend).unwrap();
    let _ = tenferro::typed_tensor::cos(&x, &mut backend).unwrap();
    let _ = tenferro::typed_tensor::tanh(&x, &mut backend).unwrap();
    let _ = tenferro::typed_tensor::sqrt(&x, &mut backend).unwrap();
    let _ = tenferro::typed_tensor::rsqrt(&x, &mut backend).unwrap();
    let _ = tenferro::typed_tensor::expm1(&x, &mut backend).unwrap();
    let _ = tenferro::typed_tensor::log1p(&x, &mut backend).unwrap();
}

#[test]
fn typed_tensor_compare_returns_bool_and_where_select_accepts_bool_condition() {
    let mut backend = CpuBackend::new();
    let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]);
    let y = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]);

    let cond: TypedTensor<bool> =
        tenferro::typed_tensor::compare(&x, &y, CompareDir::Gt, &mut backend).unwrap();
    let selected = tenferro::typed_tensor::where_select(&cond, &x, &y, &mut backend).unwrap();

    assert_eq!(cond.host_data(), &[true, false]);
    assert_eq!(selected.try_into_vec_col_major().unwrap().1, vec![2.0, 8.0]);
}
