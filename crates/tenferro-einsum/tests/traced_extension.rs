#[cfg(feature = "autodiff")]
use tenferro_ad::TracedTensorAdExt;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};

#[test]
fn concrete_traced_einsum_executes_without_extension_runtime() {
    let a = TracedTensor::from_vec_col_major(vec![2, 2, 3], vec![1.0_f64; 12]);
    let b = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]);
    let mut compiler = GraphCompiler::new();

    let c = tenferro_einsum::einsum(&mut compiler, &[&a, &b], "iij,jk->ik").unwrap();
    let program = compiler.compile(&c).unwrap();

    let mut executor = GraphExecutor::new(CpuBackend::new());
    let out = executor.run(&program).unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[3.0_f64; 4]);
    assert_eq!(executor.cache_stats().extensions.entries, 0);
}

#[test]
fn traced_binary_col_major_matmul_uses_direct_dot_general_without_extension_runtime() {
    let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    let b = TracedTensor::from_vec_col_major(vec![4, 2], vec![1.0_f64; 8]);
    let mut compiler = GraphCompiler::new();

    let c = tenferro_einsum::einsum(&mut compiler, &[&a, &b], "ji,kj->ki").unwrap();
    let program = compiler.compile(&c).unwrap();

    let mut executor = GraphExecutor::new(CpuBackend::new());
    let out = executor.run(&program).unwrap();

    assert_eq!(out.shape(), &[4, 3]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0_f64; 12]);
    assert_eq!(executor.cache_stats().extensions.entries, 0);
}

#[test]
fn traced_binary_tree_col_major_matmul_uses_direct_dot_general_without_extension_runtime() {
    let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    let b = TracedTensor::from_vec_col_major(vec![4, 2], vec![1.0_f64; 8]);
    let subs = tenferro_einsum::Subscripts::parse("ji,kj->ki").unwrap();
    let lhs_shape = [2, 3];
    let rhs_shape = [4, 2];
    let shapes = [&lhs_shape[..], &rhs_shape[..]];
    let tree = tenferro_einsum::ContractionTree::from_pairs(&subs, &shapes, &[(0, 1)]).unwrap();
    let mut compiler = GraphCompiler::new();

    let c = tenferro_einsum::einsum_with(
        &mut compiler,
        &[&a, &b],
        "ji,kj->ki",
        tenferro_einsum::EinsumOptimize::Tree(tree),
    )
    .unwrap();
    let program = compiler.compile(&c).unwrap();

    let mut executor = GraphExecutor::new(CpuBackend::new());
    let out = executor.run(&program).unwrap();

    assert_eq!(out.shape(), &[4, 3]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0_f64; 12]);
    assert_eq!(executor.cache_stats().extensions.entries, 0);
}

#[test]
fn traced_symbolic_binary_tree_col_major_matmul_uses_direct_dot_general_without_extension_runtime()
{
    let a = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let b = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let subs = tenferro_einsum::Subscripts::parse("ji,kj->ki").unwrap();
    let lhs_shape = [2, 3];
    let rhs_shape = [4, 2];
    let shapes = [&lhs_shape[..], &rhs_shape[..]];
    let tree = tenferro_einsum::ContractionTree::from_pairs(&subs, &shapes, &[(0, 1)]).unwrap();
    let mut compiler = GraphCompiler::new();

    let c = tenferro_einsum::einsum_with(
        &mut compiler,
        &[&a, &b],
        "ji,kj->ki",
        tenferro_einsum::EinsumOptimize::Tree(tree),
    )
    .unwrap();
    let program = compiler
        .compile_with_input_specs(&c, &[(&a, DType::F64, &[2, 3]), (&b, DType::F64, &[4, 2])])
        .unwrap();

    let a_value = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    let b_value = Tensor::from_vec_col_major(vec![4, 2], vec![1.0_f64; 8]);
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let out = executor
        .run_with_inputs(&program, &[(&a, &a_value), (&b, &b_value)])
        .unwrap();

    assert_eq!(out.shape(), &[4, 3]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0_f64; 12]);
    assert_eq!(executor.cache_stats().extensions.entries, 0);
}

#[test]
fn runtime_registration_is_idempotent() {
    let mut executor = GraphExecutor::new(CpuBackend::new());

    executor
        .register_extension(tenferro_einsum::register_runtime)
        .unwrap();
    executor
        .register_extension(tenferro_einsum::register_runtime)
        .unwrap();

    assert_eq!(executor.extension_executor().registry().len(), 1);
}

#[test]
fn runtime_einsum_caches_are_extension_owned() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 3);
    let y = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let mut compiler = GraphCompiler::new();

    let dot = tenferro_einsum::einsum(&mut compiler, &[&x, &y], "iij,jk->ik").unwrap();
    let program = compiler
        .compile_with_input_specs(
            &dot,
            &[(&x, DType::F64, &[2, 2, 3]), (&y, DType::F64, &[3, 2])],
        )
        .unwrap();

    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(tenferro_einsum::register_runtime)
        .unwrap();
    let x_value = Tensor::from_vec_col_major(vec![2, 2, 3], vec![1.0_f64; 12]);
    let y_value = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]);
    let out = executor
        .run_with_inputs(&program, &[(&x, &x_value), (&y, &y_value)])
        .unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[3.0_f64; 4]);
    assert_eq!(executor.cache_stats().extensions.entries, 2);
}

#[test]
#[cfg(feature = "autodiff")]
fn traced_einsum_grad_uses_extension_ad_rule() {
    let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut compiler = GraphCompiler::new();

    let y = tenferro_einsum::einsum_with(
        &mut compiler,
        &[&a, &b],
        "ij,jk->ik",
        tenferro_einsum::EinsumOptimize::Path(vec![(0, 1)]),
    )
    .unwrap();
    let grad_a = y.reduce_sum(&[0, 1]).grad(&a).unwrap();
    let program = compiler.compile(&grad_a).unwrap();

    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(tenferro_einsum::register_runtime)
        .unwrap();
    let out = executor.run(&program).unwrap();

    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(
        out.as_slice::<f64>().unwrap(),
        &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]
    );
}
