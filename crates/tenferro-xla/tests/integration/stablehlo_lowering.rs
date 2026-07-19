use tenferro_einsum::GraphCompilerEinsumExt;
use tenferro_runtime::{DType, DotGeneralConfig, GraphCompiler, TracedTensor};
use tenferro_xla::lower_to_stablehlo;

#[test]
fn lowers_elementwise_reduce_and_static_shapes() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let y = (&x + &x).unwrap().reduce_sum(Some(&[0])).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2, 3])])
        .unwrap();

    let module = lower_to_stablehlo(&program).unwrap();
    let text = module.as_str();

    assert!(text.contains("func.func @main(%arg0: tensor<2x3xf64>) -> tensor<3xf64>"));
    assert!(text.contains("stablehlo.add %arg0, %arg0 : tensor<2x3xf64>"));
    assert!(text.contains("stablehlo.reduce("));
    assert!(text.contains("applies stablehlo.add across dimensions = [0]"));
    assert!(text.contains("return %"));
}

#[test]
fn lowers_phase_one_real_elementwise_ops() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let unary = x
        .abs()
        .unwrap()
        .exp()
        .unwrap()
        .log()
        .unwrap()
        .sin()
        .unwrap()
        .cos()
        .unwrap()
        .tanh()
        .unwrap()
        .sqrt()
        .unwrap()
        .rsqrt()
        .unwrap()
        .expm1()
        .unwrap()
        .log1p()
        .unwrap();
    let divided = unary.div(&x).unwrap();
    let powered = divided.pow(&x).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&powered, &[(&x, DType::F64, &[4])])
        .unwrap();

    let module = lower_to_stablehlo(&program).unwrap();
    let text = module.as_str();

    assert!(text.contains("stablehlo.abs %arg0 : tensor<4xf64>"));
    assert!(text.contains("stablehlo.exponential %"));
    assert!(text.contains("stablehlo.log %"));
    assert!(text.contains("stablehlo.sine %"));
    assert!(text.contains("stablehlo.cosine %"));
    assert!(text.contains("stablehlo.tanh %"));
    assert!(text.contains("stablehlo.sqrt %"));
    assert!(text.contains("stablehlo.rsqrt %"));
    assert!(text.contains("stablehlo.exponential_minus_one %"));
    assert!(text.contains("stablehlo.log_plus_one %"));
    assert!(text.contains("stablehlo.divide %"));
    assert!(text.contains("stablehlo.power %"));
    assert!(text.contains("-> tensor<4xf64>"));
}

#[test]
fn lowers_structural_ops_and_convert() {
    let x = TracedTensor::input_symbolic_shape(DType::F32, 1).unwrap();
    let y = x
        .broadcast_in_dim(&[2, 3], &[1])
        .unwrap()
        .transpose(&[1, 0])
        .unwrap()
        .reshape(&[6])
        .unwrap()
        .convert(DType::F64)
        .unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F32, &[3])])
        .unwrap();

    let module = lower_to_stablehlo(&program).unwrap();
    let text = module.as_str();

    assert!(text.contains("stablehlo.broadcast_in_dim %arg0, dims = [1]"));
    assert!(text.contains("stablehlo.transpose"));
    assert!(text.contains("dims = [1, 0]"));
    assert!(text.contains("stablehlo.reshape"));
    assert!(text.contains("stablehlo.convert"));
    assert!(text.contains("return %"));
    assert!(text.contains("tensor<6xf64>"));
}

#[test]
fn lowers_unbatched_dot_general() {
    let lhs = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let rhs = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let product = lhs
        .dot_general(
            &rhs,
            DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )
        .unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(
            &product,
            &[(&lhs, DType::F64, &[2, 3]), (&rhs, DType::F64, &[3, 4])],
        )
        .unwrap();

    let module = lower_to_stablehlo(&program).unwrap();
    let text = module.as_str();

    assert!(text.contains("stablehlo.dot_general %arg0, %arg1"));
    assert!(text.contains("contracting_dims = [1] x [0]"));
    assert!(text.contains("-> tensor<2x4xf64>"));
}

#[test]
fn lowers_concrete_nary_einsum_via_standard_ops() {
    let lhs = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0])
        .unwrap();
    let mid = TracedTensor::from_vec_col_major(
        vec![3, 4],
        vec![
            10.0_f64, 20.0, 30.0, 11.0, 21.0, 31.0, 12.0, 22.0, 32.0, 13.0, 23.0, 33.0,
        ],
    )
    .unwrap();
    let rhs = TracedTensor::from_vec_col_major(
        vec![4, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    .unwrap();
    let mut compiler = GraphCompiler::new();
    let product = compiler
        .einsum(&[&lhs, &mid, &rhs], "ij,jk,kl->il")
        .unwrap();
    let program = compiler.compile(&product).unwrap();

    let module = lower_to_stablehlo(&program).unwrap();
    let text = module.as_str();

    assert!(text.contains("stablehlo.dot_general"));
    assert_eq!(text.matches("stablehlo.dot_general").count(), 2);
    assert!(!text.contains("tenferro.einsum"));
    assert!(text.contains("-> tensor<2x2xf64>"));
}

#[test]
fn lowers_static_symbolic_nary_einsum_extension_via_standard_ops() {
    let lhs = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let mid = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let rhs = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let mut compiler = GraphCompiler::new();
    let product = compiler
        .einsum(&[&lhs, &mid, &rhs], "ij,jk,kl->il")
        .unwrap();
    let program = compiler
        .compile_with_input_specs(
            &product,
            &[
                (&lhs, DType::F64, &[2, 3]),
                (&mid, DType::F64, &[3, 4]),
                (&rhs, DType::F64, &[4, 2]),
            ],
        )
        .unwrap();

    let module = lower_to_stablehlo(&program).unwrap();
    let text = module.as_str();

    assert!(text.contains("stablehlo.dot_general"));
    assert_eq!(text.matches("stablehlo.dot_general").count(), 2);
    assert!(!text.contains("tenferro.einsum"));
    assert!(text.contains("-> tensor<2x2xf64>"));
}

#[test]
fn batched_dot_general_transposes_stablehlo_batch_first_result() {
    let lhs = TracedTensor::input_symbolic_shape(DType::F64, 3).unwrap();
    let rhs = TracedTensor::input_symbolic_shape(DType::F64, 3).unwrap();
    let product = lhs
        .dot_general(
            &rhs,
            DotGeneralConfig {
                lhs_contracting_dims: vec![2],
                rhs_contracting_dims: vec![1],
                lhs_batch_dims: vec![0],
                rhs_batch_dims: vec![0],
            },
        )
        .unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(
            &product,
            &[
                (&lhs, DType::F64, &[5, 2, 3]),
                (&rhs, DType::F64, &[5, 3, 4]),
            ],
        )
        .unwrap();

    let module = lower_to_stablehlo(&program).unwrap();
    let text = module.as_str();

    assert!(text.contains("batching_dims = [0] x [0]"));
    assert!(text.contains("-> tensor<5x2x4xf64>"));
    assert!(text.contains("stablehlo.transpose"));
    assert!(text.contains("dims = [1, 2, 0]"));
    assert!(text.contains("-> tensor<2x4x5xf64>"));
}

#[test]
fn equal_extent_batched_dot_general_still_transposes_batch_last_result() {
    let lhs = TracedTensor::input_symbolic_shape(DType::F64, 3).unwrap();
    let rhs = TracedTensor::input_symbolic_shape(DType::F64, 3).unwrap();
    let product = lhs
        .dot_general(
            &rhs,
            DotGeneralConfig {
                lhs_contracting_dims: vec![2],
                rhs_contracting_dims: vec![1],
                lhs_batch_dims: vec![0],
                rhs_batch_dims: vec![0],
            },
        )
        .unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(
            &product,
            &[
                (&lhs, DType::F64, &[2, 2, 2]),
                (&rhs, DType::F64, &[2, 2, 2]),
            ],
        )
        .unwrap();

    let module = lower_to_stablehlo(&program).unwrap();
    let text = module.as_str();

    assert!(text.contains("stablehlo.dot_general"));
    assert!(text.contains("stablehlo.transpose"));
    assert!(text.contains("dims = [1, 2, 0]"));
}

#[test]
fn lowers_multi_output_program_and_special_scalar_constants() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let scaled_nan = x.scale_real(f64::NAN).unwrap();
    let scaled_neg_inf = x.scale_real(f64::NEG_INFINITY).unwrap();
    let scaled_pos_inf = x.scale_real(f64::INFINITY).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_many(&[&scaled_nan, &scaled_neg_inf, &scaled_pos_inf])
        .unwrap();

    let module = lower_to_stablehlo(&program).unwrap();
    let text = module.as_str();

    assert!(text.contains(
        "func.func @main(%arg0: tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>, tensor<2xf64>)"
    ));
    assert!(text.contains("stablehlo.constant dense<0x7ff8000000000000> : tensor<f64>"));
    assert!(text.contains("stablehlo.constant dense<-0x7ff0000000000000> : tensor<f64>"));
    assert!(text.contains("stablehlo.constant dense<0x7ff0000000000000> : tensor<f64>"));
    assert_eq!(text.matches("stablehlo.multiply").count(), 3);
    assert!(text.contains("return %"));
    assert!(text.contains(": tensor<2xf64>, tensor<2xf64>, tensor<2xf64>"));
}

#[test]
fn lowers_f32_scalar_constant() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let scaled = x.scale_real(2.5).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&scaled).unwrap();

    let module = lower_to_stablehlo(&program).unwrap();
    let text = module.as_str();

    assert!(text.contains("stablehlo.constant dense<2.50000000e0> : tensor<f32>"));
    assert!(text.contains("stablehlo.multiply"));
}

#[test]
fn lowers_empty_output_program_to_unit_return() {
    let mut compiler = GraphCompiler::new();
    let outputs: [&TracedTensor; 0] = [];
    let program = compiler.compile_many(&outputs).unwrap();

    let module = lower_to_stablehlo(&program).unwrap();
    let text = module.as_str();

    assert!(text.contains("func.func @main() -> ()"));
    assert!(text.contains("    return\n"));
}
