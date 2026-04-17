use std::collections::HashMap;
use std::sync::Arc;

use computegraph::compile::compile;
use computegraph::fragment::FragmentBuilder;
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, ValRef};
use tenferro::compiler::compile_semiring_to_exec;
use tenferro::exec::eval_semiring_ir;
use tenferro_algebra::{Algebra, Semiring};
use tenferro_einsum::builder::build_einsum_fragment;
use tenferro_einsum::{ContractionTree, Subscripts};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::semiring_op::{SemiringInputKey, SemiringOp};
use tenferro_tensor::backend::SemiringBackend;
use tenferro_tensor::cpu::CpuBackend;
use tenferro_tensor::TypedTensor;

struct TropicalAlgebra;

impl Algebra for TropicalAlgebra {
    type Scalar = f64;
}

impl Semiring for TropicalAlgebra {
    fn zero() -> Self::Scalar {
        f64::NEG_INFINITY
    }

    fn one() -> Self::Scalar {
        0.0
    }

    fn add(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        a.max(b)
    }

    fn mul(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        a + b
    }
}

fn eval_fragment(
    backend: &mut CpuBackend,
    fragment: Arc<computegraph::fragment::Fragment<SemiringOp<TropicalAlgebra>>>,
    output: computegraph::LocalValId,
    inputs_map: &HashMap<SemiringInputKey, TypedTensor<f64>>,
) -> TypedTensor<f64> {
    let output_key = fragment.vals()[output].key.clone();
    let view = resolve(vec![fragment]);
    let graph = materialize_merge(&view, &[output_key]);
    let compiled = compile(&graph);
    let inputs = graph
        .inputs
        .iter()
        .map(|key| match key {
            GlobalValKey::Input(k) => inputs_map.get(k).expect("missing semiring input").clone(),
            _ => panic!("expected input key"),
        })
        .collect::<Vec<_>>();
    let input_shapes: Vec<Vec<DimExpr>> = inputs
        .iter()
        .map(|tensor| DimExpr::from_concrete(&tensor.shape))
        .collect();
    let exec = compile_semiring_to_exec(&compiled, &input_shapes);
    eval_semiring_ir::<_, TropicalAlgebra>(backend, &exec, inputs)
        .unwrap()
        .into_iter()
        .next()
        .expect("semiring output")
}

#[test]
fn tropical_matrix_multiply_via_semiring_pipeline() {
    let a = TypedTensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = TypedTensor::from_vec(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);

    let mut builder = FragmentBuilder::<SemiringOp<TropicalAlgebra>>::new();
    let key_a = SemiringInputKey { id: 0 };
    let key_b = SemiringInputKey { id: 1 };
    let val_a = builder.add_input(key_a.clone());
    let val_b = builder.add_input(key_b.clone());

    let subs = Subscripts::parse("ij,jk->ik").unwrap();
    let shapes = vec![a.shape.clone(), b.shape.clone()];
    let shape_refs: Vec<&[usize]> = shapes.iter().map(|s| s.as_slice()).collect();
    let tree = ContractionTree::from_pairs(&subs, &shape_refs, &[(0, 1)]).unwrap();
    let out = build_einsum_fragment(
        &mut builder,
        &tree,
        &[ValRef::Local(val_a), ValRef::Local(val_b)],
        &shapes,
    );
    let out = match out {
        ValRef::Local(local) => local,
        _ => panic!("expected local output"),
    };
    builder.set_outputs(vec![out]);
    let fragment = Arc::new(builder.build());

    let mut inputs = HashMap::new();
    inputs.insert(key_a, a);
    inputs.insert(key_b, b);

    let mut backend = CpuBackend::new();
    let result = eval_fragment(&mut backend, fragment, out, &inputs);

    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(*result.get(&[0, 0]), 9.0);
    assert_eq!(*result.get(&[1, 0]), 10.0);
    assert_eq!(*result.get(&[0, 1]), 11.0);
    assert_eq!(*result.get(&[1, 1]), 12.0);
}

#[test]
fn tropical_default_semiring_ops_work_on_cpu_backend() {
    let a = TypedTensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = TypedTensor::from_vec(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let mut backend = CpuBackend::new();

    let sum = SemiringBackend::<TropicalAlgebra>::add(&mut backend, &a, &b).unwrap();
    let prod = SemiringBackend::<TropicalAlgebra>::mul(&mut backend, &a, &b).unwrap();
    let reduced = SemiringBackend::<TropicalAlgebra>::reduce_sum(&mut backend, &a, &[0]).unwrap();

    assert_eq!(sum.host_data(), &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(prod.host_data(), &[6.0, 8.0, 10.0, 12.0]);
    assert_eq!(reduced.shape, vec![2]);
    assert_eq!(reduced.host_data(), &[2.0, 4.0]);
}
