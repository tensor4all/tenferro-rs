//! Ported v1 einsum tests adapted for the v2 traced pipeline.
//!
//! Data is stored in **column-major** order. A helper `row_major_to_column_major` is
//! available for converting row-major test data when needed.

use num_complex::Complex64;
use tenferro_einsum::EinsumOptimize;
use tenferro_einsum::{
    ContractionTree, NestedEinsum, Subscripts, TensorDotAxes, EINSUM_EXTENSION_FAMILY_ID,
};
use tenferro_runtime::error::Result;
use tenferro_runtime::{CpuBackend, GraphCompiler, GraphExecutor, Tensor, TracedTensor};
use tenferro_tensor::TypedTensor;

#[path = "traced_correctness/ported_and_paths.rs"]
mod ported_and_paths;

// ============================================================================
// Helpers
// ============================================================================

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data))
}

fn get_f64_data(t: &Tensor) -> &[f64] {
    match t {
        Tensor::F64(inner) => inner.host_data(),
        _ => panic!("expected F64"),
    }
}

trait TestEinsumContext {
    fn with_compiler<R>(&mut self, f: impl FnOnce(&mut GraphCompiler) -> Result<R>) -> Result<R>;
}

impl TestEinsumContext for GraphCompiler {
    fn with_compiler<R>(&mut self, f: impl FnOnce(&mut GraphCompiler) -> Result<R>) -> Result<R> {
        f(self)
    }
}

impl TestEinsumContext for GraphExecutor<CpuBackend> {
    fn with_compiler<R>(&mut self, f: impl FnOnce(&mut GraphCompiler) -> Result<R>) -> Result<R> {
        let mut compiler = GraphCompiler::new();
        f(&mut compiler)
    }
}

fn einsum<C: TestEinsumContext>(
    ctx: &mut C,
    inputs: &[&TracedTensor],
    subscripts: &str,
) -> Result<TracedTensor> {
    ctx.with_compiler(|compiler| tenferro_einsum::einsum(compiler, inputs, subscripts))
}

fn einsum_with<C: TestEinsumContext>(
    ctx: &mut C,
    inputs: &[&TracedTensor],
    subscripts: &str,
    optimize: EinsumOptimize,
) -> Result<TracedTensor> {
    ctx.with_compiler(|compiler| {
        tenferro_einsum::einsum_with(compiler, inputs, subscripts, optimize)
    })
}

trait RunTraced {
    fn run_with(&self, executor: &mut GraphExecutor<CpuBackend>) -> Result<Tensor>;
}

impl RunTraced for TracedTensor {
    fn run_with(&self, executor: &mut GraphExecutor<CpuBackend>) -> Result<Tensor> {
        if !executor
            .extension_executor()
            .registry()
            .contains(EINSUM_EXTENSION_FAMILY_ID)
        {
            executor
                .register_extension(tenferro_einsum::register_runtime)
                .expect("register einsum runtime");
        }
        let mut compiler = GraphCompiler::new();
        let program = compiler.compile(self)?;
        executor.run(&program)
    }
}

fn extension_cache_entries(compiler: &GraphCompiler) -> usize {
    compiler.cache_stats().extensions.entries
}

/// Read a single element from a v2 Tensor by multi-index (col-major).
fn get_v2(t: &Tensor, idx: &[usize]) -> f64 {
    match t {
        Tensor::F64(inner) => *inner.get(idx),
        _ => panic!("expected F64"),
    }
}

/// Convert row-major data to column-major for a given shape.
#[allow(dead_code)]
fn row_major_to_column_major(data: &[f64], shape: &[usize]) -> Vec<f64> {
    let n: usize = shape.iter().product();
    let mut col_data = vec![0.0; n];
    let rank = shape.len();
    if rank <= 1 {
        return data.to_vec();
    }
    for rm_flat in 0..n {
        let mut idx = vec![0usize; rank];
        let mut rem = rm_flat;
        for d in (0..rank).rev() {
            idx[d] = rem % shape[d];
            rem /= shape[d];
        }
        let mut cm_flat = 0;
        let mut stride = 1;
        for d in 0..rank {
            cm_flat += idx[d] * stride;
            stride *= shape[d];
        }
        col_data[cm_flat] = data[rm_flat];
    }
    col_data
}

fn assert_close(a: f64, b: f64, label: &str) {
    assert!((a - b).abs() < 1e-10, "{label}: got {a}, expected {b}");
}

fn symbolic_identity_2d(input: &TracedTensor) -> TracedTensor {
    input
        .reshape_sym(&[input.sym_size(0), input.sym_size(1)])
        .unwrap()
}

// ============================================================================
// Group 1: Basic unary operations
// ============================================================================

#[test]
fn traced_tensor_namespace_exposes_einsum() {
    let mut compiler = GraphCompiler::new();
    let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]);
    let b = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]);
    let y = tenferro_einsum::traced_tensor::einsum(&mut compiler, &[&a, &b], "ij,jk->ik").unwrap();

    assert_eq!(y.rank, 2);
}

#[test]
fn traced_tensor_namespace_tensordot_count_contracts_last_lhs_with_first_rhs_axes() {
    let lhs = TracedTensor::from_vec_col_major(
        vec![2, 3, 4],
        (1..=24).map(f64::from).collect::<Vec<_>>(),
    );
    let rhs = TracedTensor::from_vec_col_major(
        vec![3, 4, 2],
        (1..=24).map(|value| f64::from(value) * 0.5).collect(),
    );

    let out =
        tenferro_einsum::traced_tensor::tensordot(&lhs, &rhs, TensorDotAxes::Count(2)).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let result = out.run_with(&mut executor).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    assert_close(get_v2(&result, &[0, 0]), 611.0, "tensordot_count[0,0]");
    assert_close(get_v2(&result, &[1, 0]), 650.0, "tensordot_count[1,0]");
    assert_close(get_v2(&result, &[0, 1]), 1475.0, "tensordot_count[0,1]");
    assert_close(get_v2(&result, &[1, 1]), 1586.0, "tensordot_count[1,1]");
}

#[test]
fn traced_tensor_namespace_tensordot_explicit_axes_accept_negative_indices() {
    let lhs = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let out = tenferro_einsum::traced_tensor::tensordot(
        &lhs,
        &rhs,
        TensorDotAxes::Axes {
            lhs: &[-1],
            rhs: &[0],
        },
    )
    .unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let result = out.run_with(&mut executor).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(get_f64_data(&result), &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn traced_tensor_namespace_tensordot_rejects_invalid_axes() {
    let lhs = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    let rhs = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]);

    let duplicate = match tenferro_einsum::traced_tensor::tensordot(
        &lhs,
        &rhs,
        TensorDotAxes::Axes {
            lhs: &[0, 0],
            rhs: &[0, 1],
        },
    ) {
        Ok(_) => panic!("expected duplicate tensordot axis error"),
        Err(err) => err,
    };
    assert!(duplicate.to_string().contains("duplicate lhs axis"));

    let out_of_bounds =
        match tenferro_einsum::traced_tensor::tensordot(&lhs, &rhs, TensorDotAxes::Count(3)) {
            Ok(_) => panic!("expected Count(3) tensordot axis error"),
            Err(err) => err,
        };
    assert!(out_of_bounds.to_string().contains("Count(3)"));

    let explicit_out_of_bounds = match tenferro_einsum::traced_tensor::tensordot(
        &lhs,
        &rhs,
        TensorDotAxes::Axes {
            lhs: &[2],
            rhs: &[0],
        },
    ) {
        Ok(_) => panic!("expected explicit tensordot axis bounds error"),
        Err(err) => err,
    };
    assert!(explicit_out_of_bounds
        .to_string()
        .contains("lhs axis 2 out of bounds"));
}

#[test]
fn einsum_identity() {
    // "ij->ij" — identity copy
    // v1 data: col-major [1,2,3,4,5,6] shape [2,3]
    // a[0,0]=1, a[1,0]=2, a[0,1]=3, a[1,1]=4, a[0,2]=5, a[1,2]=6
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = einsum(&mut engine, &[&ta], "ij->ij").unwrap();
    let result = tb.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            assert_close(
                get_v2(&result, &[i, j]),
                get_v2(&a, &[i, j]),
                &format!("identity[{i},{j}]"),
            );
        }
    }
}

#[test]
fn einsum_transpose() {
    // "ij->ji"
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = einsum(&mut engine, &[&ta], "ij->ji").unwrap();
    let result = tb.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[3, 2]);
    for i in 0..2 {
        for j in 0..3 {
            assert_close(
                get_v2(&result, &[j, i]),
                get_v2(&a, &[i, j]),
                &format!("transpose[{j},{i}]"),
            );
        }
    }
}

#[test]
fn einsum_sum_reduce() {
    // "ij->i" — sum over j
    // a col-major: a[0,0]=1, a[1,0]=2, a[0,1]=3, a[1,1]=4, a[0,2]=5, a[1,2]=6
    // b[0] = 1+3+5 = 9, b[1] = 2+4+6 = 12
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let tb = einsum(&mut engine, &[&ta], "ij->i").unwrap();
    let result = tb.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[2]);
    let data = get_f64_data(&result);
    assert_close(data[0], 9.0, "sum_reduce[0]");
    assert_close(data[1], 12.0, "sum_reduce[1]");
}

#[test]
fn einsum_full_contraction() {
    // "ij->" — sum all elements = 1+2+3+4+5+6 = 21
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let tb = einsum(&mut engine, &[&ta], "ij->").unwrap();
    let result = tb.run_with(&mut engine).unwrap();

    assert!(result.shape().is_empty());
    let data = get_f64_data(&result);
    assert_close(data[0], 21.0, "full_contraction");
}

#[test]
fn einsum_trace() {
    // "ii->" — trace of 2x2 matrix
    // col-major [1,2,3,4]: a[0,0]=1, a[1,0]=2, a[0,1]=3, a[1,1]=4
    // trace = 1 + 4 = 5
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let tb = einsum(&mut engine, &[&ta], "ii->").unwrap();
    let result = tb.run_with(&mut engine).unwrap();

    assert!(result.shape().is_empty());
    let data = get_f64_data(&result);
    assert_close(data[0], 5.0, "trace");
}

#[test]
fn einsum_diagonal_extraction() {
    // "ii->i" — extract diagonal of 3x3 matrix
    // col-major [1..9]: a[0,0]=1, a[1,0]=2, ..., a[1,1]=5, ..., a[2,2]=9
    let a = f64_tensor(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let tb = einsum(&mut engine, &[&ta], "ii->i").unwrap();
    let result = tb.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[3]);
    let data = get_f64_data(&result);
    assert_close(data[0], 1.0, "diag[0]");
    assert_close(data[1], 5.0, "diag[1]");
    assert_close(data[2], 9.0, "diag[2]");
}

#[test]
fn einsum_diagonal_embedding() {
    // "i->ii" — diagonal embedding: produces a diagonal matrix from a vector.
    let v = f64_tensor(vec![3], vec![2.0, 3.0, 5.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let tv = TracedTensor::from_tensor_concrete_shape(v);
    let td = einsum(&mut engine, &[&tv], "i->ii").unwrap();
    let result = td.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[3, 3]);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { [2.0, 3.0, 5.0][i] } else { 0.0 };
            assert_close(
                get_v2(&result, &[i, j]),
                expected,
                &format!("embed_diag[{i},{j}]"),
            );
        }
    }
}

// ============================================================================
// Group 2: Binary operations
// ============================================================================

#[test]
fn einsum_matmul() {
    // "ij,jk->ik"
    // Same data as v1: A[2,3] col-major [1,2,3,4,5,6], B[3,4] col-major [1..12]
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(
        vec![3, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tc = einsum(&mut engine, &[&ta, &tb], "ij,jk->ik").unwrap();
    let result = tc.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[2, 4]);
    // Verify against manual computation
    for i in 0..2 {
        for k in 0..4 {
            let mut expected = 0.0;
            for j in 0..3 {
                expected += get_v2(&a, &[i, j]) * get_v2(&b, &[j, k]);
            }
            assert_close(
                get_v2(&result, &[i, k]),
                expected,
                &format!("matmul[{i},{k}]"),
            );
        }
    }
}

#[test]
fn einsum_symbolic_matmul_matches_static_path() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let a_symbolic = symbolic_identity_2d(&a);
    let b_symbolic = symbolic_identity_2d(&b);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let expected = einsum(&mut engine, &[&a, &b], "ij,jk->ik").unwrap();
    let expected = expected.run_with(&mut engine).unwrap().clone();

    let actual = einsum(&mut engine, &[&a_symbolic, &b_symbolic], "ij,jk->ik").unwrap();
    let actual = actual.run_with(&mut engine).unwrap();

    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(get_f64_data(&actual), get_f64_data(&expected));
}

#[test]
fn einsum_symbolic_three_tensor_chain_matches_static_path() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 2],
        vec![1.0, -0.5, 2.0, 0.75],
    ));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 2],
        vec![0.5, 1.5, -1.0, 0.25],
    ));
    let c = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 2],
        vec![2.0, -1.5, 0.75, 1.25],
    ));
    let a_symbolic = symbolic_identity_2d(&a);
    let b_symbolic = symbolic_identity_2d(&b);
    let c_symbolic = symbolic_identity_2d(&c);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let expected = einsum(&mut engine, &[&a, &b, &c], "ij,jk,kl->il").unwrap();
    let expected = expected.run_with(&mut engine).unwrap().clone();

    let actual = einsum(
        &mut engine,
        &[&a_symbolic, &b_symbolic, &c_symbolic],
        "ij,jk,kl->il",
    )
    .unwrap();
    let actual = actual.run_with(&mut engine).unwrap();

    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(get_f64_data(&actual), get_f64_data(&expected));
}

#[test]
fn einsum_cache_reuses_contraction_path() {
    let mut compiler = GraphCompiler::new();

    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 4],
        (1..=12).map(|x| x as f64).collect(),
    ));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![4, 5],
        (1..=20).map(|x| (x * 2) as f64).collect(),
    ));

    let c1 = einsum(&mut compiler, &[&a, &b], "ij,jk->ik").unwrap();
    assert_eq!(c1.rank, 2);
    let entries_after_first = extension_cache_entries(&compiler);
    assert!(entries_after_first >= 2);

    let a2 = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 4],
        (21..=32).map(|x| x as f64).collect(),
    ));
    let b2 = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![4, 5],
        (41..=60).map(|x| x as f64).collect(),
    ));

    let c2 = einsum(&mut compiler, &[&a2, &b2], "ij,jk->ik").unwrap();
    assert_eq!(c2.rank, 2);
    assert_eq!(extension_cache_entries(&compiler), entries_after_first);

    let a3 = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![5, 6],
        (1..=30).map(|x| (x as f64) / 10.0).collect(),
    ));
    let b3 = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![6, 7],
        (1..=42).map(|x| (x as f64) / 5.0).collect(),
    ));

    let c3 = einsum(&mut compiler, &[&a3, &b3], "ij,jk->ik").unwrap();
    assert_eq!(c3.rank, 2);
    assert!(extension_cache_entries(&compiler) > entries_after_first);
}

#[test]
fn einsum_outer_product() {
    // "i,j->ij"
    let u = f64_tensor(vec![2], vec![1.0, 2.0]);
    let v = f64_tensor(vec![3], vec![3.0, 4.0, 5.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let tu = TracedTensor::from_tensor_concrete_shape(u.clone());
    let tv = TracedTensor::from_tensor_concrete_shape(v.clone());
    let tm = einsum(&mut engine, &[&tu, &tv], "i,j->ij").unwrap();
    let result = tm.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            let expected = get_v2(&u, &[i]) * get_v2(&v, &[j]);
            assert_close(
                get_v2(&result, &[i, j]),
                expected,
                &format!("outer[{i},{j}]"),
            );
        }
    }
}

#[test]
fn einsum_dot_product() {
    // "i,i->"
    let u = f64_tensor(vec![3], vec![1.0, 2.0, 3.0]);
    let v = f64_tensor(vec![3], vec![4.0, 5.0, 6.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let tu = TracedTensor::from_tensor_concrete_shape(u);
    let tv = TracedTensor::from_tensor_concrete_shape(v);
    let td = einsum(&mut engine, &[&tu, &tv], "i,i->").unwrap();
    let result = td.run_with(&mut engine).unwrap();

    // 1*4 + 2*5 + 3*6 = 32
    let data = get_f64_data(&result);
    assert_close(data[0], 32.0, "dot_product");
}

#[test]
fn einsum_matvec() {
    // "ij,j->i"
    // A[2,3] col-major [1,2,3,4,5,6], x = [1,2,3]
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let x = f64_tensor(vec![3], vec![1.0, 2.0, 3.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tx = TracedTensor::from_tensor_concrete_shape(x.clone());
    let ty = einsum(&mut engine, &[&ta, &tx], "ij,j->i").unwrap();
    let result = ty.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[2]);
    for i in 0..2 {
        let mut expected = 0.0;
        for j in 0..3 {
            expected += get_v2(&a, &[i, j]) * get_v2(&x, &[j]);
        }
        assert_close(get_v2(&result, &[i]), expected, &format!("matvec[{i}]"));
    }
}

#[test]
fn einsum_elementwise_mul() {
    // "ij,ij->ij" — Hadamard product
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tc = einsum(&mut engine, &[&ta, &tb], "ij,ij->ij").unwrap();
    let result = tc.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    for i in 0..2 {
        for j in 0..2 {
            let expected = get_v2(&a, &[i, j]) * get_v2(&b, &[i, j]);
            assert_close(
                get_v2(&result, &[i, j]),
                expected,
                &format!("hadamard[{i},{j}]"),
            );
        }
    }
}

// ============================================================================
// Group 3: N-ary operations
// ============================================================================

#[test]
fn einsum_three_matrices() {
    // "ij,jk,kl->il" — chain multiply 3 matrices
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let c = f64_tensor(vec![2, 2], vec![9.0, 10.0, 11.0, 12.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tc = TracedTensor::from_tensor_concrete_shape(c.clone());
    let td = einsum(&mut engine, &[&ta, &tb, &tc], "ij,jk,kl->il").unwrap();
    let result_d = td.run_with(&mut engine).unwrap();

    assert_eq!(result_d.shape(), &[2, 2]);

    // Verify D = A @ B @ C by computing step-by-step
    // First: AB
    let ta2 = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb2 = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tab = einsum(&mut engine, &[&ta2, &tb2], "ij,jk->ik").unwrap();
    let ab = tab.run_with(&mut engine).unwrap().clone();

    // Then: (AB) @ C
    let tab2 = TracedTensor::from_tensor_concrete_shape(ab);
    let tc2 = TracedTensor::from_tensor_concrete_shape(c);
    let tabc = einsum(&mut engine, &[&tab2, &tc2], "ij,jk->ik").unwrap();
    let abc = tabc.run_with(&mut engine).unwrap();

    for i in 0..2 {
        for j in 0..2 {
            assert_close(
                get_v2(&result_d, &[i, j]),
                get_v2(&abc, &[i, j]),
                &format!("three_mat[{i},{j}]"),
            );
        }
    }
}

#[test]
fn einsum_batched_three_matrix_chain_matches_pairwise_reference() {
    // "bik,bkj,bjl->bil" — batched three-matrix chain with deterministic data.
    let a = f64_tensor(vec![2, 2, 3], (1..=12).map(|x| x as f64).collect());
    let b = f64_tensor(vec![2, 3, 4], (13..=36).map(|x| x as f64).collect());
    let c = f64_tensor(vec![2, 4, 2], (37..=52).map(|x| x as f64).collect());

    let mut direct_engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tc = TracedTensor::from_tensor_concrete_shape(c.clone());
    let direct = einsum(&mut direct_engine, &[&ta, &tb, &tc], "bik,bkj,bjl->bil").unwrap();
    let direct_result = direct.run_with(&mut direct_engine).unwrap();

    let mut pairwise_engine = GraphExecutor::new(CpuBackend::new());
    let pa = TracedTensor::from_tensor_concrete_shape(a);
    let pb = TracedTensor::from_tensor_concrete_shape(b);
    let pc = TracedTensor::from_tensor_concrete_shape(c);
    let first = einsum(&mut pairwise_engine, &[&pa, &pb], "bik,bkj->bij").unwrap();
    let first_result = first.run_with(&mut pairwise_engine).unwrap().clone();
    let first_tensor = TracedTensor::from_tensor_concrete_shape(first_result);
    let reference = einsum(&mut pairwise_engine, &[&first_tensor, &pc], "bij,bjl->bil").unwrap();
    let reference_result = reference.run_with(&mut pairwise_engine).unwrap();

    assert_eq!(direct_result.shape(), reference_result.shape());
    assert_eq!(direct_result.shape(), &[2, 2, 2]);
    assert_close(
        get_v2(&direct_result, &[0, 0, 0]),
        61060.0,
        "batched_chain_manual[0,0,0]",
    );
    assert_close(
        get_v2(&direct_result, &[1, 1, 1]),
        122176.0,
        "batched_chain_manual[1,1,1]",
    );
    for b in 0..2 {
        for i in 0..2 {
            for l in 0..2 {
                assert_close(
                    get_v2(&direct_result, &[b, i, l]),
                    get_v2(&reference_result, &[b, i, l]),
                    &format!("batched_chain[{b},{i},{l}]"),
                );
            }
        }
    }
}

// ============================================================================
// Group 4: Contraction tree / path tests
// ============================================================================

#[test]
fn einsum_with_path_matches_flat_nary() {
    // Verify that an explicit JAX path produces the same result as auto-optimized.
    // A[2,2], B[2,2], C[2,2]; "ij,jk,kl->il"
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let c = f64_tensor(vec![2, 2], vec![9.0, 10.0, 11.0, 12.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());

    // Auto-optimized
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tc = TracedTensor::from_tensor_concrete_shape(c.clone());
    let auto = einsum(&mut engine, &[&ta, &tb, &tc], "ij,jk,kl->il").unwrap();
    let auto_result = auto.run_with(&mut engine).unwrap().clone();

    // Explicit path: contract B*C first (positions 1,2), then A*result (positions 0,1)
    let ta2 = TracedTensor::from_tensor_concrete_shape(a);
    let tb2 = TracedTensor::from_tensor_concrete_shape(b);
    let tc2 = TracedTensor::from_tensor_concrete_shape(c);
    let via_path = einsum_with(
        &mut engine,
        &[&ta2, &tb2, &tc2],
        "ij,jk,kl->il",
        EinsumOptimize::Path(vec![(1, 2), (0, 1)]),
    )
    .unwrap();
    let path_result = via_path.run_with(&mut engine).unwrap();

    for i in 0..2 {
        for j in 0..2 {
            assert_close(
                get_v2(&path_result, &[i, j]),
                get_v2(&auto_result, &[i, j]),
                &format!("path_vs_auto[{i},{j}]"),
            );
        }
    }
}

#[test]
fn contraction_tree_from_pairs() {
    // Build contraction tree from explicit pairs and verify shape.
    // A[2,3] B[3,4] C[4,5] -> D[2,5]
    // Contract B*C first (pair 1,2 -> index 3), then A*T (pair 0,3)
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4], &[4, 5]], &[(1, 2), (0, 3)])
        .unwrap();

    // Use the tree with v2 API
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(
        vec![3, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    let c = f64_tensor(
        vec![4, 5],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
        ],
    );

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let tb = TracedTensor::from_tensor_concrete_shape(b);
    let tc = TracedTensor::from_tensor_concrete_shape(c);
    let td = einsum_with(
        &mut engine,
        &[&ta, &tb, &tc],
        "ij,jk,kl->il",
        EinsumOptimize::Tree(tree),
    )
    .unwrap();
    let result = td.run_with(&mut engine).unwrap();
    assert_eq!(result.shape(), &[2, 5]);
}

#[test]
fn contraction_tree_from_pairs_rejects_wrong_step_count() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let result = ContractionTree::from_pairs(&subs, &[&[2, 2], &[2, 2], &[2, 2]], &[(1, 2)]);
    assert!(result.is_err(), "wrong number of path steps must error");
}

// ============================================================================
// Group 5: Complex contraction patterns
// ============================================================================

#[test]
fn einsum_partial_trace_with_free_index() {
    // "iij->j" — partial trace: v[j] = sum_i T[i,i,j]
    // T[2,2,3] col-major: data 1..12
    let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
    let t = f64_tensor(vec![2, 2, 3], data);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let tt = TracedTensor::from_tensor_concrete_shape(t.clone());
    let tv = einsum(&mut engine, &[&tt], "iij->j").unwrap();
    let result = tv.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[3]);
    for j in 0..3 {
        let mut expected = 0.0;
        for i in 0..2 {
            expected += get_v2(&t, &[i, i, j]);
        }
        assert_close(
            get_v2(&result, &[j]),
            expected,
            &format!("partial_trace[{j}]"),
        );
    }
}

#[test]
fn einsum_batched_matmul() {
    // "bij,bjk->bik" — batched matrix multiply
    // batch=2, each is 2x2
    // A col-major [2,2,2]:
    //   batch 0: I = [[1,0],[0,1]]
    //   batch 1: 2I = [[2,0],[0,2]]
    // Col-major for [2,2,2]: leftmost varies fastest
    //   A[b,i,j] with strides [1,2,4]
    //   data[0]=A[0,0,0]=1, data[1]=A[1,0,0]=2
    //   data[2]=A[0,1,0]=0, data[3]=A[1,1,0]=0
    //   data[4]=A[0,0,1]=0, data[5]=A[1,0,1]=0
    //   data[6]=A[0,1,1]=1, data[7]=A[1,1,1]=2
    let a = f64_tensor(vec![2, 2, 2], vec![1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0]);
    // B col-major [2,2,2]:
    //   B[b,j,k] with strides [1,2,4]
    //   batch 0: [[1,3],[2,4]], batch 1: [[5,7],[6,8]]
    //   data[0]=B[0,0,0]=1, data[1]=B[1,0,0]=5
    //   data[2]=B[0,1,0]=2, data[3]=B[1,1,0]=6
    //   data[4]=B[0,0,1]=3, data[5]=B[1,0,1]=7
    //   data[6]=B[0,1,1]=4, data[7]=B[1,1,1]=8
    let b = f64_tensor(vec![2, 2, 2], vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tc = einsum(&mut engine, &[&ta, &tb], "bij,bjk->bik").unwrap();
    let result = tc.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[2, 2, 2]);
    for batch in 0..2 {
        for i in 0..2 {
            for k in 0..2 {
                let mut expected = 0.0;
                for j in 0..2 {
                    expected += get_v2(&a, &[batch, i, j]) * get_v2(&b, &[batch, j, k]);
                }
                assert_close(
                    get_v2(&result, &[batch, i, k]),
                    expected,
                    &format!("batched_matmul[{batch},{i},{k}]"),
                );
            }
        }
    }
}

#[test]
fn einsum_reduce_first_axis() {
    // "ij->j" — sum over first axis
    // A[3,4] col-major: data [0..12] with strides [1,3]
    //   A[0,0]=0, A[1,0]=1, A[2,0]=2, A[0,1]=3, A[1,1]=4, A[2,1]=5, ...
    let a = f64_tensor(
        vec![3, 4],
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    );

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let ty = einsum(&mut engine, &[&ta], "ij->j").unwrap();
    let result = ty.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[4]);
    for j in 0..4 {
        let expected = get_v2(&a, &[0, j]) + get_v2(&a, &[1, j]) + get_v2(&a, &[2, j]);
        assert_close(
            get_v2(&result, &[j]),
            expected,
            &format!("reduce_first[{j}]"),
        );
    }
}

#[test]
fn einsum_self_contraction_trace() {
    // "ijk->j" — self-contraction: sum over i and k (not a trace — just reduction)
    // T[2,3,2] col-major: data 1..12
    let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
    let t = f64_tensor(vec![2, 3, 2], data);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let tt = TracedTensor::from_tensor_concrete_shape(t.clone());
    let tv = einsum(&mut engine, &[&tt], "ijk->j").unwrap();
    let result = tv.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[3]);
    for j in 0..3 {
        let mut expected = 0.0;
        for i in 0..2 {
            for k in 0..2 {
                expected += get_v2(&t, &[i, j, k]);
            }
        }
        assert_close(
            get_v2(&result, &[j]),
            expected,
            &format!("self_contraction[{j}]"),
        );
    }
}

// ============================================================================
// Group 6: EinsumOptimize variants (v2-specific)
// ============================================================================
