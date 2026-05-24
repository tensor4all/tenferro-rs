//! Ported v1 einsum tests adapted for the v2 traced pipeline.
//!
//! Data is stored in **column-major** order. A helper `row_major_to_column_major` is
//! available for converting row-major test data when needed.

use num_complex::Complex64;
use tenferro::error::Result;
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, Tensor, TracedTensor};
use tenferro_einsum::EinsumOptimize;
use tenferro_einsum::{ContractionTree, NestedEinsum, Subscripts, EINSUM_EXTENSION_FAMILY_ID};
use tenferro_tensor::TypedTensor;

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
    fn run_with(&self, executor: &mut GraphExecutor<CpuBackend>)
        -> tenferro::error::Result<Tensor>;
}

impl RunTraced for TracedTensor {
    fn run_with(
        &self,
        executor: &mut GraphExecutor<CpuBackend>,
    ) -> tenferro::error::Result<Tensor> {
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

#[test]
fn test_optimize_false_left_to_right() {
    // EinsumOptimize::False contracts left-to-right
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(
        vec![3, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    let c = f64_tensor(vec![4, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());

    // Reference: auto optimization
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tc = TracedTensor::from_tensor_concrete_shape(c.clone());
    let auto = einsum(&mut engine, &[&ta, &tb, &tc], "ij,jk,kl->il").unwrap();
    let auto_result = auto.run_with(&mut engine).unwrap().clone();

    // False: left-to-right
    let ta2 = TracedTensor::from_tensor_concrete_shape(a);
    let tb2 = TracedTensor::from_tensor_concrete_shape(b);
    let tc2 = TracedTensor::from_tensor_concrete_shape(c);
    let ltr = einsum_with(
        &mut engine,
        &[&ta2, &tb2, &tc2],
        "ij,jk,kl->il",
        EinsumOptimize::False,
    )
    .unwrap();
    let ltr_result = ltr.run_with(&mut engine).unwrap();

    for i in 0..2 {
        for l in 0..2 {
            assert_close(
                get_v2(&ltr_result, &[i, l]),
                get_v2(&auto_result, &[i, l]),
                &format!("false_ltr[{i},{l}]"),
            );
        }
    }
}

#[test]
fn test_optimize_path_jax_compatible() {
    // EinsumOptimize::Path with JAX-style indices
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(
        vec![3, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    let c = f64_tensor(vec![4, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());

    // Reference
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tc = TracedTensor::from_tensor_concrete_shape(c.clone());
    let auto = einsum(&mut engine, &[&ta, &tb, &tc], "ij,jk,kl->il").unwrap();
    let auto_result = auto.run_with(&mut engine).unwrap().clone();

    // Path: contract A*B first (0,1), then result*C (0,1)
    let ta2 = TracedTensor::from_tensor_concrete_shape(a);
    let tb2 = TracedTensor::from_tensor_concrete_shape(b);
    let tc2 = TracedTensor::from_tensor_concrete_shape(c);
    let path = einsum_with(
        &mut engine,
        &[&ta2, &tb2, &tc2],
        "ij,jk,kl->il",
        EinsumOptimize::Path(vec![(0, 1), (0, 1)]),
    )
    .unwrap();
    let path_result = path.run_with(&mut engine).unwrap();

    for i in 0..2 {
        for l in 0..2 {
            assert_close(
                get_v2(&path_result, &[i, l]),
                get_v2(&auto_result, &[i, l]),
                &format!("path_jax[{i},{l}]"),
            );
        }
    }
}

#[test]
fn test_optimize_nested() {
    // EinsumOptimize::Nested with parenthesized notation
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(
        vec![3, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    let c = f64_tensor(vec![4, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());

    // Reference
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tc = TracedTensor::from_tensor_concrete_shape(c.clone());
    let auto = einsum(&mut engine, &[&ta, &tb, &tc], "ij,jk,kl->il").unwrap();
    let auto_result = auto.run_with(&mut engine).unwrap().clone();

    // Nested: "(ij,jk),kl->il" = contract A*B first, then result*C
    let nested = NestedEinsum::parse("(ij,jk),kl->il").unwrap();
    let ta2 = TracedTensor::from_tensor_concrete_shape(a);
    let tb2 = TracedTensor::from_tensor_concrete_shape(b);
    let tc2 = TracedTensor::from_tensor_concrete_shape(c);
    let nested_result = einsum_with(
        &mut engine,
        &[&ta2, &tb2, &tc2],
        "ij,jk,kl->il",
        EinsumOptimize::Nested(nested),
    )
    .unwrap();
    let result = nested_result.run_with(&mut engine).unwrap();

    for i in 0..2 {
        for l in 0..2 {
            assert_close(
                get_v2(&result, &[i, l]),
                get_v2(&auto_result, &[i, l]),
                &format!("nested[{i},{l}]"),
            );
        }
    }
}

#[test]
fn test_optimize_tree() {
    // EinsumOptimize::Tree with pre-computed ContractionTree
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(
        vec![3, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    let c = f64_tensor(vec![4, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let subs = Subscripts::parse("ij,jk,kl->il").unwrap();
    let shapes: &[&[usize]] = &[&[2, 3], &[3, 4], &[4, 2]];
    let tree = ContractionTree::optimize(&subs, shapes).unwrap();

    let mut engine = GraphExecutor::new(CpuBackend::new());

    // Reference
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tc = TracedTensor::from_tensor_concrete_shape(c.clone());
    let auto = einsum(&mut engine, &[&ta, &tb, &tc], "ij,jk,kl->il").unwrap();
    let auto_result = auto.run_with(&mut engine).unwrap().clone();

    // Tree
    let ta2 = TracedTensor::from_tensor_concrete_shape(a);
    let tb2 = TracedTensor::from_tensor_concrete_shape(b);
    let tc2 = TracedTensor::from_tensor_concrete_shape(c);
    let tree_result = einsum_with(
        &mut engine,
        &[&ta2, &tb2, &tc2],
        "ij,jk,kl->il",
        EinsumOptimize::Tree(tree),
    )
    .unwrap();
    let result = tree_result.run_with(&mut engine).unwrap();

    for i in 0..2 {
        for l in 0..2 {
            assert_close(
                get_v2(&result, &[i, l]),
                get_v2(&auto_result, &[i, l]),
                &format!("tree[{i},{l}]"),
            );
        }
    }
}

// ============================================================================
// Group 7: Error cases
// ============================================================================

#[test]
fn einsum_error_rank_mismatch() {
    // Subscripts say "ij" (rank 2) but tensor is rank 1
    let v = f64_tensor(vec![3], vec![1.0, 2.0, 3.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let tv = TracedTensor::from_tensor_concrete_shape(v);
    // v2 returns Err during contraction tree optimization because the subscript
    // label count doesn't match the tensor's rank.
    let result = einsum(&mut engine, &[&tv], "ij->i");
    assert!(result.is_err());
}

#[test]
fn einsum_error_empty_inputs() {
    // No input tensors -- v2 returns Err during tree construction
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let empty: &[&TracedTensor] = &[];
    let result = einsum(&mut engine, empty, "->");
    assert!(result.is_err());
}

#[test]
fn einsum_wrong_operand_count() {
    // Subscripts say 2 inputs but only 1 provided
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let result = einsum(&mut engine, &[&ta], "ij,jk->ik");
    assert!(result.is_err());
}

#[test]
fn einsum_shape_mismatch() {
    // j=3 in A but j=2 in B -> shape mismatch
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let tb = TracedTensor::from_tensor_concrete_shape(b);
    let result = einsum(&mut engine, &[&ta, &tb], "ij,jk->ik");
    assert!(result.is_err());
}

// ============================================================================
// Group 8: Ported repeated-index and diagonal-extract patterns
// ============================================================================

#[test]
fn einsum_trace_rank3_to_vector() {
    // "iji->j" — trace of rank-3 tensor: v[j] = sum_i T[i,j,i]
    // T[3,2,3] col-major: data 1..18
    let data: Vec<f64> = (1..=18).map(|x| x as f64).collect();
    let t = f64_tensor(vec![3, 2, 3], data);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let tt = TracedTensor::from_tensor_concrete_shape(t.clone());
    let tv = einsum(&mut engine, &[&tt], "iji->j").unwrap();
    let result = tv.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[2]);
    for j in 0..2 {
        let mut expected = 0.0;
        for i in 0..3 {
            expected += get_v2(&t, &[i, j, i]);
        }
        assert_close(
            get_v2(&result, &[j]),
            expected,
            &format!("trace_rank3[{j}]"),
        );
    }
}

#[test]
fn einsum_multi_pair_trace_iijj() {
    // "iijj->" — trace over two pairs of repeated indices
    let n = 3;
    let total = n * n * n * n;
    let data: Vec<f64> = (0..total).map(|x| x as f64).collect();
    let a = f64_tensor(vec![n, n, n, n], data);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let ts = einsum(&mut engine, &[&ta], "iijj->").unwrap();
    let result = ts.run_with(&mut engine).unwrap();

    assert!(result.shape().is_empty());
    let mut expected = 0.0;
    for i in 0..n {
        for j in 0..n {
            expected += get_v2(&a, &[i, i, j, j]);
        }
    }
    assert_close(get_f64_data(&result)[0], expected, "multi_pair_trace_iijj");
}

#[test]
fn einsum_diag_extract_reduce_ijj_to_i() {
    // "ijj->i" — extract diagonal along axes 1,2, then free axis 0
    let data: Vec<f64> = (0..18).map(|x| x as f64).collect();
    let a = f64_tensor(vec![2, 3, 3], data);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let ty = einsum(&mut engine, &[&ta], "ijj->i").unwrap();
    let result = ty.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[2]);
    for i in 0..2 {
        let mut expected = 0.0;
        for j in 0..3 {
            expected += get_v2(&a, &[i, j, j]);
        }
        assert_close(get_v2(&result, &[i]), expected, &format!("ijj_to_i[{i}]"));
    }
}

#[test]
fn einsum_diag_extract_no_reduce_ijj_to_ij() {
    // "ijj->ij" — extract diagonal along axes 1,2, keep both i and j
    let data: Vec<f64> = (0..18).map(|x| x as f64).collect();
    let a = f64_tensor(vec![2, 3, 3], data);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let ty = einsum(&mut engine, &[&ta], "ijj->ij").unwrap();
    let result = ty.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            let expected = get_v2(&a, &[i, j, j]);
            assert_close(
                get_v2(&result, &[i, j]),
                expected,
                &format!("ijj_to_ij[{i},{j}]"),
            );
        }
    }
}

#[test]
fn einsum_diag_extract_permuted_jii_to_j() {
    // "jii->j" — extract diagonal along axes 1,2, reduce over i
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let a = f64_tensor(vec![3, 2, 2], data);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let ty = einsum(&mut engine, &[&ta], "jii->j").unwrap();
    let result = ty.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[3]);
    for j in 0..3 {
        let expected = get_v2(&a, &[j, 0, 0]) + get_v2(&a, &[j, 1, 1]);
        assert_close(get_v2(&result, &[j]), expected, &format!("jii_to_j[{j}]"));
    }
}

#[test]
fn einsum_scalar_vector_products() {
    // v2 scalars use shape [] (rank 0, 1 element)
    let three = f64_tensor(vec![], vec![3.0]);
    let two = f64_tensor(vec![], vec![2.0]);
    let v4 = f64_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let v3 = f64_tensor(vec![3], vec![10.0, 20.0, 30.0]);
    let seven = f64_tensor(vec![], vec![7.0]);

    // ",k->k": scalar * vector
    {
        let mut engine = GraphExecutor::new(CpuBackend::new());
        let ts = TracedTensor::from_tensor_concrete_shape(three.clone());
        let tv = TracedTensor::from_tensor_concrete_shape(v4.clone());
        let result = einsum(&mut engine, &[&ts, &tv], ",k->k").unwrap();
        let r = result.run_with(&mut engine).unwrap();
        assert_eq!(r.shape(), &[4]);
        assert_close(get_v2(&r, &[0]), 3.0, "scalar*vec[0]");
        assert_close(get_v2(&r, &[1]), 6.0, "scalar*vec[1]");
        assert_close(get_v2(&r, &[2]), 9.0, "scalar*vec[2]");
        assert_close(get_v2(&r, &[3]), 12.0, "scalar*vec[3]");
    }

    // "i,->i": vector * scalar
    {
        let mut engine = GraphExecutor::new(CpuBackend::new());
        let tv = TracedTensor::from_tensor_concrete_shape(v3.clone());
        let ts = TracedTensor::from_tensor_concrete_shape(two.clone());
        let result = einsum(&mut engine, &[&tv, &ts], "i,->i").unwrap();
        let r = result.run_with(&mut engine).unwrap();
        assert_eq!(r.shape(), &[3]);
        assert_close(get_v2(&r, &[0]), 20.0, "vec*scalar[0]");
        assert_close(get_v2(&r, &[1]), 40.0, "vec*scalar[1]");
        assert_close(get_v2(&r, &[2]), 60.0, "vec*scalar[2]");
    }

    // ",->": scalar * scalar
    {
        let mut engine = GraphExecutor::new(CpuBackend::new());
        let ts = TracedTensor::from_tensor_concrete_shape(three.clone());
        let t7 = TracedTensor::from_tensor_concrete_shape(seven.clone());
        let result = einsum(&mut engine, &[&ts, &t7], ",->").unwrap();
        let r = result.run_with(&mut engine).unwrap();
        assert!(r.shape().is_empty());
        assert_close(get_f64_data(&r)[0], 21.0, "scalar*scalar");
    }
}

// ============================================================================
// Group 9: Error cases (ported from v1)
// ============================================================================

#[test]
#[should_panic(expected = "einsum output label")]
fn einsum_error_output_label_not_in_input() {
    // Output references label 'k' which is not in any input
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let r = einsum(&mut engine, &[&ta], "ij->ik").unwrap();
    r.run_with(&mut engine).unwrap();
}

#[test]
fn einsum_error_non_square_trace() {
    // Trace "ii->" but matrix is 2x3 (non-square)
    // v2 catches this at contraction tree optimization time
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let result = einsum(&mut engine, &[&ta], "ii->");
    assert!(result.is_err());
}

// ============================================================================
// Group 10: Path tests (ported from v1)
// ============================================================================

#[test]
#[should_panic(expected = "references operand positions")]
fn einsum_with_path_invalid_pairs_errors() {
    // Invalid JAX path with out-of-bounds indices should panic during tree build
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let c = f64_tensor(vec![2, 2], vec![9.0, 10.0, 11.0, 12.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let tb = TracedTensor::from_tensor_concrete_shape(b);
    let tc = TracedTensor::from_tensor_concrete_shape(c);
    let r = einsum_with(
        &mut engine,
        &[&ta, &tb, &tc],
        "ij,jk,kl->il",
        EinsumOptimize::Path(vec![(0, 99), (0, 1)]),
    )
    .unwrap();
    r.run_with(&mut engine).unwrap();
}

#[test]
fn einsum_with_path_rejects_structurally_invalid_paths() {
    // Wrong step count: 3 operands needs 2 steps, giving 1 is structurally wrong
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let c = f64_tensor(vec![2, 2], vec![9.0, 10.0, 11.0, 12.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let tb = TracedTensor::from_tensor_concrete_shape(b);
    let tc = TracedTensor::from_tensor_concrete_shape(c);
    let result = einsum_with(
        &mut engine,
        &[&ta, &tb, &tc],
        "ij,jk,kl->il",
        EinsumOptimize::Path(vec![(1, 2)]),
    );
    assert!(result.is_err());
}

// ============================================================================
// Group 11: Ported v1 tests — batched, transposed, binary diagonal patterns
// ============================================================================

fn c64_tensor(shape: Vec<usize>, data: Vec<Complex64>) -> Tensor {
    Tensor::C64(TypedTensor::from_vec_col_major(shape, data))
}

fn get_c64(t: &Tensor, idx: &[usize]) -> Complex64 {
    match t {
        Tensor::C64(inner) => *inner.get(idx),
        _ => panic!("expected C64"),
    }
}

fn assert_close_c64(a: Complex64, b: Complex64, label: &str) {
    assert!((a - b).norm() < 1e-10, "{label}: got {a}, expected {b}");
}

#[test]
fn einsum_batched_dot_product() {
    // "bi,bi->b" — batched dot product
    let a = f64_tensor(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tc = einsum(&mut engine, &[&ta, &tb], "bi,bi->b").unwrap();
    let result = tc.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[3]);
    for batch in 0..3 {
        let mut expected = 0.0;
        for i in 0..2 {
            expected += get_v2(&a, &[batch, i]) * get_v2(&b, &[batch, i]);
        }
        assert_close(
            get_v2(&result, &[batch]),
            expected,
            &format!("batched_dot[{batch}]"),
        );
    }
}

#[test]
fn einsum_transposed_rhs_contraction() {
    // "ij,kj->ik" — contraction with transposed RHS
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(vec![2, 3], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tc = einsum(&mut engine, &[&ta, &tb], "ij,kj->ik").unwrap();
    let result = tc.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    for i in 0..2 {
        for k in 0..2 {
            let mut expected = 0.0;
            for j in 0..3 {
                expected += get_v2(&a, &[i, j]) * get_v2(&b, &[k, j]);
            }
            assert_close(
                get_v2(&result, &[i, k]),
                expected,
                &format!("transposed_rhs[{i},{k}]"),
            );
        }
    }
}

#[test]
fn einsum_cyclic_trace() {
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(
        vec![3, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    let c = f64_tensor(vec![4, 2], vec![2.0, 1.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tc = TracedTensor::from_tensor_concrete_shape(c.clone());
    let out = einsum(&mut engine, &[&ta, &tb, &tc], "ij,jk,ki->").unwrap();
    let result = out.run_with(&mut engine).unwrap();

    assert!(result.shape().is_empty());
    let mut expected = 0.0;
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                expected += get_v2(&a, &[i, j]) * get_v2(&b, &[j, k]) * get_v2(&c, &[k, i]);
            }
        }
    }
    assert_close(get_f64_data(&result)[0], expected, "cyclic_trace");
}

#[test]
fn einsum_three_way_repeated_label_trace() {
    // "iii->" — triple-repeated index trace
    let data: Vec<f64> = (1..=27).map(|x| x as f64).collect();
    let t = f64_tensor(vec![3, 3, 3], data);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let tt = TracedTensor::from_tensor_concrete_shape(t.clone());
    let ts = einsum(&mut engine, &[&tt], "iii->").unwrap();
    let result = ts.run_with(&mut engine).unwrap();

    assert!(result.shape().is_empty());
    let mut expected = 0.0;
    for i in 0..3 {
        expected += get_v2(&t, &[i, i, i]);
    }
    assert_close(get_f64_data(&result)[0], expected, "iii_trace");
}

#[test]
fn einsum_binary_diag_ii_jk_to_ijk() {
    // "ii,jk->ijk" — extract diagonal from A, outer product with B
    let a = f64_tensor(
        vec![3, 3],
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
    );
    let b = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tc = einsum(&mut engine, &[&ta, &tb], "ii,jk->ijk").unwrap();
    let result = tc.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[3, 2, 2]);
    let diag_a = [1.0, 2.0, 3.0];
    for i in 0..3 {
        for j in 0..2 {
            for k in 0..2 {
                let expected = diag_a[i] * get_v2(&b, &[j, k]);
                assert_close(
                    get_v2(&result, &[i, j, k]),
                    expected,
                    &format!("ii_jk_ijk[{i},{j},{k}]"),
                );
            }
        }
    }
}

#[test]
fn einsum_binary_diag_iij_jk_to_ik() {
    // "iij,jk->ik" — diagonal extraction + contraction
    let a = {
        let mut t = TypedTensor::<f64>::zeros(vec![2, 2, 3]);
        *t.get_mut(&[0, 0, 0]) = 1.0;
        *t.get_mut(&[1, 1, 0]) = 4.0;
        *t.get_mut(&[0, 0, 1]) = 2.0;
        *t.get_mut(&[1, 1, 1]) = 5.0;
        *t.get_mut(&[0, 0, 2]) = 3.0;
        *t.get_mut(&[1, 1, 2]) = 6.0;
        Tensor::F64(t)
    };
    let b = {
        let mut t = TypedTensor::<f64>::zeros(vec![3, 2]);
        *t.get_mut(&[0, 0]) = 1.0;
        *t.get_mut(&[1, 1]) = 1.0;
        Tensor::F64(t)
    };

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let tb = TracedTensor::from_tensor_concrete_shape(b);
    let tc = einsum(&mut engine, &[&ta, &tb], "iij,jk->ik").unwrap();
    let result = tc.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    assert_close(get_v2(&result, &[0, 0]), 1.0, "iij_jk_ik[0,0]");
    assert_close(get_v2(&result, &[0, 1]), 2.0, "iij_jk_ik[0,1]");
    assert_close(get_v2(&result, &[1, 0]), 4.0, "iij_jk_ik[1,0]");
    assert_close(get_v2(&result, &[1, 1]), 5.0, "iij_jk_ik[1,1]");
}

#[test]
fn einsum_binary_diag_ii_jj_to_ij() {
    // "ii,jj->ij" — outer product of two traces
    let a = {
        let mut t = TypedTensor::<f64>::zeros(vec![3, 3]);
        *t.get_mut(&[0, 0]) = 1.0;
        *t.get_mut(&[1, 1]) = 2.0;
        *t.get_mut(&[2, 2]) = 3.0;
        Tensor::F64(t)
    };
    let b = {
        let mut t = TypedTensor::<f64>::zeros(vec![2, 2]);
        *t.get_mut(&[0, 0]) = 10.0;
        *t.get_mut(&[1, 1]) = 20.0;
        Tensor::F64(t)
    };

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let tb = TracedTensor::from_tensor_concrete_shape(b);
    let tc = einsum(&mut engine, &[&ta, &tb], "ii,jj->ij").unwrap();
    let result = tc.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[3, 2]);
    let diag_a = [1.0, 2.0, 3.0];
    let diag_b = [10.0, 20.0];
    for i in 0..3 {
        for j in 0..2 {
            assert_close(
                get_v2(&result, &[i, j]),
                diag_a[i] * diag_b[j],
                &format!("ii_jj_ij[{i},{j}]"),
            );
        }
    }
}

#[test]
fn einsum_binary_diag_ii_j_to_j() {
    // "ii,j->j" — trace times vector
    let a = f64_tensor(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
    let b = f64_tensor(vec![2], vec![2.0, 3.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b);
    let tc = einsum(&mut engine, &[&ta, &tb], "ii,j->j").unwrap();
    let result = tc.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[2]);
    let trace = get_v2(&a, &[0, 0]) + get_v2(&a, &[1, 1]) + get_v2(&a, &[2, 2]);
    assert_close(get_v2(&result, &[0]), trace * 2.0, "ii_j_to_j[0]");
    assert_close(get_v2(&result, &[1]), trace * 3.0, "ii_j_to_j[1]");
}

#[test]
fn einsum_input_output_repeated_iij_to_jj() {
    // "iij->jj" — trace over i, then embed diagonal over j
    let data: Vec<f64> = (0..18).map(|x| x as f64).collect();
    let a = f64_tensor(vec![3, 3, 2], data);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let ty = einsum(&mut engine, &[&ta], "iij->jj").unwrap();
    let result = ty.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    for j1 in 0..2 {
        for j2 in 0..2 {
            let expected = if j1 == j2 {
                let mut s = 0.0;
                for i in 0..3 {
                    s += get_v2(&a, &[i, i, j1]);
                }
                s
            } else {
                0.0
            };
            assert_close(
                get_v2(&result, &[j1, j2]),
                expected,
                &format!("iij_to_jj[{j1},{j2}]"),
            );
        }
    }
}

#[test]
fn einsum_input_output_repeated_ii_to_ii() {
    // "ii->ii" — extract diagonal then re-embed
    let data: Vec<f64> = (0..9).map(|x| x as f64).collect();
    let a = f64_tensor(vec![3, 3], data);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let ty = einsum(&mut engine, &[&ta], "ii->ii").unwrap();
    let result = ty.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[3, 3]);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { get_v2(&a, &[i, i]) } else { 0.0 };
            assert_close(
                get_v2(&result, &[i, j]),
                expected,
                &format!("ii_to_ii[{i},{j}]"),
            );
        }
    }
}

#[test]
fn einsum_unit_extent_contraction() {
    // "abcdef,ace->bdf" — contraction with unit-extent dimensions
    let a = f64_tensor(vec![1, 1, 1, 1, 1, 1], vec![2.0]);
    let b = f64_tensor(vec![1, 1, 1], vec![3.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let tb = TracedTensor::from_tensor_concrete_shape(b);
    let tc = einsum(&mut engine, &[&ta, &tb], "abcdef,ace->bdf").unwrap();
    let result = tc.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[1, 1, 1]);
    assert_close(get_v2(&result, &[0, 0, 0]), 6.0, "unit_extent");
}

// ============================================================================
// Group 12: Complex number einsum tests
// ============================================================================

#[test]
fn einsum_complex_diagonal_extraction() {
    // "ii->i" with complex numbers
    let data: Vec<Complex64> = (1..=9)
        .map(|x| Complex64::new(x as f64, -(x as f64)))
        .collect();
    let a = c64_tensor(vec![3, 3], data);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = einsum(&mut engine, &[&ta], "ii->i").unwrap();
    let result = tb.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[3]);
    for i in 0..3 {
        assert_close_c64(
            get_c64(&result, &[i]),
            get_c64(&a, &[i, i]),
            &format!("complex_diag[{i}]"),
        );
    }
}

#[test]
fn einsum_complex_diagonal_embedding() {
    // "i->ii" with complex numbers
    let v = c64_tensor(
        vec![3],
        vec![
            Complex64::new(2.0, 1.0),
            Complex64::new(3.0, -1.0),
            Complex64::new(5.0, 0.5),
        ],
    );

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let tv = TracedTensor::from_tensor_concrete_shape(v.clone());
    let td = einsum(&mut engine, &[&tv], "i->ii").unwrap();
    let result = td.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[3, 3]);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j {
                get_c64(&v, &[i])
            } else {
                Complex64::new(0.0, 0.0)
            };
            assert_close_c64(
                get_c64(&result, &[i, j]),
                expected,
                &format!("complex_embed[{i},{j}]"),
            );
        }
    }
}

#[test]
fn einsum_complex_trace() {
    // "ii->" with complex numbers
    let data: Vec<Complex64> = (1..=4)
        .map(|x| Complex64::new(x as f64, 0.5 * x as f64))
        .collect();
    let a = c64_tensor(vec![2, 2], data);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let ts = einsum(&mut engine, &[&ta], "ii->").unwrap();
    let result = ts.run_with(&mut engine).unwrap();

    assert!(result.shape().is_empty());
    let expected = get_c64(&a, &[0, 0]) + get_c64(&a, &[1, 1]);
    match result {
        Tensor::C64(inner) => {
            assert_close_c64(inner.host_data()[0], expected, "complex_trace");
        }
        _ => panic!("expected C64"),
    }
}
