//! Integration tests for iterative AD with `eval()` mid-loop (issue #654).
//!
//! Verifies that:
//! 1. `eval()` does not break graph connectivity — TracedTensor remains usable
//!    as input to subsequent operations after evaluation.
//! 2. `grad()` produces correct gradients through multiple unrolled iterations,
//!    validated against central finite differences.
//! 3. `run_many_traced_with()` successfully evaluates both primal and gradient in a single
//!    merged execution (graph deduplication via `materialize_merge`).
//! 4. Graph size grows linearly with iteration count.

use tenferro_ad::TracedTensorAdExt;
mod support;
use std::collections::HashSet;
use support::{run_many_traced_with, RunTraced};

use computegraph::graph::Graph;
use tenferro_cpu::CpuBackend;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_runtime::traced::TracedTensor;
use tenferro_runtime::GraphExecutor;
use tenferro_runtime::{Tensor, TypedTensor};

const TOL: f64 = 1e-6;
const FD_H: f64 = 1e-6;

fn f64_scalar(val: f64) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![val]).unwrap())
}

fn get_f64_scalar(t: &Tensor) -> f64 {
    match t {
        Tensor::F64(inner) => inner.host_data().unwrap()[0],
        _ => panic!("expected F64"),
    }
}

/// Count total ops across all graphs in the graph tree, deduplicating
/// by pointer identity (the same `Arc<Graph>` may appear multiple times).
fn total_ops(graph: &Graph<StdTensorOp>) -> usize {
    let mut visited = HashSet::new();
    let mut count = 0;
    count_ops_recursive(graph, &mut visited, &mut count);
    count
}

// Deduplication uses raw pointer identity, which works because Graph
// parents share the same Arc allocation (not cloned data).
fn count_ops_recursive(
    graph: &Graph<StdTensorOp>,
    visited: &mut HashSet<*const Graph<StdTensorOp>>,
    count: &mut usize,
) {
    let ptr: *const Graph<StdTensorOp> = graph;
    if !visited.insert(ptr) {
        return;
    }
    *count += graph.operations().len();
    for parent in graph.parents() {
        count_ops_recursive(parent, visited, count);
    }
}

/// Run the fixed-point iteration `x_{k+1} = a * cos(x_k)` using concrete f64
/// arithmetic. Returns the converged value.
fn fixed_point_concrete(a_val: f64, x0: f64, max_iter: usize, tol: f64) -> f64 {
    let mut x = x0;
    for _ in 0..max_iter {
        let x_new = a_val * x.cos();
        if (x_new - x).abs() < tol {
            return x_new;
        }
        x = x_new;
    }
    x
}

/// Run the fixed-point iteration on the traced graph.
///
/// Returns `(x_final, iteration_count)` where `x_final` is the TracedTensor
/// still connected to the full unrolled computation graph.
fn fixed_point_traced(
    a: &TracedTensor,
    x0: f64,
    max_iter: usize,
    conv_tol: f64,
    engine: &mut GraphExecutor<CpuBackend>,
) -> (TracedTensor, usize) {
    let mut x = TracedTensor::from_tensor_concrete_shape(f64_scalar(x0)).unwrap();
    let mut prev_val = x0;

    for iter in 1..=max_iter {
        // Build graph: x_new = a * cos(x)
        let x_new = (a * &x.cos()).unwrap();

        // Convergence check — eval() must NOT break the graph
        let x_check = x_new.clone();
        let val = get_f64_scalar(&x_check.run_with(engine).unwrap());

        if (val - prev_val).abs() < conv_tol {
            return (x_new, iter);
        }

        prev_val = val;
        x = x_new;
    }
    (x, max_iter)
}

// ─────────────────────────────────────────────────────────────
// Test 1: Graph continuity — eval() does not break TracedTensor
// ─────────────────────────────────────────────────────────────

#[test]
fn eval_mid_loop_preserves_graph_connectivity() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_scalar(0.8)).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());

    let (x_final, iters) = fixed_point_traced(&a, 0.5, 100, 1e-12, &mut engine);
    assert!(iters < 100, "iteration did not converge");

    // x_final is still a TracedTensor connected to `a`.
    // Verify by computing grad — if the graph were broken, this would give 0.
    let grad = x_final.grad(&a).unwrap();
    let grad_clone = grad.clone();
    let grad_val = get_f64_scalar(&grad_clone.run_with(&mut engine).unwrap());
    assert!(
        grad_val.abs() > 1e-10,
        "gradient is near zero — graph connectivity likely broken; grad={grad_val}"
    );
}

// ─────────────────────────────────────────────────────────────
// Test 2: Gradient correctness via finite differences
// ─────────────────────────────────────────────────────────────

#[test]
fn iterative_ad_gradient_matches_finite_diff() {
    let a_val = 0.8_f64;
    let x0 = 0.5_f64;
    let max_iter = 100;
    let conv_tol = 1e-12;

    // --- AD gradient ---
    let a = TracedTensor::from_tensor_concrete_shape(f64_scalar(a_val)).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let (x_final, _) = fixed_point_traced(&a, x0, max_iter, conv_tol, &mut engine);

    let loss = x_final;
    let grad = loss.grad(&a).unwrap();
    let results = run_many_traced_with(&mut engine, &[&loss, &grad]).unwrap();
    let ad_grad = get_f64_scalar(&results[1]);

    // --- Finite difference gradient ---
    let f_plus = fixed_point_concrete(a_val + FD_H, x0, max_iter, conv_tol);
    let f_minus = fixed_point_concrete(a_val - FD_H, x0, max_iter, conv_tol);
    let fd_grad = (f_plus - f_minus) / (2.0 * FD_H);

    let diff = (ad_grad - fd_grad).abs();
    assert!(
        diff < TOL,
        "AD gradient ({ad_grad}) does not match finite diff ({fd_grad}), diff={diff}"
    );
}

// ─────────────────────────────────────────────────────────────
// Test 3: run_many_traced_with evaluates primal + gradient together
// ─────────────────────────────────────────────────────────────

#[test]
fn eval_all_evaluates_primal_and_gradient() {
    let a_val = 0.8_f64;
    let a = TracedTensor::from_tensor_concrete_shape(f64_scalar(a_val)).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());

    let (x_final, _) = fixed_point_traced(&a, 0.5, 100, 1e-12, &mut engine);

    let primal = x_final.clone();
    let grad = x_final.grad(&a).unwrap();

    // run_many_traced_with merges graphs and deduplicates shared subexpressions.
    // If this panics or returns wrong values, deduplication is broken.
    let results = run_many_traced_with(&mut engine, &[&primal, &grad]).unwrap();
    assert_eq!(results.len(), 2);

    let primal_val = get_f64_scalar(&results[0]);
    let grad_val = get_f64_scalar(&results[1]);

    // Primal should match concrete computation
    let expected = fixed_point_concrete(a_val, 0.5, 100, 1e-12);
    assert!(
        (primal_val - expected).abs() < 1e-10,
        "primal mismatch: {primal_val} vs {expected}"
    );

    // Gradient should be nonzero (sanity)
    assert!(
        grad_val.abs() > 1e-10,
        "gradient unexpectedly zero: {grad_val}"
    );
}

// ─────────────────────────────────────────────────────────────
// Test 4: Graph size grows linearly with iteration count
// ─────────────────────────────────────────────────────────────

#[test]
fn graph_size_grows_linearly() {
    // Run with a fixed number of iterations (no convergence check)
    // and measure graph op count.
    fn ops_for_k_iters(k: usize) -> usize {
        let a = TracedTensor::from_tensor_concrete_shape(f64_scalar(0.8)).unwrap();
        let mut x = TracedTensor::from_tensor_concrete_shape(f64_scalar(0.5)).unwrap();
        for _ in 0..k {
            x = (&a * &x.cos()).unwrap();
        }
        total_ops(x.graph())
    }

    let ops_10 = ops_for_k_iters(10);
    let ops_20 = ops_for_k_iters(20);
    let ops_30 = ops_for_k_iters(30);

    // ops_per_iter should be approximately constant
    let per_iter_10_20 = (ops_20 - ops_10) as f64 / 10.0;
    let per_iter_20_30 = (ops_30 - ops_20) as f64 / 10.0;

    assert!(
        per_iter_10_20 > 0.0,
        "ops must grow between 10 and 20 iterations"
    );
    assert!(
        per_iter_20_30 > 0.0,
        "ops must grow between 20 and 30 iterations"
    );

    let ratio = per_iter_10_20 / per_iter_20_30;
    assert!(
        (0.8..=1.2).contains(&ratio),
        "growth is not linear: ops(10)={ops_10}, ops(20)={ops_20}, ops(30)={ops_30}, \
         per_iter ratio={ratio}"
    );
}
