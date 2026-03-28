//! Integration tests for einsum with tropical algebras.
//!
//! These tests verify that the fallback contraction path
//! (permute view + MakeContiguous + BatchedGemm) works correctly when the
//! Contract extension is unavailable.

use tenferro_einsum::einsum;
use tenferro_ext_tropical::{
    MaxMul, MaxMulAlgebra, MaxPlus, MaxPlusAlgebra, MinPlus, MinPlusAlgebra,
};
use tenferro_prims::CpuBackend;
use tenferro_tensor::{MemoryOrder, Tensor};

const COL: MemoryOrder = MemoryOrder::ColumnMajor;

/// Helper: create a CpuContext for tropical einsum calls.
fn ctx() -> tenferro_prims::CpuContext {
    tenferro_prims::CpuContext::new(1)
}

/// Helper: read element at multi-index from a tensor (stride-aware).
fn get<T: tenferro_algebra::Scalar>(t: &Tensor<T>, idx: &[usize]) -> T {
    let data = t.buffer().as_slice().unwrap();
    let pos = t.offset()
        + idx
            .iter()
            .zip(t.strides())
            .map(|(&i, &s)| i as isize * s)
            .sum::<isize>();
    data[pos as usize]
}

// ============================================================================
// MaxPlus matmul: C[i,k] = max_j(A[i,j] + B[j,k])
// ============================================================================

#[test]
fn tropical_maxplus_matmul() {
    let mut ctx = ctx();

    // A = [[1, 3],
    //      [2, 4]]  (2x2, column-major: data = [1,2,3,4])
    let a = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(1.0), MaxPlus(2.0), MaxPlus(3.0), MaxPlus(4.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    // B = [[5, 7],
    //      [6, 8]]  (2x2, column-major: data = [5,6,7,8])
    let b = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(5.0), MaxPlus(6.0), MaxPlus(7.0), MaxPlus(8.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    // C[i,k] = max_j(A[i,j] + B[j,k])
    // C[0,0] = max(1+5, 3+6) = max(6, 9) = 9
    // C[1,0] = max(2+5, 4+6) = max(7, 10) = 10
    // C[0,1] = max(1+7, 3+8) = max(8, 11) = 11
    // C[1,1] = max(2+7, 4+8) = max(9, 12) = 12
    let c =
        einsum::<MaxPlusAlgebra<f64>, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();

    assert_eq!(c.dims(), &[2, 2]);
    let data = c.buffer().as_slice().unwrap();
    // Column-major: [C[0,0], C[1,0], C[0,1], C[1,1]]
    assert_eq!(data[0], MaxPlus(9.0));
    assert_eq!(data[1], MaxPlus(10.0));
    assert_eq!(data[2], MaxPlus(11.0));
    assert_eq!(data[3], MaxPlus(12.0));
}

// ============================================================================
// Trace: "ii->" reduces to max of diagonal elements
// ============================================================================

#[test]
fn tropical_maxplus_trace() {
    let mut ctx = ctx();

    // A = [[1, 3],
    //      [2, 4]]  column-major
    let a = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(1.0), MaxPlus(2.0), MaxPlus(3.0), MaxPlus(4.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    // trace = A[0,0] ⊕ A[1,1] = max(1, 4) = 4
    let tr = einsum::<MaxPlusAlgebra<f64>, CpuBackend>(&mut ctx, "ii->", &[&a], None).unwrap();

    assert_eq!(tr.dims(), &[] as &[usize]);
    let data = tr.buffer().as_slice().unwrap();
    assert_eq!(data[0], MaxPlus(4.0));
}

// ============================================================================
// Batched matmul: "bij,bjk->bik"
// ============================================================================

#[test]
fn tropical_maxplus_batch_matmul() {
    let mut ctx = ctx();

    // batch=2, m=2, k=2 → A is [2,2,2]
    // Batch 0: [[1,3],[2,4]], Batch 1: [[5,7],[6,8]]
    // Column-major: innermost dims vary first
    // Shape [m=2, k=2, b=2] → wait, "bij" means dims [b, i, j]
    // Let's use shape [b=2, i=2, j=2]
    let a = Tensor::<MaxPlus<f64>>::from_slice(
        &[
            // b=0,i=0,j=0; b=1,i=0,j=0; b=0,i=1,j=0; b=1,i=1,j=0;
            // b=0,i=0,j=1; b=1,i=0,j=1; b=0,i=1,j=1; b=1,i=1,j=1
            MaxPlus(1.0),
            MaxPlus(5.0),
            MaxPlus(2.0),
            MaxPlus(6.0),
            MaxPlus(3.0),
            MaxPlus(7.0),
            MaxPlus(4.0),
            MaxPlus(8.0),
        ],
        &[2, 2, 2],
        COL,
    )
    .unwrap();

    let b = Tensor::<MaxPlus<f64>>::from_slice(
        &[
            MaxPlus(0.5),
            MaxPlus(1.5),
            MaxPlus(0.5),
            MaxPlus(1.5),
            MaxPlus(1.0),
            MaxPlus(2.0),
            MaxPlus(1.0),
            MaxPlus(2.0),
        ],
        &[2, 2, 2],
        COL,
    )
    .unwrap();

    let c = einsum::<MaxPlusAlgebra<f64>, CpuBackend>(&mut ctx, "bij,bjk->bik", &[&a, &b], None)
        .unwrap();

    assert_eq!(c.dims(), &[2, 2, 2]);

    // Verify batch 0:
    // A0 = [[1,3],[2,4]], B0 = [[0.5,1.0],[0.5,1.0]]
    // C0[0,0] = max(1+0.5, 3+0.5) = max(1.5, 3.5) = 3.5
    // C0[1,0] = max(2+0.5, 4+0.5) = max(2.5, 4.5) = 4.5
    // C0[0,1] = max(1+1.0, 3+1.0) = max(2.0, 4.0) = 4.0
    // C0[1,1] = max(2+1.0, 4+1.0) = max(3.0, 5.0) = 5.0
    // Use stride-aware indexing (output may be a lazy-permuted view)
    assert_eq!(get(&c, &[0, 0, 0]), MaxPlus(3.5)); // b=0,i=0,k=0
    assert_eq!(get(&c, &[0, 1, 0]), MaxPlus(4.5)); // b=0,i=1,k=0
    assert_eq!(get(&c, &[0, 0, 1]), MaxPlus(4.0)); // b=0,i=0,k=1
    assert_eq!(get(&c, &[0, 1, 1]), MaxPlus(5.0)); // b=0,i=1,k=1
}

// ============================================================================
// Element-wise: "ij,ij->ij" (degenerate BatchedGemm with m=n=k=1)
// ============================================================================

#[test]
fn tropical_maxplus_elementwise() {
    let mut ctx = ctx();

    let a = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(1.0), MaxPlus(2.0), MaxPlus(3.0), MaxPlus(4.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    let b = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(10.0), MaxPlus(20.0), MaxPlus(30.0), MaxPlus(40.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    // Element-wise tropical mul: C[i,j] = A[i,j] ⊗ B[i,j] = A[i,j] + B[i,j]
    let c =
        einsum::<MaxPlusAlgebra<f64>, CpuBackend>(&mut ctx, "ij,ij->ij", &[&a, &b], None).unwrap();

    assert_eq!(c.dims(), &[2, 2]);
    let data = c.buffer().as_slice().unwrap();
    assert_eq!(data[0], MaxPlus(11.0));
    assert_eq!(data[1], MaxPlus(22.0));
    assert_eq!(data[2], MaxPlus(33.0));
    assert_eq!(data[3], MaxPlus(44.0));
}

// ============================================================================
// Outer product: "i,j->ij"
// ============================================================================

#[test]
fn tropical_maxplus_outer_product() {
    let mut ctx = ctx();

    let a = Tensor::<MaxPlus<f64>>::from_slice(&[MaxPlus(1.0), MaxPlus(2.0)], &[2], COL).unwrap();

    let b = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(10.0), MaxPlus(20.0), MaxPlus(30.0)],
        &[3],
        COL,
    )
    .unwrap();

    // C[i,j] = A[i] ⊗ B[j] = A[i] + B[j]  (no sum modes, k=1)
    let c =
        einsum::<MaxPlusAlgebra<f64>, CpuBackend>(&mut ctx, "i,j->ij", &[&a, &b], None).unwrap();

    assert_eq!(c.dims(), &[2, 3]);
    let data = c.buffer().as_slice().unwrap();
    // Column-major: [C[0,0], C[1,0], C[0,1], C[1,1], C[0,2], C[1,2]]
    assert_eq!(data[0], MaxPlus(11.0)); // 1+10
    assert_eq!(data[1], MaxPlus(12.0)); // 2+10
    assert_eq!(data[2], MaxPlus(21.0)); // 1+20
    assert_eq!(data[3], MaxPlus(22.0)); // 2+20
    assert_eq!(data[4], MaxPlus(31.0)); // 1+30
    assert_eq!(data[5], MaxPlus(32.0)); // 2+30
}

// ============================================================================
// Vector-matrix: "j,jk->k" (m=1)
// ============================================================================

#[test]
fn tropical_maxplus_vecmat() {
    let mut ctx = ctx();

    let v = Tensor::<MaxPlus<f64>>::from_slice(&[MaxPlus(1.0), MaxPlus(3.0)], &[2], COL).unwrap();

    // M = [[5, 7],
    //      [6, 8]]  column-major
    let m = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(5.0), MaxPlus(6.0), MaxPlus(7.0), MaxPlus(8.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    // C[k] = max_j(v[j] + M[j,k])
    // C[0] = max(1+5, 3+6) = max(6, 9) = 9
    // C[1] = max(1+7, 3+8) = max(8, 11) = 11
    let c =
        einsum::<MaxPlusAlgebra<f64>, CpuBackend>(&mut ctx, "j,jk->k", &[&v, &m], None).unwrap();

    assert_eq!(c.dims(), &[2]);
    let data = c.buffer().as_slice().unwrap();
    assert_eq!(data[0], MaxPlus(9.0));
    assert_eq!(data[1], MaxPlus(11.0));
}

// ============================================================================
// Full contraction (scalar result): "ij,ij->"
// ============================================================================

#[test]
fn tropical_maxplus_full_contraction() {
    let mut ctx = ctx();

    let a = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(1.0), MaxPlus(2.0), MaxPlus(3.0), MaxPlus(4.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    let b = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(10.0), MaxPlus(20.0), MaxPlus(30.0), MaxPlus(40.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    // scalar = max_{i,j}(A[i,j] + B[i,j])
    // = max(1+10, 2+20, 3+30, 4+40) = max(11, 22, 33, 44) = 44
    let c =
        einsum::<MaxPlusAlgebra<f64>, CpuBackend>(&mut ctx, "ij,ij->", &[&a, &b], None).unwrap();

    assert_eq!(c.dims(), &[] as &[usize]);
    let data = c.buffer().as_slice().unwrap();
    assert_eq!(data[0], MaxPlus(44.0));
}

// ============================================================================
// Three-tensor chain: "ij,jk,kl->il"
// ============================================================================

#[test]
fn tropical_maxplus_three_chain() {
    let mut ctx = ctx();

    let a = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(1.0), MaxPlus(2.0), MaxPlus(3.0), MaxPlus(4.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    let b = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(1.0), MaxPlus(1.0), MaxPlus(1.0), MaxPlus(1.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    let c_mat = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(0.0), MaxPlus(0.0), MaxPlus(0.0), MaxPlus(0.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    // D = A ⊗ B ⊗ C  (tropical chain multiplication)
    let d = einsum::<MaxPlusAlgebra<f64>, CpuBackend>(
        &mut ctx,
        "ij,jk,kl->il",
        &[&a, &b, &c_mat],
        None,
    )
    .unwrap();

    assert_eq!(d.dims(), &[2, 2]);

    // AB[i,k] = max_j(A[i,j] + B[j,k])
    // AB[0,0] = max(1+1, 3+1) = 4
    // AB[1,0] = max(2+1, 4+1) = 5
    // AB[0,1] = max(1+1, 3+1) = 4
    // AB[1,1] = max(2+1, 4+1) = 5
    //
    // D[i,l] = max_k(AB[i,k] + C[k,l])
    // D[0,0] = max(4+0, 4+0) = 4
    // D[1,0] = max(5+0, 5+0) = 5
    // D[0,1] = max(4+0, 4+0) = 4
    // D[1,1] = max(5+0, 5+0) = 5
    // Compare logical elements (stride-aware, layout-independent)
    assert_eq!(get(&d, &[0, 0]), MaxPlus(4.0));
    assert_eq!(get(&d, &[1, 0]), MaxPlus(5.0));
    assert_eq!(get(&d, &[0, 1]), MaxPlus(4.0));
    assert_eq!(get(&d, &[1, 1]), MaxPlus(5.0));
}

// ============================================================================
// MinPlus matmul: shortest-path distance via adjacency matrix squaring
// C[i,k] = min_j(A[i,j] + B[j,k])
// ============================================================================

#[test]
fn tropical_minplus_matmul_shortest_path() {
    let mut ctx = ctx();
    let inf = f64::INFINITY;

    // Weighted adjacency matrix W (3 nodes, edge weights, +inf = no edge):
    //      to 0   to 1   to 2
    // W = [[ 0,    1,   +inf],   from 0
    //      [+inf,  0,    3  ],   from 1
    //      [ 2,  +inf,   0  ]]   from 2
    //
    // Column-major 3x3: data[i + 3*j] = W[i,j]
    let w = Tensor::<MinPlus<f64>>::from_slice(
        &[
            MinPlus(0.0),
            MinPlus(inf),
            MinPlus(2.0), // col 0
            MinPlus(1.0),
            MinPlus(0.0),
            MinPlus(inf), // col 1
            MinPlus(inf),
            MinPlus(3.0),
            MinPlus(0.0), // col 2
        ],
        &[3, 3],
        COL,
    )
    .unwrap();

    // W^2[i,k] = min_j(W[i,j] + W[j,k]) = 2-hop shortest paths
    // W^2[0,0] = min(0+0, 1+inf, inf+2) = 0
    // W^2[0,1] = min(0+1, 1+0, inf+inf) = 1
    // W^2[0,2] = min(0+inf, 1+3, inf+0) = 4
    // W^2[1,0] = min(inf+0, 0+inf, 3+2) = 5
    // W^2[1,1] = min(inf+1, 0+0, 3+inf) = 0
    // W^2[1,2] = min(inf+inf, 0+3, 3+0) = 3
    // W^2[2,0] = min(2+0, inf+inf, 0+2) = 2
    // W^2[2,1] = min(2+1, inf+0, 0+inf) = 3
    // W^2[2,2] = min(2+inf, inf+3, 0+0) = 0
    let w2 =
        einsum::<MinPlusAlgebra<f64>, CpuBackend>(&mut ctx, "ij,jk->ik", &[&w, &w], None).unwrap();

    assert_eq!(w2.dims(), &[3, 3]);
    let data = w2.buffer().as_slice().unwrap();
    // Column-major: data[i + 3*j]
    assert_eq!(data[0], MinPlus(0.0)); // W^2[0,0]
    assert_eq!(data[1], MinPlus(5.0)); // W^2[1,0]
    assert_eq!(data[2], MinPlus(2.0)); // W^2[2,0]
    assert_eq!(data[3], MinPlus(1.0)); // W^2[0,1]
    assert_eq!(data[4], MinPlus(0.0)); // W^2[1,1]
    assert_eq!(data[5], MinPlus(3.0)); // W^2[2,1]
    assert_eq!(data[6], MinPlus(4.0)); // W^2[0,2]
    assert_eq!(data[7], MinPlus(3.0)); // W^2[1,2]
    assert_eq!(data[8], MinPlus(0.0)); // W^2[2,2]
}

// ============================================================================
// MaxMul matmul: Viterbi-like max-probability path computation
// C[i,k] = max_j(A[i,j] * B[j,k])
// ============================================================================

#[test]
fn tropical_maxmul_matmul_viterbi() {
    let mut ctx = ctx();

    // Transition matrix T (3 states):
    //      to 0   to 1   to 2
    // T = [[0.1,  0.7,   0.2],   from 0
    //      [0.3,  0.4,   0.3],   from 1
    //      [0.5,  0.1,   0.4]]   from 2
    //
    // Emission probability matrix E (3 states, 2 observations):
    //       obs 0  obs 1
    // E = [[0.9,   0.1],   state 0
    //      [0.2,   0.8],   state 1
    //      [0.5,   0.5]]   state 2
    //
    // MaxMul(T^T * E) gives max transition-to-emission probabilities.
    // P[i,k] = max_j(T[j,i] * E[j,k])  (T transposed so we look at incoming)

    // T^T in column-major 3x3
    let tt = Tensor::<MaxMul<f64>>::from_slice(
        &[
            MaxMul(0.1),
            MaxMul(0.7),
            MaxMul(0.2), // col 0 = row 0 of T
            MaxMul(0.3),
            MaxMul(0.4),
            MaxMul(0.3), // col 1 = row 1 of T
            MaxMul(0.5),
            MaxMul(0.1),
            MaxMul(0.4), // col 2 = row 2 of T
        ],
        &[3, 3],
        COL,
    )
    .unwrap();

    // E in column-major 3x2
    let e = Tensor::<MaxMul<f64>>::from_slice(
        &[
            MaxMul(0.9),
            MaxMul(0.2),
            MaxMul(0.5), // col 0
            MaxMul(0.1),
            MaxMul(0.8),
            MaxMul(0.5), // col 1
        ],
        &[3, 2],
        COL,
    )
    .unwrap();

    // P[i,k] = max_j(T^T[i,j] * E[j,k])
    // P[0,0] = max(0.1*0.9, 0.3*0.2, 0.5*0.5) = max(0.09, 0.06, 0.25) = 0.25
    // P[1,0] = max(0.7*0.9, 0.4*0.2, 0.1*0.5) = max(0.63, 0.08, 0.05) = 0.63
    // P[2,0] = max(0.2*0.9, 0.3*0.2, 0.4*0.5) = max(0.18, 0.06, 0.20) = 0.20
    // P[0,1] = max(0.1*0.1, 0.3*0.8, 0.5*0.5) = max(0.01, 0.24, 0.25) = 0.25
    // P[1,1] = max(0.7*0.1, 0.4*0.8, 0.1*0.5) = max(0.07, 0.32, 0.05) = 0.32
    // P[2,1] = max(0.2*0.1, 0.3*0.8, 0.4*0.5) = max(0.02, 0.24, 0.20) = 0.24
    let p =
        einsum::<MaxMulAlgebra<f64>, CpuBackend>(&mut ctx, "ij,jk->ik", &[&tt, &e], None).unwrap();

    assert_eq!(p.dims(), &[3, 2]);
    let data = p.buffer().as_slice().unwrap();
    let eps = 1e-14;
    assert!((data[0].0 - 0.25).abs() < eps); // P[0,0]
    assert!((data[1].0 - 0.63).abs() < eps); // P[1,0]
    assert!((data[2].0 - 0.20).abs() < eps); // P[2,0]
    assert!((data[3].0 - 0.25).abs() < eps); // P[0,1]
    assert!((data[4].0 - 0.32).abs() < eps); // P[1,1]
    assert!((data[5].0 - 0.24).abs() < eps); // P[2,1]
}

// ============================================================================
// MinPlus trace: minimum of diagonal elements (self-loop distances)
// ============================================================================

#[test]
fn tropical_minplus_trace() {
    let mut ctx = ctx();
    let inf = f64::INFINITY;

    // Distance matrix D (diagonal = self-loop costs):
    // D = [[2,  5 ],
    //      [inf, 3 ]]
    let d = Tensor::<MinPlus<f64>>::from_slice(
        &[MinPlus(2.0), MinPlus(inf), MinPlus(5.0), MinPlus(3.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    // trace = D[0,0] ⊕ D[1,1] = min(2, 3) = 2
    let tr = einsum::<MinPlusAlgebra<f64>, CpuBackend>(&mut ctx, "ii->", &[&d], None).unwrap();

    assert_eq!(tr.dims(), &[] as &[usize]);
    let data = tr.buffer().as_slice().unwrap();
    assert_eq!(data[0], MinPlus(2.0));
}
