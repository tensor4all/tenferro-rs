//! Tests for tropical automatic differentiation (backward pass with argmax routing).
//!
//! These tests verify that gradients are correctly routed through tropical
//! operations to only the winning elements.

use tenferro_device::Error;
use tenferro_ext_tropical::ad::{
    extract_inner, promote_to_tropical, tropical_einsum_rrule, TropicalScalar,
};
use tenferro_ext_tropical::{
    MaxMul, MaxMulAlgebra, MaxPlus, MaxPlusAlgebra, MinPlus, MinPlusAlgebra,
};
use tenferro_tensor::{MemoryOrder, Tensor};
use tidu::expert::Tape;

const COL: MemoryOrder = MemoryOrder::ColumnMajor;

fn ctx() -> tenferro_prims::CpuContext {
    tenferro_prims::CpuContext::new(1)
}

// ============================================================================
// Promote / extract roundtrip tests
// ============================================================================

#[test]
fn promote_extract_maxplus_roundtrip() {
    let t = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let tropical = promote_to_tropical::<MaxPlus<f64>>(&t).unwrap();
    let back = extract_inner::<MaxPlus<f64>>(&tropical).unwrap();
    let orig = t.buffer().as_slice().unwrap();
    let result = back.buffer().as_slice().unwrap();
    for i in 0..4 {
        assert_eq!(orig[i], result[i]);
    }
}

#[test]
fn promote_extract_minplus_roundtrip() {
    let t = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let tropical = promote_to_tropical::<MinPlus<f64>>(&t).unwrap();
    let back = extract_inner::<MinPlus<f64>>(&tropical).unwrap();
    let orig = t.buffer().as_slice().unwrap();
    let result = back.buffer().as_slice().unwrap();
    for i in 0..4 {
        assert_eq!(orig[i], result[i]);
    }
}

#[test]
fn promote_extract_maxmul_roundtrip() {
    let t = Tensor::<f64>::from_slice(&[0.1, 0.2, 0.3, 0.4], &[2, 2], COL).unwrap();
    let tropical = promote_to_tropical::<MaxMul<f64>>(&t).unwrap();
    let back = extract_inner::<MaxMul<f64>>(&tropical).unwrap();
    let orig = t.buffer().as_slice().unwrap();
    let result = back.buffer().as_slice().unwrap();
    for i in 0..4 {
        assert_eq!(orig[i], result[i]);
    }
}

#[test]
fn promote_extract_roundtrip_preserves_non_contiguous_view_order() {
    let base = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let view = base.permute(&[1, 0]).unwrap();
    let tropical = promote_to_tropical::<MaxPlus<f64>>(&view).unwrap();
    let back = extract_inner::<MaxPlus<f64>>(&tropical).unwrap();
    let contiguous = view.contiguous(MemoryOrder::ColumnMajor);
    let expected = contiguous.buffer().as_slice().unwrap();
    assert_eq!(back.buffer().as_slice().unwrap(), expected);
}

#[test]
fn extract_inner_preserves_non_contiguous_tropical_view_order() {
    let base = Tensor::<MaxPlus<f64>>::from_slice(
        &[
            MaxPlus(1.0),
            MaxPlus(2.0),
            MaxPlus(3.0),
            MaxPlus(4.0),
            MaxPlus(5.0),
            MaxPlus(6.0),
        ],
        &[2, 3],
        COL,
    )
    .unwrap();
    let view = base.permute(&[1, 0]).unwrap();
    let inner = extract_inner::<MaxPlus<f64>>(&view).unwrap();
    let expected = view
        .contiguous(MemoryOrder::ColumnMajor)
        .buffer()
        .as_slice()
        .unwrap()
        .iter()
        .map(|v| v.inner())
        .collect::<Vec<_>>();
    assert_eq!(inner.buffer().as_slice().unwrap(), expected);
}

// ============================================================================
// MaxPlus matmul backward: "ij,jk->ik"
// Gradient routes only to the winning k for each (i,k) output element.
// For MaxPlus (mul = +), backward through + gives dA = dC, dB = dC at winner.
// ============================================================================

#[test]
fn maxplus_matmul_backward_routes_to_winner() {
    let mut ctx = ctx();

    // A = [[1, 3],    (column-major: [1, 2, 3, 4])
    //      [2, 4]]
    let a = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(1.0), MaxPlus(2.0), MaxPlus(3.0), MaxPlus(4.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    // B = [[5, 7],    (column-major: [5, 6, 7, 8])
    //      [6, 8]]
    let b = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(5.0), MaxPlus(6.0), MaxPlus(7.0), MaxPlus(8.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    // C[i,k] = max_j(A[i,j] + B[j,k])
    // C[0,0] = max(1+5, 3+6) = max(6, 9) = 9    → winner j=1
    // C[1,0] = max(2+5, 4+6) = max(7, 10) = 10   → winner j=1
    // C[0,1] = max(1+7, 3+8) = max(8, 11) = 11   → winner j=1
    // C[1,1] = max(2+7, 4+8) = max(9, 12) = 12   → winner j=1

    // Cotangent: all ones
    let grad_c = Tensor::<f64>::from_slice(&[1.0, 1.0, 1.0, 1.0], &[2, 2], COL).unwrap();

    let grads = tropical_einsum_rrule::<
        MaxPlus<f64>,
        MaxPlusAlgebra<f64>,
        tenferro_prims::CpuBackend,
    >(&mut ctx, "ij,jk->ik", &[&a, &b], &grad_c)
    .unwrap();

    assert_eq!(grads.len(), 2);

    let da = grads[0].buffer().as_slice().unwrap();
    let db = grads[1].buffer().as_slice().unwrap();

    // For MaxPlus: dA[i,j*] += dC[i,k] for each k where j* won
    // All winners are j=1, so:
    // dA[0,0] = 0 (j=0 never won for i=0)
    // dA[1,0] = 0 (j=0 never won for i=1)
    // dA[0,1] = dC[0,0] + dC[0,1] = 1 + 1 = 2  (j=1 won for i=0, both k=0 and k=1)
    // dA[1,1] = dC[1,0] + dC[1,1] = 1 + 1 = 2  (j=1 won for i=1, both k=0 and k=1)
    // Column-major [i + 2*j]: da[0]=dA[0,0], da[1]=dA[1,0], da[2]=dA[0,1], da[3]=dA[1,1]
    assert_eq!(da[0], 0.0); // dA[0,0]
    assert_eq!(da[1], 0.0); // dA[1,0]
    assert_eq!(da[2], 2.0); // dA[0,1]
    assert_eq!(da[3], 2.0); // dA[1,1]

    // dB[j*,k] += dC[i,k] for each i where j* won
    // All winners are j=1, so:
    // dB[0,0] = 0 (j=0 never won)
    // dB[0,1] = 0 (j=0 never won)
    // dB[1,0] = dC[0,0] + dC[1,0] = 1 + 1 = 2  (j=1 won for both i=0,i=1 at k=0)
    // dB[1,1] = dC[0,1] + dC[1,1] = 1 + 1 = 2  (j=1 won for both i=0,i=1 at k=1)
    // Column-major [j + 2*k]: db[0]=dB[0,0], db[1]=dB[1,0], db[2]=dB[0,1], db[3]=dB[1,1]
    assert_eq!(db[0], 0.0); // dB[0,0]
    assert_eq!(db[1], 2.0); // dB[1,0]
    assert_eq!(db[2], 0.0); // dB[0,1]
    assert_eq!(db[3], 2.0); // dB[1,1]
}

#[test]
fn maxplus_matmul_backward_mixed_winners() {
    let mut ctx = ctx();

    // Design a case where different output elements have different winners
    // A = [[10, 1],    (column-major: [10, 0, 1, 5])
    //      [0,  5]]
    let a = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(10.0), MaxPlus(0.0), MaxPlus(1.0), MaxPlus(5.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    // B = [[1, 0],    (column-major: [1, 10, 0, 1])
    //      [10, 1]]
    let b = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(1.0), MaxPlus(10.0), MaxPlus(0.0), MaxPlus(1.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    // C[i,k] = max_j(A[i,j] + B[j,k])
    // C[0,0] = max(10+1, 1+10) = max(11, 11) = 11  → winner j=0 (smallest index wins tie)
    // C[1,0] = max(0+1, 5+10) = max(1, 15) = 15    → winner j=1
    // C[0,1] = max(10+0, 1+1) = max(10, 2) = 10    → winner j=0
    // C[1,1] = max(0+0, 5+1) = max(0, 6) = 6       → winner j=1

    let grad_c = Tensor::<f64>::from_slice(&[1.0, 1.0, 1.0, 1.0], &[2, 2], COL).unwrap();

    let grads = tropical_einsum_rrule::<
        MaxPlus<f64>,
        MaxPlusAlgebra<f64>,
        tenferro_prims::CpuBackend,
    >(&mut ctx, "ij,jk->ik", &[&a, &b], &grad_c)
    .unwrap();

    let da = grads[0].buffer().as_slice().unwrap();
    let db = grads[1].buffer().as_slice().unwrap();

    // Winners: C[0,0]→j=0, C[1,0]→j=1, C[0,1]→j=0, C[1,1]→j=1
    // dA[i,j] gets dC[i,k] when j was the winner for (i,k)
    // dA[0,0] = dC[0,0] + dC[0,1] = 2  (j=0 won for both outputs in row 0)
    // dA[1,0] = 0             (j=0 never won for i=1)
    // dA[0,1] = 0             (j=1 never won for i=0)
    // dA[1,1] = dC[1,0] + dC[1,1] = 2  (j=1 won for (1,0) and (1,1))
    assert_eq!(da[0], 2.0); // dA[0,0]
    assert_eq!(da[1], 0.0); // dA[1,0]
    assert_eq!(da[2], 0.0); // dA[0,1]
    assert_eq!(da[3], 2.0); // dA[1,1]

    // dB[j,k] gets dC[i,k] when j was the winner for (i,k)
    // dB[0,0] = dC[0,0] = 1  (j=0 won for (0,0))
    // dB[1,0] = dC[1,0] = 1  (j=1 won for (1,0))
    // dB[0,1] = dC[0,1] = 1  (j=0 won for (0,1))
    // dB[1,1] = dC[1,1] = 1  (j=1 won for (1,1))
    assert_eq!(db[0], 1.0); // dB[0,0]
    assert_eq!(db[1], 1.0); // dB[1,0]
    assert_eq!(db[2], 1.0); // dB[0,1]
    assert_eq!(db[3], 1.0); // dB[1,1]
}

// ============================================================================
// MinPlus matmul backward: "ij,jk->ik"
// Same structure as MaxPlus but winner is argmin instead of argmax.
// ============================================================================

#[test]
fn minplus_matmul_backward_routes_to_argmin() {
    let mut ctx = ctx();

    // A = [[1, 3],    (column-major: [1, 2, 3, 4])
    //      [2, 4]]
    let a = Tensor::<MinPlus<f64>>::from_slice(
        &[MinPlus(1.0), MinPlus(2.0), MinPlus(3.0), MinPlus(4.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    // B = [[5, 7],    (column-major: [5, 6, 7, 8])
    //      [6, 8]]
    let b = Tensor::<MinPlus<f64>>::from_slice(
        &[MinPlus(5.0), MinPlus(6.0), MinPlus(7.0), MinPlus(8.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    // C[i,k] = min_j(A[i,j] + B[j,k])
    // C[0,0] = min(1+5, 3+6) = min(6, 9) = 6     → winner j=0
    // C[1,0] = min(2+5, 4+6) = min(7, 10) = 7    → winner j=0
    // C[0,1] = min(1+7, 3+8) = min(8, 11) = 8    → winner j=0
    // C[1,1] = min(2+7, 4+8) = min(9, 12) = 9    → winner j=0

    let grad_c = Tensor::<f64>::from_slice(&[1.0, 1.0, 1.0, 1.0], &[2, 2], COL).unwrap();

    let grads = tropical_einsum_rrule::<
        MinPlus<f64>,
        MinPlusAlgebra<f64>,
        tenferro_prims::CpuBackend,
    >(&mut ctx, "ij,jk->ik", &[&a, &b], &grad_c)
    .unwrap();

    let da = grads[0].buffer().as_slice().unwrap();
    let db = grads[1].buffer().as_slice().unwrap();

    // All winners are j=0 (argmin), so:
    // dA[0,0] = dC[0,0] + dC[0,1] = 2  (j=0 won for i=0, both k)
    // dA[1,0] = dC[1,0] + dC[1,1] = 2  (j=0 won for i=1, both k)
    // dA[0,1] = 0 (j=1 never won)
    // dA[1,1] = 0 (j=1 never won)
    assert_eq!(da[0], 2.0); // dA[0,0]
    assert_eq!(da[1], 2.0); // dA[1,0]
    assert_eq!(da[2], 0.0); // dA[0,1]
    assert_eq!(da[3], 0.0); // dA[1,1]

    // dB[0,0] = dC[0,0] + dC[1,0] = 2
    // dB[1,0] = 0
    // dB[0,1] = dC[0,1] + dC[1,1] = 2
    // dB[1,1] = 0
    assert_eq!(db[0], 2.0); // dB[0,0]
    assert_eq!(db[1], 0.0); // dB[1,0]
    assert_eq!(db[2], 2.0); // dB[0,1]
    assert_eq!(db[3], 0.0); // dB[1,1]
}

// ============================================================================
// MaxMul matmul backward: "ij,jk->ik"
// For MaxMul (mul = *), backward through * uses product rule:
// dA[i,k*] = dC[i,j] * B[k*,j], dB[k*,j] = dC[i,j] * A[i,k*]
// ============================================================================

#[test]
fn maxmul_matmul_backward_product_rule() {
    let mut ctx = ctx();

    // A = [[0.3, 0.7],    (column-major: [0.3, 0.1, 0.7, 0.9])
    //      [0.1, 0.9]]
    let a = Tensor::<MaxMul<f64>>::from_slice(
        &[MaxMul(0.3), MaxMul(0.1), MaxMul(0.7), MaxMul(0.9)],
        &[2, 2],
        COL,
    )
    .unwrap();

    // B = [[0.5, 0.2],    (column-major: [0.5, 0.8, 0.2, 0.6])
    //      [0.8, 0.6]]
    let b = Tensor::<MaxMul<f64>>::from_slice(
        &[MaxMul(0.5), MaxMul(0.8), MaxMul(0.2), MaxMul(0.6)],
        &[2, 2],
        COL,
    )
    .unwrap();

    // C[i,k] = max_j(A[i,j] * B[j,k])
    // C[0,0] = max(0.3*0.5, 0.7*0.8) = max(0.15, 0.56) = 0.56  → winner j=1
    // C[1,0] = max(0.1*0.5, 0.9*0.8) = max(0.05, 0.72) = 0.72  → winner j=1
    // C[0,1] = max(0.3*0.2, 0.7*0.6) = max(0.06, 0.42) = 0.42  → winner j=1
    // C[1,1] = max(0.1*0.2, 0.9*0.6) = max(0.02, 0.54) = 0.54  → winner j=1

    let grad_c = Tensor::<f64>::from_slice(&[1.0, 1.0, 1.0, 1.0], &[2, 2], COL).unwrap();

    let grads =
        tropical_einsum_rrule::<MaxMul<f64>, MaxMulAlgebra<f64>, tenferro_prims::CpuBackend>(
            &mut ctx,
            "ij,jk->ik",
            &[&a, &b],
            &grad_c,
        )
        .unwrap();

    let da = grads[0].buffer().as_slice().unwrap();
    let db = grads[1].buffer().as_slice().unwrap();

    let eps = 1e-12;

    // All winners are j=1.
    // dA[i,j*] += dC[i,k] * B[j*,k]
    // dA[0,0] = 0 (j=0 never won)
    // dA[1,0] = 0 (j=0 never won)
    // dA[0,1] = dC[0,0]*B[1,0] + dC[0,1]*B[1,1] = 1*0.8 + 1*0.6 = 1.4
    // dA[1,1] = dC[1,0]*B[1,0] + dC[1,1]*B[1,1] = 1*0.8 + 1*0.6 = 1.4
    assert!((da[0] - 0.0).abs() < eps); // dA[0,0]
    assert!((da[1] - 0.0).abs() < eps); // dA[1,0]
    assert!((da[2] - 1.4).abs() < eps); // dA[0,1]
    assert!((da[3] - 1.4).abs() < eps); // dA[1,1]

    // dB[j*,k] += dC[i,k] * A[i,j*]
    // dB[0,0] = 0 (j=0 never won)
    // dB[1,0] = dC[0,0]*A[0,1] + dC[1,0]*A[1,1] = 1*0.7 + 1*0.9 = 1.6
    // dB[0,1] = 0 (j=0 never won)
    // dB[1,1] = dC[0,1]*A[0,1] + dC[1,1]*A[1,1] = 1*0.7 + 1*0.9 = 1.6
    assert!((db[0] - 0.0).abs() < eps); // dB[0,0]
    assert!((db[1] - 1.6).abs() < eps); // dB[1,0]
    assert!((db[2] - 0.0).abs() < eps); // dB[0,1]
    assert!((db[3] - 1.6).abs() < eps); // dB[1,1]
}

// ============================================================================
// Tape-based backward tests via tracked_tropical_einsum
// ============================================================================

#[test]
fn tracked_maxplus_matmul_pullback() {
    // Verify that tracked_tropical_einsum correctly records on the tape
    // and pullback produces correct gradients.
    //
    // We compute C = tropical_matmul(A, B) then compute loss = C[0,0]
    // by selecting a single element using standard einsum on a 1-element
    // slice, or by building a dot-product with a selector vector.
    //
    // For simplicity, we'll use a 1x1 contraction to extract a scalar:
    // Chain: tropical matmul -> standard dot with ones -> scalar loss
    let tape = Tape::<Tensor<f64>>::new();

    let a_data = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b_data = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();

    let a = tape.leaf(a_data);
    let b = tape.leaf(b_data);

    // C = MaxPlus matmul(A, B)
    let c =
        tracked_tropical_einsum::<MaxPlus<f64>, MaxPlusAlgebra<f64>, tenferro_prims::CpuBackend>(
            "ij,jk->ik",
            &[&a, &b],
        )
        .unwrap();

    // Verify the forward pass produced correct inner values
    let c_data = c.value().buffer().as_slice().unwrap();
    // C[0,0] = max(1+5, 3+6) = 9,  C[1,0] = max(2+5, 4+6) = 10
    // C[0,1] = max(1+7, 3+8) = 11, C[1,1] = max(2+7, 4+8) = 12
    assert_eq!(c_data[0], 9.0); // C[0,0]
    assert_eq!(c_data[1], 10.0); // C[1,0]
    assert_eq!(c_data[2], 11.0); // C[0,1]
    assert_eq!(c_data[3], 12.0); // C[1,1]

    // Compute loss = sum(C) using standard einsum with ones vector:
    // loss = C . ones = einsum("ik,ik->", C, ones_2x2)
    use std::sync::{Arc, Mutex};
    use tenferro_device::LogicalMemorySpace;
    use tenferro_einsum::tracked_einsum;

    let ones = Tensor::<f64>::ones(&[2, 2], LogicalMemorySpace::MainMemory, COL).unwrap();
    let ones_tracked = tidu::expert::TrackedValue::new(ones);

    let ctx = Arc::new(Mutex::new(ctx()));
    let loss = tracked_einsum::<tenferro_algebra::Standard<f64>, tenferro_prims::CpuBackend>(
        ctx.clone(),
        "ik,ik->",
        &[&c, &ones_tracked],
    )
    .unwrap();

    // loss = 9 + 10 + 11 + 12 = 42
    assert_eq!(loss.value().buffer().as_slice().unwrap()[0], 42.0);

    let grads = tape.pullback(&loss).unwrap();

    // The loss cotangent is all ones for each element of C (dot product with ones).
    // So the gradients through the tropical matmul are:
    // All winners j=1, so same as the standalone test above.
    let ga = grads.get(a.node_id().unwrap()).unwrap();
    let gb = grads.get(b.node_id().unwrap()).unwrap();

    let ga_data = ga.buffer().as_slice().unwrap();
    let gb_data = gb.buffer().as_slice().unwrap();

    assert_eq!(ga_data[0], 0.0); // dA[0,0]
    assert_eq!(ga_data[1], 0.0); // dA[1,0]
    assert_eq!(ga_data[2], 2.0); // dA[0,1]
    assert_eq!(ga_data[3], 2.0); // dA[1,1]

    assert_eq!(gb_data[0], 0.0); // dB[0,0]
    assert_eq!(gb_data[1], 2.0); // dB[1,0]
    assert_eq!(gb_data[2], 0.0); // dB[0,1]
    assert_eq!(gb_data[3], 2.0); // dB[1,1]
}

// ============================================================================
// Vector-matrix backward: "j,jk->k" (m=1)
// ============================================================================

#[test]
fn maxplus_vecmat_backward() {
    let mut ctx = ctx();

    // v = [1, 3]   (length 2)
    let v = Tensor::<MaxPlus<f64>>::from_slice(&[MaxPlus(1.0), MaxPlus(3.0)], &[2], COL).unwrap();

    // M = [[5, 7],    (column-major: [5, 6, 7, 8])
    //      [6, 8]]
    let m = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(5.0), MaxPlus(6.0), MaxPlus(7.0), MaxPlus(8.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    // C[k] = max_j(v[j] + M[j,k])
    // C[0] = max(1+5, 3+6) = max(6, 9) = 9   → winner j=1
    // C[1] = max(1+7, 3+8) = max(8, 11) = 11  → winner j=1

    let grad_c = Tensor::<f64>::from_slice(&[1.0, 1.0], &[2], COL).unwrap();

    let grads = tropical_einsum_rrule::<
        MaxPlus<f64>,
        MaxPlusAlgebra<f64>,
        tenferro_prims::CpuBackend,
    >(&mut ctx, "j,jk->k", &[&v, &m], &grad_c)
    .unwrap();

    let dv = grads[0].buffer().as_slice().unwrap();
    let dm = grads[1].buffer().as_slice().unwrap();

    // Both winners are j=1
    // dv[0] = 0 (j=0 never won)
    // dv[1] = dC[0] + dC[1] = 2 (j=1 won for both k=0 and k=1)
    assert_eq!(dv[0], 0.0);
    assert_eq!(dv[1], 2.0);

    // dM[0,0] = 0, dM[1,0] = dC[0] = 1, dM[0,1] = 0, dM[1,1] = dC[1] = 1
    assert_eq!(dm[0], 0.0); // dM[0,0]
    assert_eq!(dm[1], 1.0); // dM[1,0]
    assert_eq!(dm[2], 0.0); // dM[0,1]
    assert_eq!(dm[3], 1.0); // dM[1,1]
}

// ============================================================================
// Outer product backward: "i,j->ij" (no contraction, k=1)
// When there are no contracted indices, every element contributes.
// ============================================================================

#[test]
fn maxplus_outer_product_backward() {
    let mut ctx = ctx();

    let a = Tensor::<MaxPlus<f64>>::from_slice(&[MaxPlus(1.0), MaxPlus(2.0)], &[2], COL).unwrap();
    let b = Tensor::<MaxPlus<f64>>::from_slice(&[MaxPlus(10.0), MaxPlus(20.0)], &[2], COL).unwrap();

    // C[i,j] = A[i] * B[j] (no tropical addition, just tropical mul)
    // For MaxPlus: C[i,j] = A[i] + B[j] (ordinary +)
    // C[0,0] = 11, C[1,0] = 12, C[0,1] = 21, C[1,1] = 22

    let grad_c = Tensor::<f64>::from_slice(&[1.0, 1.0, 1.0, 1.0], &[2, 2], COL).unwrap();

    let grads = tropical_einsum_rrule::<
        MaxPlus<f64>,
        MaxPlusAlgebra<f64>,
        tenferro_prims::CpuBackend,
    >(&mut ctx, "i,j->ij", &[&a, &b], &grad_c)
    .unwrap();

    let da = grads[0].buffer().as_slice().unwrap();
    let db = grads[1].buffer().as_slice().unwrap();

    // No contraction means every pair contributes (k dimension is trivially 1)
    // Since contracted_total = max(1, 0) = 1 with k_flat = 0:
    // dA[i] += sum_j dC[i,j] * 1 = 2 for each i
    // dB[j] += sum_i dC[i,j] * 1 = 2 for each j
    assert_eq!(da[0], 2.0);
    assert_eq!(da[1], 2.0);
    assert_eq!(db[0], 2.0);
    assert_eq!(db[1], 2.0);
}

// ============================================================================
// Non-uniform cotangent backward
// ============================================================================

#[test]
fn maxplus_matmul_backward_nonuniform_cotangent() {
    let mut ctx = ctx();

    // A = [[1, 3],
    //      [2, 4]]
    let a = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(1.0), MaxPlus(2.0), MaxPlus(3.0), MaxPlus(4.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    // B = [[5, 7],
    //      [6, 8]]
    let b = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(5.0), MaxPlus(6.0), MaxPlus(7.0), MaxPlus(8.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    // Non-uniform cotangent: different weights for each output element
    // dC = [[10, 30],
    //       [20, 40]]
    let grad_c = Tensor::<f64>::from_slice(&[10.0, 20.0, 30.0, 40.0], &[2, 2], COL).unwrap();

    let grads = tropical_einsum_rrule::<
        MaxPlus<f64>,
        MaxPlusAlgebra<f64>,
        tenferro_prims::CpuBackend,
    >(&mut ctx, "ij,jk->ik", &[&a, &b], &grad_c)
    .unwrap();

    let da = grads[0].buffer().as_slice().unwrap();
    let db = grads[1].buffer().as_slice().unwrap();

    // All winners j=1 (same as before)
    // dA[0,0] = 0
    // dA[1,0] = 0
    // dA[0,1] = dC[0,0] + dC[0,1] = 10 + 30 = 40
    // dA[1,1] = dC[1,0] + dC[1,1] = 20 + 40 = 60
    assert_eq!(da[0], 0.0);
    assert_eq!(da[1], 0.0);
    assert_eq!(da[2], 40.0);
    assert_eq!(da[3], 60.0);

    // dB[0,0] = 0
    // dB[1,0] = dC[0,0] + dC[1,0] = 10 + 20 = 30
    // dB[0,1] = 0
    // dB[1,1] = dC[0,1] + dC[1,1] = 30 + 40 = 70
    assert_eq!(db[0], 0.0);
    assert_eq!(db[1], 30.0);
    assert_eq!(db[2], 0.0);
    assert_eq!(db[3], 70.0);
}

// ============================================================================
// Full contraction backward: "ij,ij->" (scalar output)
// ============================================================================

#[test]
fn maxplus_full_contraction_backward() {
    let mut ctx = ctx();

    // A = [[1, 3],
    //      [2, 4]]
    let a = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(1.0), MaxPlus(2.0), MaxPlus(3.0), MaxPlus(4.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    // B = [[10, 30],
    //      [20, 40]]
    let b = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(10.0), MaxPlus(20.0), MaxPlus(30.0), MaxPlus(40.0)],
        &[2, 2],
        COL,
    )
    .unwrap();

    // "ij,ij->" = element-wise tropical mul then tropical sum
    // Products: (1+10, 2+20, 3+30, 4+40) = (11, 22, 33, 44)
    // Scalar = max(11, 22, 33, 44) = 44
    // Winner: the element (i=1, j=1) with product 44
    //   → contracted indices are both i and j (since neither appears in output)

    let grad_c = Tensor::<f64>::from_slice(&[1.0], &[], COL).unwrap();

    let grads = tropical_einsum_rrule::<
        MaxPlus<f64>,
        MaxPlusAlgebra<f64>,
        tenferro_prims::CpuBackend,
    >(&mut ctx, "ij,ij->", &[&a, &b], &grad_c)
    .unwrap();

    let da = grads[0].buffer().as_slice().unwrap();
    let db = grads[1].buffer().as_slice().unwrap();

    // Winner is (i=1,j=1) which corresponds to flat index 3 in column-major
    // dA[1,1] = dC * 1 = 1 (MaxPlus backward through + is identity)
    // All other dA = 0
    assert_eq!(da[0], 0.0); // dA[0,0]
    assert_eq!(da[1], 0.0); // dA[1,0]
    assert_eq!(da[2], 0.0); // dA[0,1]
    assert_eq!(da[3], 1.0); // dA[1,1]

    assert_eq!(db[0], 0.0); // dB[0,0]
    assert_eq!(db[1], 0.0); // dB[1,0]
    assert_eq!(db[2], 0.0); // dB[0,1]
    assert_eq!(db[3], 1.0); // dB[1,1]
}

// ============================================================================
// Rectangular matmul backward: "ij,jk->ik" with non-square matrices
// ============================================================================

#[test]
fn maxplus_rectangular_matmul_backward() {
    let mut ctx = ctx();

    // A: 2x3 (column-major)
    // A = [[1, 5, 3],
    //      [2, 4, 6]]
    let a = Tensor::<MaxPlus<f64>>::from_slice(
        &[
            MaxPlus(1.0),
            MaxPlus(2.0),
            MaxPlus(5.0),
            MaxPlus(4.0),
            MaxPlus(3.0),
            MaxPlus(6.0),
        ],
        &[2, 3],
        COL,
    )
    .unwrap();

    // B: 3x2 (column-major)
    // B = [[1, 2],
    //      [3, 1],
    //      [2, 4]]
    let b = Tensor::<MaxPlus<f64>>::from_slice(
        &[
            MaxPlus(1.0),
            MaxPlus(3.0),
            MaxPlus(2.0),
            MaxPlus(2.0),
            MaxPlus(1.0),
            MaxPlus(4.0),
        ],
        &[3, 2],
        COL,
    )
    .unwrap();

    // C[i,k] = max_j(A[i,j] + B[j,k])
    // C[0,0] = max(1+1, 5+3, 3+2) = max(2, 8, 5) = 8  → winner j=1
    // C[1,0] = max(2+1, 4+3, 6+2) = max(3, 7, 8) = 8  → winner j=2
    // C[0,1] = max(1+2, 5+1, 3+4) = max(3, 6, 7) = 7  → winner j=2
    // C[1,1] = max(2+2, 4+1, 6+4) = max(4, 5, 10) = 10 → winner j=2

    let grad_c = Tensor::<f64>::from_slice(&[1.0, 1.0, 1.0, 1.0], &[2, 2], COL).unwrap();

    let grads = tropical_einsum_rrule::<
        MaxPlus<f64>,
        MaxPlusAlgebra<f64>,
        tenferro_prims::CpuBackend,
    >(&mut ctx, "ij,jk->ik", &[&a, &b], &grad_c)
    .unwrap();

    let da = grads[0].buffer().as_slice().unwrap();
    let db = grads[1].buffer().as_slice().unwrap();

    // Winners: C[0,0]→j=1, C[1,0]→j=2, C[0,1]→j=2, C[1,1]→j=2
    // dA[0,0] = 0
    // dA[1,0] = 0
    // dA[0,1] = dC[0,0] = 1  (j=1 won for (0,0))
    // dA[1,1] = 0             (j=1 never won for i=1)
    // dA[0,2] = dC[0,1] = 1  (j=2 won for (0,1))
    // dA[1,2] = dC[1,0] + dC[1,1] = 2  (j=2 won for (1,0) and (1,1))
    assert_eq!(da[0], 0.0); // dA[0,0]
    assert_eq!(da[1], 0.0); // dA[1,0]
    assert_eq!(da[2], 1.0); // dA[0,1]
    assert_eq!(da[3], 0.0); // dA[1,1]
    assert_eq!(da[4], 1.0); // dA[0,2]
    assert_eq!(da[5], 2.0); // dA[1,2]

    // dB[j,k] gets dC[i,k] when j was the winner for (i,k)
    // dB[0,0] = 0
    // dB[1,0] = dC[0,0] = 1
    // dB[2,0] = dC[1,0] = 1
    // dB[0,1] = 0
    // dB[1,1] = 0
    // dB[2,1] = dC[0,1] + dC[1,1] = 2
    assert_eq!(db[0], 0.0); // dB[0,0]
    assert_eq!(db[1], 1.0); // dB[1,0]
    assert_eq!(db[2], 1.0); // dB[2,0]
    assert_eq!(db[3], 0.0); // dB[0,1]
    assert_eq!(db[4], 0.0); // dB[1,1]
    assert_eq!(db[5], 2.0); // dB[2,1]
}

// ============================================================================
// Error path tests
// ============================================================================

#[test]
fn rrule_accepts_single_operand() {
    let mut ctx = ctx();

    let a = Tensor::<MaxPlus<f64>>::from_slice(&[MaxPlus(1.0), MaxPlus(2.0)], &[2], COL).unwrap();
    let grad = Tensor::<f64>::from_slice(&[1.0], &[], COL).unwrap();

    let result = tropical_einsum_rrule::<
        MaxPlus<f64>,
        MaxPlusAlgebra<f64>,
        tenferro_prims::CpuBackend,
    >(&mut ctx, "i->", &[&a], &grad);
    assert!(result.is_ok());
}

#[test]
fn rrule_rejects_three_operands() {
    let mut ctx = ctx();

    let a = Tensor::<MaxPlus<f64>>::from_slice(&[MaxPlus(1.0), MaxPlus(2.0)], &[2], COL).unwrap();
    let b = Tensor::<MaxPlus<f64>>::from_slice(&[MaxPlus(3.0), MaxPlus(4.0)], &[2], COL).unwrap();
    let c = Tensor::<MaxPlus<f64>>::from_slice(&[MaxPlus(5.0), MaxPlus(6.0)], &[2], COL).unwrap();
    let grad = Tensor::<f64>::from_slice(&[1.0], &[], COL).unwrap();

    let result = tropical_einsum_rrule::<
        MaxPlus<f64>,
        MaxPlusAlgebra<f64>,
        tenferro_prims::CpuBackend,
    >(&mut ctx, "i,i,i->", &[&a, &b, &c], &grad);
    assert!(result.is_err());
}

#[test]
fn rrule_rejects_cotangent_shape_mismatch() {
    let mut ctx = ctx();

    let a = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(1.0), MaxPlus(2.0), MaxPlus(3.0), MaxPlus(4.0)],
        &[2, 2],
        COL,
    )
    .unwrap();
    let b = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(5.0), MaxPlus(6.0), MaxPlus(7.0), MaxPlus(8.0)],
        &[2, 2],
        COL,
    )
    .unwrap();
    let bad_cotangent = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], COL).unwrap();

    let result = tropical_einsum_rrule::<
        MaxPlus<f64>,
        MaxPlusAlgebra<f64>,
        tenferro_prims::CpuBackend,
    >(&mut ctx, "ij,jk->ik", &[&a, &b], &bad_cotangent);

    assert!(
        matches!(result, Err(Error::InvalidArgument(message)) if message.contains("cotangent"))
    );
}

#[test]
fn tracked_accepts_single_operand() {
    let tape = Tape::<Tensor<f64>>::new();
    let a_data = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], COL).unwrap();
    let a = tape.leaf(a_data);

    let result = tracked_tropical_einsum::<
        MaxPlus<f64>,
        MaxPlusAlgebra<f64>,
        tenferro_prims::CpuBackend,
    >("i->", &[&a]);
    assert!(result.is_ok());
}

#[test]
fn tracked_no_grad_returns_plain_tensor() {
    // When operands don't require grad, tracked_tropical_einsum should
    // return a TrackedValue without node_id (not on tape).
    let a_data = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b_data = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();

    // Non-tracked (no tape, no gradient)
    let a = tidu::expert::TrackedValue::new(a_data);
    let b = tidu::expert::TrackedValue::new(b_data);

    let c =
        tracked_tropical_einsum::<MaxPlus<f64>, MaxPlusAlgebra<f64>, tenferro_prims::CpuBackend>(
            "ij,jk->ik",
            &[&a, &b],
        )
        .unwrap();

    // Result should not have a node ID (not recorded on tape)
    assert!(c.node_id().is_none());

    // But should have correct values
    let c_data = c.value().buffer().as_slice().unwrap();
    assert_eq!(c_data[0], 9.0); // max(1+5, 3+6) = 9
    assert_eq!(c_data[1], 10.0); // max(2+5, 4+6) = 10
    assert_eq!(c_data[2], 11.0); // max(1+7, 3+8) = 11
    assert_eq!(c_data[3], 12.0); // max(2+7, 4+8) = 12
}

// ============================================================================
// f32 backward test
// ============================================================================

#[test]
fn maxplus_f32_matmul_backward() {
    let mut ctx = ctx();

    // A = [[1, 3],    (column-major: [1, 2, 3, 4])
    //      [2, 4]]
    let a = Tensor::<MaxPlus<f32>>::from_slice(
        &[
            MaxPlus(1.0f32),
            MaxPlus(2.0f32),
            MaxPlus(3.0f32),
            MaxPlus(4.0f32),
        ],
        &[2, 2],
        COL,
    )
    .unwrap();

    // B = [[5, 7],    (column-major: [5, 6, 7, 8])
    //      [6, 8]]
    let b = Tensor::<MaxPlus<f32>>::from_slice(
        &[
            MaxPlus(5.0f32),
            MaxPlus(6.0f32),
            MaxPlus(7.0f32),
            MaxPlus(8.0f32),
        ],
        &[2, 2],
        COL,
    )
    .unwrap();

    // C[i,k] = max_j(A[i,j] + B[j,k])
    // C[0,0] = max(1+5, 3+6) = max(6, 9) = 9    -> winner j=1
    // C[1,0] = max(2+5, 4+6) = max(7, 10) = 10   -> winner j=1
    // C[0,1] = max(1+7, 3+8) = max(8, 11) = 11   -> winner j=1
    // C[1,1] = max(2+7, 4+8) = max(9, 12) = 12   -> winner j=1

    // Cotangent: all ones
    let grad_c = Tensor::<f32>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[2, 2], COL).unwrap();

    let grads = tropical_einsum_rrule::<
        MaxPlus<f32>,
        MaxPlusAlgebra<f32>,
        tenferro_prims::CpuBackend,
    >(&mut ctx, "ij,jk->ik", &[&a, &b], &grad_c)
    .unwrap();

    assert_eq!(grads.len(), 2);
    let da = grads[0].buffer().as_slice().unwrap();
    let db = grads[1].buffer().as_slice().unwrap();

    // All winners are j=1 (second row of A, second column of B in col-major).
    // For MaxPlus: backward through + is identity, so dA[i,j*] += dC[i,k], dB[j*,k] += dC[i,k].
    // dA[0,0] = 0 (never winner), dA[1,0] = 0
    // dA[0,1] = dC[0,0]+dC[0,1] = 2, dA[1,1] = dC[1,0]+dC[1,1] = 2
    // col-major dA: [dA[0,0], dA[1,0], dA[0,1], dA[1,1]] = [0, 0, 2, 2]
    assert_eq!(da[0], 0.0f32);
    assert_eq!(da[1], 0.0f32);
    assert_eq!(da[2], 2.0f32);
    assert_eq!(da[3], 2.0f32);

    // dB[0,0] = 0, dB[1,0] = 0 (j=0 never won)
    // dB[0,1] = 0, dB[1,1] = 0
    // Wait: j=1 won for all outputs. So:
    // dB[j*=1,k=0] += dC[0,0] + dC[1,0] = 2
    // dB[j*=1,k=1] += dC[0,1] + dC[1,1] = 2
    // col-major dB: [dB[0,0], dB[1,0], dB[0,1], dB[1,1]] = [0, 2, 0, 2]
    assert_eq!(db[0], 0.0f32);
    assert_eq!(db[1], 2.0f32);
    assert_eq!(db[2], 0.0f32);
    assert_eq!(db[3], 2.0f32);
}

// ============================================================================
// Unary tropical backward tests
// ============================================================================

#[test]
fn maxplus_unary_trace_backward() {
    // ii-> : max of diagonal elements
    let mut ctx = ctx();
    // A = [[1, 3],    (col-major: [1, 2, 3, 4])
    //      [2, 4]]
    // Diagonal: A[0,0]=1, A[1,1]=4. MaxPlus sum = max(1, 4) = 4
    // Winner: (i=1) → flat index k=1 in contracted dim (size 2)
    let a = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(1.0), MaxPlus(2.0), MaxPlus(3.0), MaxPlus(4.0)],
        &[2, 2],
        COL,
    )
    .unwrap();
    let grad = Tensor::<f64>::from_slice(&[1.0], &[], COL).unwrap();

    let grads = tropical_einsum_rrule::<
        MaxPlus<f64>,
        MaxPlusAlgebra<f64>,
        tenferro_prims::CpuBackend,
    >(&mut ctx, "ii->", &[&a], &grad)
    .unwrap();

    assert_eq!(grads.len(), 1);
    let da = grads[0].buffer().as_slice().unwrap();
    // Only the winner diagonal element (1,1) = flat index 3 gets gradient
    assert_eq!(da[0], 0.0); // A[0,0]
    assert_eq!(da[1], 0.0); // A[1,0]
    assert_eq!(da[2], 0.0); // A[0,1]
    assert_eq!(da[3], 1.0); // A[1,1] — winner
}

#[test]
fn maxplus_unary_full_contraction_backward() {
    // ij-> : max of all elements
    let mut ctx = ctx();
    // A = [[1, 5],    (col-major: [1, 4, 5, 2])
    //      [4, 2]]
    // A[0,0]=1, A[1,0]=4, A[0,1]=5, A[1,1]=2
    // Max = 5 at A[0,1] = flat index 2
    let a = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(1.0), MaxPlus(4.0), MaxPlus(5.0), MaxPlus(2.0)],
        &[2, 2],
        COL,
    )
    .unwrap();
    let grad = Tensor::<f64>::from_slice(&[1.0], &[], COL).unwrap();

    let grads = tropical_einsum_rrule::<
        MaxPlus<f64>,
        MaxPlusAlgebra<f64>,
        tenferro_prims::CpuBackend,
    >(&mut ctx, "ij->", &[&a], &grad)
    .unwrap();

    assert_eq!(grads.len(), 1);
    let da = grads[0].buffer().as_slice().unwrap();
    // Winner is A[0,1] = 5.0 at flat index 2
    assert_eq!(da[0], 0.0); // A[0,0]
    assert_eq!(da[1], 0.0); // A[1,0]
    assert_eq!(da[2], 1.0); // A[0,1] — winner
    assert_eq!(da[3], 0.0); // A[1,1]
}

#[test]
fn maxplus_unary_row_max_backward() {
    // ij->i : max over j for each i (row-wise max)
    let mut ctx = ctx();
    // A = [[1, 5],    (col-major: [1, 4, 5, 2])
    //      [4, 2]]
    // Row 0 (i=0): max(A[0,0]=1, A[0,1]=5) = 5, winner j=1
    // Row 1 (i=1): max(A[1,0]=4, A[1,1]=2) = 4, winner j=0
    let a = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(1.0), MaxPlus(4.0), MaxPlus(5.0), MaxPlus(2.0)],
        &[2, 2],
        COL,
    )
    .unwrap();
    let grad = Tensor::<f64>::from_slice(&[1.0, 1.0], &[2], COL).unwrap();

    let grads = tropical_einsum_rrule::<
        MaxPlus<f64>,
        MaxPlusAlgebra<f64>,
        tenferro_prims::CpuBackend,
    >(&mut ctx, "ij->i", &[&a], &grad)
    .unwrap();

    assert_eq!(grads.len(), 1);
    let da = grads[0].buffer().as_slice().unwrap();
    // dA[0,0] = 0 (j=0 didn't win for i=0)
    // dA[1,0] = 1 (j=0 won for i=1)
    // dA[0,1] = 1 (j=1 won for i=0)
    // dA[1,1] = 0 (j=1 didn't win for i=1)
    assert_eq!(da[0], 0.0); // A[0,0]
    assert_eq!(da[1], 1.0); // A[1,0]
    assert_eq!(da[2], 1.0); // A[0,1]
    assert_eq!(da[3], 0.0); // A[1,1]
}

#[test]
fn maxplus_unary_col_max_backward() {
    // ij->j : max over i for each j (column-wise max)
    let mut ctx = ctx();
    // A = [[1, 5],    (col-major: [1, 4, 5, 2])
    //      [4, 2]]
    // Col 0 (j=0): max(A[0,0]=1, A[1,0]=4) = 4, winner i=1
    // Col 1 (j=1): max(A[0,1]=5, A[1,1]=2) = 5, winner i=0
    let a = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(1.0), MaxPlus(4.0), MaxPlus(5.0), MaxPlus(2.0)],
        &[2, 2],
        COL,
    )
    .unwrap();
    let grad = Tensor::<f64>::from_slice(&[1.0, 1.0], &[2], COL).unwrap();

    let grads = tropical_einsum_rrule::<
        MaxPlus<f64>,
        MaxPlusAlgebra<f64>,
        tenferro_prims::CpuBackend,
    >(&mut ctx, "ij->j", &[&a], &grad)
    .unwrap();

    assert_eq!(grads.len(), 1);
    let da = grads[0].buffer().as_slice().unwrap();
    // dA[0,0] = 0 (i=0 didn't win for j=0)
    // dA[1,0] = 1 (i=1 won for j=0)
    // dA[0,1] = 1 (i=0 won for j=1)
    // dA[1,1] = 0 (i=1 didn't win for j=1)
    assert_eq!(da[0], 0.0); // A[0,0]
    assert_eq!(da[1], 1.0); // A[1,0]
    assert_eq!(da[2], 1.0); // A[0,1]
    assert_eq!(da[3], 0.0); // A[1,1]
}
