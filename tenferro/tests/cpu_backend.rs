//! Integration tests for the v2 CpuBackend.
//!
//! Verifies faer GEMM dispatch, batched GEMM, and stride-aware input handling.

use tenferro::buffer_pool::BufferPool;
use tenferro::einsum::einsum;
use tenferro::engine::Engine;
use tenferro::traced::TracedTensor;
use tenferro_tensor::{cpu::CpuBackend, DotGeneralConfig, LayoutOrder, Tensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn f32_tensor(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    Tensor::F32(TypedTensor::from_vec(shape, data))
}

fn get_f64_data(t: &Tensor) -> Vec<f64> {
    match t {
        Tensor::F64(inner) => inner
            .to_contiguous(LayoutOrder::ColumnMajor)
            .unwrap()
            .host_data()
            .to_vec(),
        _ => panic!("expected F64"),
    }
}

fn get_f32_data(t: &Tensor) -> Vec<f32> {
    match t {
        Tensor::F32(inner) => inner
            .to_contiguous(LayoutOrder::ColumnMajor)
            .unwrap()
            .host_data()
            .to_vec(),
        _ => panic!("expected F32"),
    }
}

// ============================================================================
// Basic faer GEMM tests
// ============================================================================

#[test]
fn test_faer_gemm_basic_f64() {
    // A[2,3] x B[3,2] -> C[2,2]
    // A = [[1,3,5],[2,4,6]] col-major: [1,2,3,4,5,6]
    // B = [[1,4],[2,5],[3,6]] col-major: [1,2,3,4,5,6]
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let ta = TracedTensor::from_tensor(a);
    let tb = TracedTensor::from_tensor(b);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
        lhs_rank: 2,
        rhs_rank: 2,
    };
    let mut tc = ta.dot_general(&tb, config);

    let mut engine = Engine::new(CpuBackend::new());
    let result = tc.eval(&mut engine).unwrap();
    let data = get_f64_data(result);
    // C = A*B = [[22,49],[28,64]] col-major: [22,28,49,64]
    assert_eq!(data, &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn test_faer_gemm_basic_f32() {
    let a = f32_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f32_tensor(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let ta = TracedTensor::from_tensor(a);
    let tb = TracedTensor::from_tensor(b);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
        lhs_rank: 2,
        rhs_rank: 2,
    };
    let mut tc = ta.dot_general(&tb, config);

    let mut engine = Engine::new(CpuBackend::new());
    let result = tc.eval(&mut engine).unwrap();
    let data = get_f32_data(result);
    assert_eq!(data, &[22.0f32, 28.0, 49.0, 64.0]);
}

#[test]
fn test_faer_gemm_identity() {
    // Multiply by identity: A[3,3] * I[3,3] = A[3,3]
    // A = [[1,4,7],[2,5,8],[3,6,9]] col-major: [1,2,3,4,5,6,7,8,9]
    let a = f64_tensor(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
    // I = [[1,0,0],[0,1,0],[0,0,1]] col-major
    let i = f64_tensor(
        vec![3, 3],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    );

    let ta = TracedTensor::from_tensor(a.clone());
    let ti = TracedTensor::from_tensor(i);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
        lhs_rank: 2,
        rhs_rank: 2,
    };
    let mut tc = ta.dot_general(&ti, config);

    let mut engine = Engine::new(CpuBackend::new());
    let result = tc.eval(&mut engine).unwrap();
    let data = get_f64_data(result);
    let expected = get_f64_data(&a);
    assert_eq!(data, expected);
}

// ============================================================================
// Batched GEMM tests
// ============================================================================

#[test]
fn test_batched_gemm() {
    // Batch of 2 matrix multiplications: A[2,2,2] x B[2,2,2] -> C[2,2,2]
    // where batch dim is the last (dim 2).
    //
    // batch 0: A0 = [[1,3],[2,4]], B0 = [[5,7],[6,8]]
    // batch 1: A1 = [[9,11],[10,12]], B1 = [[13,15],[14,16]]
    //
    // Col-major for [2,2,2]:
    // A = [1,2,3,4, 9,10,11,12]
    // B = [5,6,7,8, 13,14,15,16]
    let a = f64_tensor(
        vec![2, 2, 2],
        vec![1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0],
    );
    let b = f64_tensor(
        vec![2, 2, 2],
        vec![5.0, 6.0, 7.0, 8.0, 13.0, 14.0, 15.0, 16.0],
    );

    let ta = TracedTensor::from_tensor(a);
    let tb = TracedTensor::from_tensor(b);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1], // contract over dim 1 (K)
        rhs_contracting_dims: vec![0], // contract over dim 0 (K)
        lhs_batch_dims: vec![2],       // batch over dim 2
        rhs_batch_dims: vec![2],       // batch over dim 2
        lhs_rank: 3,
        rhs_rank: 3,
    };
    let mut tc = ta.dot_general(&tb, config);

    let mut engine = Engine::new(CpuBackend::new());
    let result = tc.eval(&mut engine).unwrap();

    // Batch 0: C0 = A0*B0 = [[1*5+3*6, 1*7+3*8],[2*5+4*6, 2*7+4*8]]
    //             = [[23,31],[34,46]]
    // Batch 1: C1 = A1*B1 = [[9*13+11*14, 9*15+11*16],[10*13+12*14, 10*15+12*16]]
    //             = [[271,311],[298,342]]
    //
    // Output shape: [M=2, N=2, batch=2]
    // (lhs_free ++ rhs_free ++ batch)
    //
    // Col-major: data[m + n*2 + b*4]:
    //   batch 0: [23, 34, 31, 46]
    //   batch 1: [271, 298, 311, 342]
    let data = get_f64_data(result);
    assert_eq!(data, &[23.0, 34.0, 31.0, 46.0, 271.0, 298.0, 311.0, 342.0]);
}

#[test]
fn test_batched_gemm_via_einsum() {
    // Same as above but via einsum notation: "ijk,jlk->ilk"
    // where i=M, j=K, l=N, k=batch
    let a = f64_tensor(
        vec![2, 2, 2],
        vec![1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0],
    );
    let b = f64_tensor(
        vec![2, 2, 2],
        vec![5.0, 6.0, 7.0, 8.0, 13.0, 14.0, 15.0, 16.0],
    );

    let mut engine = Engine::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor(a);
    let tb = TracedTensor::from_tensor(b);
    let mut tc = einsum(&mut engine, &[&ta, &tb], "ijk,jlk->ilk").unwrap();

    let result = tc.eval(&mut engine).unwrap();
    let data = get_f64_data(result);
    // Same result as above, potentially in a different layout due to einsum
    // reordering. The einsum output indices are "ilk" so shape is [i=2, l=2, k=2].
    // col-major: i fastest, then l, then k.
    // batch 0 (k=0): C[i,l] = [[23,31],[34,46]]
    //   col-major: [23, 34, 31, 46]
    // batch 1 (k=1): C[i,l] = [[271,311],[298,342]]
    //   col-major: [271, 298, 311, 342]
    // full: [23, 34, 31, 46, 271, 298, 311, 342]
    assert_eq!(data, &[23.0, 34.0, 31.0, 46.0, 271.0, 298.0, 311.0, 342.0]);
}

// ============================================================================
// Strided input tests
// ============================================================================

#[test]
fn test_strided_input_via_einsum() {
    // Multiply A^T * B where A is transposed via einsum.
    // A[3,2] with data = [[1,4],[2,5],[3,6]] col-major: [1,2,3,4,5,6]
    // The transpose "ji->ij" makes A^T[2,3].
    // Then A^T[2,3] * B[3,2] -> C[2,2].
    let a = f64_tensor(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // A^T * B via einsum: "ji,jk->ik"
    let mut engine = Engine::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor(a);
    let tb = TracedTensor::from_tensor(b);
    let mut tc = einsum(&mut engine, &[&ta, &tb], "ji,jk->ik").unwrap();

    let result = tc.eval(&mut engine).unwrap();
    let data = get_f64_data(result);

    // A^T = [[1,2,3],[4,5,6]]
    // A^T * B = [[1*1+2*2+3*3, 1*4+2*5+3*6],[4*1+5*2+6*3, 4*4+5*5+6*6]]
    //         = [[14, 32],[32, 77]]
    // col-major: [14, 32, 32, 77]
    assert_eq!(data, &[14.0, 32.0, 32.0, 77.0]);
}

#[test]
fn test_vector_dot_product() {
    // Inner product via dot_general: v[3] . w[3] -> scalar[]
    let v = f64_tensor(vec![3], vec![1.0, 2.0, 3.0]);
    let w = f64_tensor(vec![3], vec![4.0, 5.0, 6.0]);

    let tv = TracedTensor::from_tensor(v);
    let tw = TracedTensor::from_tensor(w);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![0],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
        lhs_rank: 1,
        rhs_rank: 1,
    };
    let mut tc = tv.dot_general(&tw, config);

    let mut engine = Engine::new(CpuBackend::new());
    let result = tc.eval(&mut engine).unwrap();
    assert!(result.shape().is_empty());
    let data = get_f64_data(result);
    // 1*4 + 2*5 + 3*6 = 32
    assert_eq!(data, &[32.0]);
}

// ============================================================================
// Buffer pool tests
// ============================================================================

#[test]
fn test_buffer_pool_allocate_fresh() {
    let mut pool = BufferPool::new();
    let buf = pool.allocate(64);
    assert_eq!(buf.len(), 64);
    assert!(pool.is_empty());
}

#[test]
fn test_buffer_pool_default_is_empty() {
    let pool = BufferPool::default();
    assert_eq!(pool.len(), 0);
    assert!(pool.is_empty());
}

#[test]
fn test_buffer_pool_reuse() {
    let mut pool = BufferPool::new();
    let buf = pool.allocate(128);
    assert_eq!(buf.len(), 128);

    pool.return_buffer(buf);
    assert_eq!(pool.len(), 1);

    // Allocate smaller: should reuse the 128-byte buffer
    let buf2 = pool.allocate(64);
    assert_eq!(buf2.len(), 64);
    assert!(pool.is_empty());
}

#[test]
fn test_buffer_pool_no_fit() {
    let mut pool = BufferPool::new();
    let buf = pool.allocate(32);
    pool.return_buffer(buf);

    // Request larger than any pooled buffer: fresh allocation
    let buf2 = pool.allocate(256);
    assert_eq!(buf2.len(), 256);
    // The 32-byte buffer is still in the pool
    assert_eq!(pool.len(), 1);
}

#[test]
fn test_buffer_pool_best_fit() {
    let mut pool = BufferPool::new();

    let b1 = pool.allocate(100);
    let b2 = pool.allocate(200);
    let b3 = pool.allocate(300);
    pool.return_buffer(b1);
    pool.return_buffer(b2);
    pool.return_buffer(b3);
    assert_eq!(pool.len(), 3);

    // Request 150 bytes: should pick the 200-byte buffer (smallest that fits)
    let reused = pool.allocate(150);
    assert_eq!(reused.len(), 150); // resized to requested size
    assert_eq!(pool.len(), 2); // two buffers left
}

#[test]
fn buffer_pool_reclaims_intermediate_buffers() {
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let c = f64_tensor(vec![2, 2], vec![9.0, 10.0, 11.0, 12.0]);

    let mut engine = Engine::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor(a);
    let tb = TracedTensor::from_tensor(b);
    let tc = TracedTensor::from_tensor(c);
    let mut out = einsum(&mut engine, &[&ta, &tb, &tc], "ij,jk,kl->il").unwrap();

    let result = out.eval(&mut engine).unwrap();

    assert_eq!(get_f64_data(result), &[517.0, 766.0, 625.0, 926.0]);
    assert!(engine.buffer_pool_len() > 0);
}
