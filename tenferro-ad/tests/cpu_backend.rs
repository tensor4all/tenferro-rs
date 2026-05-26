//! Integration tests for the v2 CpuBackend.
//!
//! Verifies faer GEMM dispatch, batched GEMM, and stride-aware input handling.

mod support;
use support::RunTraced;
use tenferro_runtime::traced::TracedTensor;
use tenferro_runtime::GraphExecutor;
use tenferro_tensor::{cpu::CpuBackend, DotGeneralConfig, Tensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data))
}

fn f32_tensor(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    Tensor::F32(TypedTensor::from_vec_col_major(shape, data))
}

fn get_f64_data(t: &Tensor) -> &[f64] {
    match t {
        Tensor::F64(inner) => inner.host_data(),
        _ => panic!("expected F64"),
    }
}

fn get_f32_data(t: &Tensor) -> &[f32] {
    match t {
        Tensor::F32(inner) => inner.host_data(),
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

    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let tb = TracedTensor::from_tensor_concrete_shape(b);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let tc = ta.dot_general(&tb, config);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = tc.run_with(&mut engine).unwrap();
    let data = get_f64_data(&result);
    // C = A*B = [[22,49],[28,64]] col-major: [22,28,49,64]
    assert_eq!(data, &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn test_faer_gemm_basic_f32() {
    let a = f32_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f32_tensor(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let tb = TracedTensor::from_tensor_concrete_shape(b);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let tc = ta.dot_general(&tb, config);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = tc.run_with(&mut engine).unwrap();
    let data = get_f32_data(&result);
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

    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let ti = TracedTensor::from_tensor_concrete_shape(i);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let tc = ta.dot_general(&ti, config);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = tc.run_with(&mut engine).unwrap();
    let data = get_f64_data(&result);
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

    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let tb = TracedTensor::from_tensor_concrete_shape(b);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1], // contract over dim 1 (K)
        rhs_contracting_dims: vec![0], // contract over dim 0 (K)
        lhs_batch_dims: vec![2],       // batch over dim 2
        rhs_batch_dims: vec![2],       // batch over dim 2
    };
    let tc = ta.dot_general(&tb, config);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = tc.run_with(&mut engine).unwrap();

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
    let data = get_f64_data(&result);
    assert_eq!(data, &[23.0, 34.0, 31.0, 46.0, 271.0, 298.0, 311.0, 342.0]);
}

// ============================================================================
// Strided input tests
// ============================================================================

#[test]
fn test_strided_input_via_transpose_and_dot_general() {
    // Multiply A^T * B where A is transposed by the graph.
    // A[3,2] with data = [[1,4],[2,5],[3,6]] col-major: [1,2,3,4,5,6]
    // The transpose "ji->ij" makes A^T[2,3].
    // Then A^T[2,3] * B[3,2] -> C[2,2].
    let a = f64_tensor(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let tb = TracedTensor::from_tensor_concrete_shape(b);
    let ta_t = ta.transpose(&[1, 0]);
    let tc = ta_t.dot_general(
        &tb,
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    );

    let result = tc.run_with(&mut engine).unwrap();
    let data = get_f64_data(&result);

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

    let tv = TracedTensor::from_tensor_concrete_shape(v);
    let tw = TracedTensor::from_tensor_concrete_shape(w);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![0],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let tc = tv.dot_general(&tw, config);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = tc.run_with(&mut engine).unwrap();
    assert!(result.shape().is_empty());
    let data = get_f64_data(&result);
    // 1*4 + 2*5 + 3*6 = 32
    assert_eq!(data, &[32.0]);
}

// ============================================================================
