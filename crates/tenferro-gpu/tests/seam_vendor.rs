#![cfg(feature = "cuda")]

//! U4: cuBLAS GEMM and strided-batched GEMM on the runtime's own stream, matched against
//! host references (issue #2208 U4).

use t4a_cuda_kernels::DType;
use tenferro_gpu::cuda::{
    gpu_available, Cublas, CudaDeviceId, CudaEngine, CudaRuntime, GemmRequest,
};

fn runtime() -> Option<CudaRuntime> {
    if !gpu_available() {
        return None;
    }
    let engine = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine");
    Some(engine.runtime().clone())
}

fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn from_f64_bytes(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|word| f64::from_le_bytes(word.try_into().expect("8 bytes")))
        .collect()
}

/// Column-major `c = a * b` for one `(m, k)` times `(k, n)` product.
fn host_gemm(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for column in 0..n {
        for row in 0..m {
            let mut sum = 0.0;
            for inner in 0..k {
                sum += a[row + inner * m] * b[inner + column * k];
            }
            c[row + column * m] = sum;
        }
    }
    c
}

#[test]
fn a_plain_gemm_matches_the_host() {
    let Some(runtime) = runtime() else { return };
    let (m, n, k) = (37usize, 29usize, 41usize);
    let a: Vec<f64> = (0..m * k)
        .map(|index| (index % 13) as f64 * 0.5 - 2.0)
        .collect();
    let b: Vec<f64> = (0..k * n).map(|index| (index % 7) as f64 - 1.5).collect();
    let a_buf = runtime.upload_bytes(&f64_bytes(&a)).expect("a");
    let b_buf = runtime.upload_bytes(&f64_bytes(&b)).expect("b");
    let c_buf = runtime.upload_bytes(&vec![0u8; m * n * 8]).expect("c");
    let mut cublas = Cublas::new(&runtime).expect("cuBLAS");
    cublas
        .gemm(&GemmRequest {
            dtype: DType::F64,
            a: &a_buf,
            trans_a: false,
            b: &b_buf,
            trans_b: false,
            c: &c_buf,
            shape: (m as i64, n as i64, k as i64),
            leading: (m as i64, k as i64, m as i64),
            batches: (1, 0, 0, 0),
        })
        .expect("gemm");
    let observed = from_f64_bytes(&runtime.download_bytes(&c_buf).expect("download"));
    assert_eq!(observed, host_gemm(&a, &b, m, n, k));
}

#[test]
fn a_transposed_operand_matches_the_host() {
    let Some(runtime) = runtime() else { return };
    // `a` is stored as `(k, m)` and transposed into the product.
    let (m, n, k) = (16usize, 12usize, 20usize);
    let a: Vec<f64> = (0..k * m).map(|index| (index % 5) as f64 + 0.25).collect();
    let b: Vec<f64> = (0..k * n).map(|index| (index % 3) as f64 - 1.0).collect();
    // The host reference wants `a` in `(m, k)` order, which is the transposed view.
    let mut a_transposed = vec![0.0f64; m * k];
    for row in 0..m {
        for inner in 0..k {
            a_transposed[row + inner * m] = a[inner + row * k];
        }
    }
    let a_buf = runtime.upload_bytes(&f64_bytes(&a)).expect("a");
    let b_buf = runtime.upload_bytes(&f64_bytes(&b)).expect("b");
    let c_buf = runtime.upload_bytes(&vec![0u8; m * n * 8]).expect("c");
    let mut cublas = Cublas::new(&runtime).expect("cuBLAS");
    cublas
        .gemm(&GemmRequest {
            dtype: DType::F64,
            a: &a_buf,
            trans_a: true,
            b: &b_buf,
            trans_b: false,
            c: &c_buf,
            shape: (m as i64, n as i64, k as i64),
            leading: (k as i64, k as i64, m as i64),
            batches: (1, 0, 0, 0),
        })
        .expect("gemm");
    let observed = from_f64_bytes(&runtime.download_bytes(&c_buf).expect("download"));
    assert_eq!(observed, host_gemm(&a_transposed, &b, m, n, k));
}

#[test]
fn a_strided_batched_gemm_matches_the_host() {
    let Some(runtime) = runtime() else { return };
    let (m, n, k, batches) = (8usize, 6usize, 5usize, 3usize);
    let a: Vec<f64> = (0..batches * m * k)
        .map(|index| (index % 11) as f64 * 0.25)
        .collect();
    let b: Vec<f64> = (0..batches * k * n)
        .map(|index| (index % 9) as f64 - 2.0)
        .collect();
    let a_buf = runtime.upload_bytes(&f64_bytes(&a)).expect("a");
    let b_buf = runtime.upload_bytes(&f64_bytes(&b)).expect("b");
    let c_buf = runtime
        .upload_bytes(&vec![0u8; batches * m * n * 8])
        .expect("c");
    let mut cublas = Cublas::new(&runtime).expect("cuBLAS");
    cublas
        .gemm(&GemmRequest {
            dtype: DType::F64,
            a: &a_buf,
            trans_a: false,
            b: &b_buf,
            trans_b: false,
            c: &c_buf,
            shape: (m as i64, n as i64, k as i64),
            leading: (m as i64, k as i64, m as i64),
            batches: (
                batches as i64,
                (m * k) as i64,
                (k * n) as i64,
                (m * n) as i64,
            ),
        })
        .expect("batched gemm");
    let observed = from_f64_bytes(&runtime.download_bytes(&c_buf).expect("download"));
    let mut expected = Vec::with_capacity(batches * m * n);
    for batch in 0..batches {
        expected.extend(host_gemm(
            &a[batch * m * k..(batch + 1) * m * k],
            &b[batch * k * n..(batch + 1) * k * n],
            m,
            n,
            k,
        ));
    }
    assert_eq!(observed, expected);
}

#[test]
fn an_impossible_leading_dimension_is_refused() {
    let Some(runtime) = runtime() else { return };
    let a_buf = runtime.upload_bytes(&f64_bytes(&[1.0; 4])).expect("a");
    let b_buf = runtime.upload_bytes(&f64_bytes(&[1.0; 4])).expect("b");
    let c_buf = runtime.upload_bytes(&[0u8; 4 * 8]).expect("c");
    let mut cublas = Cublas::new(&runtime).expect("cuBLAS");
    // A 4-element operand cannot serve an 4x4 product with `lda = 4` when it needs 16.
    assert!(cublas
        .gemm(&GemmRequest {
            dtype: DType::F64,
            a: &a_buf,
            trans_a: false,
            b: &b_buf,
            trans_b: false,
            c: &c_buf,
            shape: (4, 4, 4),
            leading: (2, 2, 2),
            batches: (1, 0, 0, 0),
        })
        .is_err());
}
