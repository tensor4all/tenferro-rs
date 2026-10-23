#![cfg(feature = "cuda-next")]

//! U4: cuTENSOR contraction with shape-keyed plans, matched against host references
//! (issue #2208 U4).

use t4a_cuda_kernels::DType;
use tenferro_gpu::cuda_next::{
    gpu_available, Contraction, CudaDeviceId, CudaEngine, CudaRuntime, Cutensor, Operand,
};

fn runtime() -> Option<CudaRuntime> {
    if !gpu_available() {
        return None;
    }
    let engine = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine");
    Some(engine.runtime().clone())
}

fn bytes(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn values(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|word| f64::from_le_bytes(word.try_into().expect("8 bytes")))
        .collect()
}

#[test]
fn a_matrix_product_as_a_contraction_matches_the_host() {
    let Some(runtime) = runtime() else { return };
    let (m, n, k) = (23usize, 17usize, 31usize);
    // `a` is `(m, k)` and `b` is `(k, n)`, both column-major.
    let a: Vec<f64> = (0..m * k)
        .map(|index| (index % 11) as f64 * 0.5 - 2.0)
        .collect();
    let b: Vec<f64> = (0..k * n).map(|index| (index % 7) as f64 - 1.5).collect();
    let a_buf = runtime.upload_bytes(&bytes(&a)).expect("a");
    let b_buf = runtime.upload_bytes(&bytes(&b)).expect("b");
    let d_buf = runtime
        .upload_bytes(&bytes(&vec![0.0f64; m * n]))
        .expect("d");
    let mut cutensor = Cutensor::new(&runtime).expect("cuTENSOR");
    let request = Contraction {
        dtype: DType::F64,
        a: Operand {
            buffer: &a_buf,
            modes: &[0, 1],
            extents: &[m as i64, k as i64],
            strides: &[1, m as i64],
        },
        b: Operand {
            buffer: &b_buf,
            modes: &[1, 2],
            extents: &[k as i64, n as i64],
            strides: &[1, k as i64],
        },
        c: None,
        d: Operand {
            buffer: &d_buf,
            modes: &[0, 2],
            extents: &[m as i64, n as i64],
            strides: &[1, m as i64],
        },
    };
    cutensor.contract(&request).expect("contract");
    let observed = values(&runtime.download_bytes(&d_buf).expect("download"));
    let mut expected = vec![0.0f64; m * n];
    for column in 0..n {
        for row in 0..m {
            let mut sum = 0.0;
            for inner in 0..k {
                sum += a[row + inner * m] * b[inner + column * k];
            }
            expected[row + column * m] = sum;
        }
    }
    assert_eq!(observed, expected);
    assert_eq!(cutensor.cold_plans(), 1);
    assert_eq!(cutensor.warm_hits(), 0);
}

#[test]
fn a_repeated_signature_reuses_the_plan() {
    let Some(runtime) = runtime() else { return };
    let (m, n, k) = (9usize, 8usize, 7usize);
    let a = runtime
        .upload_bytes(&bytes(&vec![1.0f64; m * k]))
        .expect("a");
    let b = runtime
        .upload_bytes(&bytes(&vec![2.0f64; k * n]))
        .expect("b");
    let d = runtime
        .upload_bytes(&bytes(&vec![0.0f64; m * n]))
        .expect("d");
    let mut cutensor = Cutensor::new(&runtime).expect("cuTENSOR");
    let request = Contraction {
        dtype: DType::F64,
        a: Operand {
            buffer: &a,
            modes: &[0, 1],
            extents: &[m as i64, k as i64],
            strides: &[1, m as i64],
        },
        b: Operand {
            buffer: &b,
            modes: &[1, 2],
            extents: &[k as i64, n as i64],
            strides: &[1, k as i64],
        },
        c: None,
        d: Operand {
            buffer: &d,
            modes: &[0, 2],
            extents: &[m as i64, n as i64],
            strides: &[1, m as i64],
        },
    };
    for _ in 0..3 {
        cutensor.contract(&request).expect("contract");
    }
    assert_eq!(cutensor.cold_plans(), 1, "one plan for one signature");
    assert_eq!(cutensor.warm_hits(), 2, "two later calls hit the plan");
    assert_eq!(cutensor.cached_plans(), 1);
    let observed = values(&runtime.download_bytes(&d).expect("download"));
    assert_eq!(observed, vec![2.0 * k as f64; m * n]);
}

#[test]
fn a_permuted_destination_mode_matches_the_host() {
    let Some(runtime) = runtime() else { return };
    let (m, n, k) = (5usize, 4usize, 6usize);
    let a: Vec<f64> = (0..m * k).map(|index| index as f64 * 0.25).collect();
    let b: Vec<f64> = (0..k * n).map(|index| index as f64 - 3.0).collect();
    let a_buf = runtime.upload_bytes(&bytes(&a)).expect("a");
    let b_buf = runtime.upload_bytes(&bytes(&b)).expect("b");
    // `d` is laid out `(n, m)`: modes `[2, 0]` with strides `[1, n]`.
    let d_buf = runtime
        .upload_bytes(&bytes(&vec![0.0f64; m * n]))
        .expect("d");
    let mut cutensor = Cutensor::new(&runtime).expect("cuTENSOR");
    cutensor
        .contract(&Contraction {
            dtype: DType::F64,
            a: Operand {
                buffer: &a_buf,
                modes: &[0, 1],
                extents: &[m as i64, k as i64],
                strides: &[1, m as i64],
            },
            b: Operand {
                buffer: &b_buf,
                modes: &[1, 2],
                extents: &[k as i64, n as i64],
                strides: &[1, k as i64],
            },
            c: None,
            d: Operand {
                buffer: &d_buf,
                modes: &[2, 0],
                extents: &[n as i64, m as i64],
                strides: &[1, n as i64],
            },
        })
        .expect("contract");
    let observed = values(&runtime.download_bytes(&d_buf).expect("download"));
    let mut expected = vec![0.0f64; m * n];
    for column in 0..n {
        for row in 0..m {
            let mut sum = 0.0;
            for inner in 0..k {
                sum += a[row + inner * m] * b[inner + column * k];
            }
            // `d`'s element `(n_index = column, m_index = row)` sits at `column + row * n`.
            expected[column + row * n] = sum;
        }
    }
    assert_eq!(observed, expected);
}

#[test]
fn disagreeing_operand_lists_are_refused() {
    let Some(runtime) = runtime() else { return };
    let buffer = runtime.upload_bytes(&bytes(&[0.0f64; 16])).expect("buffer");
    let mut cutensor = Cutensor::new(&runtime).expect("cuTENSOR");
    assert!(cutensor
        .contract(&Contraction {
            dtype: DType::F64,
            a: Operand {
                buffer: &buffer,
                modes: &[0, 1],
                extents: &[4],
                strides: &[1, 4],
            },
            b: Operand {
                buffer: &buffer,
                modes: &[1, 2],
                extents: &[4, 4],
                strides: &[1, 4],
            },
            c: None,
            d: Operand {
                buffer: &buffer,
                modes: &[0, 2],
                extents: &[4, 4],
                strides: &[1, 4],
            },
        })
        .is_err());
}
