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
        alpha: (1.0, 0.0),
        beta: (0.0, 0.0),
        conj_a: false,
        conj_b: false,
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
        alpha: (1.0, 0.0),
        beta: (0.0, 0.0),
        conj_a: false,
        conj_b: false,
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
            alpha: (1.0, 0.0),
            beta: (0.0, 0.0),
            conj_a: false,
            conj_b: false,
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
            alpha: (1.0, 0.0),
            beta: (0.0, 0.0),
            conj_a: false,
            conj_b: false,
        })
        .is_err());
}

/// Little-endian bytes of `f32` values.
fn bytes32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

/// Little-endian bytes of `(re, im)` pairs, as `cuFloatComplex` lays them out.
fn bytes_complex32(values: &[(f32, f32)]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|(re, im)| [re.to_ne_bytes(), im.to_ne_bytes()].concat())
        .collect()
}

#[test]
fn an_f32_contraction_reads_its_scalars_at_the_float_width() {
    let Some(runtime) = runtime() else { return };
    let (m, n, k) = (5usize, 4usize, 3usize);
    let a: Vec<f32> = (0..m * k).map(|index| index as f32 * 0.5 - 1.0).collect();
    let b: Vec<f32> = (0..k * n).map(|index| index as f32 - 2.0).collect();
    let a_buf = runtime.upload_bytes(&bytes32(&a)).expect("a");
    let b_buf = runtime.upload_bytes(&bytes32(&b)).expect("b");
    let d_buf = runtime
        .upload_bytes(&bytes32(&vec![0.0f32; m * n]))
        .expect("d");
    let mut cutensor = Cutensor::new(&runtime).expect("cuTENSOR");
    cutensor
        .contract(&Contraction {
            dtype: DType::F32,
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
            alpha: (1.0, 0.0),
            beta: (0.0, 0.0),
            conj_a: false,
            conj_b: false,
        })
        .expect("contract");
    let observed: Vec<f32> = runtime
        .download_bytes(&d_buf)
        .expect("download")
        .chunks_exact(4)
        .map(|word| f32::from_ne_bytes(word.try_into().expect("4 bytes")))
        .collect();
    let mut expected = vec![0.0f32; m * n];
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
}

#[test]
fn a_complex_contraction_scales_by_its_complex_alpha() {
    let Some(runtime) = runtime() else { return };
    let (m, n, k) = (2usize, 2usize, 2usize);
    // Identity-like operands with one non-trivial entry, so `i *` is visible on the diagonal.
    let a = [(1.0f32, 0.0f32), (0.0, 1.0), (2.0, 0.0), (1.0, -1.0)];
    let b = [(1.0f32, 0.0f32), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)];
    let a_buf = runtime.upload_bytes(&bytes_complex32(&a)).expect("a");
    let b_buf = runtime.upload_bytes(&bytes_complex32(&b)).expect("b");
    let d_buf = runtime
        .upload_bytes(&bytes_complex32(&vec![(0.0f32, 0.0f32); m * n]))
        .expect("d");
    let mut cutensor = Cutensor::new(&runtime).expect("cuTENSOR");
    // alpha = i: every result component is rotated by 90 degrees.
    cutensor
        .contract(&Contraction {
            dtype: DType::C32,
            a: Operand {
                buffer: &a_buf,
                modes: &[0, 2],
                extents: &[m as i64, k as i64],
                strides: &[1, m as i64],
            },
            b: Operand {
                buffer: &b_buf,
                modes: &[2, 1],
                extents: &[k as i64, n as i64],
                strides: &[1, k as i64],
            },
            c: None,
            d: Operand {
                buffer: &d_buf,
                modes: &[0, 1],
                extents: &[m as i64, n as i64],
                strides: &[1, m as i64],
            },
            alpha: (0.0, 1.0),
            beta: (0.0, 0.0),
            conj_a: false,
            conj_b: false,
        })
        .expect("contract");
    let observed: Vec<(f32, f32)> = runtime
        .download_bytes(&d_buf)
        .expect("download")
        .chunks_exact(8)
        .map(|word| {
            (
                f32::from_ne_bytes(word[0..4].try_into().expect("re")),
                f32::from_ne_bytes(word[4..8].try_into().expect("im")),
            )
        })
        .collect();
    // b is the identity, so `i * a` is the expected product.
    let expected = [(0.0f32, 1.0f32), (-1.0, 0.0), (0.0, 2.0), (1.0, 1.0)];
    assert_eq!(observed, expected);
}

#[test]
fn a_nonzero_beta_accumulates_into_the_addend() {
    let Some(runtime) = runtime() else { return };
    let (m, n, k) = (3usize, 2usize, 2usize);
    let a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [1.0f64, 0.0, 0.0, 1.0];
    let addend = [10.0f64, 20.0, 30.0, 40.0, 50.0, 60.0];
    let a_buf = runtime.upload_bytes(&bytes(&a)).expect("a");
    let b_buf = runtime.upload_bytes(&bytes(&b)).expect("b");
    let c_buf = runtime.upload_bytes(&bytes(&addend)).expect("c");
    let d_buf = runtime
        .upload_bytes(&bytes(&vec![0.0f64; m * n]))
        .expect("d");
    let mut cutensor = Cutensor::new(&runtime).expect("cuTENSOR");
    cutensor
        .contract(&Contraction {
            dtype: DType::F64,
            a: Operand {
                buffer: &a_buf,
                modes: &[0, 2],
                extents: &[m as i64, k as i64],
                strides: &[1, m as i64],
            },
            b: Operand {
                buffer: &b_buf,
                modes: &[2, 1],
                extents: &[k as i64, n as i64],
                strides: &[1, k as i64],
            },
            c: Some(Operand {
                buffer: &c_buf,
                modes: &[0, 1],
                extents: &[m as i64, n as i64],
                strides: &[1, m as i64],
            }),
            d: Operand {
                buffer: &d_buf,
                modes: &[0, 1],
                extents: &[m as i64, n as i64],
                strides: &[1, m as i64],
            },
            alpha: (1.0, 0.0),
            beta: (1.0, 0.0),
            conj_a: false,
            conj_b: false,
        })
        .expect("contract");
    let observed = values(&runtime.download_bytes(&d_buf).expect("download"));
    // b is the identity, so the product is `a` scaled by alpha and added to the addend.
    let expected = vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0];
    assert_eq!(observed, expected);

    // A non-zero beta without an addend is refused instead of silently ignoring the scale.
    assert!(cutensor
        .contract(&Contraction {
            dtype: DType::F64,
            a: Operand {
                buffer: &a_buf,
                modes: &[0, 2],
                extents: &[m as i64, k as i64],
                strides: &[1, m as i64],
            },
            b: Operand {
                buffer: &b_buf,
                modes: &[2, 1],
                extents: &[k as i64, n as i64],
                strides: &[1, k as i64],
            },
            c: None,
            d: Operand {
                buffer: &d_buf,
                modes: &[0, 1],
                extents: &[m as i64, n as i64],
                strides: &[1, m as i64],
            },
            alpha: (1.0, 0.0),
            beta: (1.0, 0.0),
            conj_a: false,
            conj_b: false,
        })
        .is_err());
}

#[test]
fn a_new_shape_grows_one_shared_workspace() {
    let Some(runtime) = runtime() else { return };
    let mut cutensor = Cutensor::new(&runtime).expect("cuTENSOR");
    // Two different shapes plan two plans; the handle keeps one workspace, not one each.
    for (m, n, k) in [(4usize, 3usize, 2usize), (32, 16, 8)] {
        let a = runtime
            .upload_bytes(&bytes(&vec![1.0f64; m * k]))
            .expect("a");
        let b = runtime
            .upload_bytes(&bytes(&vec![2.0f64; k * n]))
            .expect("b");
        let d = runtime
            .upload_bytes(&bytes(&vec![0.0f64; m * n]))
            .expect("d");
        cutensor
            .contract(&Contraction {
                dtype: DType::F64,
                a: Operand {
                    buffer: &a,
                    modes: &[0, 2],
                    extents: &[m as i64, k as i64],
                    strides: &[1, m as i64],
                },
                b: Operand {
                    buffer: &b,
                    modes: &[2, 1],
                    extents: &[k as i64, n as i64],
                    strides: &[1, k as i64],
                },
                c: None,
                d: Operand {
                    buffer: &d,
                    modes: &[0, 1],
                    extents: &[m as i64, n as i64],
                    strides: &[1, m as i64],
                },
                alpha: (1.0, 0.0),
                beta: (0.0, 0.0),
                conj_a: false,
                conj_b: false,
            })
            .expect("contract");
    }
    assert_eq!(cutensor.cold_plans(), 2);
    assert_eq!(cutensor.cached_plans(), 2);
}
