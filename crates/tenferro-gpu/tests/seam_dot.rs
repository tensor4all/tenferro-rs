#![cfg(feature = "cuda-next")]

//! U5 tail: `TensorDot` on the new engine — plain, batched, conjugated and accumulating
//! contractions against host references (issue #2208 U5).

use tenferro_gpu::cuda_next::{
    download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine,
};
use tenferro_tensor::{
    Complex64, ContractionScalar, DotGeneralAccumulation, DotGeneralConfig, Tensor, TensorDot,
    TensorRead, TensorWrite,
};
use tinyvec::TinyVec;

fn engine() -> Option<CudaEngine> {
    if !gpu_available() {
        return None;
    }
    Some(CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine"))
}

fn device(engine: &CudaEngine, shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    let host = Tensor::from_vec_col_major(shape, data).expect("host tensor");
    upload_tensor(engine.runtime(), &host).expect("upload")
}

fn read(engine: &CudaEngine, tensor: &Tensor) -> Vec<f64> {
    download_tensor(engine.runtime(), tensor)
        .expect("download")
        .as_slice::<f64>()
        .expect("f64 payload")
        .to_vec()
}

fn dims(values: &[usize]) -> TinyVec<[usize; 4]> {
    values.iter().copied().collect()
}

fn plain(
    lhs_contracting: &[usize],
    rhs_contracting: &[usize],
    lhs_batch: &[usize],
    rhs_batch: &[usize],
) -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: dims(lhs_contracting),
        rhs_contracting_dims: dims(rhs_contracting),
        lhs_batch_dims: dims(lhs_batch),
        rhs_batch_dims: dims(rhs_batch),
    }
}

#[test]
fn a_matrix_product_matches_the_host() {
    let Some(mut engine) = engine() else { return };
    let (m, n, k) = (7usize, 5usize, 3usize);
    // `a` is `(m, k)`, `b` is `(k, n)`, both column-major.
    let a: Vec<f64> = (0..m * k).map(|index| (index % 5) as f64 - 2.0).collect();
    let b: Vec<f64> = (0..k * n).map(|index| (index % 3) as f64 * 0.5).collect();
    let lhs = device(&engine, vec![m, k], a.clone());
    let rhs = device(&engine, vec![k, n], b.clone());
    let product = engine
        .dot_general_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &plain(&[1], &[0], &[], &[]),
        )
        .expect("dot_general");
    assert_eq!(product.shape(), &[m, n]);
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
    assert_eq!(read(&engine, &product), expected);
}

#[test]
fn a_batched_product_matches_the_host() {
    let Some(mut engine) = engine() else { return };
    let (batches, m, n, k) = (2usize, 2usize, 3usize, 2usize);
    // `a` is `(batches, m, k)`, `b` is `(batches, k, n)`, column-major.
    let a: Vec<f64> = (0..batches * m * k)
        .map(|index| index as f64 * 0.25 + 1.0)
        .collect();
    let b: Vec<f64> = (0..batches * k * n)
        .map(|index| index as f64 * 0.5 - 1.0)
        .collect();
    let lhs = device(&engine, vec![batches, m, k], a.clone());
    let rhs = device(&engine, vec![batches, k, n], b.clone());
    let product = engine
        .dot_general_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &plain(&[2], &[1], &[0], &[0]),
        )
        .expect("batched dot_general");
    // `dot_general_output_shape` lists the free axes first and the batch axes last.
    assert_eq!(product.shape(), &[m, n, batches]);
    let mut expected = vec![0.0f64; m * n * batches];
    for batch in 0..batches {
        for column in 0..n {
            for row in 0..m {
                let mut sum = 0.0;
                for inner in 0..k {
                    sum += a[batch + row * batches + inner * batches * m]
                        * b[batch + inner * batches + column * batches * k];
                }
                expected[batch * (m * n) + row + column * m] = sum;
            }
        }
    }
    assert_eq!(read(&engine, &product), expected);
}

#[test]
fn an_accumulating_dot_adds_scaled_product_to_the_output() {
    let Some(mut engine) = engine() else { return };
    let (m, n, k) = (2usize, 2usize, 2usize);
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 0.0, 0.0, 1.0];
    let lhs = device(&engine, vec![m, k], a);
    let rhs = device(&engine, vec![k, n], b);
    let mut out = device(&engine, vec![m, n], vec![10.0, 20.0, 30.0, 40.0]);
    engine
        .dot_general_read_into_accum(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &plain(&[1], &[0], &[], &[]),
            DotGeneralAccumulation {
                lhs_conj: false,
                rhs_conj: false,
                alpha: ContractionScalar::F64(1.0),
                beta: ContractionScalar::F64(2.0),
            },
            TensorWrite::from_tensor(&mut out),
        )
        .expect("accumulate");
    // `b` is the identity, so `out = a + 2 * out` elementwise.
    assert_eq!(read(&engine, &out), vec![21.0, 42.0, 63.0, 84.0]);
}

#[test]
fn a_conjugated_left_operand_is_conjugated_before_the_product() {
    let Some(mut engine) = engine() else { return };
    let lhs_host = Tensor::from_vec_col_major(
        vec![1, 2],
        vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
    )
    .expect("lhs");
    let rhs_host = Tensor::from_vec_col_major(
        vec![2, 1],
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
    )
    .expect("rhs");
    let lhs = upload_tensor(engine.runtime(), &lhs_host).expect("upload lhs");
    let rhs = upload_tensor(engine.runtime(), &rhs_host).expect("upload rhs");
    let product = engine
        .dot_general_with_conj_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &plain(&[1], &[0], &[], &[]),
            true,
            false,
        )
        .expect("conjugated dot_general");
    assert_eq!(product.shape(), &[1, 1]);
    let observed = download_tensor(engine.runtime(), &product).expect("download");
    let observed = observed.as_slice::<Complex64>().expect("c64 payload");
    // conj(1 + i) * 1 + conj(2 - i) * i = (1 - i) + (-1 + 2i) = i.
    assert_eq!(observed, &[Complex64::new(0.0, 1.0)]);
}

#[test]
fn a_mispaired_configuration_is_refused() {
    let Some(mut engine) = engine() else { return };
    let lhs = device(&engine, vec![2, 3], vec![1.0; 6]);
    let rhs = device(&engine, vec![2, 4], vec![1.0; 8]);
    // The contracting extents disagree (3 against 2), and the batch extents differ too.
    assert!(engine
        .dot_general_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &plain(&[1], &[0], &[], &[]),
        )
        .is_err());
    // An axis past the operand rank is refused rather than reinterpreted.
    assert!(engine
        .dot_general_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &plain(&[2], &[0], &[], &[]),
        )
        .is_err());
}

#[test]
fn an_axis_holding_two_roles_is_refused() {
    let Some(mut engine) = engine() else { return };
    let lhs = device(&engine, vec![2, 2], vec![1.0; 4]);
    let rhs = device(&engine, vec![2, 2], vec![1.0; 4]);
    // Axis 0 of `lhs` is both a batch and a contracting axis; the shared validator refuses
    // it instead of letting one role silently overwrite the other.
    assert!(engine
        .dot_general_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &plain(&[0], &[1], &[0], &[0]),
        )
        .is_err());
}

#[test]
fn an_empty_contraction_yields_zeros_or_the_scaled_output() {
    let Some(mut engine) = engine() else { return };
    let lhs = device(&engine, vec![2, 0], Vec::new());
    let rhs = device(&engine, vec![0, 3], Vec::new());
    let product = engine
        .dot_general_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &plain(&[1], &[0], &[], &[]),
        )
        .expect("empty contraction");
    assert_eq!(product.shape(), &[2, 3]);
    assert_eq!(read(&engine, &product), vec![0.0; 6]);

    // Accumulating with an empty product is `beta * out`, which the surface scales in place.
    let mut out = device(&engine, vec![2, 3], vec![1.0; 6]);
    engine
        .dot_general_read_into_accum(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &plain(&[1], &[0], &[], &[]),
            DotGeneralAccumulation {
                lhs_conj: false,
                rhs_conj: false,
                alpha: ContractionScalar::F64(1.0),
                beta: ContractionScalar::F64(3.0),
            },
            TensorWrite::from_tensor(&mut out),
        )
        .expect("scaled accumulation");
    assert_eq!(read(&engine, &out), vec![3.0; 6]);
}

#[test]
fn an_empty_contraction_zeroes_an_existing_destination() {
    let Some(mut engine) = engine() else { return };
    let lhs = device(&engine, vec![2, 0], Vec::new());
    let rhs = device(&engine, vec![0, 3], Vec::new());
    // `read_into` overwrites, so a prefilled destination must come back zero even though the
    // contraction has no element to compute.
    let mut out = device(&engine, vec![2, 3], vec![7.0; 6]);
    engine
        .dot_general_read_into(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &plain(&[1], &[0], &[], &[]),
            TensorWrite::from_tensor(&mut out),
        )
        .expect("empty overwrite");
    assert_eq!(read(&engine, &out), vec![0.0; 6]);

    // A zero-beta accumulation also overwrites, whatever the destination held.
    let mut out = device(&engine, vec![2, 3], vec![7.0; 6]);
    engine
        .dot_general_read_into_accum(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &plain(&[1], &[0], &[], &[]),
            DotGeneralAccumulation {
                lhs_conj: false,
                rhs_conj: false,
                alpha: ContractionScalar::F64(1.0),
                beta: ContractionScalar::F64(0.0),
            },
            TensorWrite::from_tensor(&mut out),
        )
        .expect("empty zero-beta accumulation");
    assert_eq!(read(&engine, &out), vec![0.0; 6]);
}
