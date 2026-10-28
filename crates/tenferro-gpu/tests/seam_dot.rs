#![cfg(feature = "cuda")]

//! U5 tail: `TensorDot` on the new engine — plain, batched, conjugated and accumulating
//! contractions against host references (issue #2208 U5).

use tenferro_gpu::cuda::{download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine};
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
#[ignore = "requires CUDA"]
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
#[ignore = "requires CUDA"]
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
#[ignore = "requires CUDA"]
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
#[ignore = "requires CUDA"]
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
#[ignore = "requires CUDA"]
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
#[ignore = "requires CUDA"]
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
#[ignore = "requires CUDA"]
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
#[ignore = "requires CUDA"]
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

// --- Write-side views: a contraction into a borrowed destination region (#2223 item 3) ---

use tenferro_tensor::TensorViewMut;

/// The three destination flavours the write-side tests use: a transposed stride pair, a
/// nonzero offset and a negative stride, all over a `6 x 5` column-major root of thirty
/// sentinel entries. Each covers six entries, so the other twenty-four must survive untouched.
fn write_view_flavours() -> Vec<(&'static str, Vec<usize>, Vec<isize>, isize)> {
    vec![
        ("transposed", vec![2, 3], vec![5, 1], 2),
        ("offset", vec![2, 3], vec![1, 6], 7),
        ("negative", vec![2, 3], vec![-1, 6], 5),
    ]
}

fn sentinel(index: usize) -> f64 {
    100.0 + index as f64
}

/// A `6 x 5` column-major device tensor of sentinels.
fn write_root(engine: &CudaEngine) -> Tensor {
    device(engine, vec![6, 5], (0..30).map(sentinel).collect())
}

/// The root's linear indices in logical column-major order for one region.
fn region_indices(shape: &[usize], strides: &[isize], offset: isize) -> Vec<usize> {
    let count: usize = shape.iter().product();
    (0..count)
        .map(|flat| {
            let mut remaining = flat;
            let mut index = offset;
            for (&extent, &stride) in shape.iter().zip(strides) {
                let coordinate = remaining % extent;
                remaining /= extent;
                index += coordinate as isize * stride;
            }
            usize::try_from(index).expect("the region starts at a nonnegative element")
        })
        .collect()
}

/// The root a write should have produced: `written` at the region and the sentinel elsewhere.
fn expected_root(region: &[usize], written: &[f64]) -> Vec<f64> {
    assert_eq!(region.len(), written.len());
    let mut root: Vec<f64> = (0..30).map(sentinel).collect();
    for (&index, &value) in region.iter().zip(written) {
        root[index] = value;
    }
    root
}

/// Write through one destination region of `root`, then read the whole root back.
fn write_through_view(
    engine: &mut CudaEngine,
    root: &mut Tensor,
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    write: impl FnOnce(&mut CudaEngine, TensorWrite<'_>) -> tenferro_tensor::Result<()>,
) -> Vec<f64> {
    {
        let region = root
            .as_typed_mut::<f64>()
            .expect("f64 payload")
            .backend_region_view_mut(shape, strides, offset)
            .expect("mutable region view");
        write(engine, TensorWrite::from_view(TensorViewMut::F64(region)))
            .expect("write through the destination view");
    }
    read(engine, root)
}

/// The `[2, 3]` column-major product of a `[2, 2]` and a `[2, 3]` operand.
fn product(lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; 6];
    for column in 0..3 {
        for row in 0..2 {
            out[row + column * 2] = lhs[row] * rhs[column * 2] + lhs[row + 2] * rhs[column * 2 + 1];
        }
    }
    out
}

/// The `[2, 2]` and `[2, 3]` operands and the product they must produce.
fn operands(engine: &CudaEngine) -> (Tensor, Tensor, Vec<f64>) {
    let a: Vec<f64> = (1..=4).map(|index| index as f64).collect();
    let b: Vec<f64> = vec![1.0, 0.0, 0.5, 2.0, -1.0, 3.0];
    let expected = product(&a, &b);
    let lhs = device(engine, vec![2, 2], a);
    let rhs = device(engine, vec![2, 3], b);
    (lhs, rhs, expected)
}

fn accumulation(alpha: f64, beta: f64) -> DotGeneralAccumulation {
    DotGeneralAccumulation {
        lhs_conj: false,
        rhs_conj: false,
        alpha: ContractionScalar::F64(alpha),
        beta: ContractionScalar::F64(beta),
    }
}

#[test]
#[ignore = "requires CUDA"]
fn dot_general_read_into_writes_a_strided_destination_view() {
    let Some(mut engine) = engine() else { return };
    for (flavour, shape, strides, offset) in write_view_flavours() {
        let mut root = write_root(&engine);
        let (lhs, rhs, expected) = operands(&engine);
        let observed = write_through_view(
            &mut engine,
            &mut root,
            shape.clone(),
            strides.clone(),
            offset,
            |engine, out| {
                engine.dot_general_read_into(
                    TensorRead::from_tensor(&lhs),
                    TensorRead::from_tensor(&rhs),
                    &plain(&[1], &[0], &[], &[]),
                    out,
                )
            },
        );
        let region = region_indices(&shape, &strides, offset);
        assert_eq!(
            observed,
            expected_root(&region, &expected),
            "the {flavour} destination"
        );
    }
}

#[test]
#[ignore = "requires CUDA"]
fn dot_general_read_into_accum_scales_a_strided_destination_view() {
    let Some(mut engine) = engine() else { return };
    for (flavour, shape, strides, offset) in write_view_flavours() {
        let mut root = write_root(&engine);
        let (lhs, rhs, expected) = operands(&engine);
        let region = region_indices(&shape, &strides, offset);
        // `out = 0.5 * A B + 2 * out`, where the old destination is the root's sentinel.
        let scaled: Vec<f64> = expected
            .iter()
            .zip(&region)
            .map(|(&product, &index)| 0.5 * product + 2.0 * sentinel(index))
            .collect();
        let observed = write_through_view(
            &mut engine,
            &mut root,
            shape.clone(),
            strides.clone(),
            offset,
            |engine, out| {
                engine.dot_general_read_into_accum(
                    TensorRead::from_tensor(&lhs),
                    TensorRead::from_tensor(&rhs),
                    &plain(&[1], &[0], &[], &[]),
                    accumulation(0.5, 2.0),
                    out,
                )
            },
        );
        assert_eq!(
            observed,
            expected_root(&region, &scaled),
            "the {flavour} destination"
        );
    }
}

#[test]
#[ignore = "requires CUDA"]
fn a_zero_beta_accumulation_does_not_read_a_nan_destination_view() {
    let Some(mut engine) = engine() else { return };
    for (flavour, shape, strides, offset) in write_view_flavours() {
        let region = region_indices(&shape, &strides, offset);
        // A NaN in every destination entry: `beta * NaN` would survive an accumulation that
        // read the old value, which a zero beta must not do (BLAS's "do not read C").
        let mut data: Vec<f64> = (0..30).map(sentinel).collect();
        for &index in &region {
            data[index] = f64::NAN;
        }
        let mut root = device(&engine, vec![6, 5], data);
        let (lhs, rhs, expected) = operands(&engine);
        let observed = write_through_view(
            &mut engine,
            &mut root,
            shape.clone(),
            strides.clone(),
            offset,
            |engine, out| {
                engine.dot_general_read_into_accum(
                    TensorRead::from_tensor(&lhs),
                    TensorRead::from_tensor(&rhs),
                    &plain(&[1], &[0], &[], &[]),
                    accumulation(1.0, 0.0),
                    out,
                )
            },
        );
        assert_eq!(
            observed,
            expected_root(&region, &expected),
            "the {flavour} destination kept a NaN through a zero beta"
        );
    }
}

#[test]
#[ignore = "requires CUDA"]
fn an_empty_contraction_keeps_the_beta_semantics_on_a_destination_view() {
    let Some(mut engine) = engine() else { return };
    for (flavour, shape, strides, offset) in write_view_flavours() {
        let region = region_indices(&shape, &strides, offset);
        let lhs = device(&engine, vec![2, 0], Vec::new());
        let rhs = device(&engine, vec![0, 3], Vec::new());
        let zeros = vec![0.0; region.len()];

        // A nonzero beta scales what the region held; nothing outside it moves.
        let scaled: Vec<f64> = region.iter().map(|&index| 3.0 * sentinel(index)).collect();
        let mut root = write_root(&engine);
        let observed = write_through_view(
            &mut engine,
            &mut root,
            shape.clone(),
            strides.clone(),
            offset,
            |engine, out| {
                engine.dot_general_read_into_accum(
                    TensorRead::from_tensor(&lhs),
                    TensorRead::from_tensor(&rhs),
                    &plain(&[1], &[0], &[], &[]),
                    accumulation(1.0, 3.0),
                    out,
                )
            },
        );
        assert_eq!(
            observed,
            expected_root(&region, &scaled),
            "the {flavour} destination"
        );

        // A zero beta overwrites the region with the empty product's zeros.
        let mut root = write_root(&engine);
        let observed = write_through_view(
            &mut engine,
            &mut root,
            shape.clone(),
            strides.clone(),
            offset,
            |engine, out| {
                engine.dot_general_read_into_accum(
                    TensorRead::from_tensor(&lhs),
                    TensorRead::from_tensor(&rhs),
                    &plain(&[1], &[0], &[], &[]),
                    accumulation(1.0, 0.0),
                    out,
                )
            },
        );
        assert_eq!(
            observed,
            expected_root(&region, &zeros),
            "the {flavour} destination"
        );

        // The overwrite form leaves zeros as well.
        let mut root = write_root(&engine);
        let observed = write_through_view(
            &mut engine,
            &mut root,
            shape.clone(),
            strides.clone(),
            offset,
            |engine, out| {
                engine.dot_general_read_into(
                    TensorRead::from_tensor(&lhs),
                    TensorRead::from_tensor(&rhs),
                    &plain(&[1], &[0], &[], &[]),
                    out,
                )
            },
        );
        assert_eq!(
            observed,
            expected_root(&region, &zeros),
            "the {flavour} destination"
        );
    }
}
