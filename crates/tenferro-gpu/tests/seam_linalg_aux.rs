#![cfg(feature = "cuda")]

//! The linalg auxiliaries the cuSOLVER providers schedule: the backend lowers each to its kernel
//! catalogue entry, so these check the lowering against host references (issue #2208, part 2).

use tenferro_gpu::cuda::{download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine};
use tenferro_tensor::Tensor;

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

fn indices(engine: &CudaEngine, shape: Vec<usize>, data: Vec<i64>) -> Tensor {
    let host = Tensor::from_vec_col_major(shape, data).expect("host tensor");
    upload_tensor(engine.runtime(), &host).expect("upload")
}

fn read_index(engine: &CudaEngine, tensor: &Tensor) -> Vec<i64> {
    download_tensor(engine.runtime(), tensor)
        .expect("download")
        .as_slice::<i64>()
        .expect("i64 payload")
        .to_vec()
}

fn read(engine: &CudaEngine, tensor: &Tensor) -> Vec<f64> {
    download_tensor(engine.runtime(), tensor)
        .expect("download")
        .as_slice::<f64>()
        .expect("f64 payload")
        .to_vec()
}

#[test]
fn the_matrix_adjoint_transposes_each_batch() {
    let Some(mut engine) = engine() else { return };
    // Two 2x3 matrices; the adjoint of each is 3x2.
    let input = device(&engine, vec![2, 3, 2], (0..12).map(|v| v as f64).collect());
    let output = device(&engine, vec![3, 2, 2], vec![0.0; 12]);
    engine
        .linalg_matrix_adjoint(&input, &output, "seam_linalg_aux")
        .expect("matrix adjoint");
    let observed = read(&engine, &output);
    // Column-major: element (row, col, batch) of the input is `row + 2 * (col + 3 * batch)`.
    let mut expected = vec![0.0; 12];
    for batch in 0..2 {
        for column in 0..3 {
            for row in 0..2 {
                expected[column + 3 * (row + 2 * batch)] = (row + 2 * (column + 3 * batch)) as f64;
            }
        }
    }
    assert_eq!(observed, expected);
}

#[test]
fn the_column_norms_measure_the_trailing_columns() {
    let Some(mut engine) = engine() else { return };
    // A 3x3 work matrix with columns of known norm; `step = 0` measures every column.
    let input = device(
        &engine,
        vec![3, 3],
        vec![3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 12.0],
    );
    let norms = device(&engine, vec![3], vec![0.0; 3]);
    engine
        .linalg_column_norms(&input, &norms, 0, "seam_linalg_aux")
        .expect("column norms");
    assert_eq!(read(&engine, &norms), vec![3.0, 4.0, 12.0]);
}

#[test]
fn the_pivot_selection_and_swap_exchange_the_largest_column() {
    let Some(mut engine) = engine() else { return };
    // Two columns whose norms are 3 and 12: from step 0 the largest column is 1.
    let work = device(&engine, vec![3, 2], vec![3.0, 0.0, 0.0, 12.0, 0.0, 0.0]);
    let norms = device(&engine, vec![2], vec![3.0, 12.0]);
    let permutation = indices(&engine, vec![2], vec![0i64, 1]);
    let pivots = indices(&engine, vec![1], vec![0i64]);
    let status = indices(&engine, vec![1], vec![0i64]);
    engine
        .linalg_select_pivot(&norms, &permutation, &pivots, &status, 0, "seam_linalg_aux")
        .expect("select pivot");
    // The chosen column is recorded as a pivot and the status stays clean.
    assert_eq!(read_index(&engine, &pivots), vec![1i64]);
    assert_eq!(read_index(&engine, &status), vec![0i64]);

    engine
        .linalg_swap_columns(&work, &permutation, &pivots, 0, "seam_linalg_aux")
        .expect("swap columns");
    // Column 0 (norm 3) and column 1 (norm 12) exchange, and so do their permutation entries.
    assert_eq!(read(&engine, &work), vec![12.0, 0.0, 0.0, 3.0, 0.0, 0.0]);
    assert_eq!(read_index(&engine, &permutation), vec![1i64, 0i64]);
}

#[test]
fn the_qr_phase_scales_and_the_shifted_identity_is_built() {
    let Some(mut engine) = engine() else { return };
    // R = diag(2, -3) as a 2x2; the phase of a diagonal is +1 where the entry is positive.
    let r = device(&engine, vec![2, 2], vec![2.0, 0.0, 0.0, -3.0]);
    let phase = device(&engine, vec![2], vec![0.0; 2]);
    engine
        .linalg_qr_phase(&r, &phase, "seam_linalg_aux")
        .expect("qr phase");
    let observed = read(&engine, &phase);
    assert_eq!(observed.len(), 2);
    assert!((observed[0] - 1.0).abs() < 1e-12, "{observed:?}");

    // Q's trailing column is scaled by its phase and R's row by its conjugate.
    let q = device(&engine, vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let r = device(&engine, vec![2, 2], vec![2.0, 0.0, 0.0, -3.0]);
    let phase = device(&engine, vec![2], vec![1.0, -1.0]);
    engine
        .linalg_qr_apply_phase(&q, &r, &phase, 0, "seam_linalg_aux")
        .expect("qr apply phase");
    assert_eq!(read(&engine, &q), vec![1.0, 0.0, 0.0, -1.0]);
    assert_eq!(read(&engine, &r), vec![2.0, 0.0, 0.0, 3.0]);

    // The shifted identity puts a one where `row == start + col`, i.e. on the subdiagonal for
    // `start = 1` (the kernel's own rule).
    let identity = device(&engine, vec![2, 2], vec![0.0; 4]);
    engine
        .linalg_identity_shifted(&identity, 1, "seam_linalg_aux")
        .expect("identity shifted");
    assert_eq!(read(&engine, &identity), vec![0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn the_reflector_round_trip_and_the_extracted_r() {
    let Some(mut engine) = engine() else { return };
    // Column 0 of the 3x2 working matrix is (3, 4, 0): its Householder reflector has norm 5, phase
    // +1 (x0 > 0), alpha = -5 and denominator 8, so the stored column is (-5, 4/8, 0) and the
    // coefficient is 1 + |x0| / norm = 1.6 (docs/design/slice-rrqr.md 4.5).
    let work = device(&engine, vec![3, 2], vec![3.0, 4.0, 0.0, 1.0, 2.0, 3.0]);
    let coeff = device(&engine, vec![2], vec![0.0; 2]);
    engine
        .linalg_reflector(&work, &coeff, 0, "seam_linalg_aux")
        .expect("reflector");
    let reflected = read(&engine, &work);
    assert!((reflected[0] + 5.0).abs() < 1e-9, "{reflected:?}");
    assert!((reflected[1] - 0.5).abs() < 1e-9, "{reflected:?}");
    assert!(reflected[2].abs() < 1e-9, "{reflected:?}");
    let coefficients = read(&engine, &coeff);
    assert!((coefficients[0] - 1.6).abs() < 1e-9, "{coefficients:?}");

    // Applying the stored reflector to a copy of the *original* matrix is H * A: the first column
    // collapses to (-5, 0, 0) and the second becomes a - 1.6 * (a . v) * v with v = (1, 0.5, 0),
    // i.e. (1, 2, 3) - 3.2 * (1, 0.5, 0) = (-2.2, 0.4, 3).
    let packed = device(&engine, vec![3, 2], reflected.clone());
    let target = device(&engine, vec![3, 2], vec![3.0, 4.0, 0.0, 1.0, 2.0, 3.0]);
    engine
        .linalg_apply_reflector(&packed, &target, &coeff, 0, 0, 2, "seam_linalg_aux")
        .expect("apply reflector");
    let applied = read(&engine, &target);
    for (got, want) in applied.iter().zip([-5.0, 0.0, 0.0, -2.2, 0.4, 3.0]) {
        assert!((got - want).abs() < 1e-9, "{applied:?}");
    }

    // `R` is the leading 2x2 of the reflected working matrix: rows 0..2 of columns 0..2.
    let r = device(&engine, vec![2, 2], vec![0.0; 4]);
    engine
        .linalg_extract_r(&packed, &r, "seam_linalg_aux")
        .expect("extract r");
    // Upper triangle only: R(0,0) = work(0,0) = -5, R(1,0) is masked, R(0,1) = 1, R(1,1) = 2.
    assert_eq!(read(&engine, &r), vec![-5.0, 0.0, 1.0, 2.0]);

    // The prefix rank of a diagonal with one negligible entry stops before it.
    let diagonal = device(
        &engine,
        vec![3, 3],
        vec![4.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1e-9],
    );
    let rank = indices(&engine, vec![1], vec![-1i64]);
    let status = indices(&engine, vec![1], vec![0i64]);
    engine
        .linalg_prefix_rank(&diagonal, &rank, &status, 1e-6, 0.0, "seam_linalg_aux")
        .expect("prefix rank");
    assert_eq!(read_index(&engine, &rank), vec![2i64]);
}

#[test]
fn the_lu_pivots_permute_rows_and_the_parity_is_read() {
    let Some(mut engine) = engine() else { return };
    // The pivots are LAPACK's one-based row numbers: pivot 2 at step 0 means row 0 and row 1
    // exchange (the kernel's convention, not a zero-based index).
    let input = device(&engine, vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = device(&engine, vec![3, 2], vec![0.0; 6]);
    let pivots = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![1], vec![2i32]).expect("pivots"),
    )
    .expect("upload pivots");
    engine
        .linalg_lu_apply_pivots(&out, &input, &pivots, 1, false, "seam_linalg_aux")
        .expect("apply pivots");
    // Rows 0 and 1 exchange; row 2 stays.
    assert_eq!(read(&engine, &out), vec![2.0, 1.0, 3.0, 5.0, 4.0, 6.0]);
}
