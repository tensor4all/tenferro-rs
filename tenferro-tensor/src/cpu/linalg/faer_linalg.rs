use faer::{diag::DiagRef, MatMut, MatRef, Par, Side};
use num_complex::Complex64;

use crate::{Buffer, LayoutOrder, Tensor, TypedTensor};

pub(crate) trait FaerLinalg: Copy + Clone {
    fn parity_one() -> Self;
    fn cholesky_2d(input: &TypedTensor<Self>) -> crate::Result<TypedTensor<Self>>;
    fn lu_2d(input: &TypedTensor<Self>) -> Vec<TypedTensor<Self>>;
    fn triangular_solve_2d(
        a: &TypedTensor<Self>,
        b: &TypedTensor<Self>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> TypedTensor<Self>;
    fn svd_2d(input: &TypedTensor<Self>) -> Vec<TypedTensor<Self>>;
    fn qr_2d(input: &TypedTensor<Self>) -> Vec<TypedTensor<Self>>;
    fn eigh_2d(input: &TypedTensor<Self>) -> Vec<TypedTensor<Self>>;
}

fn matrix_dims<T>(input: &TypedTensor<T>, op: &str) -> (usize, usize) {
    assert_eq!(input.shape.len(), 2, "{op}: expected a 2D matrix");
    (input.shape[0], input.shape[1])
}

fn square_matrix_dim<T>(input: &TypedTensor<T>, op: &str) -> usize {
    let (rows, cols) = matrix_dims(input, op);
    assert_eq!(rows, cols, "{op}: expected a square matrix");
    rows
}

fn tensor_from_vec_with_template<T: Clone, U>(
    shape: Vec<usize>,
    data: Vec<T>,
    template: &TypedTensor<U>,
) -> TypedTensor<T> {
    TypedTensor {
        buffer: Buffer::Host(data.into()),
        strides: crate::col_major_strides(&shape),
        offset: 0,
        shape,
        placement: template.placement.clone(),
    }
}

fn with_col_major_input<T, R>(input: &TypedTensor<T>, f: impl FnOnce(&TypedTensor<T>) -> R) -> R
where
    T: Copy + Default,
{
    if input.is_contiguous_col_major() {
        f(input)
    } else {
        let materialized = input
            .to_contiguous(LayoutOrder::ColumnMajor)
            .expect("linalg input materialization must succeed");
        f(&materialized)
    }
}

fn col_major_vec_from_mat<T: Copy>(mat: MatRef<'_, T>) -> Vec<T> {
    let (rows, cols) = mat.shape();
    let mut data = Vec::with_capacity(rows * cols);
    for j in 0..cols {
        for i in 0..rows {
            data.push(mat[(i, j)]);
        }
    }
    data
}

fn tensor_from_mat<T: Copy + Clone, U>(
    mat: MatRef<'_, T>,
    shape: Vec<usize>,
    template: &TypedTensor<U>,
) -> TypedTensor<T> {
    tensor_from_vec_with_template(shape, col_major_vec_from_mat(mat), template)
}

fn vec_from_diag<T: Copy>(diag: DiagRef<'_, T>) -> Vec<T> {
    let col = diag.column_vector();
    let mut data = Vec::with_capacity(col.nrows());
    for i in 0..col.nrows() {
        data.push(col[i]);
    }
    data
}

fn complex64_to_faer_slice(data: &[Complex64]) -> &[faer::c64] {
    debug_assert_eq!(
        std::mem::size_of::<Complex64>(),
        std::mem::size_of::<faer::c64>()
    );
    debug_assert_eq!(
        std::mem::align_of::<Complex64>(),
        std::mem::align_of::<faer::c64>()
    );

    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<faer::c64>(), data.len()) }
}

fn complex64_to_faer_slice_mut(data: &mut [Complex64]) -> &mut [faer::c64] {
    debug_assert_eq!(
        std::mem::size_of::<Complex64>(),
        std::mem::size_of::<faer::c64>()
    );
    debug_assert_eq!(
        std::mem::align_of::<Complex64>(),
        std::mem::align_of::<faer::c64>()
    );

    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<faer::c64>(), data.len()) }
}

fn complex_vec_from_real_diag(diag: DiagRef<'_, faer::c64>) -> Vec<Complex64> {
    let col = diag.column_vector();
    let mut data = Vec::with_capacity(col.nrows());
    for i in 0..col.nrows() {
        data.push(Complex64::new(col[i].re, 0.0));
    }
    data
}

fn complex_vec_from_diag(diag: DiagRef<'_, faer::c64>) -> Vec<Complex64> {
    let col = diag.column_vector();
    let mut data = Vec::with_capacity(col.nrows());
    for i in 0..col.nrows() {
        data.push(Complex64::new(col[i].re, col[i].im));
    }
    data
}

fn complex_vec_from_mat(mat: MatRef<'_, faer::c64>) -> Vec<Complex64> {
    let (rows, cols) = mat.shape();
    let mut data = Vec::with_capacity(rows * cols);
    for j in 0..cols {
        for i in 0..rows {
            let value = mat[(i, j)];
            data.push(Complex64::new(value.re, value.im));
        }
    }
    data
}

fn split_core_and_batch<'a, T>(
    input: &'a TypedTensor<T>,
    core_rank: usize,
    op: &str,
) -> (&'a [usize], &'a [usize]) {
    assert!(
        input.shape.len() >= core_rank,
        "{op}: expected rank >= {core_rank}"
    );
    input.shape.split_at(core_rank)
}

fn transpose_col_major_data<T: Copy>(data: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut transposed = Vec::with_capacity(data.len());
    for j in 0..rows {
        for i in 0..cols {
            transposed.push(data[j + i * rows]);
        }
    }
    transposed
}

fn batched_single<T, F>(
    input: &TypedTensor<T>,
    core_rank: usize,
    op: F,
) -> crate::Result<TypedTensor<T>>
where
    T: Clone,
    F: Fn(&TypedTensor<T>) -> crate::Result<TypedTensor<T>>,
{
    let (core_shape, batch_shape) = split_core_and_batch(input, core_rank, "batched_single");
    if batch_shape.is_empty() {
        return op(input);
    }

    let slice_size: usize = core_shape.iter().product();
    let batch_count: usize = batch_shape.iter().product();
    assert!(
        batch_count > 0,
        "batched_single: zero-sized batch dims are unsupported"
    );

    let mut out_core_shape: Option<Vec<usize>> = None;
    let mut out_data = Vec::new();

    for batch_idx in 0..batch_count {
        let start = batch_idx * slice_size;
        let end = start + slice_size;
        let batch_input = tensor_from_vec_with_template(
            core_shape.to_vec(),
            input.host_data()[start..end].to_vec(),
            input,
        );
        let batch_output = op(&batch_input)?;

        if let Some(expected_shape) = &out_core_shape {
            assert_eq!(
                batch_output.shape.as_slice(),
                expected_shape.as_slice(),
                "batched_single: output core shape mismatch across batches"
            );
        } else {
            out_data.reserve(batch_output.n_elements() * batch_count);
            out_core_shape = Some(batch_output.shape.clone());
        }

        out_data.extend_from_slice(batch_output.host_data());
    }

    let mut out_shape = out_core_shape.expect("batched_single: missing output shape");
    out_shape.extend_from_slice(batch_shape);
    Ok(tensor_from_vec_with_template(out_shape, out_data, input))
}

fn batched_multi<T, F>(input: &TypedTensor<T>, core_rank: usize, op: F) -> Vec<TypedTensor<T>>
where
    T: Clone,
    F: Fn(&TypedTensor<T>) -> Vec<TypedTensor<T>>,
{
    let (core_shape, batch_shape) = split_core_and_batch(input, core_rank, "batched_multi");
    if batch_shape.is_empty() {
        return op(input);
    }

    let slice_size: usize = core_shape.iter().product();
    let batch_count: usize = batch_shape.iter().product();
    assert!(
        batch_count > 0,
        "batched_multi: zero-sized batch dims are unsupported"
    );

    let mut out_shapes: Vec<Vec<usize>> = Vec::new();
    let mut out_data: Vec<Vec<T>> = Vec::new();

    for batch_idx in 0..batch_count {
        let start = batch_idx * slice_size;
        let end = start + slice_size;
        let batch_input = tensor_from_vec_with_template(
            core_shape.to_vec(),
            input.host_data()[start..end].to_vec(),
            input,
        );
        let batch_outputs = op(&batch_input);

        if out_shapes.is_empty() {
            out_shapes = batch_outputs
                .iter()
                .map(|tensor| tensor.shape.clone())
                .collect();
            out_data = batch_outputs
                .iter()
                .map(|tensor| Vec::with_capacity(tensor.n_elements() * batch_count))
                .collect();
        } else {
            assert_eq!(
                batch_outputs.len(),
                out_shapes.len(),
                "batched_multi: output count mismatch across batches"
            );
        }

        for (idx, batch_output) in batch_outputs.iter().enumerate() {
            assert_eq!(
                batch_output.shape.as_slice(),
                out_shapes[idx].as_slice(),
                "batched_multi: output core shape mismatch across batches"
            );
            out_data[idx].extend_from_slice(batch_output.host_data());
        }
    }

    out_shapes
        .into_iter()
        .zip(out_data)
        .map(|(mut out_shape, out_data)| {
            out_shape.extend_from_slice(batch_shape);
            tensor_from_vec_with_template(out_shape, out_data, input)
        })
        .collect()
}

fn batched_multi_convert<InT, OutT, F>(
    input: &TypedTensor<InT>,
    core_rank: usize,
    op: F,
) -> Vec<TypedTensor<OutT>>
where
    InT: Clone,
    OutT: Clone,
    F: Fn(&TypedTensor<InT>) -> Vec<TypedTensor<OutT>>,
{
    let (core_shape, batch_shape) = split_core_and_batch(input, core_rank, "batched_multi");
    if batch_shape.is_empty() {
        return op(input);
    }

    let slice_size: usize = core_shape.iter().product();
    let batch_count: usize = batch_shape.iter().product();
    assert!(
        batch_count > 0,
        "batched_multi: zero-sized batch dims are unsupported"
    );

    let mut out_shapes: Vec<Vec<usize>> = Vec::new();
    let mut out_data: Vec<Vec<OutT>> = Vec::new();

    for batch_idx in 0..batch_count {
        let start = batch_idx * slice_size;
        let end = start + slice_size;
        let batch_input = tensor_from_vec_with_template(
            core_shape.to_vec(),
            input.host_data()[start..end].to_vec(),
            input,
        );
        let batch_outputs = op(&batch_input);

        if out_shapes.is_empty() {
            out_shapes = batch_outputs
                .iter()
                .map(|tensor| tensor.shape.clone())
                .collect();
            out_data = batch_outputs
                .iter()
                .map(|tensor| Vec::with_capacity(tensor.n_elements() * batch_count))
                .collect();
        } else {
            assert_eq!(
                batch_outputs.len(),
                out_shapes.len(),
                "batched_multi: output count mismatch across batches"
            );
        }

        for (idx, batch_output) in batch_outputs.iter().enumerate() {
            assert_eq!(
                batch_output.shape.as_slice(),
                out_shapes[idx].as_slice(),
                "batched_multi: output core shape mismatch across batches"
            );
            out_data[idx].extend_from_slice(batch_output.host_data());
        }
    }

    out_shapes
        .into_iter()
        .zip(out_data)
        .map(|(mut out_shape, out_data)| {
            out_shape.extend_from_slice(batch_shape);
            tensor_from_vec_with_template(out_shape, out_data, input)
        })
        .collect()
}

fn batched_binary<T, F>(
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    core_rank_a: usize,
    core_rank_b: usize,
    op: F,
) -> TypedTensor<T>
where
    T: Clone,
    F: Fn(&TypedTensor<T>, &TypedTensor<T>) -> TypedTensor<T>,
{
    let (a_core_shape, a_batch_shape) = split_core_and_batch(a, core_rank_a, "batched_binary");
    let (b_core_shape, b_batch_shape) = split_core_and_batch(b, core_rank_b, "batched_binary");
    assert_eq!(
        a_batch_shape, b_batch_shape,
        "batched_binary: batch shape mismatch"
    );

    if a_batch_shape.is_empty() {
        return op(a, b);
    }

    let a_slice_size: usize = a_core_shape.iter().product();
    let b_slice_size: usize = b_core_shape.iter().product();
    let batch_count: usize = a_batch_shape.iter().product();
    assert!(
        batch_count > 0,
        "batched_binary: zero-sized batch dims are unsupported"
    );

    let mut out_core_shape: Option<Vec<usize>> = None;
    let mut out_data = Vec::new();

    for batch_idx in 0..batch_count {
        let a_start = batch_idx * a_slice_size;
        let a_end = a_start + a_slice_size;
        let b_start = batch_idx * b_slice_size;
        let b_end = b_start + b_slice_size;

        let batch_a = tensor_from_vec_with_template(
            a_core_shape.to_vec(),
            a.host_data()[a_start..a_end].to_vec(),
            a,
        );
        let batch_b = tensor_from_vec_with_template(
            b_core_shape.to_vec(),
            b.host_data()[b_start..b_end].to_vec(),
            b,
        );
        let batch_output = op(&batch_a, &batch_b);

        if let Some(expected_shape) = &out_core_shape {
            assert_eq!(
                batch_output.shape.as_slice(),
                expected_shape.as_slice(),
                "batched_binary: output core shape mismatch across batches"
            );
        } else {
            out_data.reserve(batch_output.n_elements() * batch_count);
            out_core_shape = Some(batch_output.shape.clone());
        }

        out_data.extend_from_slice(batch_output.host_data());
    }

    let mut out_shape = out_core_shape.expect("batched_binary: missing output shape");
    out_shape.extend_from_slice(a_batch_shape);
    tensor_from_vec_with_template(out_shape, out_data, b)
}

impl FaerLinalg for f64 {
    fn parity_one() -> Self {
        1.0
    }

    fn cholesky_2d(input: &TypedTensor<Self>) -> crate::Result<TypedTensor<Self>> {
        let n = square_matrix_dim(input, "cholesky");
        let mat = MatRef::from_column_major_slice(input.host_data(), n, n);
        let chol = match mat.llt(Side::Lower) {
            Ok(chol) => chol,
            Err(_) => {
                return Err(crate::Error::BackendFailure {
                    op: "cholesky",
                    message: "matrix is not positive definite".into(),
                });
            }
        };
        Ok(tensor_from_mat(chol.L(), vec![n, n], input))
    }

    fn lu_2d(input: &TypedTensor<Self>) -> Vec<TypedTensor<Self>> {
        let (m, n) = matrix_dims(input, "lu");
        let k = m.min(n);
        let mat = MatRef::from_column_major_slice(input.host_data(), m, n);
        let lu = mat.partial_piv_lu();
        let perm: Vec<usize> = lu.P().arrays().0.iter().copied().collect();

        let mut p_data = vec![0.0; m * m];
        for (row, &col) in perm.iter().enumerate() {
            p_data[row + col * m] = 1.0;
        }

        let mut parity = 1.0;
        let mut visited = vec![false; m];
        for start in 0..m {
            if visited[start] {
                continue;
            }
            let mut current = start;
            let mut cycle_len = 0usize;
            while !visited[current] {
                visited[current] = true;
                current = perm[current];
                cycle_len += 1;
            }
            if cycle_len > 0 && (cycle_len - 1) % 2 == 1 {
                parity = -parity;
            }
        }

        vec![
            tensor_from_vec_with_template(vec![m, m], p_data, input),
            tensor_from_mat(lu.L(), vec![m, k], input),
            tensor_from_mat(lu.U(), vec![k, n], input),
            tensor_from_vec_with_template(vec![], vec![parity], input),
        ]
    }

    fn triangular_solve_2d(
        a: &TypedTensor<Self>,
        b: &TypedTensor<Self>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> TypedTensor<Self> {
        let n = square_matrix_dim(a, "triangular_solve");
        let (b_rows, b_cols) = matrix_dims(b, "triangular_solve");
        let a_mat = MatRef::from_column_major_slice(a.host_data(), n, n);

        if left_side {
            assert_eq!(b_rows, n, "triangular_solve: rhs row count mismatch");
            let mut rhs_data = b.host_data().to_vec();
            let rhs = MatMut::from_column_major_slice_mut(&mut rhs_data, n, b_cols);
            match (transpose_a, lower, unit_diagonal) {
                (false, true, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        Par::Seq,
                    );
                }
                (false, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        Par::Seq,
                    );
                }
                (false, false, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        Par::Seq,
                    );
                }
                (false, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        Par::Seq,
                    );
                }
                (true, true, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        Par::Seq,
                    );
                }
                (true, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        Par::Seq,
                    );
                }
                (true, false, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        Par::Seq,
                    );
                }
                (true, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        Par::Seq,
                    );
                }
            }
            tensor_from_vec_with_template(vec![n, b_cols], rhs_data, b)
        } else {
            assert_eq!(b_cols, n, "triangular_solve: rhs column count mismatch");
            let nrhs = b_rows;
            let mut rhs_transposed = transpose_col_major_data(b.host_data(), nrhs, n);
            let rhs = MatMut::from_column_major_slice_mut(&mut rhs_transposed, n, nrhs);
            match (transpose_a, lower, unit_diagonal) {
                (false, true, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        Par::Seq,
                    );
                }
                (false, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        Par::Seq,
                    );
                }
                (false, false, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        Par::Seq,
                    );
                }
                (false, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        Par::Seq,
                    );
                }
                (true, true, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        Par::Seq,
                    );
                }
                (true, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        Par::Seq,
                    );
                }
                (true, false, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        Par::Seq,
                    );
                }
                (true, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        Par::Seq,
                    );
                }
            }
            let result = transpose_col_major_data(&rhs_transposed, n, nrhs);
            tensor_from_vec_with_template(vec![nrhs, n], result, b)
        }
    }

    fn svd_2d(input: &TypedTensor<Self>) -> Vec<TypedTensor<Self>> {
        let (m, n) = matrix_dims(input, "svd");
        let k = m.min(n);
        let mat = MatRef::from_column_major_slice(input.host_data(), m, n);
        let svd = match mat.thin_svd() {
            Ok(svd) => svd,
            Err(_) => panic!("svd: decomposition failed"),
        };

        let u = tensor_from_mat(svd.U(), vec![m, k], input);
        let s = tensor_from_vec_with_template(vec![k], vec_from_diag(svd.S()), input);

        let v = svd.V();
        let mut vt_data = Vec::with_capacity(k * n);
        for j in 0..n {
            for i in 0..k {
                vt_data.push(v[(j, i)]);
            }
        }
        let vt = tensor_from_vec_with_template(vec![k, n], vt_data, input);

        vec![u, s, vt]
    }

    fn qr_2d(input: &TypedTensor<Self>) -> Vec<TypedTensor<Self>> {
        let (m, n) = matrix_dims(input, "qr");
        let k = m.min(n);
        let mat = MatRef::from_column_major_slice(input.host_data(), m, n);
        let qr = mat.qr();

        let q_mat = qr.compute_thin_Q();
        let q = tensor_from_mat(q_mat.as_ref(), vec![m, k], input);
        let r = tensor_from_mat(qr.thin_R(), vec![k, n], input);

        vec![q, r]
    }

    fn eigh_2d(input: &TypedTensor<Self>) -> Vec<TypedTensor<Self>> {
        let n = square_matrix_dim(input, "eigh");
        let mat = MatRef::from_column_major_slice(input.host_data(), n, n);
        let eig = match mat.self_adjoint_eigen(Side::Lower) {
            Ok(eig) => eig,
            Err(_) => panic!("eigh: decomposition failed"),
        };

        let values = tensor_from_vec_with_template(vec![n], vec_from_diag(eig.S()), input);
        let vectors = tensor_from_mat(eig.U(), vec![n, n], input);

        vec![values, vectors]
    }
}

impl FaerLinalg for Complex64 {
    fn parity_one() -> Self {
        Complex64::new(1.0, 0.0)
    }

    fn cholesky_2d(input: &TypedTensor<Self>) -> crate::Result<TypedTensor<Self>> {
        let n = square_matrix_dim(input, "cholesky");
        let mat = MatRef::from_column_major_slice(complex64_to_faer_slice(input.host_data()), n, n);
        let chol = match mat.llt(Side::Lower) {
            Ok(chol) => chol,
            Err(_) => {
                return Err(crate::Error::BackendFailure {
                    op: "cholesky",
                    message: "matrix is not positive definite".into(),
                });
            }
        };
        Ok(tensor_from_mat(chol.L(), vec![n, n], input))
    }

    fn lu_2d(input: &TypedTensor<Self>) -> Vec<TypedTensor<Self>> {
        let (m, n) = matrix_dims(input, "lu");
        let k = m.min(n);
        let mat = MatRef::from_column_major_slice(complex64_to_faer_slice(input.host_data()), m, n);
        let lu = mat.partial_piv_lu();
        let perm: Vec<usize> = lu.P().arrays().0.iter().copied().collect();

        let mut p_data = vec![Complex64::new(0.0, 0.0); m * m];
        for (row, &col) in perm.iter().enumerate() {
            p_data[row + col * m] = Complex64::new(1.0, 0.0);
        }

        let mut parity = Complex64::new(1.0, 0.0);
        let mut visited = vec![false; m];
        for start in 0..m {
            if visited[start] {
                continue;
            }
            let mut current = start;
            let mut cycle_len = 0usize;
            while !visited[current] {
                visited[current] = true;
                current = perm[current];
                cycle_len += 1;
            }
            if cycle_len > 0 && (cycle_len - 1) % 2 == 1 {
                parity = -parity;
            }
        }

        vec![
            tensor_from_vec_with_template(vec![m, m], p_data, input),
            tensor_from_mat(lu.L(), vec![m, k], input),
            tensor_from_mat(lu.U(), vec![k, n], input),
            tensor_from_vec_with_template(vec![], vec![parity], input),
        ]
    }

    fn triangular_solve_2d(
        a: &TypedTensor<Self>,
        b: &TypedTensor<Self>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> TypedTensor<Self> {
        let n = square_matrix_dim(a, "triangular_solve");
        let (b_rows, b_cols) = matrix_dims(b, "triangular_solve");
        let a_mat = MatRef::from_column_major_slice(complex64_to_faer_slice(a.host_data()), n, n);

        if left_side {
            assert_eq!(b_rows, n, "triangular_solve: rhs row count mismatch");
            let mut rhs_data = b.host_data().to_vec();
            let rhs = MatMut::from_column_major_slice_mut(
                complex64_to_faer_slice_mut(&mut rhs_data),
                n,
                b_cols,
            );
            match (transpose_a, lower, unit_diagonal) {
                (false, true, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        Par::Seq,
                    );
                }
                (false, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        Par::Seq,
                    );
                }
                (false, false, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        Par::Seq,
                    );
                }
                (false, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        Par::Seq,
                    );
                }
                (true, true, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        Par::Seq,
                    );
                }
                (true, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        Par::Seq,
                    );
                }
                (true, false, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        Par::Seq,
                    );
                }
                (true, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        Par::Seq,
                    );
                }
            }
            tensor_from_vec_with_template(vec![n, b_cols], rhs_data, b)
        } else {
            assert_eq!(b_cols, n, "triangular_solve: rhs column count mismatch");
            let nrhs = b_rows;
            let mut rhs_transposed = transpose_col_major_data(b.host_data(), nrhs, n);
            let rhs = MatMut::from_column_major_slice_mut(
                complex64_to_faer_slice_mut(&mut rhs_transposed),
                n,
                nrhs,
            );
            match (transpose_a, lower, unit_diagonal) {
                (false, true, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        Par::Seq,
                    );
                }
                (false, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        Par::Seq,
                    );
                }
                (false, false, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        Par::Seq,
                    );
                }
                (false, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        Par::Seq,
                    );
                }
                (true, true, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        Par::Seq,
                    );
                }
                (true, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        Par::Seq,
                    );
                }
                (true, false, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        Par::Seq,
                    );
                }
                (true, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        Par::Seq,
                    );
                }
            }
            let result = transpose_col_major_data(&rhs_transposed, n, nrhs);
            tensor_from_vec_with_template(vec![nrhs, n], result, b)
        }
    }

    fn svd_2d(input: &TypedTensor<Self>) -> Vec<TypedTensor<Self>> {
        let (m, n) = matrix_dims(input, "svd");
        let k = m.min(n);
        let mat = MatRef::from_column_major_slice(complex64_to_faer_slice(input.host_data()), m, n);
        let svd = match mat.thin_svd() {
            Ok(svd) => svd,
            Err(_) => panic!("svd: decomposition failed"),
        };

        let u = tensor_from_mat(svd.U(), vec![m, k], input);
        let s = tensor_from_vec_with_template(vec![k], complex_vec_from_real_diag(svd.S()), input);

        let v = svd.V();
        let mut vt_data = Vec::with_capacity(k * n);
        for j in 0..n {
            for i in 0..k {
                vt_data.push(v[(j, i)].conj());
            }
        }
        let vt = tensor_from_vec_with_template(vec![k, n], vt_data, input);

        vec![u, s, vt]
    }

    fn qr_2d(input: &TypedTensor<Self>) -> Vec<TypedTensor<Self>> {
        let (m, n) = matrix_dims(input, "qr");
        let k = m.min(n);
        let mat = MatRef::from_column_major_slice(complex64_to_faer_slice(input.host_data()), m, n);
        let qr = mat.qr();

        let q_mat = qr.compute_thin_Q();
        let q = tensor_from_mat(q_mat.as_ref(), vec![m, k], input);
        let r = tensor_from_mat(qr.thin_R(), vec![k, n], input);

        vec![q, r]
    }

    fn eigh_2d(input: &TypedTensor<Self>) -> Vec<TypedTensor<Self>> {
        let n = square_matrix_dim(input, "eigh");
        let mat = MatRef::from_column_major_slice(complex64_to_faer_slice(input.host_data()), n, n);
        let eig = match mat.self_adjoint_eigen(Side::Lower) {
            Ok(eig) => eig,
            Err(_) => panic!("eigh: decomposition failed"),
        };

        let values =
            tensor_from_vec_with_template(vec![n], complex_vec_from_real_diag(eig.S()), input);
        let vectors = tensor_from_mat(eig.U(), vec![n, n], input);

        vec![values, vectors]
    }
}

pub(crate) fn cholesky<T: FaerLinalg + Default>(
    input: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>> {
    with_col_major_input(input, |input| {
        if has_zero_dim(&input.shape) {
            return Ok(tensor_from_vec_with_template(
                input.shape.clone(),
                Vec::new(),
                input,
            ));
        }
        batched_single(input, 2, T::cholesky_2d)
    })
}

pub(crate) fn lu<T: FaerLinalg + Default>(input: &TypedTensor<T>) -> Vec<TypedTensor<T>> {
    with_col_major_input(input, |input| {
        if has_zero_dim(&input.shape) {
            let m = input.shape[0];
            let n = input.shape[1];
            let k = m.min(n);
            let batch_shape = &input.shape[2..];
            let parity_elements: usize = batch_shape.iter().product::<usize>().max(1);
            return vec![
                tensor_from_vec_with_template(
                    matrix_with_batch_shape(m, m, batch_shape),
                    Vec::new(),
                    input,
                ),
                tensor_from_vec_with_template(
                    matrix_with_batch_shape(m, k, batch_shape),
                    Vec::new(),
                    input,
                ),
                tensor_from_vec_with_template(
                    matrix_with_batch_shape(k, n, batch_shape),
                    Vec::new(),
                    input,
                ),
                tensor_from_vec_with_template(
                    batch_shape.to_vec(),
                    vec![T::parity_one(); parity_elements],
                    input,
                ),
            ];
        }
        batched_multi(input, 2, T::lu_2d)
    })
}

pub(crate) fn triangular_solve<T: FaerLinalg + Default>(
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> TypedTensor<T> {
    with_col_major_input(a, |a| {
        with_col_major_input(b, |b| {
            if has_zero_dim(&a.shape) || has_zero_dim(&b.shape) {
                return tensor_from_vec_with_template(b.shape.clone(), Vec::new(), b);
            }
            batched_binary(a, b, 2, 2, |a, b| {
                T::triangular_solve_2d(a, b, left_side, lower, transpose_a, unit_diagonal)
            })
        })
    })
}

pub(crate) fn svd<T: FaerLinalg + Default>(input: &TypedTensor<T>) -> Vec<TypedTensor<T>> {
    with_col_major_input(input, |input| {
        if has_zero_dim(&input.shape) {
            let (matrix_shape, batch_shape) = split_core_and_batch(input, 2, "svd");
            let m = matrix_shape[0];
            let n = matrix_shape[1];
            let k = m.min(n);
            return vec![
                tensor_from_vec_with_template(
                    matrix_with_batch_shape(m, k, batch_shape),
                    Vec::new(),
                    input,
                ),
                tensor_from_vec_with_template(
                    vector_with_batch_shape(k, batch_shape),
                    Vec::new(),
                    input,
                ),
                tensor_from_vec_with_template(
                    matrix_with_batch_shape(k, n, batch_shape),
                    Vec::new(),
                    input,
                ),
            ];
        }
        batched_multi(input, 2, T::svd_2d)
    })
}

pub(crate) fn qr<T: FaerLinalg + Default>(input: &TypedTensor<T>) -> Vec<TypedTensor<T>> {
    with_col_major_input(input, |input| {
        if has_zero_dim(&input.shape) {
            let (matrix_shape, batch_shape) = split_core_and_batch(input, 2, "qr");
            let m = matrix_shape[0];
            let n = matrix_shape[1];
            let k = m.min(n);
            return vec![
                tensor_from_vec_with_template(
                    matrix_with_batch_shape(m, k, batch_shape),
                    Vec::new(),
                    input,
                ),
                tensor_from_vec_with_template(
                    matrix_with_batch_shape(k, n, batch_shape),
                    Vec::new(),
                    input,
                ),
            ];
        }
        batched_multi(input, 2, T::qr_2d)
    })
}

pub(crate) fn eigh<T: FaerLinalg + Default>(input: &TypedTensor<T>) -> Vec<TypedTensor<T>> {
    with_col_major_input(input, |input| {
        if has_zero_dim(&input.shape) {
            let n = input.shape[0];
            let batch_shape = &input.shape[2..];
            return vec![
                tensor_from_vec_with_template(
                    vector_with_batch_shape(n, batch_shape),
                    Vec::new(),
                    input,
                ),
                tensor_from_vec_with_template(
                    matrix_with_batch_shape(n, n, batch_shape),
                    Vec::new(),
                    input,
                ),
            ];
        }
        batched_multi(input, 2, T::eigh_2d)
    })
}

fn eig_real_2d(input: &TypedTensor<f64>) -> Vec<TypedTensor<Complex64>> {
    let n = square_matrix_dim(input, "eig");
    let mat = MatRef::from_column_major_slice(input.host_data(), n, n);
    let eig = match mat.eigen() {
        Ok(eig) => eig,
        Err(_) => panic!("eig: decomposition failed"),
    };

    vec![
        tensor_from_vec_with_template(vec![n], complex_vec_from_diag(eig.S()), input),
        tensor_from_vec_with_template(vec![n, n], complex_vec_from_mat(eig.U()), input),
    ]
}

fn eig_complex_2d(input: &TypedTensor<Complex64>) -> Vec<TypedTensor<Complex64>> {
    let n = square_matrix_dim(input, "eig");
    let mat = MatRef::from_column_major_slice(complex64_to_faer_slice(input.host_data()), n, n);
    let eig = match mat.eigen() {
        Ok(eig) => eig,
        Err(_) => panic!("eig: decomposition failed"),
    };

    vec![
        tensor_from_vec_with_template(vec![n], complex_vec_from_diag(eig.S()), input),
        tensor_from_vec_with_template(vec![n, n], complex_vec_from_mat(eig.U()), input),
    ]
}

pub(crate) fn eig(input: &Tensor) -> Vec<Tensor> {
    match input {
        Tensor::F64(t) => with_col_major_input(t, |t| {
            if has_zero_dim(&t.shape) {
                let n = t.shape[0];
                let batch_shape = &t.shape[2..];
                return vec![
                    Tensor::C64(TypedTensor::from_vec(
                        vector_with_batch_shape(n, batch_shape),
                        Vec::new(),
                    )),
                    Tensor::C64(TypedTensor::from_vec(
                        matrix_with_batch_shape(n, n, batch_shape),
                        Vec::new(),
                    )),
                ];
            }
            batched_multi_convert(t, 2, eig_real_2d)
                .into_iter()
                .map(Tensor::C64)
                .collect()
        }),
        Tensor::C64(t) => with_col_major_input(t, |t| {
            if has_zero_dim(&t.shape) {
                let n = t.shape[0];
                let batch_shape = &t.shape[2..];
                return vec![
                    Tensor::C64(TypedTensor::from_vec(
                        vector_with_batch_shape(n, batch_shape),
                        Vec::new(),
                    )),
                    Tensor::C64(TypedTensor::from_vec(
                        matrix_with_batch_shape(n, n, batch_shape),
                        Vec::new(),
                    )),
                ];
            }
            batched_multi_convert(t, 2, eig_complex_2d)
                .into_iter()
                .map(Tensor::C64)
                .collect()
        }),
        _ => todo!("eig: unsupported dtype"),
    }
}

fn has_zero_dim(shape: &[usize]) -> bool {
    shape.contains(&0)
}

fn matrix_with_batch_shape(rows: usize, cols: usize, batch_shape: &[usize]) -> Vec<usize> {
    let mut shape = vec![rows, cols];
    shape.extend_from_slice(batch_shape);
    shape
}

fn vector_with_batch_shape(len: usize, batch_shape: &[usize]) -> Vec<usize> {
    let mut shape = vec![len];
    shape.extend_from_slice(batch_shape);
    shape
}
