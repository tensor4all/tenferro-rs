use faer::linalg::solvers::Solve;
use faer::{diag::DiagRef, MatRef, Side};

use crate::{Buffer, TypedTensor};

fn matrix_dims<T>(input: &TypedTensor<T>, op: &str) -> (usize, usize) {
    assert_eq!(input.shape.len(), 2, "{op}: expected a 2D matrix");
    (input.shape[0], input.shape[1])
}

fn square_matrix_dim<T>(input: &TypedTensor<T>, op: &str) -> usize {
    let (rows, cols) = matrix_dims(input, op);
    assert_eq!(rows, cols, "{op}: expected a square matrix");
    rows
}

fn tensor_from_vec_with_template<T: Clone>(
    shape: Vec<usize>,
    data: Vec<T>,
    template: &TypedTensor<T>,
) -> TypedTensor<T> {
    TypedTensor {
        buffer: Buffer::Host(data),
        shape,
        placement: template.placement.clone(),
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

fn tensor_from_mat<T: Copy + Clone>(
    mat: MatRef<'_, T>,
    shape: Vec<usize>,
    template: &TypedTensor<T>,
) -> TypedTensor<T> {
    tensor_from_vec_with_template(shape, col_major_vec_from_mat(mat), template)
}

fn vec_from_diag(diag: DiagRef<'_, f64>) -> Vec<f64> {
    let col = diag.column_vector();
    let mut data = Vec::with_capacity(col.nrows());
    for i in 0..col.nrows() {
        data.push(col[i]);
    }
    data
}

pub(crate) fn cholesky(input: &TypedTensor<f64>) -> TypedTensor<f64> {
    let n = square_matrix_dim(input, "cholesky");
    let mat = MatRef::from_column_major_slice(input.host_data(), n, n);
    let chol = match mat.llt(Side::Lower) {
        Ok(chol) => chol,
        Err(_) => panic!("cholesky: matrix is not positive definite"),
    };
    tensor_from_mat(chol.L(), vec![n, n], input)
}

pub(crate) fn svd(input: &TypedTensor<f64>) -> Vec<TypedTensor<f64>> {
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

pub(crate) fn qr(input: &TypedTensor<f64>) -> Vec<TypedTensor<f64>> {
    let (m, n) = matrix_dims(input, "qr");
    let k = m.min(n);
    let mat = MatRef::from_column_major_slice(input.host_data(), m, n);
    let qr = mat.qr();

    let q_mat = qr.compute_thin_Q();
    let q = tensor_from_mat(q_mat.as_ref(), vec![m, k], input);
    let r = tensor_from_mat(qr.thin_R(), vec![k, n], input);

    vec![q, r]
}

pub(crate) fn eigh(input: &TypedTensor<f64>) -> Vec<TypedTensor<f64>> {
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

pub(crate) fn solve(a: &TypedTensor<f64>, b: &TypedTensor<f64>) -> TypedTensor<f64> {
    let n = square_matrix_dim(a, "solve");
    let (b_rows, nrhs) = matrix_dims(b, "solve");
    assert_eq!(b_rows, n, "solve: rhs row count mismatch");

    let a_mat = MatRef::from_column_major_slice(a.host_data(), n, n);
    let b_mat = MatRef::from_column_major_slice(b.host_data(), n, nrhs);
    let lu = a_mat.partial_piv_lu();
    let u_mat = lu.U();
    for i in 0..n {
        let diag = u_mat[(i, i)];
        assert!(diag.is_finite() && diag != 0.0, "solve: singular matrix");
    }

    let x = lu.solve(&b_mat);
    tensor_from_mat(x.as_ref(), vec![n, nrhs], b)
}
