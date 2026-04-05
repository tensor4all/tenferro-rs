use crate::TypedTensor;

pub(crate) fn cholesky(_input: &TypedTensor<f64>) -> TypedTensor<f64> {
    todo!("lapack linalg cholesky")
}

pub(crate) fn svd(_input: &TypedTensor<f64>) -> Vec<TypedTensor<f64>> {
    todo!("lapack linalg svd")
}

pub(crate) fn qr(_input: &TypedTensor<f64>) -> Vec<TypedTensor<f64>> {
    todo!("lapack linalg qr")
}

pub(crate) fn eigh(_input: &TypedTensor<f64>) -> Vec<TypedTensor<f64>> {
    todo!("lapack linalg eigh")
}

pub(crate) fn solve(_a: &TypedTensor<f64>, _b: &TypedTensor<f64>) -> TypedTensor<f64> {
    todo!("lapack linalg solve")
}
