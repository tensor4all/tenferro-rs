use super::*;

pub(crate) fn pack_lu_factors<T: LinalgScalar>(l: &Tensor<T>, u: &Tensor<T>) -> Result<Tensor<T>> {
    Tensor::merge_strict_lower_and_upper(l, u)
}
