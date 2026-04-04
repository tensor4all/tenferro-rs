use num_traits::Zero;

use super::tensor_data::TensorData;
use super::types::{
    col_major_strides, dispatch_binary, dispatch_tensor, flat_to_multi, ConjElem, Tensor,
    TypedTensor,
};
use computegraph::Operand;

impl Operand for Tensor {
    fn zero(shape: &[usize]) -> Self {
        Tensor::F64(TypedTensor::zeros(shape.to_vec()))
    }

    fn one(shape: &[usize]) -> Self {
        Tensor::F64(TypedTensor::ones(shape.to_vec()))
    }

    fn reshape(&self, shape: &[usize]) -> Self {
        dispatch_tensor!(self, t => typed_reshape(t, shape))
    }

    fn broadcast_in_dim(&self, shape: &[usize], dims: &[usize]) -> Self {
        dispatch_tensor!(self, t => typed_broadcast_in_dim(t, shape, dims))
    }

    fn add(&self, other: &Self) -> Self {
        dispatch_binary!(self, other, |a, b| typed_add(a, b))
    }

    fn multiply(&self, other: &Self) -> Self {
        dispatch_binary!(self, other, |a, b| typed_mul(a, b))
    }

    fn reduce_sum(&self, axes: &[usize]) -> Self {
        dispatch_tensor!(self, t => typed_reduce_sum(t, axes))
    }

    fn dot_general(
        &self,
        other: &Self,
        lhs_contracting: &[usize],
        rhs_contracting: &[usize],
        lhs_batch: &[usize],
        rhs_batch: &[usize],
    ) -> Self {
        dispatch_binary!(self, other, |a, b| typed_dot_general(
            a,
            b,
            lhs_contracting,
            rhs_contracting,
            lhs_batch,
            rhs_batch,
        ))
    }

    fn conj(&self) -> Self {
        dispatch_tensor!(self, t => typed_conj(t))
    }
}

fn typed_reshape<T: Clone + Zero>(t: &TypedTensor<T>, new_shape: &[usize]) -> TypedTensor<T> {
    let old_n: usize = t.shape.iter().product();
    let new_n: usize = new_shape.iter().product();
    assert_eq!(old_n, new_n, "reshape: element count mismatch");
    let data = linearize_to_col_major(t);
    TypedTensor::from_vec(new_shape.to_vec(), data)
}

fn typed_broadcast_in_dim<T: Clone + Zero>(
    t: &TypedTensor<T>,
    out_shape: &[usize],
    dims: &[usize],
) -> TypedTensor<T> {
    assert_eq!(dims.len(), t.shape.len());
    let out_rank = out_shape.len();
    let out_n: usize = out_shape.iter().product();
    let mut result = TypedTensor::zeros(out_shape.to_vec());
    let mut out_idx = vec![0usize; out_rank];
    let mut in_idx = vec![0usize; t.shape.len()];
    for flat in 0..out_n {
        flat_to_multi(flat, out_shape, &mut out_idx);
        for (in_dim, &out_dim) in dims.iter().enumerate() {
            in_idx[in_dim] = if t.shape[in_dim] == 1 {
                0
            } else {
                out_idx[out_dim]
            };
        }
        let val = t.get(&in_idx).clone();
        let off = result.linear_offset(&out_idx);
        result.host_data_mut()[off] = val;
    }
    result
}

fn typed_add<T>(a: &TypedTensor<T>, b: &TypedTensor<T>) -> TypedTensor<T>
where
    T: Clone + Zero + std::ops::Add<Output = T>,
{
    assert_eq!(a.shape, b.shape, "add: shape mismatch");
    let n = a.n_elements();
    let mut result = TypedTensor::zeros(a.shape.clone());
    let mut idx = vec![0usize; a.shape.len()];
    for flat in 0..n {
        flat_to_multi(flat, &a.shape, &mut idx);
        let va = a.get(&idx).clone();
        let vb = b.get(&idx).clone();
        let off = result.linear_offset(&idx);
        result.host_data_mut()[off] = va + vb;
    }
    result
}

fn typed_mul<T>(a: &TypedTensor<T>, b: &TypedTensor<T>) -> TypedTensor<T>
where
    T: Clone + Zero + std::ops::Mul<Output = T>,
{
    assert_eq!(a.shape, b.shape, "mul: shape mismatch");
    let n = a.n_elements();
    let mut result = TypedTensor::zeros(a.shape.clone());
    let mut idx = vec![0usize; a.shape.len()];
    for flat in 0..n {
        flat_to_multi(flat, &a.shape, &mut idx);
        let va = a.get(&idx).clone();
        let vb = b.get(&idx).clone();
        let off = result.linear_offset(&idx);
        result.host_data_mut()[off] = va * vb;
    }
    result
}

fn typed_reduce_sum<T>(t: &TypedTensor<T>, axes: &[usize]) -> TypedTensor<T>
where
    T: Clone + Zero + std::ops::Add<Output = T>,
{
    let rank = t.shape.len();
    let out_shape: Vec<usize> = (0..rank)
        .filter(|d| !axes.contains(d))
        .map(|d| t.shape[d])
        .collect();
    let out_shape = if out_shape.is_empty() {
        vec![1]
    } else {
        out_shape
    };
    let kept_dims: Vec<usize> = (0..rank).filter(|d| !axes.contains(d)).collect();
    let n_in = t.n_elements();
    let mut result: TypedTensor<T> = TypedTensor::zeros(out_shape.clone());
    let mut in_idx = vec![0usize; rank];
    let mut out_idx = vec![0usize; kept_dims.len().max(1)];
    for flat in 0..n_in {
        flat_to_multi(flat, &t.shape, &mut in_idx);
        if kept_dims.is_empty() {
            out_idx[0] = 0;
        } else {
            for (oi, &d) in kept_dims.iter().enumerate() {
                out_idx[oi] = in_idx[d];
            }
        }
        let val = t.get(&in_idx).clone();
        let off = result.linear_offset(&out_idx);
        let cur = result.host_data()[off].clone();
        result.host_data_mut()[off] = cur + val;
    }
    result
}

fn typed_dot_general<T>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    lhs_contracting: &[usize],
    rhs_contracting: &[usize],
    lhs_batch: &[usize],
    rhs_batch: &[usize],
) -> TypedTensor<T>
where
    T: Clone + Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    assert_eq!(lhs_contracting.len(), rhs_contracting.len());
    assert_eq!(lhs_batch.len(), rhs_batch.len());

    let lhs_rank = lhs.shape.len();
    let rhs_rank = rhs.shape.len();

    let lhs_free: Vec<usize> = (0..lhs_rank)
        .filter(|d| !lhs_contracting.contains(d) && !lhs_batch.contains(d))
        .collect();
    let rhs_free: Vec<usize> = (0..rhs_rank)
        .filter(|d| !rhs_contracting.contains(d) && !rhs_batch.contains(d))
        .collect();

    let batch_shape: Vec<usize> = lhs_batch.iter().map(|&d| lhs.shape[d]).collect();
    let lhs_free_shape: Vec<usize> = lhs_free.iter().map(|&d| lhs.shape[d]).collect();
    let rhs_free_shape: Vec<usize> = rhs_free.iter().map(|&d| rhs.shape[d]).collect();
    let contract_shape: Vec<usize> = lhs_contracting.iter().map(|&d| lhs.shape[d]).collect();

    // Output shape: batch_dims + lhs_free_dims + rhs_free_dims
    let mut out_shape = Vec::new();
    out_shape.extend_from_slice(&batch_shape);
    out_shape.extend_from_slice(&lhs_free_shape);
    out_shape.extend_from_slice(&rhs_free_shape);
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    let out_n: usize = out_shape.iter().product();
    let contract_n: usize = contract_shape.iter().product();

    let mut result = TypedTensor::zeros(out_shape.clone());
    let n_batch = batch_shape.len();
    let n_lhs_free = lhs_free_shape.len();

    let mut out_idx = vec![0usize; out_shape.len()];
    let mut lhs_idx = vec![0usize; lhs_rank];
    let mut rhs_idx = vec![0usize; rhs_rank];
    let mut contract_idx = vec![0usize; contract_shape.len()];

    for flat_out in 0..out_n {
        flat_to_multi(flat_out, &out_shape, &mut out_idx);

        // Extract batch, lhs_free, rhs_free from output index
        let batch_vals = &out_idx[..n_batch];
        let lhs_free_vals = &out_idx[n_batch..n_batch + n_lhs_free];
        let rhs_free_vals = &out_idx[n_batch + n_lhs_free..];

        // Set batch dims in lhs/rhs
        for (bi, &ld) in lhs_batch.iter().enumerate() {
            lhs_idx[ld] = batch_vals[bi];
        }
        for (bi, &rd) in rhs_batch.iter().enumerate() {
            rhs_idx[rd] = batch_vals[bi];
        }
        // Set free dims
        for (fi, &ld) in lhs_free.iter().enumerate() {
            lhs_idx[ld] = lhs_free_vals[fi];
        }
        for (fi, &rd) in rhs_free.iter().enumerate() {
            rhs_idx[rd] = rhs_free_vals[fi];
        }

        let mut acc = T::zero();
        for flat_k in 0..contract_n {
            flat_to_multi(flat_k, &contract_shape, &mut contract_idx);
            for (ci, &ld) in lhs_contracting.iter().enumerate() {
                lhs_idx[ld] = contract_idx[ci];
            }
            for (ci, &rd) in rhs_contracting.iter().enumerate() {
                rhs_idx[rd] = contract_idx[ci];
            }
            let lv = lhs.get(&lhs_idx).clone();
            let rv = rhs.get(&rhs_idx).clone();
            acc = acc + lv * rv;
        }

        let off = result.linear_offset(&out_idx);
        result.host_data_mut()[off] = acc;
    }

    // If the output was a scalar contraction with shape [1], unwrap to []
    // Actually, keep as-is: dot_general output shape follows the spec exactly.
    // The [1] fallback only activates when all dims are contracted, which is fine.
    result
}

fn typed_conj<T: Clone + ConjElem + Zero>(t: &TypedTensor<T>) -> TypedTensor<T> {
    let n = t.n_elements();
    let mut result = TypedTensor::zeros(t.shape.clone());
    let mut idx = vec![0usize; t.shape.len()];
    for flat in 0..n {
        flat_to_multi(flat, &t.shape, &mut idx);
        let val = t.get(&idx).conj_elem();
        let off = result.linear_offset(&idx);
        result.host_data_mut()[off] = val;
    }
    result
}

fn linearize_to_col_major<T: Clone + Zero>(t: &TypedTensor<T>) -> Vec<T> {
    let n = t.n_elements();
    let new_strides = col_major_strides(&t.shape);
    if t.strides == new_strides {
        return t.host_data().to_vec();
    }
    let mut data = vec![T::zero(); n];
    let mut idx = vec![0usize; t.shape.len()];
    for flat in 0..n {
        flat_to_multi(flat, &t.shape, &mut idx);
        let src_off = t.linear_offset(&idx);
        let dst_off = idx
            .iter()
            .zip(new_strides.iter())
            .map(|(&i, &s)| i as isize * s)
            .sum::<isize>() as usize;
        data[dst_off] = t.host_data()[src_off].clone();
    }
    data
}

pub fn generic_transpose<T: TensorData>(t: &T, perm: &[usize]) -> T
where
    T::Scalar: Clone + Zero,
{
    let rank = t.shape().len();
    assert_eq!(perm.len(), rank);
    let old_shape = t.shape();
    let new_shape: Vec<usize> = perm.iter().map(|&p| old_shape[p]).collect();
    let n: usize = old_shape.iter().product();
    let new_strides = col_major_strides(&new_shape);

    let mut data = vec![T::Scalar::zero(); n];
    let mut src_idx = vec![0usize; rank];
    let mut dst_idx = vec![0usize; rank];
    let src_data = t.as_slice();

    for flat in 0..n {
        flat_to_multi_generic(flat, old_shape, &mut src_idx);
        for (d, &p) in perm.iter().enumerate() {
            dst_idx[d] = src_idx[p];
        }
        let src_off = src_idx
            .iter()
            .zip(t.strides().iter())
            .map(|(&i, &s)| i as isize * s)
            .sum::<isize>() as usize;
        let dst_off = dst_idx
            .iter()
            .zip(new_strides.iter())
            .map(|(&i, &s)| i as isize * s)
            .sum::<isize>() as usize;
        data[dst_off] = src_data[src_off].clone();
    }
    T::from_dense(new_shape, data)
}

pub fn generic_reshape<T: TensorData>(t: &T, new_shape: &[usize]) -> T
where
    T::Scalar: Clone + Zero,
{
    let old_n: usize = t.shape().iter().product();
    let new_n: usize = new_shape.iter().product();
    assert_eq!(old_n, new_n, "reshape element count mismatch");

    let old_shape = t.shape();
    let old_strides_expected = col_major_strides(old_shape);

    let data = if t.strides() == old_strides_expected.as_slice() {
        t.as_slice().to_vec()
    } else {
        let mut out = vec![T::Scalar::zero(); old_n];
        let mut idx = vec![0usize; old_shape.len()];
        let new_col_strides = col_major_strides(old_shape);
        for flat in 0..old_n {
            flat_to_multi_generic(flat, old_shape, &mut idx);
            let src_off = idx
                .iter()
                .zip(t.strides().iter())
                .map(|(&i, &s)| i as isize * s)
                .sum::<isize>() as usize;
            let dst_off = idx
                .iter()
                .zip(new_col_strides.iter())
                .map(|(&i, &s)| i as isize * s)
                .sum::<isize>() as usize;
            out[dst_off] = t.as_slice()[src_off].clone();
        }
        out
    };
    T::from_dense(new_shape.to_vec(), data)
}

pub fn generic_broadcast_in_dim<T: TensorData>(t: &T, out_shape: &[usize], dims: &[usize]) -> T
where
    T::Scalar: Clone + Zero,
{
    let in_shape = t.shape();
    let in_rank = in_shape.len();
    let out_rank = out_shape.len();
    assert_eq!(dims.len(), in_rank);

    let out_n: usize = out_shape.iter().product();
    let mut data = vec![T::Scalar::zero(); out_n];
    let out_strides = col_major_strides(out_shape);

    let mut out_idx = vec![0usize; out_rank];
    let mut in_idx = vec![0usize; in_rank];

    for flat in 0..out_n {
        flat_to_multi_generic(flat, out_shape, &mut out_idx);
        for (in_dim, &out_dim) in dims.iter().enumerate() {
            in_idx[in_dim] = if in_shape[in_dim] == 1 {
                0
            } else {
                out_idx[out_dim]
            };
        }
        let src_off = in_idx
            .iter()
            .zip(t.strides().iter())
            .map(|(&i, &s)| i as isize * s)
            .sum::<isize>() as usize;
        let dst_off = out_idx
            .iter()
            .zip(out_strides.iter())
            .map(|(&i, &s)| i as isize * s)
            .sum::<isize>() as usize;
        data[dst_off] = t.as_slice()[src_off].clone();
    }
    T::from_dense(out_shape.to_vec(), data)
}

fn flat_to_multi_generic(mut flat: usize, shape: &[usize], out: &mut [usize]) {
    for i in 0..shape.len() {
        out[i] = flat % shape[i];
        flat /= shape[i];
    }
}
