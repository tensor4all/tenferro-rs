use super::*;

pub(crate) fn pack_lu_factors<T: LinalgScalar>(l: &Tensor<T>, u: &Tensor<T>) -> Result<Tensor<T>> {
    Tensor::merge_strict_lower_and_upper(l, u)
}

pub(crate) fn lu_solve_impl<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    factors: &Tensor<T>,
    pivots: &Tensor<i32>,
    b: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    let (n, batch_dims) = validate_square(factors)?;
    let rhs = backend::tensor_helpers::validate_solve_rhs_shape(b, n, batch_dims, "lu_solve")?;
    let bc = batch_count(batch_dims);
    let pivot_perm = backend::tensor_helpers::backend_pivots_to_forward_perm(pivots, n)?;
    let expected_pivots = n * bc;
    if pivot_perm.len() != expected_pivots {
        return Err(Error::InvalidArgument(format!(
            "lu_solve expects pivots.len() == {expected_pivots}, got {}",
            pivot_perm.len()
        )));
    }

    let factors_input = ensure_col_major(factors);
    let rhs_input = ensure_col_major(b);
    let factors_data = extract_slice(&factors_input)?;
    let rhs_data = extract_slice(&rhs_input)?;
    let factors_offset = factors_input.offset() as usize;
    let rhs_offset = rhs_input.offset() as usize;

    let mat_size = n * n;
    let rhs_size = n * rhs.nrhs;
    let mut out = vec![T::zero(); rhs_size * bc];
    let mut lower = vec![T::zero(); mat_size];
    let mut upper = vec![T::zero(); mat_size];
    let mut permuted_rhs = vec![T::zero(); rhs_size];
    let mut tmp = vec![T::zero(); rhs_size];

    for batch in 0..bc {
        let factor_start = factors_offset + batch * mat_size;
        let rhs_start = rhs_offset + batch * rhs_size;
        let factor_slice = &factors_data[factor_start..factor_start + mat_size];
        let rhs_slice = &rhs_data[rhs_start..rhs_start + rhs_size];
        let pivot_slice = &pivot_perm[batch * n..(batch + 1) * n];

        unpack_packed_lu_square(factor_slice, n, &mut lower, &mut upper);
        apply_lu_permutation(pivot_slice, rhs_slice, n, rhs.nrhs, &mut permuted_rhs)?;
        let tmp_solution = backend::slice_bridge::solve_triangular_vec(
            ctx,
            &lower,
            &permuted_rhs,
            n,
            rhs.nrhs,
            false,
        )?;
        tmp.copy_from_slice(&tmp_solution);
        let out_solution =
            backend::slice_bridge::solve_triangular_vec(ctx, &upper, &tmp, n, rhs.nrhs, true)?;
        out[batch * rhs_size..(batch + 1) * rhs_size].copy_from_slice(&out_solution);
    }

    tensor_from_data(out, &rhs.output_dims)
}

pub(crate) fn unpack_packed_lu_square<T: LinalgScalar>(
    factors: &[T],
    n: usize,
    lower: &mut [T],
    upper: &mut [T],
) {
    lower.fill(T::zero());
    upper.fill(T::zero());
    for j in 0..n {
        for i in 0..n {
            let value = factors[i + j * n];
            if i > j {
                lower[i + j * n] = value;
            } else {
                upper[i + j * n] = value;
                if i == j {
                    lower[i + j * n] = T::one();
                }
            }
        }
    }
}

pub(crate) fn apply_lu_permutation<T: LinalgScalar>(
    pivots: &[usize],
    rhs: &[T],
    n: usize,
    nrhs: usize,
    out: &mut [T],
) -> Result<()> {
    for &pivot in pivots {
        if pivot >= n {
            return Err(Error::InvalidArgument(format!(
                "lu_solve pivot index {pivot} is out of range for n={n}"
            )));
        }
    }

    for col in 0..nrhs {
        let col_offset = col * n;
        for row in 0..n {
            out[row + col_offset] = rhs[pivots[row] + col_offset];
        }
    }

    Ok(())
}
