use super::*;

pub(crate) fn lu_factor_impl<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<LuFactorExResult<T>>
where
    C: backend::TensorLinalgContextFor<T>,
{
    let (m, n, batch_dims) = validate_2d(tensor)?;
    let bc = batch_count(batch_dims);
    let k = m.min(n);
    let result = <C::Backend as backend::TensorLinalgBackend<T>>::lu_factor(ctx, tensor)?;
    let factors = pack_lu_factors(&result.l, &result.u, m, n, batch_dims)?;

    let u_input = ensure_col_major(&result.u);
    let u_data = extract_slice(&u_input)?;
    let u_offset = u_input.offset() as usize;
    let mut info = vec![0_i32; bc];

    for (batch, info_slot) in info.iter_mut().enumerate().take(bc) {
        let start = u_offset + batch * k * n;
        let u_slice = &u_data[start..start + k * n];
        for i in 0..k {
            if u_slice[i + i * k].abs_real() <= T::real_epsilon() {
                *info_slot = (i + 1) as i32;
                break;
            }
        }
    }

    Ok(LuFactorExResult {
        factors,
        pivots: result
            .pivots
            .into_iter()
            .map(|pivot| pivot as usize)
            .collect(),
        info,
    })
}

pub(crate) fn pack_lu_factors<T: LinalgScalar>(
    l: &Tensor<T>,
    u: &Tensor<T>,
    m: usize,
    n: usize,
    batch_dims: &[usize],
) -> Result<Tensor<T>> {
    let bc = batch_count(batch_dims);
    let k = m.min(n);
    let l_input = ensure_col_major(l);
    let u_input = ensure_col_major(u);
    let l_data = extract_slice(&l_input)?;
    let u_data = extract_slice(&u_input)?;
    let l_offset = l_input.offset() as usize;
    let u_offset = u_input.offset() as usize;
    let mut packed = vec![T::zero(); m * n * bc];

    for batch in 0..bc {
        let l_start = l_offset + batch * m * k;
        let u_start = u_offset + batch * k * n;
        let l_slice = &l_data[l_start..l_start + m * k];
        let u_slice = &u_data[u_start..u_start + k * n];
        let packed_slice = &mut packed[batch * m * n..(batch + 1) * m * n];
        for j in 0..n {
            for i in 0..m {
                packed_slice[i + j * m] = if i > j {
                    l_slice[i + j * m]
                } else {
                    u_slice[i + j * k]
                };
            }
        }
    }

    tensor_from_data(packed, &output_dims(&[m, n], batch_dims))
}

pub(crate) fn lu_solve_impl<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    factors: &Tensor<T>,
    pivots: &[usize],
    b: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    let (n, batch_dims) = validate_square(factors)?;
    let rhs = backend::tensor_helpers::validate_solve_rhs_shape(b, n, batch_dims, "lu_solve")?;
    let bc = batch_count(batch_dims);
    let expected_pivots = n * bc;
    if pivots.len() != expected_pivots {
        return Err(Error::InvalidArgument(format!(
            "lu_solve expects pivots.len() == {expected_pivots}, got {}",
            pivots.len()
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
        let pivot_slice = &pivots[batch * n..(batch + 1) * n];

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
