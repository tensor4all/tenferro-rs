use super::*;

/// Forward-mode AD rule for norm (JVP / pushforward).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{norm_frule, NormKind};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col).unwrap();
/// let da = Tensor::<f64>::ones(&[3, 4], mem, col).unwrap();
/// let (n, dn) = norm_frule(&mut ctx, &a, &da, NormKind::Fro).unwrap();
/// ```
pub fn norm_frule<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
    kind: NormKind,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>>::ScalarBackend:
        tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<T>, Context = C>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Norm, "norm_frule")
        .map_err(to_ad_err)?;

    let nrm = crate::primal::norm_real_impl(ctx, tensor, kind)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;

    if tensor.ndim() == 1 {
        let (a_data, _) = extract_data(tensor)?;
        let (nrm_data, _) = extract_data(&nrm)?;
        let (da_data, _) = extract_data(tangent)?;
        let len = tensor.dims()[0];
        let mut dnrm = T::zero();

        match kind {
            NormKind::Fro => {
                let nv = nrm_data[0];
                if nv > T::zero() {
                    for i in 0..len {
                        dnrm = dnrm + a_data[i] * da_data[i];
                    }
                    dnrm = dnrm / nv;
                }
            }
            NormKind::L1 => {
                for i in 0..len {
                    let v = a_data[i];
                    let sign = if v > T::zero() {
                        T::one()
                    } else if v < T::zero() {
                        -T::one()
                    } else {
                        T::zero()
                    };
                    dnrm = dnrm + sign * da_data[i];
                }
            }
            NormKind::Inf => {
                let max_abs = a_data.iter().fold(T::zero(), |acc, &v| acc.max(v.abs()));
                let active: Vec<usize> = a_data
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| if v.abs() == max_abs { Some(i) } else { None })
                    .collect();
                if !active.is_empty() {
                    for i in active.iter().copied() {
                        let v = a_data[i];
                        let sign = if v > T::zero() {
                            T::one()
                        } else if v < T::zero() {
                            -T::one()
                        } else {
                            T::zero()
                        };
                        dnrm = dnrm + sign * da_data[i];
                    }
                    let active_count = scalar_from::<T>(active.len() as f64).map_err(to_ad_err)?;
                    dnrm = dnrm / active_count;
                }
            }
            NormKind::Lp(p) => {
                if p < 1.0 {
                    return Err(invalid_vector_lp_exponent_ad_error(p));
                }
                if p == 1.0 {
                    for i in 0..len {
                        let v = a_data[i];
                        let sign = if v > T::zero() {
                            T::one()
                        } else if v < T::zero() {
                            -T::one()
                        } else {
                            T::zero()
                        };
                        dnrm = dnrm + sign * da_data[i];
                    }
                } else {
                    let nv = nrm_data[0];
                    if nv > T::zero() {
                        let p_minus_one = scalar_from::<T>(p - 1.0).map_err(to_ad_err)?;
                        for i in 0..len {
                            let v = a_data[i];
                            let sign = if v > T::zero() {
                                T::one()
                            } else if v < T::zero() {
                                -T::one()
                            } else {
                                T::zero()
                            };
                            dnrm = dnrm + sign * v.abs().powf(p_minus_one) * da_data[i];
                        }
                        dnrm = dnrm / nv.powf(p_minus_one);
                    }
                }
            }
            NormKind::Nuclear | NormKind::Spectral => {
                return Err(matrix_only_norm_kind_ad_error(kind));
            }
        }

        let dnrm = tensor_from_data(vec![dnrm], &[]).map_err(to_ad_err)?;
        return Ok((nrm, dnrm));
    }

    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (nrm_data, _) = extract_data(&nrm)?;
    let (da_data, _) = extract_data(tangent)?;

    let mut dnrm_data = vec![T::zero(); bc];

    match kind {
        NormKind::Fro => {
            // d||A||_F = tr(A^T dA) / ||A||_F
            for batch in 0..bc {
                let nv = nrm_data[batch];
                if nv > T::zero() {
                    let mut dot = T::zero();
                    for i in 0..m * n {
                        dot = dot + a_data[batch * m * n + i] * da_data[batch * m * n + i];
                    }
                    dnrm_data[batch] = dot / nv;
                }
            }
        }
        NormKind::Nuclear => {
            // d||A||_* = tr(U^T dA V)
            for batch in 0..bc {
                let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
                let da_b = &da_data[batch * m * n..(batch + 1) * m * n];
                let (u, _s, v) = backend_thin_svd(ctx, a_b, m, n)?;
                let k = m.min(n);
                let ut_da = backend_mat_mul(ctx, &transpose(&u, m, k), k, m, da_b, n)?;
                let ut_da_v = backend_mat_mul(ctx, &ut_da, k, n, &v, k)?;
                let mut trace = T::zero();
                for i in 0..k {
                    trace = trace + ut_da_v[i + i * k];
                }
                dnrm_data[batch] = trace;
            }
        }
        NormKind::Spectral => {
            // d||A||_2 = u1^T dA v1
            for batch in 0..bc {
                let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
                let da_b = &da_data[batch * m * n..(batch + 1) * m * n];
                let (u, _s, v) = backend_thin_svd(ctx, a_b, m, n)?;
                let mut val = T::zero();
                for i in 0..m {
                    for j in 0..n {
                        val = val + u[i] * da_b[i + j * m] * v[j];
                    }
                }
                dnrm_data[batch] = val;
            }
        }
        NormKind::L1 => {
            // d||A||_1 = sum_i sign(A_ij) dA_ij on active max columns.
            // At ties, average uniformly over active columns.
            for (batch, dn_slot) in dnrm_data.iter_mut().enumerate().take(bc) {
                if m == 0 || n == 0 {
                    continue;
                }
                let base = batch * m * n;
                let mut col_sums = vec![T::zero(); n];
                for j in 0..n {
                    let mut sum = T::zero();
                    for i in 0..m {
                        sum = sum + a_data[base + i + j * m].abs();
                    }
                    col_sums[j] = sum;
                }
                let mut max_sum = T::neg_infinity();
                for &sum in &col_sums {
                    if sum > max_sum {
                        max_sum = sum;
                    }
                }
                let active_cols: Vec<usize> = col_sums
                    .iter()
                    .enumerate()
                    .filter_map(|(j, &sum)| if sum == max_sum { Some(j) } else { None })
                    .collect();
                if active_cols.is_empty() {
                    continue;
                }
                let mut accum = T::zero();
                for j in active_cols.iter().copied() {
                    for i in 0..m {
                        let v = a_data[base + i + j * m];
                        let sign = if v > T::zero() {
                            T::one()
                        } else if v < T::zero() {
                            -T::one()
                        } else {
                            T::zero()
                        };
                        accum = accum + sign * da_data[base + i + j * m];
                    }
                }
                let active_count = scalar_from::<T>(active_cols.len() as f64).map_err(to_ad_err)?;
                *dn_slot = accum / active_count;
            }
        }
        NormKind::Inf => {
            // d||A||_inf = sum_j sign(A_ij) dA_ij on active max rows.
            // At ties, average uniformly over active rows.
            for (batch, dn_slot) in dnrm_data.iter_mut().enumerate().take(bc) {
                if m == 0 || n == 0 {
                    continue;
                }
                let base = batch * m * n;
                let mut row_sums = vec![T::zero(); m];
                for i in 0..m {
                    let mut sum = T::zero();
                    for j in 0..n {
                        sum = sum + a_data[base + i + j * m].abs();
                    }
                    row_sums[i] = sum;
                }
                let mut max_sum = T::neg_infinity();
                for &sum in &row_sums {
                    if sum > max_sum {
                        max_sum = sum;
                    }
                }
                let active_rows: Vec<usize> = row_sums
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &sum)| if sum == max_sum { Some(i) } else { None })
                    .collect();
                if active_rows.is_empty() {
                    continue;
                }
                let mut accum = T::zero();
                for i in active_rows.iter().copied() {
                    for j in 0..n {
                        let v = a_data[base + i + j * m];
                        let sign = if v > T::zero() {
                            T::one()
                        } else if v < T::zero() {
                            -T::one()
                        } else {
                            T::zero()
                        };
                        accum = accum + sign * da_data[base + i + j * m];
                    }
                }
                let active_count = scalar_from::<T>(active_rows.len() as f64).map_err(to_ad_err)?;
                *dn_slot = accum / active_count;
            }
        }
        _ => {
            return Err(chainrules_core::AutodiffError::ModeNotSupported {
                mode: "norm_frule".into(),
                reason: format!("norm kind {kind:?} AD not yet implemented"),
            });
        }
    }

    let dims = output_dims(&[], batch_dims);
    let dnrm = tensor_from_data(dnrm_data, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok((nrm, dnrm))
}
