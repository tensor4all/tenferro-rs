use super::*;

/// Reverse-mode AD rule for norm (VJP / pullback).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{norm_rrule, NormKind};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
/// let cotangent = Tensor::<f64>::ones(&[], mem, col);
/// let grad_a = norm_rrule(&mut ctx, &a, &cotangent, NormKind::Fro).unwrap();
/// ```
pub fn norm_rrule<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
    kind: NormKind,
) -> AdResult<Tensor<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Norm, "norm_rrule")
        .map_err(to_ad_err)?;

    if tensor.ndim() == 1 {
        validate_norm_cotangent(cotangent, &[]).map_err(to_ad_err)?;
        let (a_data, _) = extract_data(tensor)?;
        let (dn_data, _) = extract_data(cotangent)?;
        let dn = dn_data[0];
        let len = tensor.dims()[0];
        let mut grad_a = vec![T::zero(); len];

        match kind {
            NormKind::Fro => {
                let nrm = norm(ctx, tensor, NormKind::Fro).map_err(to_ad_err)?;
                let (nrm_data, _) = extract_data(&nrm)?;
                let nv = nrm_data[0];
                let scale = if nv > T::zero() { dn / nv } else { T::zero() };
                for i in 0..len {
                    grad_a[i] = scale * a_data[i];
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
                    grad_a[i] = dn * sign;
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
                    let active_count = scalar_from::<T>(active.len() as f64).map_err(to_ad_err)?;
                    let scale = dn / active_count;
                    for i in active {
                        let v = a_data[i];
                        let sign = if v > T::zero() {
                            T::one()
                        } else if v < T::zero() {
                            -T::one()
                        } else {
                            T::zero()
                        };
                        grad_a[i] = scale * sign;
                    }
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
                        grad_a[i] = dn * sign;
                    }
                } else {
                    let nrm = norm(ctx, tensor, kind).map_err(to_ad_err)?;
                    let (nrm_data, _) = extract_data(&nrm)?;
                    let nv = nrm_data[0];
                    if nv > T::zero() {
                        let p_minus_one = scalar_from::<T>(p - 1.0).map_err(to_ad_err)?;
                        let scale = dn / nv.powf(p_minus_one);
                        for i in 0..len {
                            let v = a_data[i];
                            let sign = if v > T::zero() {
                                T::one()
                            } else if v < T::zero() {
                                -T::one()
                            } else {
                                T::zero()
                            };
                            grad_a[i] = scale * sign * v.abs().powf(p_minus_one);
                        }
                    }
                }
            }
            NormKind::Nuclear | NormKind::Spectral => {
                return Err(matrix_only_norm_kind_ad_error(kind));
            }
        }

        return tensor_from_data(grad_a, &[len]).map_err(to_ad_err);
    }

    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);
    validate_norm_cotangent(cotangent, batch_dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;

    let (a_data, _) = extract_data(tensor)?;
    let (dn_data, _) = extract_data(cotangent)?;

    let mut grad_a = vec![T::zero(); m * n * bc];

    match kind {
        NormKind::Fro => {
            // dA = dn * A / ||A||_F
            let nrm = norm(ctx, tensor, NormKind::Fro)
                .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
            let (nrm_data, _) = extract_data(&nrm)?;
            for batch in 0..bc {
                let dn = dn_data[batch];
                let nv = nrm_data[batch];
                let scale = if nv > T::zero() { dn / nv } else { T::zero() };
                for i in 0..m * n {
                    grad_a[batch * m * n + i] = scale * a_data[batch * m * n + i];
                }
            }
        }
        NormKind::Nuclear => {
            // dA = dn * U V^T
            for batch in 0..bc {
                let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
                let (u, _s, v) = backend_thin_svd(ctx, a_b, m, n)?;
                let k = m.min(n);
                let uv = backend_mat_mul(ctx, &u, m, k, &transpose(&v, n, k), n)?;
                let dn = dn_data[batch];
                for i in 0..m * n {
                    grad_a[batch * m * n + i] = dn * uv[i];
                }
            }
        }
        NormKind::Spectral => {
            // dA = dn * u1 v1^T (rank-1 outer product of leading singular vectors)
            for batch in 0..bc {
                let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
                let (u, _s, v) = backend_thin_svd(ctx, a_b, m, n)?;
                let dn = dn_data[batch];
                for j in 0..n {
                    for i in 0..m {
                        grad_a[batch * m * n + i + j * m] = dn * u[i] * v[j];
                    }
                }
            }
        }
        NormKind::L1 => {
            // dA = dn * sign(A) on columns that attain max absolute column sum.
            // At ties, average uniformly over active columns.
            for batch in 0..bc {
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
                let active_count = scalar_from::<T>(active_cols.len() as f64).map_err(to_ad_err)?;
                let dn = dn_data[batch] / active_count;
                for j in active_cols {
                    for i in 0..m {
                        let v = a_data[base + i + j * m];
                        let sign = if v > T::zero() {
                            T::one()
                        } else if v < T::zero() {
                            -T::one()
                        } else {
                            T::zero()
                        };
                        grad_a[base + i + j * m] = grad_a[base + i + j * m] + dn * sign;
                    }
                }
            }
        }
        NormKind::Inf => {
            // dA = dn * sign(A) on rows that attain max absolute row sum.
            // At ties, average uniformly over active rows.
            for batch in 0..bc {
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
                let active_count = scalar_from::<T>(active_rows.len() as f64).map_err(to_ad_err)?;
                let dn = dn_data[batch] / active_count;
                for i in active_rows {
                    for j in 0..n {
                        let v = a_data[base + i + j * m];
                        let sign = if v > T::zero() {
                            T::one()
                        } else if v < T::zero() {
                            -T::one()
                        } else {
                            T::zero()
                        };
                        grad_a[base + i + j * m] = grad_a[base + i + j * m] + dn * sign;
                    }
                }
            }
        }
        _ => {
            return Err(chainrules_core::AutodiffError::ModeNotSupported {
                mode: "norm_rrule".into(),
                reason: format!("norm kind {kind:?} AD not yet implemented"),
            });
        }
    }

    let dims = output_dims(&[m, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
}
