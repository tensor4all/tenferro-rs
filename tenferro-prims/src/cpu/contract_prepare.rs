use strided_view::{StridedView, StridedViewMut};
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};

use crate::infra::typed_dispatch::{
    cast_scalar_value, cast_strided_view, cast_strided_view_mut, dispatch_standard_scalar_type,
};

use super::common::is_supported_scalar_type;
use super::context::CpuContext;
#[cfg(feature = "gemm-blas")]
use super::contract_gemm::run_dense;
#[cfg(feature = "gemm-faer")]
use super::contract_gemm::run_strided;
use super::contract_gemm::{compute_layout, reordered_dims_strides};
use super::layout_fusion::try_fuse_group_in_order;
use super::plan::{build_contract_gemm_spec, ContractGemmSpec};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ContractOperandStrategy {
    Borrowed,
    Materialize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ContractOutputStrategy {
    Direct,
    Temporary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ContractPreparation {
    pub(crate) a_strategy: ContractOperandStrategy,
    pub(crate) b_strategy: ContractOperandStrategy,
    pub(crate) output_strategy: ContractOutputStrategy,
}

struct TempTensor<T> {
    data: Vec<T>,
    dims: Vec<usize>,
    strides: Vec<isize>,
}

impl<T> TempTensor<T> {
    fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T: Scalar> TempTensor<T> {
    fn view(&self) -> Result<StridedView<'_, T>> {
        StridedView::new(self.data.as_slice(), &self.dims, &self.strides, 0)
            .map_err(|e| Error::StrideError(e.to_string()))
    }

    fn view_mut(&mut self) -> Result<StridedViewMut<'_, T>> {
        StridedViewMut::new(self.data.as_mut_slice(), &self.dims, &self.strides, 0)
            .map_err(|e| Error::StrideError(e.to_string()))
    }
}

fn temp_view_in_modes<'a, T: Scalar>(
    temp: &'a TempTensor<T>,
    source_modes: &[u32],
    target_modes: &[u32],
    op: &'static str,
) -> Result<StridedView<'a, T>> {
    let view = temp.view()?;
    if source_modes == target_modes {
        return Ok(view);
    }
    let perm = perm_for(target_modes, source_modes).ok_or_else(|| {
        Error::InvalidArgument(format!(
            "{op}: unable to reorder temporary output modes {source_modes:?} into {target_modes:?}"
        ))
    })?;
    view.permute(&perm)
        .map_err(|e| Error::StrideError(e.to_string()))
}

fn temp_view_mut_in_modes<'a, T: Scalar>(
    temp: &'a mut TempTensor<T>,
    source_modes: &[u32],
    target_modes: &[u32],
    op: &'static str,
) -> Result<StridedViewMut<'a, T>> {
    let view = temp.view_mut()?;
    if source_modes == target_modes {
        return Ok(view);
    }
    let perm = perm_for(target_modes, source_modes).ok_or_else(|| {
        Error::InvalidArgument(format!(
            "{op}: unable to reorder temporary output modes {source_modes:?} into {target_modes:?}"
        ))
    })?;
    view.permute(&perm)
        .map_err(|e| Error::StrideError(e.to_string()))
}

fn perm_for(target: &[u32], source: &[u32]) -> Option<Vec<usize>> {
    target
        .iter()
        .map(|mode| source.iter().position(|candidate| candidate == mode))
        .collect()
}

fn col_major_strides(dims: &[usize]) -> Vec<isize> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut stride = 1isize;
    for &dim in dims {
        strides.push(stride);
        stride = stride
            .checked_mul(dim as isize)
            .expect("temporary tensor stride overflow");
    }
    strides
}

fn element_count(dims: &[usize]) -> usize {
    if dims.is_empty() {
        1
    } else {
        dims.iter().product()
    }
}

fn group_is_fusible(dims: &[usize], strides: &[isize]) -> bool {
    try_fuse_group_in_order(dims, strides).is_some()
}

fn operand_strategy(
    dims: &[usize],
    strides: &[isize],
    nb: usize,
    left: usize,
    right: usize,
) -> ContractOperandStrategy {
    if group_is_fusible(&dims[..nb], &strides[..nb])
        && group_is_fusible(&dims[nb..nb + left], &strides[nb..nb + left])
        && group_is_fusible(
            &dims[nb + left..nb + left + right],
            &strides[nb + left..nb + left + right],
        )
    {
        ContractOperandStrategy::Borrowed
    } else {
        ContractOperandStrategy::Materialize
    }
}

fn output_strategy(
    dims: &[usize],
    strides: &[isize],
    nb: usize,
    nm: usize,
    nn: usize,
) -> ContractOutputStrategy {
    if group_is_fusible(&dims[..nb], &strides[..nb])
        && group_is_fusible(&dims[nb..nb + nm], &strides[nb..nb + nm])
        && group_is_fusible(
            &dims[nb + nm..nb + nm + nn],
            &strides[nb + nm..nb + nm + nn],
        )
    {
        ContractOutputStrategy::Direct
    } else {
        ContractOutputStrategy::Temporary
    }
}

fn contract_preparation<T: Scalar>(
    inputs: &[&StridedView<T>],
    output: &StridedViewMut<T>,
    modes_a: &[u32],
    modes_b: &[u32],
    modes_c: &[u32],
    spec: &ContractGemmSpec,
) -> Result<Option<ContractPreparation>> {
    let (a_dims, a_strides) = reordered_dims_strides(
        modes_a,
        inputs[0].dims(),
        inputs[0].strides(),
        &spec.a_target,
    )
    .ok_or_else(|| {
        Error::InvalidArgument("Contract: failed to reorder A for preparation".into())
    })?;
    let (b_dims, b_strides) = reordered_dims_strides(
        modes_b,
        inputs[1].dims(),
        inputs[1].strides(),
        &spec.b_target,
    )
    .ok_or_else(|| {
        Error::InvalidArgument("Contract: failed to reorder B for preparation".into())
    })?;
    let (c_dims, c_strides) =
        reordered_dims_strides(modes_c, output.dims(), output.strides(), &spec.c_target)
            .ok_or_else(|| {
                Error::InvalidArgument("Contract: failed to reorder output for preparation".into())
            })?;

    let nb = spec.batch_modes.len();
    let nm = spec.m_modes.len();
    let nk = spec.k_modes.len();
    let nn = spec.n_modes.len();

    let preparation = ContractPreparation {
        a_strategy: operand_strategy(&a_dims, &a_strides, nb, nm, nk),
        b_strategy: operand_strategy(&b_dims, &b_strides, nb, nk, nn),
        output_strategy: output_strategy(&c_dims, &c_strides, nb, nm, nn),
    };
    Ok(Some(preparation))
}

#[cfg(test)]
pub(crate) fn inspect_contract_preparation<T: Scalar>(
    _ctx: &mut CpuContext,
    inputs: &[&StridedView<T>],
    output: &mut StridedViewMut<T>,
    modes_a: &[u32],
    modes_b: &[u32],
    modes_c: &[u32],
    spec: &ContractGemmSpec,
) -> Result<Option<ContractPreparation>> {
    contract_preparation(inputs, output, modes_a, modes_b, modes_c, spec)
}

fn materialize_permuted_input<T: Scalar + 'static>(
    ctx: &mut CpuContext,
    src: &StridedView<T>,
    source_modes: &[u32],
    target_modes: &[u32],
    op: &'static str,
) -> Result<TempTensor<T>> {
    let perm = perm_for(target_modes, source_modes).ok_or_else(|| {
        Error::InvalidArgument(format!(
            "{op}: unable to permute modes {source_modes:?} into target order {target_modes:?}"
        ))
    })?;
    let permuted = src
        .permute(&perm)
        .map_err(|e| Error::StrideError(e.to_string()))?;
    let dims = permuted.dims().to_vec();
    let strides = col_major_strides(&dims);
    let mut data = ctx.temp_pool_mut().take_vec::<T>(element_count(&dims));
    data.resize(element_count(&dims), T::zero());
    let mut output = StridedViewMut::new(data.as_mut_slice(), &dims, &strides, 0)
        .map_err(|e| Error::StrideError(e.to_string()))?;
    strided_perm::copy_into_par(&mut output, &permuted)
        .map_err(|e| Error::StrideError(e.to_string()))?;
    Ok(TempTensor {
        data,
        dims,
        strides,
    })
}

fn temporary_output<T: Scalar + 'static>(ctx: &mut CpuContext, dims: &[usize]) -> TempTensor<T> {
    let mut data = ctx.temp_pool_mut().take_vec::<T>(element_count(dims));
    data.resize(element_count(dims), T::zero());
    TempTensor {
        data,
        dims: dims.to_vec(),
        strides: col_major_strides(dims),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_execute_contract_gemm<T: Scalar + 'static>(
    ctx: &mut CpuContext,
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
    modes_a: &[u32],
    modes_b: &[u32],
    modes_c: &[u32],
    cached_spec: Option<&ContractGemmSpec>,
) -> Result<Option<()>> {
    let spec = if let Some(cached) = cached_spec {
        cached.clone()
    } else {
        match build_contract_gemm_spec(modes_a, modes_b, modes_c) {
            Some(spec) => spec,
            None => return Ok(None),
        }
    };

    if !is_supported_scalar_type::<T>() {
        return Ok(None);
    }

    let preparation = contract_preparation(inputs, output, modes_a, modes_b, modes_c, &spec)?
        .ok_or_else(|| Error::InvalidArgument("Contract: failed to inspect preparation".into()))?;

    let lhs_temp = if preparation.a_strategy == ContractOperandStrategy::Materialize {
        Some(materialize_permuted_input(
            ctx,
            inputs[0],
            modes_a,
            &spec.a_target,
            "Contract",
        )?)
    } else {
        None
    };
    let rhs_temp = if preparation.b_strategy == ContractOperandStrategy::Materialize {
        Some(materialize_permuted_input(
            ctx,
            inputs[1],
            modes_b,
            &spec.b_target,
            "Contract",
        )?)
    } else {
        None
    };
    let prepared_output_dims = if preparation.output_strategy == ContractOutputStrategy::Temporary {
        Some(
            reordered_dims_strides(modes_c, output.dims(), output.strides(), &spec.c_target)
                .ok_or_else(|| {
                    Error::InvalidArgument(
                        "Contract: failed to reorder output for temporary preparation".into(),
                    )
                })?
                .0,
        )
    } else {
        None
    };
    let mut out_temp = if let Some(dims) = prepared_output_dims.as_ref() {
        Some(temporary_output::<T>(ctx, dims))
    } else {
        None
    };

    let result = (|| {
        if let Some(temp) = out_temp.as_mut() {
            if beta != T::zero() {
                let mut temp_view =
                    temp_view_mut_in_modes(temp, &spec.c_target, modes_c, "Contract")?;
                strided_perm::copy_into_par(&mut temp_view, &output.as_view())
                    .map_err(|e| Error::StrideError(e.to_string()))?;
            }
        }

        let lhs_view = lhs_temp
            .as_ref()
            .map(TempTensor::view)
            .transpose()?
            .unwrap_or_else(|| inputs[0].clone());
        let rhs_view = rhs_temp
            .as_ref()
            .map(TempTensor::view)
            .transpose()?
            .unwrap_or_else(|| inputs[1].clone());
        let has_out_temp = out_temp.is_some();
        let mut out_view = out_temp.as_mut().map(TempTensor::view_mut).transpose()?;

        let prepared_modes_a = if lhs_temp.is_some() {
            spec.a_target.as_slice()
        } else {
            modes_a
        };
        let prepared_modes_b = if rhs_temp.is_some() {
            spec.b_target.as_slice()
        } else {
            modes_b
        };
        let prepared_modes_c = if has_out_temp {
            spec.c_target.as_slice()
        } else {
            modes_c
        };

        let layout = compute_layout(
            lhs_view.dims(),
            lhs_view.strides(),
            rhs_view.dims(),
            rhs_view.strides(),
            out_view
                .as_ref()
                .map_or_else(|| output.dims(), |view| view.dims()),
            out_view
                .as_ref()
                .map_or_else(|| output.strides(), |view| view.strides()),
            prepared_modes_a,
            prepared_modes_b,
            prepared_modes_c,
            &spec,
        )
        .ok_or_else(|| {
            Error::DeviceError(
                "Contract: supported scalar type could not lower to GEMM after prepared fallback"
                    .into(),
            )
        })?;

        dispatch_standard_scalar_type!(T, Concrete, {
            let a = cast_strided_view!(&lhs_view, T, Concrete);
            let b = cast_strided_view!(&rhs_view, T, Concrete);
            let c = match out_view.as_mut() {
                Some(view) => cast_strided_view_mut!(view, T, Concrete),
                None => cast_strided_view_mut!(output, T, Concrete),
            };
            let alpha = cast_scalar_value!(alpha, T, Concrete);
            let beta = cast_scalar_value!(beta, T, Concrete);

            #[cfg(feature = "gemm-faer")]
            {
                run_strided(ctx, alpha, a, b, beta, c, &layout)?;
            }
            #[cfg(all(feature = "gemm-blas", not(feature = "gemm-faer")))]
            {
                run_dense(
                    alpha,
                    a,
                    b,
                    beta,
                    c,
                    &layout,
                    <Concrete as BlasGemm>::contiguous_gemm,
                )?;
            }
        });

        if let Some(temp) = out_temp.as_ref() {
            let temp_view = temp_view_in_modes(temp, &spec.c_target, modes_c, "Contract")?;
            strided_perm::copy_into_par(output, &temp_view)
                .map_err(|e| Error::StrideError(e.to_string()))?;
        }
        Ok(())
    })();

    if let Some(temp) = lhs_temp {
        ctx.temp_pool_mut().put_vec(temp.into_vec());
    }
    if let Some(temp) = rhs_temp {
        ctx.temp_pool_mut().put_vec(temp.into_vec());
    }
    if let Some(temp) = out_temp {
        ctx.temp_pool_mut().put_vec(temp.into_vec());
    }

    result.map(Some)
}
