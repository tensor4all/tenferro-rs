use strided_view::{StridedView, StridedViewMut};
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};

use crate::SemiringBinaryOp;

use super::context::CpuContext;
use super::contract_prepare::try_execute_contract_gemm;
use super::plan::ContractGemmSpec;
use super::reduction::scale_output;

pub(super) fn execute_elementwise_binary<T: Scalar>(
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
    op: SemiringBinaryOp,
) -> Result<()> {
    let a = inputs[0];
    let b = inputs[1];

    if beta == T::zero() {
        let alpha_val = alpha;
        strided_kernel::zip_map2_into(output, a, b, move |a_val, b_val| {
            let fused = match op {
                SemiringBinaryOp::Add => a_val + b_val,
                SemiringBinaryOp::Mul => a_val * b_val,
            };
            alpha_val * fused
        })
        .map_err(|e| Error::DeviceError(e.to_string()))?;
    } else {
        let dims = output.dims().to_vec();
        crate::for_each_index(&dims, |idx| {
            let fused = match op {
                SemiringBinaryOp::Add => a.get(idx) + b.get(idx),
                SemiringBinaryOp::Mul => a.get(idx) * b.get(idx),
            };
            let val = alpha * fused;
            output.set(idx, val + beta * output.get(idx));
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_contract<T: Scalar + 'static>(
    ctx: &mut CpuContext,
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
    modes_a: &[u32],
    modes_b: &[u32],
    modes_c: &[u32],
    cached_gemm_spec: Option<&ContractGemmSpec>,
) -> Result<()> {
    if inputs.iter().any(|view| view.dims().contains(&0)) || output.dims().contains(&0) {
        scale_output(output, beta);
        return Ok(());
    }

    if let Some(done) = try_execute_contract_gemm(
        ctx,
        alpha,
        inputs,
        beta,
        output,
        modes_a,
        modes_b,
        modes_c,
        cached_gemm_spec,
    )? {
        return Ok(done);
    }

    let mut contracted_modes = Vec::new();
    for &mode in modes_a.iter().chain(modes_b.iter()) {
        if !modes_c.contains(&mode) && !contracted_modes.contains(&mode) {
            contracted_modes.push(mode);
        }
    }
    let mut contracted_dims = Vec::with_capacity(contracted_modes.len());
    for &mode in &contracted_modes {
        if let Some(a_pos) = modes_a.iter().position(|&mm| mm == mode) {
            contracted_dims.push(inputs[0].dims()[a_pos]);
        } else if let Some(b_pos) = modes_b.iter().position(|&mm| mm == mode) {
            contracted_dims.push(inputs[1].dims()[b_pos]);
        } else {
            return Err(Error::InvalidArgument(format!(
                "Contract reduction mode {mode} appears in neither input operand"
            )));
        }
    }
    let contracted_total: usize = if contracted_dims.is_empty() {
        1
    } else {
        contracted_dims.iter().product()
    };

    let a_axis_map: Vec<(u8, usize)> = modes_a
        .iter()
        .map(|&mode| {
            if let Some(c_pos) = modes_c.iter().position(|&m| m == mode) {
                (0, c_pos)
            } else if let Some(k_pos) = contracted_modes.iter().position(|&m| m == mode) {
                (1, k_pos)
            } else {
                unreachable!("every A-only mode absent from C must be reduced")
            }
        })
        .collect();
    let b_axis_map: Vec<(u8, usize)> = modes_b
        .iter()
        .map(|&mode| {
            if let Some(c_pos) = modes_c.iter().position(|&m| m == mode) {
                (0, c_pos)
            } else if let Some(k_pos) = contracted_modes.iter().position(|&m| m == mode) {
                (1, k_pos)
            } else {
                unreachable!("every B-only mode absent from C must be reduced")
            }
        })
        .collect();

    fn unflatten_into(mut flat: usize, dims: &[usize], out: &mut [usize]) {
        debug_assert_eq!(dims.len(), out.len());
        debug_assert!(dims.iter().all(|&d| d > 0));
        debug_assert!(flat < dims.iter().product::<usize>());
        for (i, &d) in dims.iter().enumerate() {
            out[i] = flat % d;
            flat /= d;
        }
    }

    let out_dims = output.dims().to_vec();
    let mut a_idx = vec![0usize; modes_a.len()];
    let mut b_idx = vec![0usize; modes_b.len()];
    let mut k_idx = vec![0usize; contracted_dims.len()];

    crate::for_each_index(&out_dims, |c_idx| {
        let mut sum = T::zero();
        for k_flat in 0..contracted_total {
            if !contracted_dims.is_empty() {
                unflatten_into(k_flat, &contracted_dims, &mut k_idx);
            }
            for (ax, &(src, pos)) in a_axis_map.iter().enumerate() {
                a_idx[ax] = if src == 0 { c_idx[pos] } else { k_idx[pos] };
            }
            for (ax, &(src, pos)) in b_axis_map.iter().enumerate() {
                b_idx[ax] = if src == 0 { c_idx[pos] } else { k_idx[pos] };
            }
            sum = sum + inputs[0].get(&a_idx) * inputs[1].get(&b_idx);
        }
        let old = if beta == T::zero() {
            T::zero()
        } else {
            beta * output.get(c_idx)
        };
        output.set(c_idx, alpha * sum + old);
    });
    Ok(())
}
