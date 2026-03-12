use strided_view::{StridedView, StridedViewMut};
use tenferro_algebra::Scalar;
use tenferro_device::Result;

use crate::for_each_index;

/// Unflatten a linear index into a pre-allocated buffer (column-major).
fn unflatten_index_into(mut flat: usize, dims: &[usize], out: &mut [usize]) {
    debug_assert!(
        flat < dims.iter().product::<usize>(),
        "flat index {flat} out of range for dims {dims:?}"
    );
    for d in 0..dims.len() {
        out[d] = flat % dims[d];
        flat /= dims[d];
    }
}

/// Scale all elements of the output by `beta`, or zero them if `beta == 0`.
pub(super) fn scale_output<T: Scalar>(output: &mut StridedViewMut<T>, beta: T) {
    let dims = output.dims().to_vec();
    if beta == T::zero() {
        for_each_index(&dims, |idx| {
            output.set(idx, T::zero());
        });
    } else if beta != T::one() {
        for_each_index(&dims, |idx| {
            let old = output.get(idx);
            output.set(idx, beta * old);
        });
    }
}

pub(super) fn execute_reduce_sum<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    reduced_axes: &[usize],
) -> Result<()> {
    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&ax| in_dims[ax]).collect();
    let reduced_total: usize = reduced_dims.iter().product();
    let mut red_idx = vec![0usize; reduced_dims.len()];
    let mut in_idx = vec![0usize; in_dims.len()];

    for_each_index(&out_dims, |out_idx| {
        let mut sum = T::zero();
        for red_flat in 0..reduced_total {
            unflatten_index_into(red_flat, &reduced_dims, &mut red_idx);
            let mut out_pos = 0;
            let mut red_pos = 0;
            for (ax, in_slot) in in_idx.iter_mut().enumerate().take(in_dims.len()) {
                if red_pos < reduced_axes.len() && reduced_axes[red_pos] == ax {
                    *in_slot = red_idx[red_pos];
                    red_pos += 1;
                } else {
                    *in_slot = out_idx[out_pos];
                    out_pos += 1;
                }
            }
            sum = sum + input.get(&in_idx);
        }
        let old = if beta == T::zero() {
            T::zero()
        } else {
            beta * output.get(out_idx)
        };
        output.set(out_idx, alpha * sum + old);
    });
    Ok(())
}

pub(super) fn execute_trace<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    components: &[Vec<usize>],
    comp_dims: &[usize],
    free_axes: &[usize],
) -> Result<()> {
    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let n_comps = comp_dims.len();
    let mut comp_idx = vec![0usize; n_comps];
    let mut in_idx = vec![0usize; in_dims.len()];

    for_each_index(&out_dims, |out_idx| {
        let mut sum = T::zero();
        comp_idx.fill(0);
        loop {
            for (out_pos, &in_ax) in free_axes.iter().enumerate() {
                in_idx[in_ax] = out_idx[out_pos];
            }
            for (t, comp) in components.iter().enumerate() {
                for &ax in comp {
                    in_idx[ax] = comp_idx[t];
                }
            }
            sum = sum + input.get(&in_idx);

            let mut carry = true;
            for t in 0..n_comps {
                if carry {
                    comp_idx[t] += 1;
                    if comp_idx[t] < comp_dims[t] {
                        carry = false;
                    } else {
                        comp_idx[t] = 0;
                    }
                }
            }
            if carry {
                break;
            }
        }
        let old = if beta == T::zero() {
            T::zero()
        } else {
            beta * output.get(out_idx)
        };
        output.set(out_idx, alpha * sum + old);
    });
    Ok(())
}

pub(super) fn execute_anti_trace<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    components: &[Vec<usize>],
    comp_dims: &[usize],
    free_axes: &[usize],
) -> Result<()> {
    scale_output(output, beta);

    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let n_comps = comp_dims.len();
    let mut comp_idx = vec![0usize; n_comps];
    let mut out_idx = vec![0usize; out_dims.len()];

    for_each_index(&in_dims, |in_idx| {
        let val = alpha * input.get(in_idx);
        comp_idx.fill(0);
        loop {
            for (in_pos, &out_ax) in free_axes.iter().enumerate() {
                out_idx[out_ax] = in_idx[in_pos];
            }
            for (t, comp) in components.iter().enumerate() {
                for &ax in comp {
                    out_idx[ax] = comp_idx[t];
                }
            }
            let old = output.get(&out_idx);
            output.set(&out_idx, old + val);

            let mut carry = true;
            for t in 0..n_comps {
                if carry {
                    comp_idx[t] += 1;
                    if comp_idx[t] < comp_dims[t] {
                        carry = false;
                    } else {
                        comp_idx[t] = 0;
                    }
                }
            }
            if carry {
                break;
            }
        }
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_anti_diag<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    components: &[Vec<usize>],
    comp_dims: &[usize],
    free_axes: &[usize],
    generative_comps: &[usize],
) -> Result<()> {
    scale_output(output, beta);

    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let gen_dims: Vec<usize> = generative_comps.iter().map(|&c| comp_dims[c]).collect();
    let mut generative_pos_by_component = vec![None; components.len()];
    for (gen_pos, &component) in generative_comps.iter().enumerate() {
        generative_pos_by_component[component] = Some(gen_pos);
    }

    let mut gen_idx = vec![0usize; generative_comps.len()];
    let mut out_idx = vec![0usize; out_dims.len()];

    for_each_index(&in_dims, |in_idx| {
        let val = alpha * input.get(in_idx);
        gen_idx.fill(0);
        loop {
            for (in_pos, &out_ax) in free_axes.iter().enumerate() {
                out_idx[out_ax] = in_idx[in_pos];
            }
            for (component_idx, comp) in components.iter().enumerate() {
                if let Some(gen_pos) = generative_pos_by_component[component_idx] {
                    for &ax in comp {
                        out_idx[ax] = gen_idx[gen_pos];
                    }
                } else {
                    let anchor_val = out_idx[comp[0]];
                    for &ax in &comp[1..] {
                        out_idx[ax] = anchor_val;
                    }
                }
            }
            let old = output.get(&out_idx);
            output.set(&out_idx, old + val);

            if gen_dims.is_empty() {
                break;
            }
            let mut carry = true;
            for g in 0..gen_dims.len() {
                if carry {
                    gen_idx[g] += 1;
                    if gen_idx[g] < gen_dims[g] {
                        carry = false;
                    } else {
                        gen_idx[g] = 0;
                    }
                }
            }
            if carry {
                break;
            }
        }
    });
    Ok(())
}
