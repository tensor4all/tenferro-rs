use std::collections::{HashMap, HashSet};
use std::ffi::c_void;

use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::cuda_ffi::{CUTENSOR_OP_ADD, CUTENSOR_OP_IDENTITY};
use crate::{mode_position, validate_rank, validate_shape_count, SemiringCoreDescriptor};

use super::planning::{plan_elementwise_binary, plan_permutation, plan_reduction};
use super::runtime::{
    allocate_workspace, new_gpu_tensor, null_stream, tensor_device_ptr_with_offset,
};
use super::scalar_type::{scalar_compute_descriptor, scalar_data_type};
use super::CudaContext;

pub(super) fn validate_diagonal_plan(
    desc: &SemiringCoreDescriptor,
    shapes: &[&[usize]],
) -> Result<()> {
    match desc {
        SemiringCoreDescriptor::Trace {
            modes_a,
            modes_c,
            paired,
        } => {
            validate_shape_count(shapes, 2, "Trace")?;
            validate_rank(shapes[0], modes_a.len(), "Trace input A")?;
            validate_rank(shapes[1], modes_c.len(), "Trace output C")?;
            let paired_axes = paired_axes_from_modes(modes_a, paired)?;
            ensure_paired_dims_match("Trace", &paired_axes, shapes[0])?;
        }
        SemiringCoreDescriptor::AntiTrace {
            modes_a,
            modes_c,
            paired,
        } => {
            validate_shape_count(shapes, 2, "AntiTrace")?;
            validate_rank(shapes[0], modes_a.len(), "AntiTrace input A")?;
            validate_rank(shapes[1], modes_c.len(), "AntiTrace output C")?;
            let paired_axes = paired_axes_from_modes(modes_c, paired)?;
            ensure_paired_dims_match("AntiTrace", &paired_axes, shapes[1])?;
        }
        SemiringCoreDescriptor::AntiDiag {
            modes_a,
            modes_c,
            paired,
        } => {
            validate_shape_count(shapes, 2, "AntiDiag")?;
            validate_rank(shapes[0], modes_a.len(), "AntiDiag input A")?;
            validate_rank(shapes[1], modes_c.len(), "AntiDiag output C")?;
            let paired_axes = paired_axes_from_modes(modes_c, paired)?;
            ensure_paired_dims_match("AntiDiag", &paired_axes, shapes[1])?;
        }
        _ => {}
    }
    Ok(())
}

pub(super) fn execute_trace<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    modes_a: &[u32],
    modes_c: &[u32],
    paired: &[(u32, u32)],
    alpha: S,
    input: &Tensor<S>,
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()> {
    let paired_axes = paired_axes_from_modes(modes_a, paired)?;
    let free_axes: Vec<usize> = modes_c
        .iter()
        .map(|mode| mode_position(modes_a, *mode))
        .collect::<Result<_>>()?;
    let (components, comp_dims) = compute_paired_components(&paired_axes, input.dims());
    let trace_view = build_trace_view(input, &free_axes, &components, &comp_dims)?;
    let diag_count = comp_dims.len();

    let data_type = scalar_data_type::<S>()?;
    let compute = scalar_compute_descriptor::<S>(&ctx.vtable)?;
    let modes_view: Vec<i32> = (0..trace_view.ndim() as i32).collect();
    let modes_out: Vec<i32> = (0..modes_c.len() as i32).collect();
    let native = plan_reduction(
        ctx,
        data_type,
        compute,
        &modes_view,
        trace_view.dims(),
        trace_view.strides(),
        &modes_out,
        output.dims(),
        output.strides(),
        CUTENSOR_OP_ADD,
    )?;
    let workspace = allocate_workspace(native.workspace_size)?;
    let ws_ptr = workspace.as_ref().map_or(std::ptr::null_mut(), |ws| ws.ptr);
    let input_ptr = tensor_device_ptr_with_offset("trace view", &trace_view)? as *const c_void;
    let output_ptr = tensor_device_ptr_with_offset("trace output", output)?;
    let status = unsafe {
        (ctx.vtable.reduce)(
            ctx.handle.raw,
            native.plan.raw,
            &alpha as *const S as *const c_void,
            input_ptr,
            &beta as *const S as *const c_void,
            output_ptr as *const c_void,
            output_ptr,
            ws_ptr,
            native.workspace_size,
            null_stream(),
        )
    };
    if diag_count == 0 {
        return Err(Error::InvalidArgument(
            "Trace requires at least one paired diagonal component".into(),
        ));
    }
    super::planning::check_status(status, "cutensorReduce")
}

pub(super) fn execute_anti_trace<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    modes_a: &[u32],
    modes_c: &[u32],
    paired: &[(u32, u32)],
    alpha: S,
    input: &Tensor<S>,
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()> {
    execute_diagonal_scatter(ctx, modes_a, modes_c, paired, alpha, input, beta, output)
}

pub(super) fn execute_anti_diag<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    modes_a: &[u32],
    modes_c: &[u32],
    paired: &[(u32, u32)],
    alpha: S,
    input: &Tensor<S>,
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()> {
    execute_diagonal_scatter(ctx, modes_a, modes_c, paired, alpha, input, beta, output)
}

fn execute_diagonal_scatter<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    modes_a: &[u32],
    modes_c: &[u32],
    paired: &[(u32, u32)],
    alpha: S,
    input: &Tensor<S>,
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()> {
    let mut working = new_gpu_tensor::<S>(output.dims(), ctx.device_id);
    copy_scaled(ctx, beta, output, &mut working)?;

    let paired_axes = paired_axes_from_modes(modes_c, paired)?;
    let free_axes: Vec<usize> = modes_a
        .iter()
        .map(|mode| mode_position(modes_c, *mode))
        .collect::<Result<_>>()?;
    let (components, comp_dims) = compute_paired_components(&paired_axes, output.dims());
    let (diag_view, generative_dims) =
        build_output_diagonal_view(&working, &free_axes, &components, &comp_dims, input.dims())?;
    let broadcast_input = broadcast_input(input, &generative_dims, diag_view.dims())?;
    add_into_view(ctx, alpha, &broadcast_input, &diag_view)?;
    copy_scaled(ctx, S::one(), &working, output)
}

fn copy_scaled<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    alpha: S,
    input: &Tensor<S>,
    output: &mut Tensor<S>,
) -> Result<()> {
    let data_type = scalar_data_type::<S>()?;
    let compute = scalar_compute_descriptor::<S>(&ctx.vtable)?;
    let modes: Vec<i32> = (0..input.ndim() as i32).collect();
    let native = plan_permutation(
        ctx,
        data_type,
        compute,
        &modes,
        input.dims(),
        input.strides(),
        &modes,
        output.dims(),
        output.strides(),
    )?;
    let input_ptr = tensor_device_ptr_with_offset("copy input", input)? as *const c_void;
    let output_ptr = tensor_device_ptr_with_offset("copy output", output)?;
    let status = unsafe {
        (ctx.vtable.permute)(
            ctx.handle.raw,
            native.plan.raw,
            &alpha as *const S as *const c_void,
            input_ptr,
            output_ptr,
            null_stream(),
        )
    };
    super::planning::check_status(status, "cutensorPermute")
}

fn add_into_view<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    alpha: S,
    input: &Tensor<S>,
    output_view: &Tensor<S>,
) -> Result<()> {
    let data_type = scalar_data_type::<S>()?;
    let compute = scalar_compute_descriptor::<S>(&ctx.vtable)?;
    let modes: Vec<i32> = (0..output_view.ndim() as i32).collect();
    let native = plan_elementwise_binary(
        ctx,
        data_type,
        compute,
        &modes,
        input.dims(),
        input.strides(),
        output_view.strides(),
        output_view.strides(),
        CUTENSOR_OP_IDENTITY,
        CUTENSOR_OP_IDENTITY,
        CUTENSOR_OP_ADD,
    )?;
    let input_ptr =
        tensor_device_ptr_with_offset("diagonal scatter input", input)? as *const c_void;
    let output_ptr = tensor_device_ptr_with_offset("diagonal scatter view", output_view)?;
    let one = S::one();
    let status = unsafe {
        (ctx.vtable.elementwise_binary_execute)(
            ctx.handle.raw,
            native.plan.raw,
            &alpha as *const S as *const c_void,
            input_ptr,
            &one as *const S as *const c_void,
            output_ptr as *const c_void,
            output_ptr,
            null_stream(),
        )
    };
    super::planning::check_status(status, "cutensorElementwiseBinaryExecute")
}

fn paired_axes_from_modes(modes: &[u32], paired: &[(u32, u32)]) -> Result<Vec<(usize, usize)>> {
    paired
        .iter()
        .map(|(m1, m2)| Ok((mode_position(modes, *m1)?, mode_position(modes, *m2)?)))
        .collect()
}

fn ensure_paired_dims_match(
    op_name: &str,
    paired_axes: &[(usize, usize)],
    shape: &[usize],
) -> Result<()> {
    for &(ax1, ax2) in paired_axes {
        if shape[ax1] != shape[ax2] {
            return Err(Error::InvalidArgument(format!(
                "{op_name} paired axes ({ax1}, {ax2}) have mismatched dimensions: {} vs {}",
                shape[ax1], shape[ax2]
            )));
        }
    }
    Ok(())
}

fn compute_paired_components(
    paired_axes: &[(usize, usize)],
    shape: &[usize],
) -> (Vec<Vec<usize>>, Vec<usize>) {
    if paired_axes.is_empty() {
        return (vec![], vec![]);
    }

    let mut all_axes: Vec<usize> = paired_axes.iter().flat_map(|(a, b)| [*a, *b]).collect();
    all_axes.sort_unstable();
    all_axes.dedup();

    let mut parent: HashMap<usize, usize> = all_axes.iter().map(|&ax| (ax, ax)).collect();

    fn find(parent: &mut HashMap<usize, usize>, x: usize) -> usize {
        let p = parent[&x];
        if p == x {
            x
        } else {
            let root = find(parent, p);
            parent.insert(x, root);
            root
        }
    }

    for &(ax1, ax2) in paired_axes {
        let r1 = find(&mut parent, ax1);
        let r2 = find(&mut parent, ax2);
        if r1 != r2 {
            let (lo, hi) = if r1 < r2 { (r1, r2) } else { (r2, r1) };
            parent.insert(hi, lo);
        }
    }

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for &ax in &all_axes {
        let root = find(&mut parent, ax);
        groups.entry(root).or_default().push(ax);
    }

    let mut components: Vec<Vec<usize>> = groups.into_values().collect();
    components.sort_by_key(|group| group[0]);
    let comp_dims = components.iter().map(|group| shape[group[0]]).collect();
    (components, comp_dims)
}

fn build_trace_view<T: Scalar>(
    input: &Tensor<T>,
    free_axes: &[usize],
    components: &[Vec<usize>],
    comp_dims: &[usize],
) -> Result<Tensor<T>> {
    let mut dims: Vec<usize> = free_axes.iter().map(|&axis| input.dims()[axis]).collect();
    let mut strides: Vec<isize> = free_axes
        .iter()
        .map(|&axis| input.strides()[axis])
        .collect();
    for (component, &dim) in components.iter().zip(comp_dims.iter()) {
        dims.push(dim);
        let stride = component.iter().try_fold(0isize, |acc, &axis| {
            acc.checked_add(input.strides()[axis])
                .ok_or_else(|| Error::InvalidArgument("trace diagonal stride overflow".into()))
        })?;
        strides.push(stride);
    }
    input.view_as_strided(dims, strides)
}

fn build_output_diagonal_view<T: Scalar>(
    output: &Tensor<T>,
    free_axes: &[usize],
    components: &[Vec<usize>],
    comp_dims: &[usize],
    input_dims: &[usize],
) -> Result<(Tensor<T>, Vec<usize>)> {
    let free_axis_set: HashSet<usize> = free_axes.iter().copied().collect();
    let component_strides: Vec<isize> = components
        .iter()
        .map(|component| {
            component.iter().try_fold(0isize, |acc, &axis| {
                acc.checked_add(output.strides()[axis])
                    .ok_or_else(|| Error::InvalidArgument("output diagonal stride overflow".into()))
            })
        })
        .collect::<Result<_>>()?;

    let mut axis_to_component = HashMap::new();
    for (component_idx, component) in components.iter().enumerate() {
        for &axis in component {
            axis_to_component.insert(axis, component_idx);
        }
    }

    let mut dims = Vec::with_capacity(input_dims.len());
    let mut strides = Vec::with_capacity(input_dims.len());
    for (input_axis, &output_axis) in free_axes.iter().enumerate() {
        dims.push(input_dims[input_axis]);
        let stride = axis_to_component
            .get(&output_axis)
            .map_or(output.strides()[output_axis], |&component_idx| {
                component_strides[component_idx]
            });
        strides.push(stride);
    }

    let generative_components: Vec<usize> = components
        .iter()
        .enumerate()
        .filter(|(_, component)| component.iter().all(|axis| !free_axis_set.contains(axis)))
        .map(|(component_idx, _)| component_idx)
        .collect();
    let generative_dims: Vec<usize> = generative_components
        .iter()
        .map(|&component_idx| comp_dims[component_idx])
        .collect();
    for &component_idx in &generative_components {
        dims.push(comp_dims[component_idx]);
        strides.push(component_strides[component_idx]);
    }

    Ok((output.view_as_strided(dims, strides)?, generative_dims))
}

fn broadcast_input<T: Scalar>(
    input: &Tensor<T>,
    generative_dims: &[usize],
    target_dims: &[usize],
) -> Result<Tensor<T>> {
    if generative_dims.is_empty() {
        return Ok(input.clone());
    }
    let mut view_dims = input.dims().to_vec();
    view_dims.extend(std::iter::repeat_n(1usize, generative_dims.len()));
    let mut view_strides = input.strides().to_vec();
    view_strides.extend(std::iter::repeat_n(0isize, generative_dims.len()));
    let view = input.view_as_strided(view_dims, view_strides)?;
    view.broadcast(target_dims)
}
