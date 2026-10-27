#![cfg(feature = "cuda")]

//! U2 first increment: the kernel catalogue is prepared once, bound on demand and
//! launched on the runtime's stream, proved against a host reference through the
//! movement entries (issue #2208 U2).

use t4a_cuda_kernels::{
    plan_contiguous_to_view, plan_copy_exact, plan_fill_zero, plan_fill_zero_view,
    plan_materialize_strided, CLayout, DType, Entry, Layout,
};
use t4a_cuda_runtime::{Arg, LaunchConfig};
use tenferro_gpu::cuda::{
    device_buffer, download_tensor, gpu_available, layout_arg, upload_tensor, CudaDeviceId,
    CudaEngine, KernelModules,
};

fn engine() -> Option<CudaEngine> {
    if !gpu_available() {
        return None;
    }
    Some(CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine"))
}

fn values(len: usize) -> Vec<f64> {
    (0..len).map(|index| index as f64 * 0.25 + 1.0).collect()
}

fn vector(data: &[f64]) -> tenferro_tensor::Tensor {
    tenferro_tensor::Tensor::from_vec_col_major(vec![data.len()], data.to_vec()).expect("vector")
}

fn threads(plan: &t4a_cuda_kernels::Launch) -> (u32, u32) {
    (plan.grid[0], plan.block[0])
}

#[test]
fn the_catalogue_prepares_every_module_once() {
    let Some(engine) = engine() else { return };
    let modules = KernelModules::new(engine.runtime()).expect("prepare");
    assert!(
        modules.modules() >= 10,
        "every translation unit is prepared, got {}",
        modules.modules()
    );
    // A second preparation for the same runtime serves the artifact cache.
    let before = engine.runtime().compile_attempts();
    let _second = KernelModules::new(engine.runtime()).expect("prepare again");
    assert_eq!(
        engine.runtime().compile_attempts(),
        before,
        "a warm prepare performs no NVRTC compilation"
    );
}

#[test]
fn copy_exact_reads_back_the_same_bytes() {
    let Some(engine) = engine() else { return };
    let runtime = engine.runtime().clone();
    let data = values(1024);
    let source = upload_tensor(&runtime, &vector(&data)).expect("upload");
    let destination = upload_tensor(&runtime, &vector(&vec![0.0; 1024])).expect("destination");

    let mut modules = KernelModules::new(&runtime).expect("prepare");
    let len = 1024i64;
    let plan = plan_copy_exact(DType::F64, len).expect("plan");
    let (grid, block) = threads(&plan);
    // SAFETY: `copy_exact` reads `len` f64 values from the source and writes the same
    // count into the destination, both full allocations of this runtime.
    unsafe {
        modules
            .launch(
                Entry::CopyExact,
                LaunchConfig::new(grid, block),
                &[
                    Arg::DevicePtr(device_buffer(&destination).expect("device")),
                    Arg::DevicePtr(device_buffer(&source).expect("device")),
                    Arg::U32(DType::F64.tag() as u32),
                    Arg::U64(len as u64),
                ],
            )
            .expect("launch");
    }
    let back = download_tensor(&runtime, &destination).expect("download");
    assert_eq!(
        back.as_slice::<f64>().expect("f64 payload"),
        data.as_slice()
    );
}

#[test]
fn materialize_strided_gathers_a_transposed_view() {
    let Some(engine) = engine() else { return };
    let runtime = engine.runtime().clone();
    let extents = [3i64, 4];
    let len: i64 = 12;
    let data = values(len as usize);
    let source = upload_tensor(&runtime, &vector(&data)).expect("upload");
    let destination = upload_tensor(&runtime, &vector(&vec![0.0; len as usize])).expect("output");

    // The view is the same extents with the two axes permuted: the gathered result is the
    // host transpose.
    let view = Layout::new(&extents, &[3, 1], 0).expect("view");
    let layout = CLayout::new(&view);
    let plan = plan_materialize_strided(DType::F64, &view, len).expect("plan");
    let (grid, block) = threads(&plan);
    let mut modules = KernelModules::new(&runtime).expect("prepare");
    // SAFETY: the plan proves the view's span and `len` is its element count.
    unsafe {
        modules
            .launch(
                Entry::MaterializeStrided,
                LaunchConfig::new(grid, block),
                &[
                    Arg::DevicePtr(device_buffer(&destination).expect("device")),
                    Arg::DevicePtr(device_buffer(&source).expect("device")),
                    Arg::U32(DType::F64.tag() as u32),
                    layout_arg(&layout),
                    Arg::U64(len as u64),
                ],
            )
            .expect("launch");
    }
    let back = download_tensor(&runtime, &destination).expect("download");
    let observed = back.as_slice::<f64>().expect("f64 payload").to_vec();
    // Element (a, b) of the transposed view is element (a*3 + b) of the source when read
    // with the column stride: the host reference walks the logical coordinates.
    let mut expected = Vec::with_capacity(len as usize);
    for flat in 0..len {
        let a = flat % extents[0];
        let b = flat / extents[0];
        let index = a * 3 + b;
        expected.push(data[index as usize]);
    }
    assert_eq!(observed, expected);
}

#[test]
fn fill_zero_and_contiguous_to_view_cover_the_movement_family() {
    let Some(engine) = engine() else { return };
    let runtime = engine.runtime().clone();
    let len: i64 = 8;
    let destination = upload_tensor(&runtime, &vector(&vec![7.0; len as usize])).expect("output");
    let mut modules = KernelModules::new(&runtime).expect("prepare");

    let plan = plan_fill_zero(DType::F64, len).expect("plan");
    let (grid, block) = threads(&plan);
    // SAFETY: `fill_zero` writes `len` f64 values into the destination allocation.
    unsafe {
        modules
            .launch(
                Entry::FillZero,
                LaunchConfig::new(grid, block),
                &[
                    Arg::DevicePtr(device_buffer(&destination).expect("device")),
                    Arg::U32(DType::F64.tag() as u32),
                    Arg::U64(len as u64),
                ],
            )
            .expect("launch");
    }
    let back = download_tensor(&runtime, &destination).expect("download");
    assert_eq!(back.as_slice::<f64>().expect("f64 payload"), &[0.0; 8][..]);

    // contiguous_to_view scatters a compact source into a strided destination view.
    let data = values(len as usize);
    let source = upload_tensor(&runtime, &vector(&data)).expect("upload");
    let extents = [4i64, 2];
    let view = Layout::new(&extents, &[2, 1], 0).expect("view");
    let layout = CLayout::new(&view);
    let out = upload_tensor(&runtime, &vector(&vec![0.0; len as usize])).expect("out");
    let plan = plan_contiguous_to_view(DType::F64, &view, len).expect("plan");
    let (grid, block) = threads(&plan);
    // SAFETY: the plan proves the destination span; the source is compact with `len`
    // elements.
    unsafe {
        modules
            .launch(
                Entry::ContiguousToView,
                LaunchConfig::new(grid, block),
                &[
                    Arg::DevicePtr(device_buffer(&out).expect("device")),
                    Arg::DevicePtr(device_buffer(&source).expect("device")),
                    Arg::U32(DType::F64.tag() as u32),
                    layout_arg(&layout),
                    Arg::U64(len as u64),
                ],
            )
            .expect("launch");
    }
    let back = download_tensor(&runtime, &out).expect("download");
    let observed = back.as_slice::<f64>().expect("f64 payload");
    // The kernel stores source element `flat` at the view's address for `flat`, so the
    // allocation read back holds the inverse permutation of the source order.
    let layout_strides = [2i64, 1];
    for flat in 0..len {
        let mut rest = flat;
        let mut index = 0i64;
        for axis in 0..extents.len() {
            let coordinate = rest % extents[axis];
            rest /= extents[axis];
            index += coordinate * layout_strides[axis];
        }
        assert_eq!(
            observed[index as usize], data[flat as usize],
            "element {flat} must land at the view address {index}"
        );
    }

    // fill_zero_view over the same layout zeroes every logical element.
    let plan = plan_fill_zero_view(DType::F64, &view, len).expect("plan");
    let (grid, block) = threads(&plan);
    // SAFETY: the plan proves the destination span and `len` is its element count.
    unsafe {
        modules
            .launch(
                Entry::FillZeroView,
                LaunchConfig::new(grid, block),
                &[
                    Arg::DevicePtr(device_buffer(&out).expect("device")),
                    Arg::U32(DType::F64.tag() as u32),
                    layout_arg(&layout),
                    Arg::U64(len as u64),
                ],
            )
            .expect("launch");
    }
    let back = download_tensor(&runtime, &out).expect("download");
    assert_eq!(
        back.as_slice::<f64>().expect("f64 payload"),
        &vec![0.0; 8][..]
    );
}
