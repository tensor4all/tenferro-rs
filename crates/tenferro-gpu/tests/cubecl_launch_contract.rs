use std::{fs, path::Path};

fn cubecl_source(file: &str) -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("cubecl")
            .join(file),
    )
    .unwrap_or_else(|err| panic!("CubeCL source {file} should be readable: {err}"))
}

fn gpu_source(path: &[&str]) -> String {
    let mut full_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    for component in path {
        full_path = full_path.join(component);
    }
    fs::read_to_string(&full_path)
        .unwrap_or_else(|err| panic!("GPU source {full_path:?} should be readable: {err}"))
}

fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start_idx = source
        .find(start)
        .unwrap_or_else(|| panic!("source should contain section start {start:?}"));
    let remaining = &source[start_idx..];
    let end_idx = remaining
        .find(end)
        .map(|offset| start_idx + offset)
        .unwrap_or(source.len());
    &source[start_idx..end_idx]
}

fn source_tail<'a>(source: &'a str, start: &str) -> &'a str {
    let start_idx = source
        .find(start)
        .unwrap_or_else(|| panic!("source should contain section start {start:?}"));
    &source[start_idx..]
}

fn assert_ordered_needles(source_name: &str, source: &str, needles: &[&str]) {
    let mut offset = 0;
    for needle in needles {
        let remaining = &source[offset..];
        let found = remaining.find(needle).unwrap_or_else(|| {
            panic!("{source_name} should contain {needle:?} after byte offset {offset}")
        });
        offset += found + needle.len();
    }
}

#[test]
fn cubecl_scatter_does_not_use_single_thread_launch_fallback() {
    let mod_source = cubecl_source("mod.rs");
    let scatter_source = source_section(&mod_source, "    fn scatter(", "    fn slice(");
    let dispatch_source = cubecl_source("dispatch.rs");
    let sources = [
        ("cubecl/mod.rs scatter body", scatter_source),
        ("cubecl/dispatch.rs", dispatch_source.as_str()),
    ];
    let banned = ["single_thread_launch_config", "CubeCount::new_single()"];

    let mut violations = Vec::new();
    for (name, source) in sources {
        for needle in banned {
            if source.contains(needle) {
                violations.push(format!("{name} contains {needle}"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "CubeCL scatter launch must not use a single-thread fallback:\n{}",
        violations.join("\n")
    );
}

#[test]
fn cubecl_zero_length_launches_validate_buffers_before_returning() {
    let dispatch_source = cubecl_source("dispatch.rs");
    let dispatch_contracts = [
        (
            "launch_unary",
            "pub(crate) fn launch_unary<",
            "pub(crate) fn launch_unary_tensor<",
            vec![
                "ensure_resident_on_runtime(rt, input, op)?;",
                "let input_arg = typed_tensor_array_arg(input, op)?;",
                "if len == 0",
            ],
        ),
        (
            "launch_unary_tensor",
            "pub(crate) fn launch_unary_tensor<",
            "pub(crate) fn launch_nullary_into<",
            vec![
                "ensure_resident_on_runtime(rt, input, op)?;",
                "let input_arg = typed_tensor_binding(input, op)?;",
                "if len == 0",
            ],
        ),
        (
            "launch_nullary_into",
            "pub(crate) fn launch_nullary_into<",
            "pub(crate) fn launch_unary_tensor_into<",
            vec![
                "ensure_resident_on_runtime(rt, output, op)?;",
                "let output_arg = typed_tensor_array_arg(output, op)?;",
                "if output.n_elements() == 0",
            ],
        ),
        (
            "launch_unary_tensor_into",
            "pub(crate) fn launch_unary_tensor_into<",
            "pub(crate) fn launch_binary<",
            vec![
                "ensure_resident_on_runtime(rt, output, op)?;",
                "ensure_resident_on_runtime(rt, input, op)?;",
                "let output_arg = typed_tensor_binding(output, op)?;",
                "let input_arg = typed_tensor_binding(input, op)?;",
                "if output.n_elements() == 0",
            ],
        ),
        (
            "launch_binary",
            "pub(crate) fn launch_binary<",
            "pub(crate) fn launch_compare_bool<",
            vec![
                "ensure_resident_on_runtime(rt, lhs, op)?;",
                "ensure_resident_on_runtime(rt, rhs, op)?;",
                "let lhs_arg = typed_tensor_array_arg(lhs, op)?;",
                "let rhs_arg = typed_tensor_array_arg(rhs, op)?;",
                "if len == 0",
            ],
        ),
        (
            "launch_compare_bool",
            "pub(crate) fn launch_compare_bool<",
            "pub(crate) fn launch_binary_tensor<",
            vec![
                "ensure_resident_on_runtime(rt, lhs, op)?;",
                "ensure_resident_on_runtime(rt, rhs, op)?;",
                "let lhs_arg = typed_tensor_array_arg(lhs, op)?;",
                "let rhs_arg = typed_tensor_array_arg(rhs, op)?;",
                "if len == 0",
            ],
        ),
        (
            "launch_binary_tensor",
            "pub(crate) fn launch_binary_tensor<",
            "pub(crate) fn launch_select_bool<",
            vec![
                "ensure_resident_on_runtime(rt, lhs, op)?;",
                "ensure_resident_on_runtime(rt, rhs, op)?;",
                "let lhs_arg = typed_tensor_binding(lhs, op)?;",
                "let rhs_arg = typed_tensor_binding(rhs, op)?;",
                "if len == 0",
            ],
        ),
        (
            "launch_select_bool",
            "pub(crate) fn launch_select_bool<",
            "pub(crate) fn launch_ternary<",
            vec![
                "ensure_resident_on_runtime(rt, pred, op)?;",
                "ensure_resident_on_runtime(rt, on_true, op)?;",
                "ensure_resident_on_runtime(rt, on_false, op)?;",
                "let pred_arg = bool_tensor_array_arg(pred, op)?;",
                "let true_arg = typed_tensor_array_arg(on_true, op)?;",
                "let false_arg = typed_tensor_array_arg(on_false, op)?;",
                "if len == 0",
            ],
        ),
        (
            "launch_ternary",
            "pub(crate) fn launch_ternary<",
            "pub(crate) fn dtype_mismatch(",
            vec![
                "ensure_resident_on_runtime(rt, a, op)?;",
                "ensure_resident_on_runtime(rt, b, op)?;",
                "ensure_resident_on_runtime(rt, c, op)?;",
                "let a_arg = typed_tensor_array_arg(a, op)?;",
                "let b_arg = typed_tensor_array_arg(b, op)?;",
                "let c_arg = typed_tensor_array_arg(c, op)?;",
                "if len == 0",
            ],
        ),
    ];
    for (name, start, end, needles) in dispatch_contracts {
        let section = source_section(&dispatch_source, start, end);
        assert_ordered_needles(name, section, &needles);
    }

    let fusion_source = cubecl_source("fusion/launch.rs");
    assert_ordered_needles(
        "fusion::launch",
        &fusion_source,
        &[
            "ensure_resident_on_runtime(runtime, input, \"fused_elementwise\")?;",
            "typed_tensor_array_arg(input, \"fused_elementwise\")?;",
            "typed_tensor_array_arg(output, \"fused_elementwise\")?;",
            "if classified.n_elements == 0",
        ],
    );
}

#[test]
fn cubecl_binary_elementwise_kernels_broadcast_zero_dim_scalars() {
    let dispatch_source = cubecl_source("dispatch.rs");
    let binary_macro = source_section(
        &dispatch_source,
        "macro_rules! launch_binary_elementwise_kernel",
        "macro_rules! dispatch_binary_float_complex_int",
    );

    assert_ordered_needles(
        "launch_binary_elementwise_kernel scalar lhs broadcast",
        binary_macro,
        &[
            "$lhs.shape().is_empty()",
            "let lhs = $backend.broadcast_typed($lhs, $rhs.shape(), &[])?;",
            "launch_binary(",
        ],
    );
    assert_ordered_needles(
        "launch_binary_elementwise_kernel scalar rhs broadcast",
        binary_macro,
        &[
            "$rhs.shape().is_empty()",
            "let rhs = $backend.broadcast_typed($rhs, $lhs.shape(), &[])?;",
            "launch_binary(",
        ],
    );
}

#[test]
fn cubecl_interop_download_validates_buffer_before_empty_fast_path() {
    let interop_source = cubecl_source("interop.rs");
    let download_source = source_section(
        &interop_source,
        "pub fn download_typed_tensor<",
        "/// Allocate a CubeCL-owned byte workspace",
    );

    assert_ordered_needles(
        "interop::download_typed_tensor",
        download_source,
        &[
            "dispatch::ensure_resident_on_runtime(rt, tensor, op)?;",
            "let buffer = dispatch::cubecl_buffer(tensor, op)?;",
            "if tensor.n_elements() == 0",
            "rt.synchronize()?;",
            ".read_one(buffer.handle().clone())",
        ],
    );
}

#[test]
fn cubecl_raw_device_pointer_paths_validate_runtime_residency() {
    let memory_source = cubecl_source("memory.rs");
    let memory_ptr = source_section(
        &memory_source,
        "pub fn device_ptr(rt: &CudaRuntime, tensor: &Tensor) -> crate::Result<u64> {",
        "fn upload_typed<",
    );
    assert_ordered_needles(
        "memory::device_ptr",
        memory_ptr,
        &[
            "ensure_tensor_resident_on_runtime(rt, tensor, \"device_ptr\")?;",
            "let handle = cubecl_handle(tensor)?;",
            ".get_resource(handle)",
        ],
    );

    let interop_source = cubecl_source("interop.rs");
    let interop_ptr = source_section(
        &interop_source,
        "pub fn typed_device_ptr<T: 'static>(",
        "/// Upload host data into a dense GPU tensor",
    );
    assert_ordered_needles(
        "interop::typed_device_ptr",
        interop_ptr,
        &[
            "dispatch::ensure_resident_on_runtime(rt, tensor, op)?;",
            "dispatch::cubecl_buffer(tensor, op)?;",
            ".get_resource(buffer.handle().clone())",
        ],
    );

    let gemm_source = cubecl_source("gemm.rs");
    let gemm_ptr = source_section(
        &gemm_source,
        "fn typed_device_ptr<T: 'static>(",
        "fn zero_alloc<T>",
    );
    assert_ordered_needles(
        "gemm::typed_device_ptr",
        gemm_ptr,
        &[
            "ensure_resident_on_runtime(rt, tensor, OP)?;",
            "cubecl_buffer(tensor, OP)?;",
            ".get_resource(buffer.handle().clone())",
        ],
    );
}

#[test]
fn cubecl_host_download_paths_synchronize_before_reading() {
    let memory_source = cubecl_source("memory.rs");
    let typed_download = source_section(
        &memory_source,
        "fn download_typed<T: CubeElement + Clone + 'static>(",
        "fn upload_bool(",
    );
    assert_ordered_needles(
        "memory::download_typed",
        typed_download,
        &[
            "if typed.n_elements() == 0",
            "rt.synchronize()?;",
            ".read_one(handle)",
        ],
    );

    let bool_download = source_section(&memory_source, "fn download_bool(", "fn cubecl_handle(");
    assert_ordered_needles(
        "memory::download_bool",
        bool_download,
        &[
            "if typed.n_elements() == 0",
            "rt.synchronize()?;",
            ".read_one(handle)",
        ],
    );
}

#[test]
fn cubecl_scatter_validates_all_device_inputs_before_binding() {
    let mod_source = cubecl_source("mod.rs");
    let scatter_float = source_section(
        &mod_source,
        "    fn scatter_float_typed<",
        "    fn scatter_complex_typed<",
    );
    assert_ordered_needles(
        "scatter_float_typed",
        scatter_float,
        &[
            "ensure_resident_on_runtime(self.runtime(), scatter_indices, \"scatter\")?;",
            "ensure_resident_on_runtime(self.runtime(), updates, \"scatter\")?;",
            "typed_tensor_binding(scatter_indices, \"scatter\")?;",
            "typed_tensor_binding(updates, \"scatter\")?;",
        ],
    );

    let scatter_complex = source_section(
        &mod_source,
        "    fn scatter_complex_typed<",
        "impl BackendRuntimeCache for CudaBackend",
    );
    assert_ordered_needles(
        "scatter_complex_typed",
        scatter_complex,
        &[
            "ensure_resident_on_runtime(self.runtime(), scatter_indices, \"scatter\")?;",
            "ensure_resident_on_runtime(self.runtime(), updates, \"scatter\")?;",
            "typed_tensor_binding(scatter_indices, \"scatter\")?;",
            "typed_tensor_binding(updates, \"scatter\")?;",
        ],
    );
}

#[test]
fn cubecl_indexing_kernels_use_saturating_window_arithmetic() {
    let indexing_source = gpu_source(&["kernels", "indexing.rs"]);
    let clamp = source_section(
        &indexing_source,
        "pub(crate) fn clamp_window_start",
        "#[cube]\npub(crate) fn index_component",
    );
    assert!(
        clamp.contains("dim_size.saturating_sub(window_size)"),
        "GPU gather/scatter clamp_window_start must not underflow when window_size exceeds dim_size"
    );
    assert!(
        !clamp.contains("dim_size - window_size"),
        "GPU clamp_window_start must not use unchecked usize subtraction"
    );

    let scatter_float = source_section(
        &indexing_source,
        "pub fn scatter_float_kernel",
        "#[cube(launch_unchecked)]\npub fn scatter_complex_kernel",
    );
    assert!(
        scatter_float.contains("clamp_window_start::<I>"),
        "GPU float scatter must clamp out-of-range starts like the CPU backend"
    );
    assert!(
        !scatter_float.contains("start < I::from_int(0)"),
        "GPU float scatter must not skip negative starts instead of clamping them"
    );

    let scatter_complex = source_tail(&indexing_source, "pub fn scatter_complex_kernel");
    assert!(
        scatter_complex.contains("clamp_window_start::<I>"),
        "GPU complex scatter must clamp out-of-range starts like the CPU backend"
    );
    assert!(
        !scatter_complex.contains("start < I::from_int(0)"),
        "GPU complex scatter must not skip negative starts instead of clamping them"
    );

    let structural_source = gpu_source(&["kernels", "structural.rs"]);
    let reverse = source_section(
        &structural_source,
        "pub fn reverse_kernel",
        "#[cube(launch_unchecked)]\npub fn concatenate_copy_kernel",
    );
    assert!(
        reverse.contains("dim.saturating_sub(1)"),
        "GPU reverse_kernel should guard zero-sized dimensions with saturating_sub"
    );
    assert!(
        !reverse.contains("dim - 1"),
        "GPU reverse_kernel must not compute dim - 1 directly"
    );
}

#[test]
fn cubecl_gather_and_pad_validate_shape_bounds_before_launch() {
    let mod_source = cubecl_source("mod.rs");
    let gather_meta = source_section(
        &mod_source,
        "fn gather_launch_meta(",
        "struct ScatterLaunchMeta",
    );
    assert!(
        gather_meta.contains("validate_slice_sizes_within_operand(\"gather\""),
        "GPU gather launch metadata must reject per-axis slice sizes larger than the operand"
    );

    let pad_shape = source_section(&mod_source, "fn pad_output_shape(", "fn index_vector_size(");
    assert!(
        pad_shape.contains("i64::try_from(input_shape[axis])"),
        "GPU pad output shape must not cast usize dimensions to i64 with `as`"
    );
    assert!(
        !pad_shape.contains("input_shape[axis] as i64"),
        "GPU pad output shape must use checked conversion before signed arithmetic"
    );
}

#[test]
fn cubecl_scatter_reports_unsupported_integer_operand_dtypes() {
    let mod_source = cubecl_source("mod.rs");
    let scatter_source = source_section(&mod_source, "    fn scatter(", "    fn slice(");
    for needle in [
        "(Tensor::I32(_), _, _)",
        "(Tensor::I64(_), _, _)",
        "(Tensor::Bool(_), _, _)",
    ] {
        assert!(
            scatter_source.contains(needle),
            "GPU scatter should explicitly reject unsupported operand dtype arm {needle}"
        );
    }
    assert!(
        scatter_source.contains("Err(unsupported_dtype(\"scatter\", operand.dtype()))"),
        "GPU scatter unsupported operand arms should report unsupported dtype rather than ternary mismatch"
    );
}

#[test]
fn cubecl_runtime_initializes_context_before_client_and_syncs_on_drop() {
    let runtime_source = cubecl_source("runtime.rs");
    let new_source = source_section(
        &runtime_source,
        "    pub fn new(device_ordinal: usize) -> crate::Result<Self> {",
        "    pub(crate) fn client(&self)",
    );
    assert_ordered_needles(
        "CudaRuntime::new",
        new_source,
        &[
            "cudarc::runtime::result::device::set",
            "cudarc::driver::result::init()",
            "let primary_context = CudaPrimaryContext::retain(cuda_device)?;",
            "cudarc::driver::result::ctx::set_current",
            "let device = CudaDevice::new(device_ordinal);",
            "let client =",
            "::client(&device);",
        ],
    );

    let drop_source = source_section(&runtime_source, "impl Drop for CudaRuntime", "}");
    assert_ordered_needles(
        "CudaRuntime::drop",
        drop_source,
        &[
            "if let Err(err) = self.synchronize()",
            "report_cuda_runtime_drop_error(&err);",
        ],
    );
}

#[test]
fn cubecl_gemm_zero_contracting_path_stays_device_native() {
    let gemm_source = cubecl_source("gemm.rs");
    let zero_alloc_source = source_section(&gemm_source, "fn zero_alloc<", "fn build_layout<");

    for banned in [
        "vec![T::zero(); len]",
        "create_from_slice(T::as_bytes(&zeros))",
    ] {
        assert!(
            !zero_alloc_source.contains(banned),
            "CubeCL GEMM zero-contracting fast path must not materialize host zeros: {banned}"
        );
    }
    assert_ordered_needles(
        "gemm::zero_alloc",
        zero_alloc_source,
        &[
            "alloc_output::<T>(rt, shape)",
            "structural::fill_zero_kernel",
        ],
    );
}

#[test]
fn cubecl_raw_device_pointer_paths_use_exposed_provenance() {
    for (name, source) in [
        ("cubecl/interop.rs", cubecl_source("interop.rs")),
        ("cubecl/gemm.rs", cubecl_source("gemm.rs")),
    ] {
        assert!(
            !source.contains("as usize as *mut c_void"),
            "{name} must not recreate raw CUDA pointers through an integer-pointer roundtrip"
        );
        assert!(
            source.contains("cuda_device_ptr_from_addr"),
            "{name} should centralize CUDA device address conversion through the provenance-aware helper"
        );
    }
}

#[test]
fn cubecl_runtime_uses_primary_context_guard_during_initialization() {
    let runtime_source = cubecl_source("runtime.rs");
    assert!(
        runtime_source.contains("struct CudaPrimaryContext"),
        "CudaRuntime initialization should retain the CUDA primary context through an RAII guard"
    );
    assert!(
        runtime_source.contains("primary_context: CudaPrimaryContext"),
        "CudaRuntime should own the retained primary context guard"
    );
    assert!(
        runtime_source.contains("impl Drop for CudaPrimaryContext"),
        "Cuda primary context release should be tied to the guard Drop implementation"
    );
    assert!(
        !runtime_source.contains("let _ = unsafe { cudarc::driver::result::primary_ctx::release"),
        "Cuda primary context release status should not be silently discarded from CudaRuntime::drop"
    );
}

#[test]
fn cubecl_extension_cache_guard_validates_downcast_before_deref() {
    let mod_source = cubecl_source("mod.rs");
    let cache_source = source_section(
        &mod_source,
        "pub fn get_or_try_init<T>",
        "impl Default for CudaExtensionCache",
    );
    let guard_source = source_section(
        &mod_source,
        "pub struct CudaExtensionCacheGuard",
        "impl CudaBackend",
    );

    assert!(
        cache_source.contains("downcast_ref::<T>()"),
        "CudaExtensionCacheGuard construction should validate the cached value type before returning"
    );
    assert!(
        guard_source.contains("value:"),
        "CudaExtensionCacheGuard should store a typed pointer validated during construction"
    );
    assert!(
        !guard_source
            .contains(".expect(\"CudaExtensionCache stored value under the wrong TypeId\")"),
        "CudaExtensionCacheGuard::deref should not panic on cache corruption"
    );
}

#[test]
fn cutensor_drop_paths_report_destroy_status() {
    let source = cubecl_source("ffi/cutensor.rs");
    for banned in [
        "let _ = unsafe { (self.lib.vtable.destroy)(self.raw) };",
        "let _ = unsafe { (self.lib.vtable.destroy_tensor_descriptor)(self.raw) };",
        "let _ = unsafe { (self.lib.vtable.destroy_operation_descriptor)(self.raw) };",
        "let _ = unsafe { (self.lib.vtable.destroy_plan_preference)(self.raw) };",
        "let _ = unsafe { (self.lib.vtable.destroy_plan)(self.raw) };",
    ] {
        assert!(
            !source.contains(banned),
            "cuTENSOR Drop paths must inspect destroy status instead of discarding it: found {banned}"
        );
    }
    assert!(
        source.contains("report_cutensor_destroy_status"),
        "cuTENSOR Drop paths should share a helper that reports non-success destroy statuses"
    );
}

#[test]
fn cutensor_data_symbols_validate_pointer_before_deref() {
    let source = cubecl_source("ffi/cutensor.rs");
    let section = source_section(
        &source,
        "unsafe fn load_data_symbol",
        "struct CutensorLibrary",
    );

    assert!(
        !section.contains("Ok(**symbol)"),
        "cuTENSOR data symbol loading must not blindly double-deref the exported pointer"
    );
    assert!(
        section.contains("let ptr = *symbol;"),
        "cuTENSOR data symbol loading should name the exported pointer before validation"
    );
    assert!(
        section.contains("ptr.is_null()") && section.contains("ptr.is_aligned()"),
        "cuTENSOR data symbol loading must reject null or misaligned descriptor pointers"
    );
    assert!(
        section.contains("std::ptr::read(ptr)"),
        "cuTENSOR data symbol loading should read the validated data symbol pointer explicitly"
    );
}

#[test]
fn cubecl_i64_index_conversion_does_not_roundtrip_through_host() {
    let mod_source = cubecl_source("mod.rs");
    let banned = [
        "fn i64_indices_as_f64",
        "download_tensor(self.runtime(), &Tensor::I64",
        "upload_tensor(self.runtime(), &converted",
    ];

    let mut violations = Vec::new();
    for needle in banned {
        if mod_source.contains(needle) {
            violations.push(format!("cubecl/mod.rs contains {needle}"));
        }
    }

    assert!(
        violations.is_empty(),
        "CubeCL I64 index conversion must stay on device; host roundtrips in indexing paths are performance regressions:\n{}",
        violations.join("\n")
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cubecl_runtime_exposes_explicit_synchronize() {
    let _sync: fn(&tenferro_gpu::CudaRuntime) -> tenferro_tensor::Result<()> =
        tenferro_gpu::CudaRuntime::synchronize;
}
