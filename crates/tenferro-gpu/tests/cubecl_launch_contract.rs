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
            "cudarc::driver::result::primary_ctx::retain",
            "cudarc::driver::result::ctx::set_current",
            "let device = CudaDevice::new(device_ordinal);",
            "let client = CudaRuntime::client(&device);",
        ],
    );

    let drop_source = source_section(&runtime_source, "impl Drop for CudaRuntime", "}");
    assert_ordered_needles(
        "CudaRuntime::drop",
        drop_source,
        &[
            "let _ = self.synchronize();",
            "primary_ctx::release(self.cuda_device)",
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
