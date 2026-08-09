use syn::ext::IdentExt;
use syn::visit::{self, Visit};
use syn::{Expr, ExprCall, ExprMethodCall};

#[derive(Debug, Default, PartialEq, Eq)]
struct HostAccessorCalls {
    methods: Vec<&'static str>,
    ufcs: Vec<&'static str>,
}

fn host_accessor_name(ident: &syn::Ident) -> Option<&'static str> {
    match ident.unraw().to_string().as_str() {
        "host_data" => Some("host_data"),
        "host_data_mut" => Some("host_data_mut"),
        _ => None,
    }
}

impl<'ast> Visit<'ast> for HostAccessorCalls {
    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        if let Some(name) = host_accessor_name(&node.method) {
            self.methods.push(name);
        }
        visit::visit_expr_method_call(self, node);
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(path) = node.func.as_ref() {
            if let Some(segment) = path.path.segments.last() {
                if let Some(name) = host_accessor_name(&segment.ident) {
                    self.ufcs.push(name);
                }
            }
        }
        visit::visit_expr_call(self, node);
    }
}

fn host_accessor_calls(path: &str, source: &str) -> HostAccessorCalls {
    let file = syn::parse_file(source)
        .unwrap_or_else(|error| panic!("failed to parse CUDA source {path}: {error}"));
    let mut calls = HostAccessorCalls::default();
    calls.visit_file(&file);
    calls
}

#[test]
fn plan_execution_stays_on_the_public_raw_session_and_forgets_workspace_on_failed_sync() {
    let source = include_str!("../plan.rs");
    // The cuFFT plan adapter must not depend on the hidden cuda::interop
    // bridge for FFI pointers; execution enters the credentialed raw session.
    assert!(!source.contains("cuda::interop"));
    assert!(!source.contains("CufftWorkspace"));
    assert!(!source.contains("CudaExternalUseReadLease"));
    assert!(!source.contains("CudaExternalUseWriteLease"));
    assert!(!source.contains("with_typed_device_ptr"));

    let execution_section = source
        .split_once("fn execute_pair")
        .and_then(|(_, rest)| rest.split_once("    pub(crate) fn retained_bytes"))
        .map(|(section, _)| section)
        .unwrap_or_else(|| unreachable!("execute pair source section should exist"));
    assert!(execution_section.contains("with_raw(OP, |raw|"));
    assert!(execution_section.contains("bind_plan_to_stream"));
    assert!(execution_section.contains("bind_workspace_to_plan"));
    assert!(execution_section.contains("raw.tensor(input)"));
    assert!(execution_section.contains("raw.tensor_mut(output)"));
    assert!(execution_section.contains("raw.alloc_bytes"));
    assert!(execution_section.contains("raw.synchronize()"));
    assert!(execution_section.contains("std::mem::forget(workspace)"));

    // The work-area binding helper owns the cufftSetWorkArea symbol together
    // with the match on workspace.with_ptr, outside the execution loop.
    let workspace_binding_section = source
        .split_once("fn bind_workspace_to_plan")
        .and_then(|(_, rest)| rest.split_once("/// One cached cuFFT plan"))
        .map(|(section, _)| section)
        .unwrap_or_else(|| unreachable!("workspace binding source section should exist"));
    assert!(workspace_binding_section.contains("cufftSetWorkArea"));
    assert!(workspace_binding_section.contains("workspace.with_ptr"));

    // The failed-synchronization leak is the deliberate invariant: a vendored
    // work-area reclamation cannot race an in-flight kernel.
    let forget_sentence = source
        .split_once("std::mem::forget(workspace)")
        .map(|(before, rest)| format!("{before}{}", rest.lines().next().unwrap_or("")))
        .unwrap_or_else(|| unreachable!("forget sentence should exist"));
    assert!(
        forget_sentence.contains("synchronization_error"),
        "workspace forget must be guarded by a synchronization failure"
    );

    let mod_source = include_str!("../mod.rs");
    assert!(!mod_source.contains("cuda::interop"));
    assert!(!mod_source.contains("with_typed_device_ptr"));
    assert!(!mod_source.contains("ensure_cubecl_resident_typed"));
    // Input residency is validated against the exact session runtime/domain
    // through the credentialed public seam, so host and foreign-runtime
    // tensors are rejected before any cache/plan work.
    assert!(mod_source.contains("ensure_gpu_resident(input, OP)"));
    assert!(mod_source.contains("with_cubecl"));
    assert!(mod_source.contains("alloc_zero_output"));
    assert!(mod_source.contains("scale_tensor_write"));
}

#[test]
fn plan_entry_and_session_reject_foreign_runtime_inputs_before_cache_work() {
    // The FFT placement guard must run before plan creation and cache writes,
    // so foreign-runtime tensors cannot allocate a cache entry.
    let mod_source = include_str!("../mod.rs");
    let validate_section = mod_source
        .split_once("fn validate_cuda_input")
        .and_then(|(_, rest)| rest.split_once("fn ensure_cuda_tensor_resident"))
        .map(|(section, _)| section)
        .unwrap_or_else(|| unreachable!("validate_cuda_input section should exist"));
    assert!(validate_section.contains("ensure_cuda_tensor_resident(session, input)?"));
    assert!(!validate_section.contains("extension_plan_key_for_runtime"));
    assert!(!validate_section.contains("cache.store_mut"));

    let validate_call_section = mod_source
        .split_once("fn execute_fft")
        .and_then(|(_, rest)| rest.split_once("let executed = with_cufft_plan_for_batch"))
        .map(|(section, _)| section)
        .unwrap_or_else(|| unreachable!("execute_fft head section should exist"));
    assert!(
        validate_call_section.contains("validate_cuda_input(self, input, spec)?"),
        "residency validation must run before the batch-gated plan/cache work"
    );
}

#[test]
fn ast_visitor_matches_accessor_calls_without_textual_false_positives() {
    let fixture = r#"
        fn fixture(tensor: &TypedTensor<f64>, metadata: &Metadata) {
            // tensor.host_data(); TypedTensor::<f64>::host_data_mut(&tensor)
            let _ = tensor /* whitespace */ . r#host_data :: <f64> ();
            let _ = tensor.host_data_mut();
            let _ = TypedTensor::<f64>::host_data(&tensor);
            let _ = TypedTensor::<f64>::r#host_data_mut(&tensor);
            let _ = metadata.host_data_type();
            let _ = metadata.host_data_mut_type();
            let _ = "tensor.host_data::<f64>(); TypedTensor::<f64>::host_data(&tensor)";
            /* tensor.host_data_mut(); TypedTensor::<f64>::host_data(&tensor) */
        }
    "#;

    assert_eq!(
        host_accessor_calls("fixture", fixture),
        HostAccessorCalls {
            methods: vec!["host_data", "host_data_mut"],
            ufcs: vec!["host_data", "host_data_mut"],
        }
    );
}

#[test]
fn cuda_sources_do_not_cross_the_explicit_transfer_boundary() {
    // Keep this list explicit: every CUDA production module must be reviewed
    // when it is added so transfer and host-payload calls cannot bypass this
    // contract by falling outside a recursive source scan.
    let sources = [
        ("mod.rs", include_str!("../mod.rs")),
        ("descriptor.rs", include_str!("../descriptor.rs")),
        ("error.rs", include_str!("../error.rs")),
        ("ffi.rs", include_str!("../ffi.rs")),
        ("hermitian.rs", include_str!("../hermitian.rs")),
        ("plan.rs", include_str!("../plan.rs")),
    ];
    let forbidden_transfers = [
        concat!("upload", "_tensor("),
        concat!("download", "_tensor("),
    ];
    for (path, source) in sources {
        for pattern in forbidden_transfers {
            assert!(
                !source.contains(pattern),
                "CUDA production source {path} must not contain {pattern}"
            );
        }
        assert_eq!(
            host_accessor_calls(path, source),
            HostAccessorCalls::default(),
            "CUDA production source {path} must not call host_data or host_data_mut"
        );
    }
    let cuda_module = include_str!("../mod.rs");
    assert!(cuda_module.contains("allocate_cuda_zero_output"));
    assert!(!cuda_module.contains("sub(current, current)"));
    assert!(!cuda_module.contains("reduce_sum(&zero"));
}

#[test]
fn plan_execution_section_has_no_safe_ffi_pointer_escape() {
    let source = include_str!("../plan.rs");
    let execution_section = source
        .split_once("fn execute_pair")
        .and_then(|(_, rest)| rest.split_once("    pub(crate) fn retained_bytes"))
        .map(|(section, _)| section)
        .unwrap_or_else(|| unreachable!("execute pair source section should exist"));
    // FFI pointers must remain scoped inside the with_raw closure; the only
    // raw-pointer field that may appear is the DeviceBytes work-area pointer,
    // whose lifetime is bound to the raw session.
    assert!(!execution_section.contains("unsafe impl Send"));
    assert!(!execution_section.contains("unsafe impl Sync"));
}
