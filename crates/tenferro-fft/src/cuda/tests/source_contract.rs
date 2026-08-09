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
fn workspace_and_execution_keep_ffi_pointers_scoped() {
    let source = include_str!("../plan.rs");
    let workspace_section = source
        .split_once("pub(crate) struct CufftWorkspace")
        .and_then(|(_, rest)| rest.split_once("#[derive(Default)]"))
        .map(|(section, _)| section)
        .unwrap_or_else(|| unreachable!("workspace source section should exist"));
    assert!(!workspace_section.contains("ptr: *mut c_void"));
    assert!(!workspace_section.contains("unsafe impl Send for CufftWorkspace"));
    assert!(!workspace_section.contains("unsafe impl Sync for CufftWorkspace"));

    let enqueue_section = source
        .split_once("pub(crate) fn enqueue_plan_execution")
        .and_then(|(_, rest)| rest.split_once("/// One cached cuFFT plan"))
        .map(|(section, _)| section)
        .unwrap_or_else(|| unreachable!("enqueue helper source section should exist"));
    assert!(enqueue_section.contains("bind_plan_to_stream"));
    assert!(enqueue_section.contains("input_lease.with_ptr"));
    assert!(enqueue_section.contains("output_lease.with_ptr"));
    assert!(enqueue_section.contains("synchronize"));

    let execute_pair_section = source
        .split_once("fn execute_pair")
        .and_then(|(_, rest)| rest.split_once("    pub(crate) fn retained_bytes"))
        .map(|(section, _)| section)
        .unwrap_or_else(|| unreachable!("execute pair source section should exist"));
    assert!(execute_pair_section.contains("CudaExternalUseReadLease"));
    assert!(execute_pair_section.contains("CudaExternalUseWriteLease"));
    assert!(execute_pair_section.contains("output: &mut TypedTensor"));
    assert!(execute_pair_section.contains("enqueue_plan_execution"));

    let interop_source = include_str!("../../../../tenferro-gpu/src/cubecl/interop.rs");
    assert!(interop_source.contains("CudaExternalUseReadLease"));
    assert!(interop_source.contains("tensor: &TypedTensor<T, R>"));
    assert!(interop_source.contains("prepared_tensor_access(tensor, op)"));
    assert!(interop_source.contains("CudaExternalUseWriteLease"));
    assert!(interop_source.contains("tensor: &mut TypedTensor<T, R>"));
    assert!(interop_source.contains("prepared_tensor_write_access(tensor, op)"));
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
