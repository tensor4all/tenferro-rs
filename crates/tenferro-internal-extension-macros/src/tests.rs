use super::{expand_extension_family_id, expand_extension_runtime, to_snake_case, ExtensionArgs};
use syn::DeriveInput;

#[test]
fn snake_case_type_name() {
    assert_eq!(to_snake_case("FftOp"), "fft_op");
    assert_eq!(to_snake_case("ScaleBy2"), "scale_by2");
}

#[test]
fn derive_uses_snake_case_type_name_when_name_is_omitted() {
    let input: DeriveInput = syn::parse_quote! {
        #[tenferro_extension(namespace = "my-crate", version = 2)]
        struct FftPlanOp;
    };

    let tokens = expand_extension_family_id(input).expect("derive should expand");

    assert!(tokens.to_string().contains("\"my-crate.fft_plan_op.v2\""));
}

#[test]
fn derive_reports_missing_attribute_fields() {
    let missing_attr: DeriveInput = syn::parse_quote! {
        struct MissingAttr;
    };
    assert!(expand_extension_family_id(missing_attr)
        .expect_err("missing attribute should fail")
        .to_string()
        .contains("missing #[tenferro_extension"));

    let missing_namespace: DeriveInput = syn::parse_quote! {
        #[tenferro_extension(version = 1)]
        struct MissingNamespace;
    };
    assert!(expand_extension_family_id(missing_namespace)
        .expect_err("missing namespace should fail")
        .to_string()
        .contains("missing tenferro_extension namespace"));

    let missing_version: DeriveInput = syn::parse_quote! {
        #[tenferro_extension(namespace = "my-crate")]
        struct MissingVersion;
    };
    assert!(expand_extension_family_id(missing_version)
        .expect_err("missing version should fail")
        .to_string()
        .contains("missing tenferro_extension version"));
}

#[test]
fn derive_attribute_parser_rejects_unknown_and_wrong_typed_values() {
    assert!(syn::parse_str::<ExtensionArgs>(r#"unknown = "x""#)
        .expect_err("unknown argument should fail")
        .to_string()
        .contains("unsupported tenferro_extension argument"));
    assert!(syn::parse_str::<ExtensionArgs>("namespace = 1")
        .expect_err("namespace must be string")
        .to_string()
        .contains("namespace must be a string literal"));
    assert!(syn::parse_str::<ExtensionArgs>(r#"version = "1""#)
        .expect_err("version must be integer")
        .to_string()
        .contains("version must be an integer literal"));
}

#[test]
fn extension_runtime_macro_generates_module_engine_and_prepared_op_without_register_function() {
    let tokens = expand_extension_runtime(syn::parse_quote! {
        runtime = TinyRuntime,
        family_id = TINY_EXTENSION_FAMILY_ID,
        op_type = TinyExtensionOp,
        execute = execute_tiny_extension,
        execute_reads = execute_tiny_extension_reads,
    });
    let source = tokens.to_string();

    assert!(source.contains("struct TinyRuntime"));
    assert!(source.contains("struct TinyRuntimeModule"));
    assert!(source.contains("struct TinyRuntimePlanningConfig"));
    assert!(source.contains("struct TinyRuntimePreparedOperation"));
    assert!(source.contains("impl < B : tenferro_tensor :: TensorBackend + 'static >"));
    assert!(source.contains("ExtensionEngine for TinyRuntime < B >"));
    assert!(source.contains("ExtensionModule for TinyRuntimeModule < B >"));
    assert!(source.contains("PreparedOperation for TinyRuntimePreparedOperation < B >"));
    assert!(source.contains("downcast_ref :: < TinyExtensionOp >"));
    assert!(source.contains("ExtensionExecutionContext :: new"));
    assert!(source.contains("execute_tiny_extension_reads"));
    assert!(source.contains("& self . op"));
    assert!(source.contains("pub fn extension_module"));
    assert!(!source.contains("register_tiny_runtime"));
    assert!(!source.contains("RuntimeConfigBuilder"));
}

#[test]
fn extension_runtime_macro_accepts_custom_backend_bound() {
    let tokens = expand_extension_runtime(syn::parse_quote! {
        runtime = LinalgRuntime,
        family_id = LINALG_EXTENSION_FAMILY_ID,
        op_type = LinalgExtensionOp,
        execute = execute_linalg_extension,
        execute_reads = execute_linalg_extension_reads,
        backend_bound = crate::backend::LinalgBackend,
    });
    let source = tokens.to_string();

    assert!(source.contains("impl < B : crate :: backend :: LinalgBackend + 'static >"));
}

#[test]
fn extension_runtime_macro_generates_required_prepared_read_executor() {
    let tokens = expand_extension_runtime(syn::parse_quote! {
        runtime = EinsumRuntime,
        family_id = EINSUM_EXTENSION_FAMILY_ID,
        op_type = EinsumExtensionOp,
        execute = execute_einsum_extension,
        execute_reads = execute_einsum_extension_reads,
    });
    let source = tokens.to_string();

    assert!(source.contains("fn execute"));
    assert!(source.contains("TensorRead < '_ >"));
    assert!(source.contains("execute_einsum_extension_reads"));
    assert!(source.contains("& self . op"));
}

#[test]
fn extension_runtime_macro_rejects_legacy_register_fn_argument() {
    let err = match syn::parse_str::<super::RuntimeArgs>(
        r#"
        runtime = EinsumRuntime,
        family_id = EINSUM_EXTENSION_FAMILY_ID,
        op_type = EinsumExtensionOp,
        execute = execute_einsum_extension,
        execute_reads = execute_einsum_extension_reads,
        register_fn = register_runtime,
        "#,
    ) {
        Ok(_) => panic!("legacy register_fn argument must no longer be accepted"),
        Err(err) => err,
    };

    assert!(err
        .to_string()
        .contains("unsupported define_extension_runtime argument"));
}
