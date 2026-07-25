use std::path::PathBuf;

use tenferro_runtime::{CompiledGraph, GraphCompiler, TracedTensor};

fn repo_file(path: &str) -> String {
    std::fs::read_to_string(repo_path(path)).expect("source file must be readable")
}

fn repo_path(path: &str) -> PathBuf {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.push("../..");
    root.push(path);
    root
}

fn assert_no_panic_helpers(path: &str, source: &str) {
    for (line_idx, line) in source.lines().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("///") || trimmed.starts_with("//!") || trimmed.starts_with("//") {
            continue;
        }
        assert!(
            !line.contains(".expect(") && !line.contains(".unwrap("),
            "{path}:{} must not use expect/unwrap in publicly reachable implementation paths: {line}",
            line_idx + 1
        );
    }
}

#[test]
fn compiled_graph_is_the_semantic_artifact_boundary() {
    fn accepts_compiled_graph(graph: &CompiledGraph) {
        let _ = (
            graph.program(),
            graph.bindings(),
            graph.input_count(),
            graph.output_count(),
        );
    }

    let _public_api: fn(&CompiledGraph) = accepts_compiled_graph;
}

#[test]
fn compiled_graph_exposes_only_semantic_program_structure() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y = (&x + &x).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();

    let operations = program.program().operations().collect::<Vec<_>>();

    assert_eq!(program.input_count(), 1);
    assert_eq!(program.output_count(), 1);
    assert_eq!(program.program().inputs().len(), 1);
    assert_eq!(program.program().outputs().len(), 1);
    assert_eq!(operations.len(), 1);
    assert_eq!(operations[0].outputs().len(), 1);
}

#[test]
fn graph_executor_legacy_facade_is_not_public_surface() {
    let lib = repo_file("crates/tenferro-runtime/src/lib.rs");
    let graph_mod = repo_file("crates/tenferro-runtime/src/graph/mod.rs");
    let graph_cache_path = repo_path("crates/tenferro-runtime/src/graph/cache.rs");
    let graph_cache = std::fs::read_to_string(&graph_cache_path).unwrap_or_default();
    let graph_compiler = repo_file("crates/tenferro-runtime/src/graph/compiler.rs");

    assert!(
        !lib.contains("GraphExecutor"),
        "tenferro-runtime crate root must not export or document the retired GraphExecutor facade"
    );
    assert!(
        !graph_mod.contains("pub use executor::GraphExecutor"),
        "graph module must not re-export the retired GraphExecutor facade"
    );
    assert!(
        !lib.contains("GraphExecutorCacheStats")
            && !graph_mod.contains("GraphExecutorCacheStats")
            && !graph_cache.contains("pub struct GraphExecutorCacheStats"),
        "GraphExecutor-specific cache stats must not remain public after retiring the facade"
    );
    assert!(
        !graph_cache_path.exists()
            || (!graph_cache.contains("LegacyStagingCache")
                && !graph_cache.contains("legacy_staging_cache")),
        "legacy semantic staging cache module must be removed with the retired facade"
    );
    assert!(
        !lib.contains("GraphCompilerCacheStats")
            && !graph_mod.contains("GraphCompilerCacheStats")
            && !graph_compiler.contains("compile_cache")
            && !graph_compiler.contains("get_or_compile"),
        "GraphCompiler must not retain the retired ExecProgram compile-cache API"
    );
}

#[test]
fn legacy_extension_executor_registry_is_not_public_surface() {
    let lib = repo_file("crates/tenferro-runtime/src/lib.rs");
    let extension = repo_file("crates/tenferro-runtime/src/extension.rs");
    let extension_context = repo_file("crates/tenferro-runtime/src/extension_execution_context.rs");

    for symbol in [
        "ExtensionExecutor",
        "ExtensionRegistry",
        "ExtensionRuntime",
        "ExtensionRuntimeRegistryError",
        "HostReferenceRuntime",
    ] {
        assert!(
            !lib.contains(symbol),
            "tenferro-runtime crate root must not expose retired legacy extension symbol {symbol}"
        );
        assert!(
            !extension.contains(symbol),
            "tenferro-runtime::extension must not re-export retired legacy extension symbol {symbol}"
        );
    }
    assert!(
        !extension_context.contains("pub trait ExtensionRuntime")
            && !extension_context.contains("pub struct ExtensionRegistry")
            && !extension_context.contains("pub struct ExtensionExecutor")
            && !extension_context.contains("pub struct HostReferenceRuntime")
            && !extension_context.contains("pub enum ExtensionRuntimeRegistryError"),
        "extension_execution_context.rs must only retain prepared-path context/cache helpers, not legacy executor API"
    );
}

#[test]
fn traced_tensor_graph_and_attached_data_are_accessor_based() {
    let source = repo_file("crates/tenferro-runtime/src/traced.rs");
    assert!(
        !source.contains("pub graph:"),
        "TracedTensor graph storage must not be a public field"
    );
    assert!(
        !source.contains("pub data:"),
        "TracedTensor attached data storage must not be a public field"
    );
    assert!(
        source.contains("pub fn graph(&self)"),
        "TracedTensor should expose graph inspection through an accessor"
    );
    assert!(
        source.contains("pub fn attached_data(&self)"),
        "TracedTensor should expose optional attached data through an accessor"
    );
}

#[test]
fn tensor_checkpoint_and_cpu_gemm_public_paths_avoid_panic_helpers() {
    for path in [
        "crates/tenferro-tensor/src/types.rs",
        "crates/tenferro-runtime/src/ad_support.rs",
        "crates/tenferro-cpu/src/gemm/mod.rs",
        "crates/tenferro-cpu/src/gemm/strided_dot.rs",
    ] {
        let source = repo_file(path);
        assert_no_panic_helpers(path, &source);
    }
}

#[test]
fn runtime_matmul_and_shape_inference_use_shared_validation_before_indexing() {
    for path in [
        "crates/tenferro-runtime/src/tensor.rs",
        "crates/tenferro-runtime/src/typed_tensor.rs",
    ] {
        let source = repo_file(path);
        assert!(
            source.contains("matmul_config_for_shapes(\"matmul\""),
            "{path} matmul helpers must use shared rank and dimension validation"
        );
        assert!(
            !source.contains("shape().len() - 1"),
            "{path} matmul helpers must not compute rank-minus-one before rank validation"
        );
    }

    let shape_infer = repo_file("crates/tenferro-runtime/src/shape_infer.rs");
    assert!(
        shape_infer.contains("validate_permutation_axes(\"transpose\""),
        "transpose shape inference must validate permutation metadata before indexing shapes"
    );
    assert!(
        shape_infer.contains(".validate_dims_with_ranks(lhs_rank, rhs_rank)"),
        "dot_general shape inference must validate dimension numbers before indexing shapes"
    );
}

#[test]
fn context_id_pointer_constructor_is_not_public_api() {
    let source = repo_file("crates/tenferro-runtime/src/error.rs");
    let context_id_impl = source
        .split_once("impl ContextId")
        .and_then(|(_, rest)| {
            rest.split_once("impl std::fmt::Display")
                .map(|(body, _)| body)
        })
        .expect("ContextId impl section should exist");

    assert!(
        !context_id_impl.contains("from_ptr"),
        "ContextId must not expose pointer-derived construction"
    );
    assert!(
        context_id_impl.contains("pub fn fresh"),
        "ContextId should be generated by the runtime instead of accepting arbitrary pointers"
    );
}

#[test]
fn traced_dtype_inference_must_not_fall_back_in_release_builds() {
    let source = repo_file("crates/tenferro-runtime/src/traced.rs");
    let helper = source
        .split_once("fn try_inferred_output_dtype")
        .and_then(|(_, rest)| {
            rest.split_once("fn checked_shape_product_for_graph_build")
                .map(|(body, _)| body)
        })
        .expect("traced dtype inference helper should exist");

    assert!(
        !helper.contains("debug_assert!"),
        "traced dtype inference must not hide failures behind debug_assert"
    );
    assert!(
        !helper.contains("fallback"),
        "traced dtype inference must not silently return a fallback dtype"
    );
    assert!(
        helper.contains("infer_output_dtype_at") && helper.contains("ErrorPhase::GraphBuild"),
        "traced dtype inference must preserve the graph-build discovery phase"
    );
    assert!(
        !helper.contains("panic!("),
        "traced dtype inference must return typed errors instead of panicking"
    );
}

#[test]
fn traced_metadata_registration_is_fallible_without_panic_helpers() {
    let traced = repo_file("crates/tenferro-runtime/src/traced.rs");
    let registration = traced
        .split_once("fn register_single_output_metadata(")
        .and_then(|(_, rest)| rest.split_once("impl TracedTensor").map(|(body, _)| body))
        .expect("single-output metadata helper should exist");
    assert!(registration.contains(") -> Result<GlobalMetadataScope>"));
    assert!(!registration.contains(".expect("));

    let packing = repo_file("crates/tenferro-runtime/src/shape_packing.rs");
    let concatenate = packing
        .split_once("fn apply_nary_concatenate(")
        .and_then(|(_, rest)| rest.split_once("#[cfg(test)]").map(|(body, _)| body))
        .expect("concatenate graph helper should exist");
    assert!(
        concatenate.contains(") -> Result<TracedTensor>"),
        "concatenate graph construction must preserve metadata errors"
    );
    assert!(
        !concatenate.contains(".expect("),
        "concatenate metadata registration must not panic"
    );
}

#[test]
fn semantic_program_module_stays_opaque_and_dependency_neutral() {
    for path in [
        "crates/tenferro-runtime/src/program/bindings.rs",
        "crates/tenferro-runtime/src/program/builder.rs",
        "crates/tenferro-runtime/src/program/error.rs",
        "crates/tenferro-runtime/src/program/identity.rs",
        "crates/tenferro-runtime/src/program/import.rs",
        "crates/tenferro-runtime/src/program/metadata.rs",
        "crates/tenferro-runtime/src/program/mod.rs",
        "crates/tenferro-runtime/src/program/op.rs",
        "crates/tenferro-runtime/src/program/semantic.rs",
        "crates/tenferro-runtime/src/program/transform.rs",
        "crates/tenferro-runtime/src/program/value.rs",
    ] {
        let source = repo_file(path);
        assert_no_panic_helpers(path, &source);
        for forbidden in [
            "crate::provider",
            "crate::resource",
            "crate::scheduler",
            "crate::graph",
            "crate::exec",
            "crate::ad",
            "tenferro_ad",
        ] {
            assert!(
                !source.contains(forbidden),
                "{path} must not depend on forbidden runtime/AD layer `{forbidden}`"
            );
        }
    }

    let values = repo_file("crates/tenferro-runtime/src/program/value.rs");
    for exposed in ["pub slot:", "pub owner:", "pub nonce:"] {
        assert!(
            !values.contains(exposed),
            "opaque program identities must not expose `{exposed}`"
        );
    }

    let program = repo_file("crates/tenferro-runtime/src/program/semantic.rs");
    for mutable_escape in [
        "operations_mut",
        "values_mut",
        "outputs_mut",
        "source_to_local",
    ] {
        assert!(
            !program.contains(mutable_escape),
            "frozen semantic program leaked `{mutable_escape}`"
        );
    }
}
