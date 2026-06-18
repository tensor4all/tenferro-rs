use std::fs;
use std::path::Path;

#[test]
fn typed_tensor_storage_fields_are_accessor_based() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = fs::read_to_string(crate_dir.join("src/types.rs"))
        .expect("tenferro-tensor types source must be readable");

    assert!(
        !source.contains("pub buffer: Buffer<T>"),
        "TypedTensor storage must not expose a public buffer field"
    );
    assert!(
        !source.contains("pub placement: Placement"),
        "TypedTensor placement must not expose a public field"
    );
    assert!(
        !source.contains("pub id: u64"),
        "BufferHandle ids must remain opaque"
    );
    assert!(
        source.contains("pub fn buffer(&self)"),
        "TypedTensor should expose read-only buffer inspection through an accessor"
    );
    assert!(
        source.contains("pub fn placement(&self)"),
        "TypedTensor should expose placement inspection through an accessor"
    );
}

#[test]
fn tensor_views_do_not_expose_legacy_physical_slice_names() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = fs::read_to_string(crate_dir.join("src/types.rs"))
        .expect("tenferro-tensor types source must be readable");

    assert!(
        !source.contains("as_physical_slice"),
        "typed tensor views must use explicit host_storage accessors, not legacy physical-slice names"
    );
}

#[test]
fn tensor_types_do_not_expose_row_major_compatibility_apis() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = fs::read_to_string(crate_dir.join("src/types.rs"))
        .expect("tenferro-tensor types source must be readable");

    assert!(
        !source.contains("from_vec_row_major") && !source.contains("into_vec_row_major"),
        "tensor public API must stay column-major only; row-major conversion belongs outside tenferro"
    );
}

#[test]
fn elementwise_fusion_ir_is_not_top_level_raw_api() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lib = fs::read_to_string(crate_dir.join("src/lib.rs"))
        .expect("tenferro-tensor lib source must be readable");
    let backend = fs::read_to_string(crate_dir.join("src/backend.rs"))
        .expect("tenferro-tensor backend source must be readable");

    assert!(
        !lib.contains("ElementwiseFusionInst, ElementwiseFusionOp, ElementwiseFusionPlan"),
        "elementwise fusion IR must not be re-exported as a top-level public tensor API"
    );
    assert!(
        !backend.contains("pub dtype: crate::DType"),
        "ElementwiseFusionPlan dtype storage must not be a public field"
    );
    assert!(
        !backend.contains("pub input_count: usize"),
        "ElementwiseFusionPlan input_count storage must not be a public field"
    );
    assert!(
        !backend.contains("pub outputs: Vec<usize>"),
        "ElementwiseFusionPlan output storage must not be a public field"
    );
    assert!(
        !backend.contains("pub ops: Vec<ElementwiseFusionInst>"),
        "ElementwiseFusionPlan op storage must not be a public field"
    );
    assert!(
        !backend.contains("pub op: ElementwiseFusionOp"),
        "ElementwiseFusionInst op storage must not be a public field"
    );
    assert!(
        !backend.contains("pub inputs: Vec<usize>"),
        "ElementwiseFusionInst input storage must not be a public field"
    );
}

#[test]
fn tensor_scalar_helpers_do_not_expose_cpu_conjugation_hook() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = fs::read_to_string(crate_dir.join("src/types.rs"))
        .expect("tenferro-tensor types source must be readable");

    assert!(
        !source.contains("pub trait ConjElem"),
        "CPU conjugation helpers must not be part of the public tensor scalar API"
    );
}
