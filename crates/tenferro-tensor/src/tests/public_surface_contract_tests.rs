use std::fs;
use std::path::Path;

use crate::{
    DynRank, TensorRank, TensorScalar, TensorViewCanonicalization, TypedTensor, TypedTensorView,
    TypedTensorViewMut,
};

struct CopyContractBackend;

impl TensorViewCanonicalization<i32, DynRank> for CopyContractBackend {
    fn to_contiguous(
        &mut self,
        _view: &TypedTensorView<'_, i32>,
    ) -> crate::Result<TypedTensor<i32>> {
        Err(crate::Error::backend_failure(
            "test",
            "materialization is not exercised by this copy contract test",
        ))
    }

    fn copy_into(
        &mut self,
        _src: &TypedTensorView<'_, i32>,
        _dst: &mut TypedTensorViewMut<'_, i32>,
    ) -> crate::Result<()> {
        Ok(())
    }
}

fn copy_between_views<T, R, B>(
    backend: &mut B,
    src: &TypedTensorView<'_, T, R>,
    dst: &mut TypedTensorViewMut<'_, T, R>,
) -> crate::Result<()>
where
    T: TensorScalar,
    R: TensorRank,
    B: TensorViewCanonicalization<T, R>,
{
    backend.copy_into(src, dst)
}

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

#[test]
fn crate_root_reexports_are_explicit() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = fs::read_to_string(crate_dir.join("src/lib.rs"))
        .expect("tenferro-tensor lib source must be readable");

    for forbidden in [
        "pub use config::*;",
        "pub use error::*;",
        "pub use types::*;",
        "pub use tenferro_tensor_core::*;",
    ] {
        assert!(
            !source.contains(forbidden),
            "crate-root public API must use deliberate explicit re-exports: {forbidden}"
        );
    }
}

#[test]
fn view_canonicalization_uses_symmetric_copy_into_contract() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = fs::read_to_string(crate_dir.join("src/backend.rs"))
        .expect("tenferro-tensor backend source must be readable");
    let trait_body = source
        .split_once("pub trait TensorViewCanonicalization")
        .expect("TensorViewCanonicalization trait must exist")
        .1
        .split_once("/// Optional elementwise fusion execution.")
        .expect("TensorViewCanonicalization trait must precede TensorFusion")
        .0;

    assert!(
        source.contains("TensorViewCanonicalization<T: TensorScalar, R: TensorRank>"),
        "view canonicalization must use the execution scalar contract"
    );
    assert!(
        trait_body.contains("fn copy_into(")
            && trait_body.contains("src: &TypedTensorView<'_, T, R>")
            && trait_body.contains("dst: &mut TypedTensorViewMut<'_, T, R>"),
        "copy_into must accept readable and writable views with the trait rank"
    );
    assert!(
        !trait_body.contains("copy_from_contiguous"),
        "the asymmetric copy_from_contiguous method must leave the backend trait"
    );

    let src = TypedTensor::<i32>::from_vec_col_major(vec![1], vec![1]).unwrap();
    let mut dst = TypedTensor::<i32>::from_vec_col_major(vec![1], vec![0]).unwrap();
    copy_between_views(
        &mut CopyContractBackend,
        &src.as_view(),
        &mut dst.as_view_mut(),
    )
    .unwrap();
}

#[test]
fn structural_runtime_materialization_is_erased_and_context_free_copies_are_removed() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let backend = fs::read_to_string(crate_dir.join("src/backend.rs"))
        .expect("tenferro-tensor backend source must be readable");
    let structural = backend
        .split_once("pub trait TensorStructural")
        .expect("TensorStructural trait must exist")
        .1
        .split_once("/// Reduction operations.")
        .expect("TensorStructural must precede TensorReduction")
        .0;

    assert!(
        structural.contains(
            "fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor>"
        ),
        "runtime materialization must use the erased TensorRead/Tensor result surface"
    );
    assert!(
        structural.contains("fn copy_read_into(")
            && structural.contains("src: TensorRead<'_>")
            && structural.contains("dst: TensorWrite<'_>"),
        "runtime copy must use erased read/write values"
    );
    let types = fs::read_to_string(crate_dir.join("src/types.rs"))
        .expect("tenferro-tensor types source must be readable");
    for removed in [
        "pub fn to_contiguous(&self)",
        "pub fn copy_from_contiguous(",
        "pub fn to_tensor(&self) -> crate::Result<Tensor>",
        "materialize_view_buffer_col_major",
        "materialize_typed_view_col_major",
    ] {
        assert!(
            !types.contains(removed),
            "context-free tensor movement must be absent: {removed}"
        );
    }
}
