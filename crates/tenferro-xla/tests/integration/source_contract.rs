use std::{fs, path::Path};

use std::error::Error as _;

use tenferro_tensor::{ErrorKind, ValidationKind};
use tenferro_xla::Error;

fn lowering_program_source() -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("lowering")
            .join("program.rs"),
    )
    .unwrap_or_else(|err| panic!("XLA lowering source should be readable: {err}"))
}

#[test]
fn xla_tensor_errors_preserve_kind_and_source_chain() {
    let error = Error::from(tenferro_tensor::Error::rank_mismatch("input", 2, 1));

    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::RankMismatch)
    );
    assert!(error.source().is_some());
}

fn pjrt_execute_source() -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("pjrt")
            .join("execute.rs"),
    )
    .unwrap_or_else(|err| panic!("PJRT execute source should be readable: {err}"))
}

#[test]
fn xla_sources_depend_only_on_semantic_program_views() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let sources = [
        lowering_program_source(),
        pjrt_execute_source(),
        fs::read_to_string(manifest_dir.join("src").join("lib.rs")).unwrap(),
        fs::read_to_string(manifest_dir.join("src").join("executor.rs")).unwrap(),
        fs::read_to_string(manifest_dir.join("src").join("lowering").join("mod.rs")).unwrap(),
    ];

    for forbidden in [
        "GraphProgram",
        "GraphInstructionView",
        "GraphProgramLoweringView",
        "ExecProgram",
        "lowering_view",
    ] {
        assert!(
            sources.iter().all(|source| !source.contains(forbidden)),
            "XLA production sources must not depend on legacy execution view {forbidden}"
        );
    }
}

#[test]
fn extension_lowering_input_ids_are_checked_before_usize_indexing() {
    let source = lowering_program_source();

    assert!(
        source.contains("usize::try_from(*id)"),
        "extension lowering must reject oversized TensorInputKey ids instead of truncating them"
    );
    assert!(
        !source.contains("let input_idx = *id as usize;"),
        "extension lowering must not truncate TensorInputKey ids with `as usize`"
    );
}

#[test]
fn pjrt_output_download_does_not_materialize_row_major_conversion_buffer() {
    let source = pjrt_execute_source();

    assert!(
        !source.contains("row_major_to_col_major"),
        "PJRT output download should request column-major host layout instead of post-copying"
    );
}

#[test]
fn pjrt_specs_use_semantic_metadata_without_slot_indexing() {
    let source = pjrt_execute_source();
    let section = source
        .split_once("fn output_specs(program: &SemanticProgram)")
        .and_then(|(_, rest)| {
            rest.split_once("fn validate_supported_dtype")
                .map(|(body, _)| body)
        })
        .expect("PJRT output_specs source section should exist");

    assert!(
        section.contains(".outputs()") && section.contains("semantic_tensor_spec(program, value"),
        "PJRT output_specs must derive output contracts from semantic value metadata"
    );
    assert!(
        section.contains(".value_metadata(value)"),
        "PJRT tensor specs must use the checked semantic metadata accessor"
    );
    assert!(
        !section.contains("lowering_view")
            && !section.contains("input_slots")
            && !section.contains("output_slots"),
        "PJRT output specs must not reconstruct semantic metadata through execution slots"
    );
}

#[test]
fn pjrt_download_host_vec_owns_event_before_error_check() {
    let source = pjrt_execute_source();
    let section = source
        .split_once("fn download_host_vec<T: Copy + Default>")
        .and_then(|(_, rest)| {
            rest.split_once("fn col_major_minor_to_major")
                .map(|(body, _)| body)
        })
        .expect("PJRT download_host_vec source section should exist");

    let event_idx = section
        .find("let mut event = PjrtEvent::from_raw(self.api, args.event);")
        .expect("download_host_vec should wrap the returned event in an RAII guard");
    let check_idx = section
        .find("check(self.api, \"PJRT_Buffer_ToHostBuffer\", error)?;")
        .expect("download_host_vec should check PJRT_Buffer_ToHostBuffer errors");
    assert!(
        event_idx < check_idx,
        "PJRT_Buffer_ToHostBuffer event must be owned before returning an error"
    );
    assert!(
        section.contains("event.await_ready_if_present(\"PJRT_Buffer_ToHostBuffer.event\")?;"),
        "download_host_vec should await the owned event after a successful enqueue"
    );
}

#[test]
fn extension_lowering_validates_every_output_dtype() {
    let source = lowering_program_source();
    let section = source
        .split_once("fn lower_extension_operation")
        .and_then(|(_, rest)| {
            rest.split_once("fn build_standard_semantic_subprogram")
                .map(|(body, _)| body)
        })
        .expect("extension lowering source section should exist");

    assert!(
        !section.contains("if output_idx == 0"),
        "extension lowering must not restrict dtype validation to the first output"
    );
    assert!(
        section.contains("semantic_value_type("),
        "extension lowering should load and validate every semantic output type"
    );
    assert!(
        section.contains("if value.ty != expected"),
        "extension lowering should compare every lowered output type with its semantic metadata"
    );
}
