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
fn pjrt_output_specs_validate_slot_bounds_before_indexing() {
    let source = pjrt_execute_source();
    let section = source
        .split_once("fn output_specs(program: &GraphProgram)")
        .and_then(|(_, rest)| {
            rest.split_once("fn validate_supported_dtype")
                .map(|(body, _)| body)
        })
        .expect("PJRT output_specs source section should exist");

    assert!(
        section.contains("program.input_specs().len() != view.input_slots().len()"),
        "PJRT output_specs must validate input spec/slot count before zipping or indexing"
    );
    assert!(
        !section.contains("view.input_slots()[index]"),
        "PJRT output_specs must not index input_slots by input spec position without bounds checks"
    );
    assert!(
        !section.contains("inst.output_slots()[0]"),
        "PJRT output_specs must not index output_slots[0] without get/get_mut validation"
    );
    assert!(
        section.contains("slots.get_mut(input_slot)")
            && section.contains("slots.get_mut(output_slot)"),
        "PJRT output_specs should populate slots through checked get_mut calls"
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
        .split_once("fn lower_extension_instruction")
        .and_then(|(_, rest)| rest.split_once("fn lower_constant").map(|(body, _)| body))
        .expect("extension lowering source section should exist");

    assert!(
        !section.contains("if output_idx == 0"),
        "extension lowering must not restrict dtype validation to the first output"
    );
    assert!(
        section.contains("validate_dtype(value.ty.dtype, \"extension output\")"),
        "extension lowering should validate the actual dtype of every lowered extension output"
    );
    assert!(
        section.contains("value.ty.dtype != inst.dtype()"),
        "extension lowering should compare every lowered output dtype with the parent instruction dtype"
    );
}
