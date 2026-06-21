use std::{fs, path::Path};

fn lowering_program_source() -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("lowering")
            .join("program.rs"),
    )
    .unwrap_or_else(|err| panic!("XLA lowering source should be readable: {err}"))
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
