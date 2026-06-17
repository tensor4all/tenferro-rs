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
