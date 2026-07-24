#[test]
fn signature_tensor_metadata_mapping_source_contract() {
    let source = include_str!("../signature.rs");
    let from_reads = function_body(source, "from_reads");

    for metadata_call in [".strides()", ".is_col_major_contiguous()"] {
        let call_start = from_reads
            .find(metadata_call)
            .unwrap_or_else(|| panic!("signature mapping must call {metadata_call}"));
        let statement = statement_containing(from_reads, call_start);

        assert!(
            statement.contains("map_err") && statement.contains("input"),
            "{metadata_call} must map the original input index in the metadata statement"
        );
    }

    assert!(
        from_reads.contains("PrepareError::InputSignature"),
        "metadata failures must map through PrepareError::InputSignature"
    );
    assert!(
        from_reads.contains("InputSignatureError::TensorMetadata"),
        "metadata failures must map through InputSignatureError::TensorMetadata"
    );
    assert!(
        !from_reads.contains("InputSignatureEntry::new"),
        "from_reads must not remap metadata failures through entry validation"
    );
    assert!(
        !from_reads.contains("ShapeStrideRankMismatch")
            && !from_reads.contains("InvalidAlignmentClass"),
        "from_reads must not remap metadata failures as entry-validation errors"
    );
}

fn function_body<'a>(source: &'a str, name: &str) -> &'a str {
    let signature_start = source
        .find(&format!("fn {name}"))
        .unwrap_or_else(|| panic!("missing {name} function"));
    let body_start = source[signature_start..]
        .find('{')
        .map(|offset| signature_start + offset)
        .unwrap_or_else(|| panic!("missing {name} body"));
    let mut depth = 0usize;
    for (offset, byte) in source[body_start..].bytes().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[body_start..=body_start + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated {name} body");
}

fn statement_containing(source: &str, index: usize) -> &str {
    let start = source[..index]
        .rfind([';', '{'])
        .map_or(0, |position| position + 1);
    let end = source[index..]
        .find(';')
        .map_or(source.len(), |position| index + position + 1);
    &source[start..end]
}
