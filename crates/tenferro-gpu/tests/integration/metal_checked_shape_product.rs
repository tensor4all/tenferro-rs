//! Source contracts for the Metal backend's element counts (issues #2668, #2671).
//!
//! Every site used a raw `shape.iter().product()`, so a wrapped count reached the
//! allocation check, the kernel arguments and the transfer counters. These tests keep
//! the checked forms in place and make the guard scan the whole module, not two slices.

use std::path::{Path, PathBuf};

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start = source
        .find(start)
        .unwrap_or_else(|| panic!("section start {start:?} should exist"));
    let tail = &source[start..];
    let end = tail
        .find(end)
        .unwrap_or_else(|| panic!("section end {end:?} should exist"));
    &tail[..end]
}

fn metal_sources() -> Vec<(PathBuf, String)> {
    fn walk(dir: &Path, out: &mut Vec<(PathBuf, String)>) {
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .unwrap_or_else(|error| panic!("read {}: {error}", dir.display()))
            .map(|entry| entry.expect("directory entry").path())
            .collect();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let source = std::fs::read_to_string(&path)
                    .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
                out.push((path, source));
            }
        }
    }

    let mut sources = Vec::new();
    walk(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src/metal"),
        &mut sources,
    );
    assert!(
        !sources.is_empty(),
        "the Metal source tree should not be empty"
    );
    sources
}

/// Normalized source plus, for each retained character, its 1-based source line.
fn normalized_with_lines(source: &str) -> (String, Vec<usize>) {
    let mut normalized = String::new();
    let mut line_of = Vec::new();
    for (index, line) in source.lines().enumerate() {
        for character in line.chars() {
            if !character.is_whitespace() {
                normalized.push(character);
                line_of.push(index + 1);
            }
        }
    }
    (normalized, line_of)
}

#[test]
fn metal_sources_have_no_raw_shape_product_without_an_invariant() {
    // Both the one-line `shape.iter().product()` and the multi-line
    // `.shape().iter().product::<usize>()` spellings reduce to these substrings.
    const PATTERNS: [&str; 2] = ["iter().product", "product::<usize>"];

    for (path, source) in metal_sources() {
        let lines: Vec<&str> = source.lines().collect();
        let (normalized, line_of) = normalized_with_lines(&source);
        for pattern in PATTERNS {
            let mut offset = 0;
            while let Some(found) = normalized[offset..].find(pattern) {
                let position = offset + found;
                let line = line_of[position];
                // A consciously unchecked site needs an adjacent `// INVARIANT:` naming
                // the validation that makes the product safe.
                let start = line.saturating_sub(2);
                let end = (line + 1).min(lines.len());
                let marked = lines[start..end]
                    .iter()
                    .any(|text| text.contains("// INVARIANT:"));
                assert!(
                    marked,
                    "{}:{line} computes an element count with a raw `{pattern}` and has no \
                     `// INVARIANT:` marker; use checked arithmetic",
                    path.display()
                );
                offset = position + pattern.len();
            }
        }
    }
}

#[test]
fn metal_engine_len_uses_checked_shape_product() {
    let engine = include_str!("../../src/metal/engine.rs");
    let len = section(engine, "fn len(tensor: &Tensor", "fn submit(");

    assert!(
        len.contains("checked_shape_product(op, \"shape\", tensor.shape())?"),
        "MetalEngine::len must reject a shape-product overflow before the i64 conversion"
    );
    assert!(
        !len.contains("shape().iter().product"),
        "MetalEngine::len must not use an unchecked shape product"
    );
}

#[test]
fn metal_wrap_uses_checked_element_count_and_byte_length() {
    let context = include_str!("../../src/metal/context.rs");
    let wrap = section(context, "pub(crate) fn wrap<T>", "fn allocate_typed<T>");

    assert!(
        wrap.contains("Self::element_count(&shape, op)?"),
        "MetalContext::wrap must validate the element count through the checked helper"
    );
    assert!(
        wrap.contains(".checked_mul(std::mem::size_of::<T>())"),
        "MetalContext::wrap must compute the expected byte length with checked arithmetic"
    );
    assert!(
        !wrap.contains("shape.iter().product"),
        "MetalContext::wrap must not use an unchecked shape product"
    );
    assert!(
        !wrap.contains("saturating_mul(std::mem::size_of::<T>())"),
        "MetalContext::wrap must not paper over a byte-length overflow by saturating"
    );
}

#[test]
fn metal_element_bytes_uses_checked_count_and_byte_length() {
    let context = include_str!("../../src/metal/context.rs");
    let element_bytes = section(
        context,
        "pub(crate) fn element_bytes(",
        "/// Refuse a tensor that is not in this context's domain",
    );

    assert!(
        element_bytes.contains("RuntimeCell::element_count(tensor.shape(), op)?"),
        "MetalContext::element_bytes must validate the element count through the checked helper"
    );
    assert!(
        element_bytes.contains(".checked_mul("),
        "MetalContext::element_bytes must compute the byte length with checked arithmetic"
    );
    assert!(
        element_bytes.contains("ok_or_else("),
        "MetalContext::element_bytes must return a typed error on overflow"
    );
    assert!(
        !element_bytes.contains("saturating_mul"),
        "MetalContext::element_bytes must not paper over a byte-length overflow by saturating"
    );
    assert!(
        !element_bytes.contains(".iter().product"),
        "MetalContext::element_bytes must not use an unchecked shape product"
    );
}
