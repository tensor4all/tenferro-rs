use tenferro_device::{Error, Result};

/// Convert a notation label character to internal `u32`.
///
/// Unicode alphanumeric scalars are accepted and mapped to their scalar value
/// (`char as u32`). This keeps label syntax aligned with the documented
/// "Unicode alphanumeric label" contract while still allowing digits,
/// accented letters, Greek, and CJK characters.
pub(crate) fn char_to_label(c: char) -> Result<u32> {
    if !c.is_alphanumeric() {
        return Err(Error::InvalidArgument(format!(
            "invalid einsum label character: {c:?} (U+{:04X}); labels must be Unicode alphanumeric",
            c as u32
        )));
    }
    Ok(c as u32)
}

/// Split einsum notation on `->` and validate balanced parentheses.
///
/// Returns `(lhs, rhs)` where `lhs` is the input side and `rhs` is the output side.
pub(crate) fn split_and_validate_notation(notation: &str) -> Result<(&str, &str)> {
    let parts: Vec<&str> = notation.split("->").collect();
    if parts.len() != 2 {
        return Err(Error::InvalidArgument(format!(
            "einsum notation must contain exactly one '->', got: {notation}"
        )));
    }
    let lhs = parts[0];
    let rhs = parts[1];

    // Validate balanced parentheses in lhs
    let mut depth: i32 = 0;
    for c in lhs.chars() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth < 0 {
                    return Err(Error::InvalidArgument(format!(
                        "unmatched ')' in einsum notation: {notation}"
                    )));
                }
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err(Error::InvalidArgument(format!(
            "unmatched '(' in einsum notation: {notation}"
        )));
    }

    Ok((lhs, rhs))
}
