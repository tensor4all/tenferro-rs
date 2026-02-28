use tenferro_device::{Error, Result};

/// Convert a notation label character to internal `u32`.
///
/// Any Unicode scalar except control characters is accepted and mapped to
/// its scalar value (`char as u32`). This enables einsum benchmark instances
/// that use characters like `×`, `ë`, `ð` as index labels.
pub(crate) fn char_to_label(c: char) -> Result<u32> {
    if c.is_control() {
        return Err(Error::InvalidArgument(format!(
            "invalid einsum label character: control character U+{:04X}",
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
