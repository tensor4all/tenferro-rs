use tenferro_device::{Error, Result};

const ELLIPSIS_MARKER: u32 = 0xE000;

pub(crate) fn expand_ellipsis_in_notation(
    notation: &str,
    operand_shapes: &[&[usize]],
) -> Result<String> {
    let (inputs_str, output_str) = crate::syntax::notation::split_and_validate_notation(notation)?;

    if !notation.contains("...") {
        return Ok(notation.to_string());
    }

    let clean_inputs = inputs_str.replace(['(', ')'], "");
    let input_specs: Vec<&str> = clean_inputs.split(',').collect();

    if input_specs.len() != operand_shapes.len() {
        return Err(Error::InvalidArgument(format!(
            "number of operands ({}) does not match number of subscript specs ({})",
            operand_shapes.len(),
            input_specs.len()
        )));
    }

    let mut ellipsis_ndims: Option<usize> = None;

    for (i, (spec, shape)) in input_specs.iter().zip(operand_shapes.iter()).enumerate() {
        let explicit_dims = count_explicit_dims(spec)?;
        let ellipsis_count = count_ellipsis(spec)?;

        if ellipsis_count > 1 {
            return Err(Error::InvalidArgument(format!(
                "operand {} has multiple ellipsis '...', only one is allowed per operand",
                i
            )));
        }

        if ellipsis_count == 1 {
            if shape.len() < explicit_dims {
                return Err(Error::InvalidArgument(format!(
                    "operand {} has {} dimensions but subscript '{}' requires at least {} explicit dimensions",
                    i,
                    shape.len(),
                    spec,
                    explicit_dims
                )));
            }

            let ellipsis_ndim = shape.len() - explicit_dims;

            if let Some(prev_ndim) = ellipsis_ndims {
                if prev_ndim != ellipsis_ndim {
                    return Err(Error::InvalidArgument(format!(
                        "inconsistent ellipsis dimensions: operand 0 has {} batch dims, operand {} has {} batch dims",
                        prev_ndim,
                        i,
                        ellipsis_ndim
                    )));
                }
            } else {
                ellipsis_ndims = Some(ellipsis_ndim);
            }
        } else {
            if shape.len() != explicit_dims {
                return Err(Error::InvalidArgument(format!(
                    "operand {} has {} dimensions but subscript '{}' has {} explicit dimensions",
                    i,
                    shape.len(),
                    spec,
                    explicit_dims
                )));
            }
        }
    }

    let ellipsis_ndim = ellipsis_ndims.unwrap_or(0);

    let expanded_inputs: Vec<String> = input_specs
        .iter()
        .map(|spec| expand_ellipsis_in_operand(spec, ellipsis_ndim))
        .collect::<Result<Vec<_>>>()?;

    let expanded_output = expand_ellipsis_in_operand(output_str, ellipsis_ndim)?;

    Ok(format!(
        "{}->{}",
        expanded_inputs.join(","),
        expanded_output
    ))
}

fn count_ellipsis(s: &str) -> Result<usize> {
    let mut count = 0;
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if i + 2 < chars.len() && chars[i] == '.' && chars[i + 1] == '.' && chars[i + 2] == '.' {
            count += 1;
            i += 3;
        } else if chars[i] == '.' {
            return Err(Error::InvalidArgument(
                "invalid '.' character in einsum notation; use '...' for ellipsis".to_string(),
            ));
        } else {
            i += 1;
        }
    }

    Ok(count)
}

fn count_explicit_dims(s: &str) -> Result<usize> {
    let without_ellipsis = s.replace("...", "");
    Ok(without_ellipsis
        .chars()
        .filter(|c| c.is_alphanumeric())
        .count())
}

fn expand_ellipsis_in_operand(spec: &str, ellipsis_ndim: usize) -> Result<String> {
    let mut result = String::new();
    let chars: Vec<char> = spec.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if i + 2 < chars.len() && chars[i] == '.' && chars[i + 1] == '.' && chars[i + 2] == '.' {
            for j in 0..ellipsis_ndim {
                let label_char = char::from_u32(ELLIPSIS_MARKER + j as u32).ok_or_else(|| {
                    Error::InvalidArgument("failed to create ellipsis label character".to_string())
                })?;
                result.push(label_char);
            }
            i += 3;
        } else if chars[i] == '.' {
            return Err(Error::InvalidArgument(
                "invalid '.' character; use '...' for ellipsis".to_string(),
            ));
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_ellipsis_simple() {
        let shapes: Vec<&[usize]> = vec![&[2, 3, 4, 5], &[2, 3, 5, 6]];
        let result = expand_ellipsis_in_notation("...ij,...jk->...ik", &shapes).unwrap();

        assert!(result.contains('\u{E000}'));
        assert!(result.contains('\u{E001}'));
        assert!(result.contains("->"));
    }

    #[test]
    fn test_expand_ellipsis_no_ellipsis() {
        let shapes: Vec<&[usize]> = vec![&[2, 3], &[3, 4]];
        let result = expand_ellipsis_in_notation("ij,jk->ik", &shapes).unwrap();
        assert_eq!(result, "ij,jk->ik");
    }

    #[test]
    fn test_expand_ellipsis_mismatched_batch_dims() {
        let shapes: Vec<&[usize]> = vec![&[2, 3, 4], &[2, 3, 4, 5]];
        let result = expand_ellipsis_in_notation("...ij,...jk->...ik", &shapes);
        assert!(result.is_err());
    }

    #[test]
    fn test_expand_ellipsis_insufficient_dims() {
        let shapes: Vec<&[usize]> = vec![&[2, 3], &[2, 3, 5]];
        let result = expand_ellipsis_in_notation("...ij,...jk->...ik", &shapes);
        assert!(result.is_err());
    }

    #[test]
    fn test_count_ellipsis() {
        assert_eq!(count_ellipsis("...ij").unwrap(), 1);
        assert_eq!(count_ellipsis("ij,jk").unwrap(), 0);
        assert_eq!(count_ellipsis("...ij,...jk").unwrap(), 2);
    }

    #[test]
    fn test_count_ellipsis_invalid_dot() {
        assert!(count_ellipsis(".ij").is_err());
        assert!(count_ellipsis("..ij").is_err());
    }

    #[test]
    fn test_count_explicit_dims() {
        assert_eq!(count_explicit_dims("ij").unwrap(), 2);
        assert_eq!(count_explicit_dims("...ij").unwrap(), 2);
        assert_eq!(count_explicit_dims("...ij,...jk").unwrap(), 4);
    }

    #[test]
    fn test_expand_ellipsis_in_operand() {
        let result = expand_ellipsis_in_operand("...ij", 2).unwrap();
        assert_eq!(result.chars().count(), 4);
        assert!(result.contains('\u{E000}'));
        assert!(result.contains('\u{E001}'));
    }

    #[test]
    fn test_expand_ellipsis_zero_batch_dims() {
        let result = expand_ellipsis_in_operand("...ij", 0).unwrap();
        assert_eq!(result, "ij");
    }

    #[test]
    fn test_expand_ellipsis_rejects_operand_count_mismatch() {
        let shapes: Vec<&[usize]> = vec![&[2, 3, 4]];
        let err = expand_ellipsis_in_notation("...ij,...jk->...ik", &shapes).unwrap_err();
        assert!(err
            .to_string()
            .contains("number of operands (1) does not match number of subscript specs (2)"));
    }

    #[test]
    fn test_expand_ellipsis_rejects_multiple_ellipses_per_operand() {
        let shapes: Vec<&[usize]> = vec![&[2, 3, 4, 5]];
        let err = expand_ellipsis_in_notation("...i...j->ij", &shapes).unwrap_err();
        assert!(err.to_string().contains("multiple ellipsis"));
    }

    #[test]
    fn test_expand_ellipsis_rejects_invalid_output_dots() {
        let shapes: Vec<&[usize]> = vec![&[2, 3, 4]];
        let err = expand_ellipsis_in_notation("...ij->..j", &shapes).unwrap_err();
        assert!(err.to_string().contains("invalid '.' character"));
    }

    #[test]
    fn test_expand_ellipsis_handles_parenthesized_inputs() {
        let shapes: Vec<&[usize]> = vec![&[2, 3, 4, 5], &[2, 3, 5, 6]];
        let result = expand_ellipsis_in_notation("(...ij),(...jk)->...ik", &shapes).unwrap();

        assert!(result.starts_with('\u{E000}'));
        assert!(result.contains("ij,"));
        assert!(result.ends_with("ik"));
    }

    #[test]
    fn test_expand_ellipsis_rejects_invalid_notation_before_expansion() {
        let shapes: Vec<&[usize]> = vec![&[2, 3, 4]];
        let err = expand_ellipsis_in_notation("...ij", &shapes).unwrap_err();
        assert!(err.to_string().contains("einsum notation"));
    }

    #[test]
    fn test_expand_ellipsis_rejects_non_ellipsis_rank_mismatch() {
        let shapes: Vec<&[usize]> = vec![&[2, 3, 4], &[3, 5, 7]];
        let err = expand_ellipsis_in_notation("...ij,jk->...ik", &shapes).unwrap_err();
        assert!(err.to_string().contains("explicit dimensions"));
    }

    #[test]
    fn test_expand_ellipsis_in_operand_reports_private_use_overflow() {
        let err = expand_ellipsis_in_operand("...", 0x110000usize).unwrap_err();
        assert!(err
            .to_string()
            .contains("failed to create ellipsis label character"));
    }
}
