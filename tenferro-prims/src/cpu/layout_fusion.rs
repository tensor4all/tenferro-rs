pub(super) fn try_fuse_group_in_order(dims: &[usize], strides: &[isize]) -> Option<(usize, isize)> {
    match dims.len() {
        0 => Some((1, 0)),
        1 => Some((dims[0], strides[0])),
        _ => {
            if dims.len() != strides.len() {
                return None;
            }

            let first_non_singleton = dims.iter().position(|&d| d > 1);
            let Some(base_idx) = first_non_singleton else {
                return Some((dims.iter().product(), *strides.first().unwrap_or(&0)));
            };

            let base_stride = strides[base_idx];
            let mut expected_abs = base_stride.unsigned_abs();
            for (&dim, &stride) in dims.iter().zip(strides.iter()) {
                if dim <= 1 {
                    continue;
                }
                if stride.unsigned_abs() != expected_abs {
                    return None;
                }
                expected_abs = expected_abs.checked_mul(dim)?;
            }

            Some((dims.iter().product(), base_stride))
        }
    }
}
