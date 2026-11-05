use super::dot_general_config_retained_bytes;
use tenferro_tensor::DotGeneralConfig;

#[test]
fn dot_general_retained_bytes_exclude_inline_axes() {
    for count in [0, 4, 5, 9] {
        let config = DotGeneralConfig {
            lhs_contracting_dims: (0..count).collect(),
            rhs_contracting_dims: (0..count).rev().collect(),
            lhs_batch_dims: (count..2 * count).collect(),
            rhs_batch_dims: (count..2 * count).rev().collect(),
        };
        let expected = if count <= 4 {
            0
        } else {
            4 * count * size_of::<usize>()
        };
        assert_eq!(dot_general_config_retained_bytes(&config), Some(expected));
    }
}
