//! The import forms a downstream crate uses for the default scalar set.
//!
//! #1785 predicted that a plain type alias would break `use Tensor::F64;` and
//! variant glob imports. `Tensor` is therefore a re-export of the generated set
//! rather than an alias, and this test keeps the supported forms compiling.

mod direct_path {
    use tenferro_tensor_core::Tensor;

    #[test]
    fn variant_paths_resolve_through_the_reexport() {
        let value = Tensor::F64(
            tenferro_tensor_core::HostTensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
        );
        assert_eq!(value.shape(), &[1]);
    }
}

mod variant_import {
    use tenferro_tensor_core::Tensor::F64;

    #[test]
    fn a_single_variant_can_be_imported() {
        let value = F64(tenferro_tensor_core::HostTensor::from_vec_col_major(
            vec![1],
            vec![2.0_f64],
        )
        .unwrap());
        assert_eq!(value.shape(), &[1]);
    }
}

mod glob_import {
    use tenferro_tensor_core::Tensor::*;

    #[test]
    fn variants_can_be_glob_imported() {
        let value = I32(
            tenferro_tensor_core::HostTensor::from_vec_col_major(vec![1], vec![7_i32]).unwrap(),
        );
        assert_eq!(value.shape(), &[1]);
    }
}
