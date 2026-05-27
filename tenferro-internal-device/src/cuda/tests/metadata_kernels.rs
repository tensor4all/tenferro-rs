use super::*;

#[test]
fn cuda_runtime_metadata_iota_i32_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{
        self, MetadataGenerateOp, MetadataGenerateSpec, MetadataTensorMut,
    };

    let runtime = runtime::get_or_init(0).unwrap();
    let dims = [8usize];
    let dst_strides = [1isize];
    let spec = MetadataGenerateSpec::new(&dims, &dst_strides, 0).unwrap();
    let mut dst = runtime.alloc::<i32>(dims.iter().product()).unwrap();

    runtime
        .metadata_generate(
            MetadataGenerateOp::IotaStartZero,
            MetadataTensorMut::I32(&mut dst),
            &spec,
        )
        .unwrap();

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_metadata_iota_reference(&dims, &dst_strides, 0);
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_metadata_constant_i32_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime;

    let runtime = runtime::get_or_init(0).unwrap();
    let dims = [8usize];
    let dst_strides = [1isize];
    let dst = runtime.alloc::<i32>(dims.iter().product()).unwrap();

    unsafe {
        runtime
            .metadata_generate_constant_i32(dst.device_ptr(), dst.len(), &dims, &dst_strides, 0, 7)
            .unwrap();
    }

    let got = runtime.copy_dtoh(&dst).unwrap();
    assert_eq!(got, vec![7; 8]);
}

#[test]
fn cuda_runtime_metadata_constant_bool_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime;

    let runtime = runtime::get_or_init(0).unwrap();
    let dims = [8usize];
    let dst_strides = [1isize];
    let dst = runtime.alloc::<u8>(dims.iter().product()).unwrap();

    unsafe {
        runtime
            .metadata_generate_constant_bool(
                dst.device_ptr(),
                dst.len(),
                &dims,
                &dst_strides,
                0,
                true,
            )
            .unwrap();
    }

    let got = runtime.copy_dtoh(&dst).unwrap();
    assert_eq!(got, vec![1; 8]);
}

#[test]
fn cuda_runtime_metadata_iota_i32_rejects_len_over_i32_max() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{
        self, MetadataGenerateOp, MetadataGenerateSpec, MetadataTensorMut,
    };

    let runtime = runtime::get_or_init(0).unwrap();
    let dims = [(i32::MAX as usize) + 1];
    let dst_strides = [1isize];
    let spec = MetadataGenerateSpec::new(&dims, &dst_strides, 0).unwrap();
    let mut dst = runtime.alloc::<i32>(1).unwrap();

    let err = runtime
        .metadata_generate(
            MetadataGenerateOp::IotaStartZero,
            MetadataTensorMut::I32(&mut dst),
            &spec,
        )
        .unwrap_err();
    assert!(
        err.to_string().contains("len <= i32::MAX"),
        "unexpected error: {err}"
    );
}

#[test]
fn cuda_runtime_metadata_reduction_spec_rejects_non_partition_axes() {
    use tenferro_device::cuda::runtime::MetadataReductionSpec;

    let overlap =
        MetadataReductionSpec::new(&[2usize, 3], &[1isize, 2], 0, &[2], &[1], 0, &[0], &[0]);
    assert!(overlap
        .unwrap_err()
        .to_string()
        .contains("appears in both kept_axes and reduced_axes"));

    let missing =
        MetadataReductionSpec::new(&[2usize, 3], &[1isize, 2], 0, &[2], &[1], 0, &[0], &[]);
    assert!(missing
        .unwrap_err()
        .to_string()
        .contains("is missing from kept_axes and reduced_axes"));
}

#[test]
fn cuda_runtime_metadata_iota_i32_supports_offset_noncontiguous_layout() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{
        self, MetadataGenerateOp, MetadataGenerateSpec, MetadataTensorMut,
    };

    let runtime = runtime::get_or_init(0).unwrap();
    let dims = [2usize, 2];
    let dst_strides = [1isize, 3];
    let dst_offset = 1isize;
    let spec = MetadataGenerateSpec::new(&dims, &dst_strides, dst_offset).unwrap();
    let mut dst = runtime.alloc::<i32>(6).unwrap();

    runtime
        .metadata_generate(
            MetadataGenerateOp::IotaStartZero,
            MetadataTensorMut::I32(&mut dst),
            &spec,
        )
        .unwrap();

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_metadata_iota_layout_reference(&dims, &dst_strides, dst_offset, 6);
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_metadata_bool_compare_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{
        self, MetadataBinaryOp, MetadataBinarySpec, MetadataTensorMut, MetadataTensorRef,
    };

    let runtime = runtime::get_or_init(0).unwrap();
    let dims = [6usize];
    let strides = [1isize];
    let lhs_data = [0u8, 1, 1, 0, 1, 0];
    let rhs_data = [0u8, 0, 1, 1, 1, 0];
    let lhs = runtime.alloc::<u8>(lhs_data.len()).unwrap();
    let rhs = runtime.alloc::<u8>(rhs_data.len()).unwrap();
    let mut not_equal = runtime.alloc::<u8>(lhs_data.len()).unwrap();
    let mut equal = runtime.alloc::<u8>(lhs_data.len()).unwrap();
    runtime.copy_htod(&lhs_data, &lhs).unwrap();
    runtime.copy_htod(&rhs_data, &rhs).unwrap();

    let spec = MetadataBinarySpec::new(&dims, &strides, 0, &strides, 0, &strides, 0).unwrap();
    runtime
        .metadata_binary(
            MetadataBinaryOp::NotEqual,
            MetadataTensorRef::Bool(&lhs),
            MetadataTensorRef::Bool(&rhs),
            MetadataTensorMut::Bool(&mut not_equal),
            &spec,
        )
        .unwrap();
    runtime
        .metadata_binary(
            MetadataBinaryOp::Equal,
            MetadataTensorRef::Bool(&lhs),
            MetadataTensorRef::Bool(&rhs),
            MetadataTensorMut::Bool(&mut equal),
            &spec,
        )
        .unwrap();

    let got_not_equal = runtime.copy_dtoh(&not_equal).unwrap();
    let got_equal = runtime.copy_dtoh(&equal).unwrap();
    let expected_not_equal = host_metadata_not_equal_bool_reference(
        &lhs_data, &rhs_data, &dims, &strides, &strides, &strides,
    );
    let expected_equal = host_metadata_equal_bool_reference(
        &lhs_data, &rhs_data, &dims, &strides, &strides, &strides,
    );
    assert_eq!(got_not_equal, expected_not_equal);
    assert_eq!(got_equal, expected_equal);
}

#[test]
fn cuda_runtime_metadata_i32_bitand_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{
        self, MetadataBinaryOp, MetadataBinarySpec, MetadataTensorMut, MetadataTensorRef,
    };

    let runtime = runtime::get_or_init(0).unwrap();
    let dims = [6usize];
    let strides = [1isize];
    let lhs_data = [0i32, 1, 2, 3, 4, 5];
    let rhs_data = [1i32, 1, 1, 1, 1, 1];
    let lhs = runtime.alloc::<i32>(lhs_data.len()).unwrap();
    let rhs = runtime.alloc::<i32>(rhs_data.len()).unwrap();
    let mut dst = runtime.alloc::<i32>(lhs_data.len()).unwrap();
    runtime.copy_htod(&lhs_data, &lhs).unwrap();
    runtime.copy_htod(&rhs_data, &rhs).unwrap();

    let spec = MetadataBinarySpec::new(&dims, &strides, 0, &strides, 0, &strides, 0).unwrap();
    runtime
        .metadata_binary(
            MetadataBinaryOp::BitAnd,
            MetadataTensorRef::I32(&lhs),
            MetadataTensorRef::I32(&rhs),
            MetadataTensorMut::I32(&mut dst),
            &spec,
        )
        .unwrap();

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected =
        host_metadata_bitand_reference(&lhs_data, &rhs_data, &dims, &strides, &strides, &strides);
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_metadata_where_bool_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{
        self, MetadataTensorMut, MetadataTensorRef, MetadataTernaryOp, MetadataTernarySpec,
    };

    let runtime = runtime::get_or_init(0).unwrap();
    let dims = [2usize, 2];
    let strides = [3isize, 1];
    let cond_data = [0u8, 1, 0, 0, 1];
    let true_data = [0u8, 11, 0, 0, 22];
    let false_data = [0u8, 101, 0, 0, 202];
    let cond = runtime.alloc::<u8>(cond_data.len()).unwrap();
    let on_true = runtime.alloc::<u8>(true_data.len()).unwrap();
    let on_false = runtime.alloc::<u8>(false_data.len()).unwrap();
    let mut dst = runtime.alloc::<u8>(false_data.len()).unwrap();
    runtime.copy_htod(&cond_data, &cond).unwrap();
    runtime.copy_htod(&true_data, &on_true).unwrap();
    runtime.copy_htod(&false_data, &on_false).unwrap();

    let spec = MetadataTernarySpec::new(&dims, &strides, 0, &strides, 0, &strides, 0, &strides, 0)
        .unwrap();
    runtime
        .metadata_ternary(
            MetadataTernaryOp::Where,
            MetadataTensorRef::Bool(&cond),
            MetadataTensorRef::Bool(&on_true),
            MetadataTensorRef::Bool(&on_false),
            MetadataTensorMut::Bool(&mut dst),
            &spec,
        )
        .unwrap();

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_metadata_where_bool_reference(
        &cond_data,
        &true_data,
        &false_data,
        &dims,
        &strides,
        &strides,
        &strides,
        &strides,
    );
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_metadata_not_equal_and_sum_match_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{
        self, MetadataBinaryOp, MetadataBinarySpec, MetadataReductionOp, MetadataReductionSpec,
        MetadataTensorMut, MetadataTensorRef,
    };

    let runtime = runtime::get_or_init(0).unwrap();
    let dims = [2usize, 2];
    let lhs_strides = [3isize, 1];
    let rhs_strides = [1isize, 3];
    let dst_strides = [3isize, 1];
    let lhs_data = [0i32, 1, 99, 3, 4];
    let rhs_data = [0i32, 0, 88, 9, 4];
    let lhs = runtime.alloc::<i32>(lhs_data.len()).unwrap();
    let rhs = runtime.alloc::<i32>(rhs_data.len()).unwrap();
    let mut mask = runtime.alloc::<u8>(lhs_data.len()).unwrap();
    let mut eq_mask = runtime.alloc::<u8>(lhs_data.len()).unwrap();
    let mut sum = runtime.alloc::<i32>(2).unwrap();
    let mut all = runtime.alloc::<u8>(2).unwrap();
    let mut any = runtime.alloc::<u8>(2).unwrap();
    runtime.copy_htod(&lhs_data, &lhs).unwrap();
    runtime.copy_htod(&rhs_data, &rhs).unwrap();

    let binary_spec =
        MetadataBinarySpec::new(&dims, &lhs_strides, 0, &rhs_strides, 0, &dst_strides, 0).unwrap();
    runtime
        .metadata_binary(
            MetadataBinaryOp::NotEqual,
            MetadataTensorRef::I32(&lhs),
            MetadataTensorRef::I32(&rhs),
            MetadataTensorMut::Bool(&mut mask),
            &binary_spec,
        )
        .unwrap();
    runtime
        .metadata_binary(
            MetadataBinaryOp::Equal,
            MetadataTensorRef::I32(&lhs),
            MetadataTensorRef::I32(&rhs),
            MetadataTensorMut::Bool(&mut eq_mask),
            &binary_spec,
        )
        .unwrap();

    let reduction_spec =
        MetadataReductionSpec::new(&dims, &dst_strides, 0, &[2usize], &[1], 0, &[1], &[0]).unwrap();
    runtime
        .metadata_reduction(
            MetadataReductionOp::Sum,
            MetadataTensorRef::Bool(&mask),
            MetadataTensorMut::I32(&mut sum),
            &reduction_spec,
        )
        .unwrap();
    runtime
        .metadata_reduction(
            MetadataReductionOp::All,
            MetadataTensorRef::Bool(&mask),
            MetadataTensorMut::Bool(&mut all),
            &reduction_spec,
        )
        .unwrap();
    runtime
        .metadata_reduction(
            MetadataReductionOp::Any,
            MetadataTensorRef::Bool(&mask),
            MetadataTensorMut::Bool(&mut any),
            &reduction_spec,
        )
        .unwrap();

    let got_mask = runtime.copy_dtoh(&mask).unwrap();
    let got_eq_mask = runtime.copy_dtoh(&eq_mask).unwrap();
    let got_sum = runtime.copy_dtoh(&sum).unwrap();
    let got_all = runtime.copy_dtoh(&all).unwrap();
    let got_any = runtime.copy_dtoh(&any).unwrap();
    let expected_mask = host_metadata_not_equal_reference(
        &lhs_data,
        &rhs_data,
        &dims,
        &lhs_strides,
        &rhs_strides,
        &dst_strides,
    );
    let expected_eq_mask = host_metadata_equal_reference(
        &lhs_data,
        &rhs_data,
        &dims,
        &lhs_strides,
        &rhs_strides,
        &dst_strides,
    );
    let expected_sum =
        host_metadata_sum_bool_reference(&expected_mask, &dims, &dst_strides, &[1], &[0]);
    let expected_all =
        host_metadata_all_bool_reference(&expected_mask, &dims, &dst_strides, &[1], &[0]);
    let expected_any =
        host_metadata_any_bool_reference(&expected_mask, &dims, &dst_strides, &[1], &[0]);
    assert_eq!(got_mask, expected_mask);
    assert_eq!(got_eq_mask, expected_eq_mask);
    assert_eq!(got_sum, expected_sum);
    assert_eq!(got_all, expected_all);
    assert_eq!(got_any, expected_any);
}

#[test]
fn cuda_runtime_metadata_where_selects_integer_values() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{
        self, MetadataTensorMut, MetadataTensorRef, MetadataTernaryOp, MetadataTernarySpec,
    };

    let runtime = runtime::get_or_init(0).unwrap();
    let dims = [6usize];
    let strides = [1isize];
    let cond_data = [1u8, 0, 1, 0, 0, 1];
    let true_data = [10i32, 20, 30, 40, 50, 60];
    let false_data = [-1i32, -2, -3, -4, -5, -6];
    let cond = runtime.alloc::<u8>(cond_data.len()).unwrap();
    let on_true = runtime.alloc::<i32>(true_data.len()).unwrap();
    let on_false = runtime.alloc::<i32>(false_data.len()).unwrap();
    let mut dst = runtime.alloc::<i32>(true_data.len()).unwrap();
    runtime.copy_htod(&cond_data, &cond).unwrap();
    runtime.copy_htod(&true_data, &on_true).unwrap();
    runtime.copy_htod(&false_data, &on_false).unwrap();

    let spec = MetadataTernarySpec::new(&dims, &strides, 0, &strides, 0, &strides, 0, &strides, 0)
        .unwrap();
    runtime
        .metadata_ternary(
            MetadataTernaryOp::Where,
            MetadataTensorRef::Bool(&cond),
            MetadataTensorRef::I32(&on_true),
            MetadataTensorRef::I32(&on_false),
            MetadataTensorMut::I32(&mut dst),
            &spec,
        )
        .unwrap();

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_metadata_where_i32_reference(
        &cond_data,
        &true_data,
        &false_data,
        &dims,
        &strides,
        &strides,
        &strides,
        &strides,
    );
    assert_eq!(got, expected);
}
