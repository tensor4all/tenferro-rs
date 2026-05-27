use super::*;

#[test]
fn cuda_runtime_zero_trailing_by_counts_f64_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    let src_host = vec![1.0_f64, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let keep_counts_host = vec![1.0_f64, 2.0];
    let src = runtime.alloc::<f64>(src_host.len()).unwrap();
    let keep_counts = runtime.alloc::<f64>(keep_counts_host.len()).unwrap();
    let dst = runtime.alloc::<f64>(src_host.len()).unwrap();
    runtime.copy_htod(&src_host, &src).unwrap();
    runtime.copy_htod(&keep_counts_host, &keep_counts).unwrap();

    let spec = tenferro_device::cuda::runtime::ZeroTrailingByCountsSpec::new(
        &[2, 2, 2],
        &[1, 2, 4],
        0,
        &[1, 2, 4],
        0,
        &[1],
        0,
        1,
        2,
    )
    .unwrap();

    runtime
        .zero_trailing_by_counts(&src, &dst, &keep_counts, &spec)
        .unwrap();

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_zero_trailing_by_counts_reference(
        &src_host,
        spec.dims(),
        spec.src_strides(),
        spec.src_offset(),
        &[1, 2],
        spec.keep_count_strides(),
        spec.keep_count_offset(),
        spec.axis(),
        spec.structural_rank(),
    );
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_zero_trailing_by_counts_complex64_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    let src_host = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, 2.0),
        Complex64::new(3.0, 3.0),
        Complex64::new(4.0, 4.0),
        Complex64::new(5.0, 5.0),
        Complex64::new(6.0, 6.0),
        Complex64::new(7.0, 7.0),
        Complex64::new(8.0, 8.0),
        Complex64::new(9.0, 9.0),
        Complex64::new(10.0, 10.0),
        Complex64::new(11.0, 11.0),
        Complex64::new(12.0, 12.0),
    ];
    let keep_counts_host = vec![2.0_f32, 1.0];
    let src = runtime.alloc::<Complex64>(src_host.len()).unwrap();
    let keep_counts = runtime.alloc::<f32>(keep_counts_host.len()).unwrap();
    let dst = runtime.alloc::<Complex64>(src_host.len()).unwrap();
    runtime.copy_htod(&src_host, &src).unwrap();
    runtime.copy_htod(&keep_counts_host, &keep_counts).unwrap();

    let spec = tenferro_device::cuda::runtime::ZeroTrailingByCountsSpec::new(
        &[3, 2, 2],
        &[1, 3, 6],
        0,
        &[1, 3, 6],
        0,
        &[1],
        0,
        0,
        2,
    )
    .unwrap();

    runtime
        .zero_trailing_by_counts(&src, &dst, &keep_counts, &spec)
        .unwrap();

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_zero_trailing_by_counts_reference(
        &src_host,
        spec.dims(),
        spec.src_strides(),
        spec.src_offset(),
        &[2, 1],
        spec.keep_count_strides(),
        spec.keep_count_offset(),
        spec.axis(),
        spec.structural_rank(),
    );
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_zero_trailing_by_counts_rejects_non_integer_keep_counts() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    let src_host = vec![1.0_f64, 2.0, 3.0, 4.0];
    let keep_counts_host = vec![1.5_f64];
    let src = runtime.alloc::<f64>(src_host.len()).unwrap();
    let keep_counts = runtime.alloc::<f64>(keep_counts_host.len()).unwrap();
    let dst = runtime.alloc::<f64>(src_host.len()).unwrap();
    runtime.copy_htod(&src_host, &src).unwrap();
    runtime.copy_htod(&keep_counts_host, &keep_counts).unwrap();

    let spec = tenferro_device::cuda::runtime::ZeroTrailingByCountsSpec::new(
        &[2, 2],
        &[1, 2],
        0,
        &[1, 2],
        0,
        &[],
        0,
        1,
        2,
    )
    .unwrap();

    let err = runtime
        .zero_trailing_by_counts(&src, &dst, &keep_counts, &spec)
        .unwrap_err();
    assert!(err.to_string().contains("integer-valued"));
}

#[test]
fn cuda_runtime_triangular_part_f64_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{TriangularHalf, TriangularPartSpec};

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    let src_host: Vec<f64> = (1..=24).map(|value| value as f64).collect();
    let src = runtime.alloc::<f64>(src_host.len()).unwrap();
    let dst = runtime.alloc::<f64>(src_host.len()).unwrap();
    runtime.copy_htod(&src_host, &src).unwrap();

    let dims = [3usize, 2, 4];
    let src_strides = [1isize, 3, 6];
    let spec = TriangularPartSpec::new(
        &dims,
        &src_strides,
        0,
        &[1isize, 3, 6],
        0,
        -1,
        TriangularHalf::Lower,
    )
    .unwrap();

    runtime.triangular_part(&src, &dst, &spec).unwrap();

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_triangular_part_reference(
        &src_host,
        &dims,
        &src_strides,
        0,
        &[1isize, 3, 6],
        -1,
        TriangularHalf::Lower,
    );
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_triangular_part_complex64_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{TriangularHalf, TriangularPartSpec};

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    let src_host: Vec<Complex64> = (1..=18)
        .map(|value| Complex64::new(value as f64, -(value as f64)))
        .collect();
    let src = runtime.alloc::<Complex64>(src_host.len()).unwrap();
    let dst = runtime.alloc::<Complex64>(src_host.len()).unwrap();
    runtime.copy_htod(&src_host, &src).unwrap();

    let dims = [3usize, 3, 2];
    let src_strides = [1isize, 3, 9];
    let spec = TriangularPartSpec::new(
        &dims,
        &src_strides,
        0,
        &[1isize, 3, 9],
        0,
        1,
        TriangularHalf::Upper,
    )
    .unwrap();

    runtime.triangular_part(&src, &dst, &spec).unwrap();

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_triangular_part_reference(
        &src_host,
        &dims,
        &src_strides,
        0,
        &[1isize, 3, 9],
        1,
        TriangularHalf::Upper,
    );
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_triangular_merge_f64_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::TriangularMergeSpec;

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    let lower_host: Vec<f64> = (1..=18).map(|value| value as f64).collect();
    let upper_host: Vec<f64> = (101..=124).map(|value| value as f64).collect();
    let lower = runtime.alloc::<f64>(lower_host.len()).unwrap();
    let upper = runtime.alloc::<f64>(upper_host.len()).unwrap();
    let dst = runtime.alloc::<f64>(24).unwrap();
    runtime.copy_htod(&lower_host, &lower).unwrap();
    runtime.copy_htod(&upper_host, &upper).unwrap();

    let dims = [3usize, 4, 2];
    let lower_strides = [1isize, 3, 9];
    let upper_strides = [1isize, 3, 12];
    let dst_strides = [1isize, 3, 12];
    let spec =
        TriangularMergeSpec::new(&dims, &lower_strides, 0, &upper_strides, 0, &dst_strides, 0)
            .unwrap();

    runtime
        .triangular_merge(&lower, &upper, &dst, &spec)
        .unwrap();

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_triangular_merge_reference(
        &lower_host,
        &upper_host,
        &dims,
        &lower_strides,
        0,
        &upper_strides,
        0,
        &dst_strides,
    );
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_triangular_merge_complex64_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::TriangularMergeSpec;

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    let lower_host: Vec<Complex64> = (1..=8)
        .map(|value| Complex64::new(value as f64, value as f64 + 0.5))
        .collect();
    let upper_host: Vec<Complex64> = (21..=24)
        .map(|value| Complex64::new(value as f64, -(value as f64)))
        .collect();
    let lower = runtime.alloc::<Complex64>(lower_host.len()).unwrap();
    let upper = runtime.alloc::<Complex64>(upper_host.len()).unwrap();
    let dst = runtime.alloc::<Complex64>(8).unwrap();
    runtime.copy_htod(&lower_host, &lower).unwrap();
    runtime.copy_htod(&upper_host, &upper).unwrap();

    let dims = [4usize, 2];
    let lower_strides = [1isize, 4];
    let upper_strides = [1isize, 2];
    let dst_strides = [1isize, 4];
    let spec =
        TriangularMergeSpec::new(&dims, &lower_strides, 0, &upper_strides, 0, &dst_strides, 0)
            .unwrap();

    runtime
        .triangular_merge(&lower, &upper, &dst, &spec)
        .unwrap();

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_triangular_merge_reference(
        &lower_host,
        &upper_host,
        &dims,
        &lower_strides,
        0,
        &upper_strides,
        0,
        &dst_strides,
    );
    assert_eq!(got, expected);
}
