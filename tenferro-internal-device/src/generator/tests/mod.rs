use super::*;

#[test]
fn default_seed_changes_with_counter_and_extra_bits() {
    let a = default_seed(0);
    let b = default_seed(0);
    let c = default_seed(1);

    assert_ne!(a, b);
    assert_ne!(a, c);
    assert_ne!(b, c);
}

#[test]
fn standard_normal_sampling_uses_and_clears_cached_pair() {
    let mut generator = Generator::cpu(7);

    assert!(generator.state.cached_normal.is_none());
    let first = generator.sample_standard_normal_f64();
    let cached = generator
        .state
        .cached_normal
        .take()
        .expect("Box-Muller should cache the paired sample");
    generator.state.cached_normal = Some(cached);

    let second = generator.sample_standard_normal_f64();
    assert_eq!(second, cached);
    assert!(generator.state.cached_normal.is_none());
    assert!(first.is_finite());
}

#[test]
fn integer_sampling_rejects_invalid_ranges() {
    let mut generator = Generator::cpu(9);
    let err = generator.sample_integer_i32(4, 4).unwrap_err();
    assert!(err.to_string().contains("invalid integer sample range"));
}

#[test]
fn default_cpu_generator_is_shared_singleton() {
    let first = default_cpu_generator();
    let second = default_cpu_generator();

    assert!(std::ptr::eq(first, second));
}

#[test]
fn with_default_generator_uses_shared_cpu_state_for_host_spaces() {
    let main_value = with_default_generator(crate::LogicalMemorySpace::MainMemory, |generator| {
        Ok(generator.sample_uniform_f64())
    })
    .unwrap();
    let pinned_value =
        with_default_generator(crate::LogicalMemorySpace::PinnedMemory, |generator| {
            Ok(generator.sample_uniform_f64())
        })
        .unwrap();
    let managed_value =
        with_default_generator(crate::LogicalMemorySpace::ManagedMemory, |generator| {
            Ok(generator.sample_uniform_f64())
        })
        .unwrap();

    assert!((0.0..1.0).contains(&main_value));
    assert!((0.0..1.0).contains(&pinned_value));
    assert!((0.0..1.0).contains(&managed_value));
    assert_ne!(main_value, pinned_value);
    assert_ne!(pinned_value, managed_value);
}

#[cfg(not(feature = "cuda"))]
#[test]
fn with_default_generator_rejects_gpu_space_without_cuda() {
    let err = with_default_generator(
        crate::LogicalMemorySpace::GpuMemory { device_id: 0 },
        |_| Ok(()),
    )
    .unwrap_err();
    assert!(err.to_string().contains("requires the cuda feature"));
}

#[test]
fn generator_high_word_seed_and_integer_sampling_success_paths_are_covered() {
    let mut lhs = Generator::cpu(1_u64 << 40);
    let mut rhs = Generator::cpu(1_u64 << 40);

    assert_eq!(lhs.sample_uniform_f64(), rhs.sample_uniform_f64());

    let sample = lhs.sample_integer_i32(-3, 7).unwrap();
    assert!((-3..7).contains(&sample));
}

#[test]
fn cpu_generators_are_deterministic_for_uniform_and_normal_sampling() {
    let mut lhs = Generator::cpu(1234);
    let mut rhs = Generator::cpu(1234);

    assert_eq!(lhs.sample_uniform_f64(), rhs.sample_uniform_f64());
    assert_eq!(lhs.sample_uniform_f64(), rhs.sample_uniform_f64());
    assert_eq!(
        lhs.sample_standard_normal_f64(),
        rhs.sample_standard_normal_f64()
    );
    assert_eq!(
        lhs.sample_standard_normal_f64(),
        rhs.sample_standard_normal_f64()
    );
}

#[test]
fn with_default_generator_propagates_closure_errors() {
    let err = with_default_generator(crate::LogicalMemorySpace::MainMemory, |_| {
        Err::<(), _>(Error::InvalidArgument("sentinel".into()))
    })
    .unwrap_err();

    assert!(err.to_string().contains("sentinel"));
}
