use std::hash::{Hash, Hasher};

use super::super::descriptor::{
    CufftDirection, CufftPlanDescriptor, CufftPlanStructuralKey, CufftTransformKind,
};
use super::super::error::{into_backend_source_error, into_tensor_error, CudaFftError};

fn assert_descriptor(
    kind: CufftTransformKind,
    direction: CufftDirection,
    inembed: [i64; 1],
    onembed: [i64; 1],
) {
    let descriptor = CufftPlanDescriptor::new(kind, direction, 8, 3).unwrap();

    assert_eq!(descriptor.kind, kind);
    assert_eq!(descriptor.direction, direction);
    assert_eq!(descriptor.rank, 1);
    assert_eq!(descriptor.n, [8]);
    assert_eq!(descriptor.inembed, inembed);
    assert_eq!(descriptor.onembed, onembed);
    assert_eq!(descriptor.istride, 3);
    assert_eq!(descriptor.idist, 1);
    assert_eq!(descriptor.ostride, 3);
    assert_eq!(descriptor.odist, 1);
    assert_eq!(descriptor.batch, 3);
}

#[test]
fn descriptor_maps_all_cufft_transform_kinds_to_rank_one_layouts() {
    assert_descriptor(CufftTransformKind::C2c32, CufftDirection::Forward, [8], [8]);
    assert_descriptor(CufftTransformKind::C2c64, CufftDirection::Inverse, [8], [8]);
    assert_descriptor(CufftTransformKind::R2c32, CufftDirection::Forward, [8], [5]);
    assert_descriptor(CufftTransformKind::R2c64, CufftDirection::Forward, [8], [5]);
    assert_descriptor(CufftTransformKind::C2r32, CufftDirection::Inverse, [5], [8]);
    assert_descriptor(CufftTransformKind::C2r64, CufftDirection::Inverse, [5], [8]);
}

#[test]
fn descriptor_uses_ceil_half_spectrum_extent_for_odd_lengths() {
    let r2c =
        CufftPlanDescriptor::new(CufftTransformKind::R2c64, CufftDirection::Forward, 7, 3).unwrap();
    assert_eq!(r2c.inembed, [7]);
    assert_eq!(r2c.onembed, [4]);

    let c2r =
        CufftPlanDescriptor::new(CufftTransformKind::C2r64, CufftDirection::Inverse, 7, 3).unwrap();
    assert_eq!(c2r.inembed, [4]);
    assert_eq!(c2r.onembed, [7]);
}

fn assert_invalid(result: Result<CufftPlanDescriptor, CudaFftError>, field: &'static str) {
    assert!(matches!(
        result,
        Err(CudaFftError::InvalidConfiguration { field: actual }) if actual == field
    ));
}

#[test]
fn descriptor_rejects_zero_and_overflowing_configurations() {
    assert_invalid(
        CufftPlanDescriptor::new(CufftTransformKind::C2c32, CufftDirection::Forward, 0, 1),
        "n",
    );
    assert_invalid(
        CufftPlanDescriptor::new(CufftTransformKind::C2c32, CufftDirection::Forward, 1, 0),
        "batch",
    );
    let product_overflow_n = usize::try_from(i64::MAX).unwrap_or_else(|_| usize::MAX / 2 + 1);
    assert_invalid(
        CufftPlanDescriptor::new(
            CufftTransformKind::C2c32,
            CufftDirection::Forward,
            product_overflow_n,
            2,
        ),
        "element_count",
    );

    if let Ok(outside_signed_width) = usize::try_from(i64::MAX as u128 + 1) {
        assert_invalid(
            CufftPlanDescriptor::new(
                CufftTransformKind::C2c32,
                CufftDirection::Forward,
                outside_signed_width,
                1,
            ),
            "n",
        );
    }
}

#[derive(Default)]
struct ConstantHasher;

impl Hasher for ConstantHasher {
    fn finish(&self) -> u64 {
        0
    }

    fn write(&mut self, _bytes: &[u8]) {}
}

fn constant_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = ConstantHasher;
    value.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn distinct_structural_keys_require_exact_comparison_even_when_hashes_collide() {
    let first = CufftPlanStructuralKey::new(
        0,
        CufftTransformKind::C2c32,
        CufftDirection::Forward,
        8,
        3,
        3,
        1,
        3,
        1,
    );
    let second = CufftPlanStructuralKey::new(
        0,
        CufftTransformKind::C2c32,
        CufftDirection::Forward,
        7,
        3,
        3,
        1,
        3,
        1,
    );

    assert_eq!(constant_hash(&first), constant_hash(&second));
    assert_ne!(first, second);
}

#[test]
fn invalid_descriptor_configuration_is_structured_tensor_validation() {
    let error = into_tensor_error("fft", CudaFftError::InvalidConfiguration { field: "batch" });

    assert_eq!(
        error.kind(),
        tenferro_tensor::ErrorKind::Validation(tenferro_tensor::ValidationKind::InvalidArgument)
    );
    assert!(matches!(
        error,
        tenferro_tensor::Error::Validation {
            source: tenferro_tensor::ValidationError::InvalidArgument {
                argument: "batch",
                ..
            },
            ..
        }
    ));
}

#[test]
fn backend_source_translation_keeps_the_typed_source() {
    let error = into_backend_source_error("fft", std::io::Error::other("cuFFT loader failed"));
    let source = std::error::Error::source(&error).expect("typed backend source is retained");
    assert!(source.downcast_ref::<std::io::Error>().is_some());
}
