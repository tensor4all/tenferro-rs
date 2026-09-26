//! Mixing an external scalar with another scalar.
//!
//! Tenferro cannot relate a scalar it does not declare to anything, so the dtype
//! lattice has no answer for two distinct external tags. These tests pin what
//! actually happens: the lattice cannot express the relationship, and every
//! executing entry point rejects the conversion with a typed error instead of
//! applying one operand's kernel to the other's payload.

use tenferro_cpu::CpuBackend;
use tenferro_df64_proof::Df64;
use tenferro_tensor::validate::{can_convert_dtype, promote_dtype};
use tenferro_tensor::{BackendSessionHost, DType, Tensor, TensorRead};
use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

fn external_df64(values: &[Df64]) -> Tensor {
    Tensor::external(ErasedHostTensor::new(
        HostTensor::from_vec_col_major(vec![values.len()], values.to_vec())
            .expect("shape matches data"),
    ))
}

fn external_i64(values: &[i64]) -> Tensor {
    Tensor::external(ErasedHostTensor::new(
        HostTensor::from_vec_col_major(vec![values.len()], values.to_vec())
            .expect("shape matches data"),
    ))
}

fn df64_dtype() -> DType {
    DType::External(std::any::TypeId::of::<Df64>())
}

fn i64_dtype() -> DType {
    DType::External(std::any::TypeId::of::<i64>())
}

#[test]
fn the_lattice_cannot_relate_two_distinct_external_tags() {
    // The lattice is derived from declared facts, and an external member declares
    // no relation to another one, so it can only return one of its two inputs.
    // That is exactly why execution must not rely on it for this pair.
    assert_eq!(promote_dtype(df64_dtype(), i64_dtype()), df64_dtype());
    assert_eq!(promote_dtype(i64_dtype(), df64_dtype()), i64_dtype());

    // The same tag is a real answer, and so is an external tag against a preset.
    assert_eq!(promote_dtype(df64_dtype(), df64_dtype()), df64_dtype());
    assert_eq!(promote_dtype(df64_dtype(), DType::F64), df64_dtype());
    assert_eq!(promote_dtype(DType::F64, df64_dtype()), df64_dtype());
}

#[test]
fn a_conversion_between_distinct_external_tags_is_rejected() {
    let mut backend = CpuBackend::new();

    let source = external_df64(&[Df64::from_f64(1.0)]);

    // One external tag is not convertible into another: the conversion has no
    // table for either direction and says so instead of reinterpreting bytes.
    assert!(!can_convert_dtype(df64_dtype(), i64_dtype()));
    assert!(!can_convert_dtype(i64_dtype(), df64_dtype()));
    assert!(backend
        .with_backend_session(|__s| __s.convert(&source, i64_dtype()))
        .is_err());
    assert!(backend
        .with_backend_session(|__s| __s.convert(&external_i64(&[1]), df64_dtype()))
        .is_err());
}

#[test]
fn a_conversion_between_a_preset_and_an_external_tag_is_rejected() {
    let mut backend = CpuBackend::new();

    // An external destination has no conversion table here, and neither has an
    // external source, so neither direction guesses a representation.
    let ordinary = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).expect("shape matches data");
    assert!(backend
        .with_backend_session(|__s| __s.convert(&ordinary, df64_dtype()))
        .is_err());
    assert!(backend
        .with_backend_session(|__s| __s.convert(&external_df64(&[Df64::from_f64(1.0)]), DType::F64))
        .is_err());
    assert!(backend
        .with_backend_session(
            |__s| __s.convert(&ordinary, DType::External(std::any::TypeId::of::<f64>()))
        )
        .is_err());
}

#[test]
fn a_binary_operation_does_not_apply_one_payload_to_the_other() {
    let mut backend = CpuBackend::new();
    let external = external_df64(&[Df64::from_f64(1.0)]);
    let other = external_i64(&[1]);

    backend.with_backend_session(|session| {
        // A preset-only kernel is never instantiated for an external payload, and
        // nothing here converts one payload into the other's element type.
        assert!(session
            .add_read(
                TensorRead::from_tensor(&external),
                TensorRead::from_tensor(&other)
            )
            .is_err());
        assert!(session
            .add_read(
                TensorRead::from_tensor(&external),
                TensorRead::from_tensor(&external)
            )
            .is_err());
    });
}
