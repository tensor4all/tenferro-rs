//! Refusals: nothing is transferred, compacted, converted or overwritten silently.

use hdf5_rt::types::VarLenUnicode;
use hdf5_rt::{Dataset, Group};
use tenferro_hdf5::hdf5;
use tenferro_hdf5::{
    read_info, read_tensor, read_typed, write_tensor, write_typed, Error, ReadOptions,
    SchemaViolation,
};
use tenferro_tensor::{
    BackendStorageHandle, DynRank, Dynamic, ErasedHostTensor, Gpu, Host, Placement, Rank,
    StorageBuffer, Tensor, TypedTensor, TypedTensorView,
};

fn fixture() -> (tempfile::TempDir, hdf5::File, hdf5::Group) {
    let dir = tempfile::tempdir().unwrap();
    let file = hdf5::File::create(dir.path().join("tensors.h5")).unwrap();
    let group = file.create_group("tensors").unwrap();
    (dir, file, group)
}

fn f64_tensor(shape: Vec<usize>, len: usize) -> TypedTensor<f64, DynRank, Host> {
    TypedTensor::from_host_vec_col_major(shape, vec![1.5; len]).unwrap()
}

fn write_text(dataset: &Dataset, name: &'static str, value: &str) {
    let attr = dataset
        .new_attr::<VarLenUnicode>()
        .shape(())
        .create(name)
        .unwrap();
    let value: VarLenUnicode = value.parse().unwrap();
    attr.write_scalar(&value).unwrap();
}

/// A tenferro-shaped dataset whose attributes the caller can corrupt.
fn fake_dataset(
    group: &Group,
    name: &str,
    type_name: &str,
    dtype_name: &str,
    version: i64,
) -> Dataset {
    let dataset = group
        .new_dataset::<f64>()
        .shape(vec![2_usize])
        .create(name)
        .unwrap();
    write_text(&dataset, "type", type_name);
    let version_attr = dataset
        .new_attr::<i64>()
        .shape(())
        .create("version")
        .unwrap();
    version_attr.write_scalar(&version).unwrap();
    write_text(&dataset, "dtype", dtype_name);
    dataset
}

#[test]
fn external_dtype_is_rejected() {
    let (_dir, _file, group) = fixture();
    let typed =
        TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![0.0; 2]).unwrap();
    let external = Tensor::external(ErasedHostTensor::new(typed));
    let error = write_tensor(&group, "a", &external).unwrap_err();
    assert!(matches!(error, Error::UnsupportedDType { .. }), "{error:?}");
    assert!(!group.link_exists("a"));
}

#[test]
fn backend_resident_tensor_is_rejected() {
    let (_dir, _file, group) = fixture();
    let handle = BackendStorageHandle::<f64>::new_with_len(1, 2);
    let device = TypedTensor::<f64, DynRank, Gpu>::from_backend_buffer_col_major(
        vec![2],
        StorageBuffer::Backend(Box::new(handle)),
        Placement::default(),
    )
    .unwrap();
    let converted: TypedTensor<f64, DynRank, Dynamic> = device.into_dynamic();
    let error = write_typed(&group, "a", converted.as_view()).unwrap_err();
    assert!(matches!(error, Error::NotHostResident { .. }), "{error:?}");
    assert!(!group.link_exists("a"));
}

#[test]
fn non_compact_view_is_rejected() {
    let (_dir, _file, group) = fixture();
    let storage: Vec<f64> = (0..6).map(f64::from).collect();
    let view: TypedTensorView<'_, f64, DynRank, Dynamic> =
        TypedTensorView::from_slice(vec![3, 2], vec![2, 1], 0, &storage).unwrap();
    let error = write_typed(&group, "a", view).unwrap_err();
    assert!(matches!(error, Error::NotCompact { .. }), "{error:?}");
    assert!(!group.link_exists("a"));
}

#[test]
fn rank_above_hdf5_limit_is_rejected() {
    let (_dir, _file, group) = fixture();
    let storage = [1.0_f64];
    let view: TypedTensorView<'_, f64, DynRank, Dynamic> =
        TypedTensorView::from_slice(vec![1; 33], vec![1; 33], 0, &storage).unwrap();
    let error = write_typed(&group, "a", view).unwrap_err();
    assert!(
        matches!(error, Error::RankTooLarge { rank: 33 }),
        "{error:?}"
    );
}

#[test]
fn existing_link_is_not_overwritten() {
    let (_dir, _file, group) = fixture();
    let tensor = f64_tensor(vec![2], 2);
    write_typed(&group, "a", tensor.as_view()).unwrap();
    let error = write_typed(&group, "a", tensor.as_view()).unwrap_err();
    assert!(matches!(error, Error::AlreadyExists { .. }), "{error:?}");

    group.unlink("a").unwrap();
    write_typed(&group, "a", tensor.as_view()).unwrap();
    let back: TypedTensor<f64, DynRank, Host> = read_typed(&group, "a").unwrap();
    assert_eq!(back.as_slice(), tensor.as_slice());
}

#[test]
fn invalid_names_are_rejected() {
    let (_dir, _file, group) = fixture();
    let tensor = f64_tensor(vec![2], 2);
    for name in ["", ".", "..", "nested/a"] {
        let error = write_typed(&group, name, tensor.as_view()).unwrap_err();
        assert!(matches!(error, Error::InvalidName { .. }), "{name:?}");
    }
}

#[test]
fn typed_dtype_mismatch_is_rejected() {
    let (_dir, _file, group) = fixture();
    let tensor = f64_tensor(vec![2], 2);
    write_typed(&group, "a", tensor.as_view()).unwrap();
    let error = read_typed::<i64, DynRank>(&group, "a").unwrap_err();
    assert!(
        matches!(
            error,
            Error::DTypeMismatch {
                expected: tenferro_tensor::DType::I64,
                found: tenferro_tensor::DType::F64
            }
        ),
        "{error:?}"
    );
}

#[test]
fn static_rank_mismatch_is_rejected() {
    let (_dir, _file, group) = fixture();
    let tensor = f64_tensor(vec![2, 3], 6);
    write_typed(&group, "a", tensor.as_view()).unwrap();
    let error = read_typed::<f64, Rank<3>>(&group, "a").unwrap_err();
    assert!(
        matches!(
            error,
            Error::RankMismatch {
                expected: 3,
                found: 2
            }
        ),
        "{error:?}"
    );
}

#[test]
fn missing_and_wrong_attributes_are_rejected() {
    let (_dir, _file, group) = fixture();

    let plain = group
        .new_dataset::<f64>()
        .shape(vec![2_usize])
        .create("plain")
        .unwrap();
    let error = read_info(&group, "plain").unwrap_err();
    assert!(
        matches!(
            error,
            Error::Schema {
                violation: SchemaViolation::MissingAttribute("type"),
                ..
            }
        ),
        "{error:?}"
    );
    drop(plain);

    fake_dataset(&group, "foreign", "idx_tensor.ITensor", "f64", 1);
    let error = read_info(&group, "foreign").unwrap_err();
    assert!(
        matches!(
            error,
            Error::Schema {
                violation: SchemaViolation::NotTenferroTensor { .. },
                ..
            }
        ),
        "{error:?}"
    );

    fake_dataset(&group, "unknown", "tenferro.Tensor", "f128", 1);
    let error = read_info(&group, "unknown").unwrap_err();
    assert!(
        matches!(
            error,
            Error::Schema {
                violation: SchemaViolation::UnknownDTypeName(_),
                ..
            }
        ),
        "{error:?}"
    );

    fake_dataset(&group, "future", "tenferro.Tensor", "f64", 2);
    let error = read_info(&group, "future").unwrap_err();
    assert!(
        matches!(error, Error::UnsupportedVersion { found: 2, .. }),
        "{error:?}"
    );
}

#[test]
fn attributes_with_a_null_dataspace_are_rejected() {
    let (_dir, _file, group) = fixture();
    // A null dataspace has rank zero too, so it must be refused before the
    // binding reads a scalar out of it.
    let null = |dataset: &Dataset, name: &'static str| {
        dataset
            .new_attr::<i64>()
            .shape(hdf5_rt::Extents::Null)
            .create(name)
            .unwrap();
    };

    let dataset = group
        .new_dataset::<f64>()
        .shape(vec![2_usize])
        .create("null_type")
        .unwrap();
    null(&dataset, "type");
    let error = read_info(&group, "null_type").unwrap_err();
    assert!(
        matches!(
            error,
            Error::Schema {
                violation: SchemaViolation::AttributeNotScalar("type"),
                ..
            }
        ),
        "{error:?}"
    );

    let dataset = group
        .new_dataset::<f64>()
        .shape(vec![2_usize])
        .create("null_version")
        .unwrap();
    write_text(&dataset, "type", "tenferro.Tensor");
    null(&dataset, "version");
    let error = read_info(&group, "null_version").unwrap_err();
    assert!(
        matches!(
            error,
            Error::Schema {
                violation: SchemaViolation::AttributeNotScalar("version"),
                ..
            }
        ),
        "{error:?}"
    );

    let dataset = group
        .new_dataset::<f64>()
        .shape(vec![2_usize])
        .create("null_dtype")
        .unwrap();
    write_text(&dataset, "type", "tenferro.Tensor");
    let version = dataset
        .new_attr::<i64>()
        .shape(())
        .create("version")
        .unwrap();
    version.write_scalar(&1_i64).unwrap();
    null(&dataset, "dtype");
    let error = read_info(&group, "null_dtype").unwrap_err();
    assert!(
        matches!(
            error,
            Error::Schema {
                violation: SchemaViolation::AttributeNotScalar("dtype"),
                ..
            }
        ),
        "{error:?}"
    );
}

#[test]
fn non_integer_version_attribute_is_rejected() {
    let (_dir, _file, group) = fixture();
    let dataset = group
        .new_dataset::<f64>()
        .shape(vec![2_usize])
        .create("float_version")
        .unwrap();
    write_text(&dataset, "type", "tenferro.Tensor");
    // 1.5 would convert to 1 without a datatype check.
    let version = dataset
        .new_attr::<f64>()
        .shape(())
        .create("version")
        .unwrap();
    version.write_scalar(&1.5_f64).unwrap();
    write_text(&dataset, "dtype", "f64");

    let error = read_info(&group, "float_version").unwrap_err();
    assert!(
        matches!(
            error,
            Error::Schema {
                violation: SchemaViolation::AttributeNotInteger {
                    name: "version",
                    ..
                },
                ..
            }
        ),
        "{error:?}"
    );
}

#[test]
fn datatype_disagreeing_with_dtype_attribute_is_rejected() {
    let (_dir, _file, group) = fixture();
    // f64 datatype, but the schema claims f32.
    fake_dataset(&group, "a", "tenferro.Tensor", "f32", 1);
    let error = read_info(&group, "a").unwrap_err();
    assert!(
        matches!(
            error,
            Error::Schema {
                violation: SchemaViolation::DatatypeMismatch { .. },
                ..
            }
        ),
        "{error:?}"
    );
}

#[test]
fn null_dataspace_is_rejected() {
    let (_dir, _file, group) = fixture();
    let dataset = group
        .new_dataset::<f64>()
        .shape(hdf5_rt::Extents::Null)
        .create("a")
        .unwrap();
    write_text(&dataset, "type", "tenferro.Tensor");
    let version_attr = dataset
        .new_attr::<i64>()
        .shape(())
        .create("version")
        .unwrap();
    version_attr.write_scalar(&1_i64).unwrap();
    write_text(&dataset, "dtype", "f64");
    let error = read_info(&group, "a").unwrap_err();
    assert!(
        matches!(
            error,
            Error::Schema {
                violation: SchemaViolation::NullDataspace,
                ..
            }
        ),
        "{error:?}"
    );
}

#[test]
fn overflowing_shape_is_rejected_before_allocation() {
    let (_dir, _file, group) = fixture();
    // 2^60 elements fit `usize`, 8 bytes each do not fit `isize`.
    let dataset = group
        .new_dataset::<f64>()
        .shape(vec![1_usize << 40, 1_usize << 20])
        .create("a")
        .unwrap();
    write_text(&dataset, "type", "tenferro.Tensor");
    let version_attr = dataset
        .new_attr::<i64>()
        .shape(())
        .create("version")
        .unwrap();
    version_attr.write_scalar(&1_i64).unwrap();
    write_text(&dataset, "dtype", "f64");
    let error = read_typed::<f64, DynRank>(&group, "a").unwrap_err();
    assert!(matches!(error, Error::ShapeOverflow { .. }), "{error:?}");
    let error = read_tensor(&group, "a").unwrap_err();
    assert!(matches!(error, Error::ShapeOverflow { .. }), "{error:?}");
}

#[test]
fn write_into_a_read_only_file_leaves_no_link() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tensors.h5");
    {
        let file = hdf5::File::create(&path).unwrap();
        file.create_group("tensors").unwrap();
    }
    let reopened = hdf5::File::open(&path).unwrap();
    assert!(reopened.is_read_only());
    let readonly = reopened.group("tensors").unwrap();
    let tensor = f64_tensor(vec![2], 2);
    let error = write_typed(&readonly, "a", tensor.as_view()).unwrap_err();
    assert!(matches!(error, Error::Hdf5 { .. }), "{error:?}");
    assert!(!readonly.link_exists("a"));
}

#[test]
fn element_limit_applies_to_dtype_erased_reads() {
    let (_dir, _file, group) = fixture();
    let tensor = Tensor::from_vec_col_major(vec![4], vec![1.0_f64; 4]).unwrap();
    write_tensor(&group, "a", &tensor).unwrap();
    let error =
        tenferro_hdf5::read_tensor_with(&group, "a", &ReadOptions::default().max_elements(3))
            .unwrap_err();
    assert!(
        matches!(
            error,
            Error::ElementLimit {
                elements: 4,
                limit: 3
            }
        ),
        "{error:?}"
    );
}

#[test]
fn missing_dataset_reports_the_hdf5_error() {
    let (_dir, _file, group) = fixture();
    let error = read_info(&group, "absent").unwrap_err();
    assert!(matches!(error, Error::Hdf5 { .. }), "{error:?}");
}
