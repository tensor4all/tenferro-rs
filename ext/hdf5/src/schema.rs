//! The v1 on-disk schema: names, dataspaces and dataset attributes.

use hdf5_rt::types::TypeDescriptor;
use hdf5_rt::types::{VarLenAscii, VarLenUnicode};
use hdf5_rt::{Container, Dataset, Extents, H5Type};
use num_complex::{Complex32, Complex64};
use tenferro_tensor::DType;

use crate::error::{Error, SchemaViolation};
use crate::{SCHEMA_TYPE, SCHEMA_VERSION};

/// HDF5's own maximum dataspace rank (`H5S_MAX_RANK`).
pub(crate) const MAX_RANK: usize = 32;

const TYPE_ATTR: &str = "type";
const VERSION_ATTR: &str = "version";
const DTYPE_ATTR: &str = "dtype";

pub(crate) fn hdf5_err(op: &'static str) -> impl Fn(hdf5_rt::Error) -> Error {
    move |source| Error::Hdf5 { op, source }
}

fn schema_err(name: &str, violation: SchemaViolation) -> Error {
    Error::Schema {
        name: name.to_owned(),
        violation,
    }
}

/// Reject a name that is not one ordinary link name in the caller's group.
pub(crate) fn validate_name(name: &str) -> Result<(), Error> {
    let invalid =
        name.is_empty() || name == "." || name == ".." || name.contains('/') || name.contains('\0');
    if invalid {
        return Err(Error::InvalidName {
            name: name.to_owned(),
        });
    }
    Ok(())
}

/// The schema name of a dtype. Only the seven built-in scalar types have one.
pub(crate) fn dtype_name(dtype: DType) -> Result<&'static str, Error> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F64 => Ok("f64"),
        DType::C32 => Ok("c32"),
        DType::C64 => Ok("c64"),
        DType::I32 => Ok("i32"),
        DType::I64 => Ok("i64"),
        DType::Bool => Ok("bool"),
        DType::External(_) => Err(Error::UnsupportedDType { dtype }),
    }
}

/// Parse a schema dtype name.
pub(crate) fn dtype_from_name(name: &str) -> Option<DType> {
    match name {
        "f32" => Some(DType::F32),
        "f64" => Some(DType::F64),
        "c32" => Some(DType::C32),
        "c64" => Some(DType::C64),
        "i32" => Some(DType::I32),
        "i64" => Some(DType::I64),
        "bool" => Some(DType::Bool),
        _ => None,
    }
}

/// The HDF5 datatype a dtype must be stored as.
pub(crate) fn expected_descriptor(dtype: DType) -> Result<TypeDescriptor, Error> {
    Ok(match dtype {
        DType::F32 => f32::type_descriptor(),
        DType::F64 => f64::type_descriptor(),
        DType::I32 => i32::type_descriptor(),
        DType::I64 => i64::type_descriptor(),
        DType::Bool => bool::type_descriptor(),
        DType::C32 => Complex32::type_descriptor(),
        DType::C64 => Complex64::type_descriptor(),
        DType::External(_) => return Err(Error::UnsupportedDType { dtype }),
    })
}

/// The dataspace extents for a logical column-major shape.
///
/// HDF5 extents are C-order (last dimension fastest) and tenferro storage is
/// column-major, so the extents are stored reversed: the payload then *is* the
/// C-order buffer of the transposed array, and both directions are copy-free.
/// Rank 0 becomes a scalar dataspace.
pub(crate) fn file_extents(shape: &[usize]) -> Extents {
    Extents::from(shape.iter().rev().copied().collect::<Vec<usize>>())
}

/// Reverse the stored extents back into a logical column-major shape.
pub(crate) fn logical_shape(extents: Extents, name: &str) -> Result<Vec<usize>, Error> {
    match extents {
        Extents::Null => Err(schema_err(name, SchemaViolation::NullDataspace)),
        Extents::Scalar => Ok(Vec::new()),
        Extents::Simple(simple) => Ok(simple.iter().rev().map(|extent| extent.dim).collect()),
    }
}

/// Validate element count and byte count of a declared shape before allocation.
pub(crate) fn checked_len(shape: &[usize], element_size: usize) -> Result<usize, Error> {
    let overflow = || Error::ShapeOverflow {
        dims: shape.to_vec(),
    };
    let elements = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(overflow)?;
    let bytes = elements.checked_mul(element_size).ok_or_else(overflow)?;
    if bytes > isize::MAX as usize {
        return Err(overflow());
    }
    Ok(elements)
}

fn text_value(value: &str) -> Result<VarLenUnicode, Error> {
    value
        .parse::<VarLenUnicode>()
        .map_err(|_| Error::InvalidAttributeValue {
            name: DTYPE_ATTR,
            value: value.to_owned(),
        })
}

fn write_text(ds: &Dataset, name: &'static str, value: &str) -> Result<(), Error> {
    let attr = ds
        .new_attr::<VarLenUnicode>()
        .shape(())
        .create(name)
        .map_err(hdf5_err("write_typed"))?;
    attr.write_scalar(&text_value(value)?)
        .map_err(hdf5_err("write_typed"))
}

/// Write the three schema attributes of a freshly created dataset.
pub(crate) fn write_attributes(ds: &Dataset, dtype: DType) -> Result<(), Error> {
    write_text(ds, TYPE_ATTR, SCHEMA_TYPE)?;
    let version = ds
        .new_attr::<i64>()
        .shape(())
        .create(VERSION_ATTR)
        .map_err(hdf5_err("write_typed"))?;
    version
        .write_scalar(&SCHEMA_VERSION)
        .map_err(hdf5_err("write_typed"))?;
    write_text(ds, DTYPE_ATTR, dtype_name(dtype)?)
}

/// Reject an attribute whose dataspace is not one scalar element.
///
/// A null dataspace has rank zero too, and reading it as a scalar would take an
/// uninitialized value out of the binding.
fn require_scalar(attr: &Container, name: &'static str) -> Result<(), Error> {
    let extents = attr
        .space()
        .and_then(|space| space.extents())
        .map_err(hdf5_err("read_info"))?;
    if extents != Extents::Scalar {
        return Err(schema_err(name, SchemaViolation::AttributeNotScalar(name)));
    }
    Ok(())
}

/// Read a scalar text attribute, accepting variable-length UTF-8 or ASCII.
///
/// The bytes are validated as UTF-8 before they become a Rust string: an HDF5
/// string's character set says nothing about the bytes a writer stored.
pub(crate) fn read_text_attr(ds: &Dataset, name: &'static str) -> Result<Option<String>, Error> {
    let attr = match ds.attr(name) {
        Ok(attr) => attr,
        Err(_) => return Ok(None),
    };
    require_scalar(&attr, name)?;
    let bytes = match attr.read_scalar::<VarLenUnicode>() {
        Ok(value) => value.as_bytes().to_vec(),
        Err(_) => attr
            .read_scalar::<VarLenAscii>()
            .map_err(hdf5_err("read_info"))?
            .as_bytes()
            .to_vec(),
    };
    let text = core::str::from_utf8(&bytes)
        .map_err(|_| schema_err(name, SchemaViolation::AttributeNotUtf8(name)))?;
    Ok(Some(text.to_owned()))
}

/// Validate `type`, `version` and `dtype`, returning the declared dtype and version.
pub(crate) fn read_attributes(ds: &Dataset, name: &str) -> Result<(DType, i64), Error> {
    let found_type = read_text_attr(ds, TYPE_ATTR)?
        .ok_or_else(|| schema_err(name, SchemaViolation::MissingAttribute(TYPE_ATTR)))?;
    if found_type != SCHEMA_TYPE {
        return Err(schema_err(
            name,
            SchemaViolation::NotTenferroTensor { found: found_type },
        ));
    }
    let version_attr = ds
        .attr(VERSION_ATTR)
        .map_err(|_| schema_err(name, SchemaViolation::MissingAttribute(VERSION_ATTR)))?;
    require_scalar(&version_attr, VERSION_ATTR)?;
    let version_type = version_attr
        .dtype()
        .and_then(|datatype| datatype.to_descriptor())
        .map_err(hdf5_err("read_info"))?;
    if !matches!(version_type, TypeDescriptor::Integer(_)) {
        return Err(schema_err(
            name,
            SchemaViolation::AttributeNotInteger {
                name: VERSION_ATTR,
                found: version_type.to_string(),
            },
        ));
    }
    let version: i64 = version_attr.read_scalar().map_err(hdf5_err("read_info"))?;
    if version != SCHEMA_VERSION {
        return Err(Error::UnsupportedVersion {
            found: version,
            supported: SCHEMA_VERSION,
        });
    }
    let dtype_name = read_text_attr(ds, DTYPE_ATTR)?
        .ok_or_else(|| schema_err(name, SchemaViolation::MissingAttribute(DTYPE_ATTR)))?;
    let dtype = dtype_from_name(&dtype_name)
        .ok_or_else(|| schema_err(name, SchemaViolation::UnknownDTypeName(dtype_name)))?;
    Ok((dtype, version))
}

/// Require that the stored HDF5 datatype really encodes the declared dtype.
pub(crate) fn check_datatype(ds: &Dataset, dtype: DType, name: &str) -> Result<(), Error> {
    let found = ds
        .dtype()
        .and_then(|datatype| datatype.to_descriptor())
        .map_err(hdf5_err("read_info"))?;
    if found != expected_descriptor(dtype)? {
        return Err(schema_err(
            name,
            SchemaViolation::DatatypeMismatch {
                dtype,
                found: found.to_string(),
            },
        ));
    }
    Ok(())
}
