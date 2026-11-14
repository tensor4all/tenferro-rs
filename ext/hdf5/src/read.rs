//! The read path: validate the whole schema before allocating the payload.

use core::any::TypeId;
use core::mem::size_of;
use hdf5_rt::{Dataset, Group, H5Type};
use tenferro_tensor::{DType, DynRank, Host, Tensor, TensorRank, TensorScalar, TypedTensor};

use crate::error::Error;
use crate::schema;

/// Metadata of one stored tensor, as declared by the dataset itself.
///
/// # Examples
///
/// ```
/// use tenferro_hdf5::{hdf5, read_info, write_typed};
/// use tenferro_tensor::{DType, DynRank, Host, TypedTensor};
///
/// let dir = tempfile::tempdir().unwrap();
/// let file = hdf5::File::create(dir.path().join("tensor.h5")).unwrap();
/// let group = file.create_group("tensors").unwrap();
/// let tensor = TypedTensor::<i64, DynRank, Host>::from_host_vec_col_major(vec![3], vec![1, 2, 3])?;
///
/// write_typed(&group, "a", tensor.as_view())?;
/// let info = read_info(&group, "a")?;
/// assert_eq!(info.dtype, DType::I64);
/// assert_eq!(info.shape, vec![3]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TensorInfo {
    /// Dtype the dataset declares.
    pub dtype: DType,
    /// Logical column-major shape the dataset declares.
    pub shape: Vec<usize>,
    /// Schema version the dataset declares.
    pub version: i64,
}

/// Limit applied to a read, for files the caller does not trust.
///
/// The default applies no limit beyond the [`Error::ShapeOverflow`] check that
/// every read performs.
///
/// # Examples
///
/// ```
/// use tenferro_hdf5::ReadOptions;
///
/// let options = ReadOptions::default().max_elements(1024);
/// assert_eq!(options.max_elements, Some(1024));
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ReadOptions {
    /// Maximum number of logical elements a read may allocate.
    pub max_elements: Option<usize>,
}

impl ReadOptions {
    /// Cap the logical element count of a read.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_hdf5::ReadOptions;
    ///
    /// assert_eq!(ReadOptions::default().max_elements(8).max_elements, Some(8));
    /// ```
    #[must_use]
    pub fn max_elements(self, max_elements: usize) -> Self {
        Self {
            max_elements: Some(max_elements),
        }
    }
}

/// Read the declared metadata of a stored tensor without touching its payload.
///
/// # Examples
///
/// ```
/// use tenferro_hdf5::{hdf5, read_info};
/// use tenferro_tensor::{DynRank, Host, TypedTensor};
///
/// let dir = tempfile::tempdir().unwrap();
/// let file = hdf5::File::create(dir.path().join("tensor.h5")).unwrap();
/// let group = file.create_group("tensors").unwrap();
/// let tensor = TypedTensor::<f32, DynRank, Host>::from_host_vec_col_major(vec![4], vec![0.0; 4])?;
/// tenferro_hdf5::write_typed(&group, "a", tensor.as_view())?;
///
/// let info = read_info(&group, "a")?;
/// assert_eq!(info.shape, vec![4]);
/// assert_eq!(info.version, 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::Hdf5`] when the dataset cannot be opened, and
/// [`Error::Schema`], [`Error::UnsupportedVersion`] or
/// [`Error::InvalidAttributeValue`] when its metadata violates the v1 schema.
pub fn read_info(group: &Group, name: &str) -> Result<TensorInfo, Error> {
    let (_, dtype, shape, version) = load(group, name, "read_info")?;
    Ok(TensorInfo {
        dtype,
        shape,
        version,
    })
}

/// Read a typed tensor, allocating exactly its payload once.
///
/// # Examples
///
/// ```
/// use tenferro_hdf5::{hdf5, read_typed, write_typed};
/// use tenferro_tensor::{DynRank, Host, Rank, TypedTensor};
///
/// let dir = tempfile::tempdir().unwrap();
/// let file = hdf5::File::create(dir.path().join("tensor.h5")).unwrap();
/// let group = file.create_group("tensors").unwrap();
/// let tensor = TypedTensor::<f64, Rank<2>, Host>::from_host_vec_col_major([2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
///
/// write_typed(&group, "a", tensor.as_view())?;
/// let back: TypedTensor<f64, Rank<2>, Host> = read_typed(&group, "a")?;
/// assert_eq!(back.shape(), &[2, 2]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::DTypeMismatch`] when the dataset stores another dtype,
/// [`Error::RankMismatch`] for a statically ranked read of another rank,
/// [`Error::ShapeOverflow`], plus the metadata and HDF5 errors of [`read_info`].
pub fn read_typed<T, R>(group: &Group, name: &str) -> Result<TypedTensor<T, R, Host>, Error>
where
    T: TensorScalar + H5Type,
    R: TensorRank,
{
    read_typed_with(group, name, &ReadOptions::default())
}

/// Read a typed tensor under an explicit [`ReadOptions`].
///
/// # Examples
///
/// ```
/// use tenferro_hdf5::{hdf5, write_typed, ReadOptions};
/// use tenferro_tensor::{DynRank, Host, TypedTensor};
///
/// let dir = tempfile::tempdir().unwrap();
/// let file = hdf5::File::create(dir.path().join("tensor.h5")).unwrap();
/// let group = file.create_group("tensors").unwrap();
/// let tensor = TypedTensor::<i32, DynRank, Host>::from_host_vec_col_major(vec![3], vec![1, 2, 3])?;
/// write_typed(&group, "a", tensor.as_view())?;
///
/// let options = ReadOptions::default().max_elements(8);
/// let back: TypedTensor<i32, DynRank, Host> =
///     tenferro_hdf5::read_typed_with(&group, "a", &options)?;
/// assert_eq!(back.as_slice(), &[1, 2, 3]);
///
/// let refused: Result<TypedTensor<i32, DynRank, Host>, _> =
///     tenferro_hdf5::read_typed_with(&group, "a", &ReadOptions::default().max_elements(2));
/// assert!(refused.is_err());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// As [`read_typed`], plus [`Error::ElementLimit`] when the dataset exceeds the
/// configured element limit.
pub fn read_typed_with<T, R>(
    group: &Group,
    name: &str,
    options: &ReadOptions,
) -> Result<TypedTensor<T, R, Host>, Error>
where
    T: TensorScalar + H5Type,
    R: TensorRank,
{
    let (dataset, dtype, shape, _) = load(group, name, "read_typed")?;
    typed_payload::<T, R>(&dataset, shape, dtype, options)
}

/// Checks and payload read shared by the typed and the dtype-erased entry points,
/// so that neither opens the dataset or reads its metadata twice.
fn typed_payload<T, R>(
    dataset: &Dataset,
    shape: Vec<usize>,
    dtype: DType,
    options: &ReadOptions,
) -> Result<TypedTensor<T, R, Host>, Error>
where
    T: TensorScalar + H5Type,
    R: TensorRank,
{
    if dtype != T::dtype() {
        return Err(Error::DTypeMismatch {
            expected: T::dtype(),
            found: dtype,
        });
    }
    let elements = schema::checked_len(&shape, size_of::<T>())?;
    if let Some(limit) = options.max_elements {
        if elements > limit {
            return Err(Error::ElementLimit { elements, limit });
        }
    }
    if let Some(expected) = R::RANK {
        if expected != shape.len() {
            return Err(Error::RankMismatch {
                expected,
                found: shape.len(),
            });
        }
    }
    let data = read_payload::<T>(dataset, elements)?;
    let rank_shape =
        R::shape_from_vec(shape.iter().copied().collect()).map_err(|source| Error::Tensor {
            source: tenferro_tensor::Error::validation("read_typed", source),
        })?;
    TypedTensor::<T, R, Host>::from_host_vec_col_major(rank_shape, data)
        .map_err(|source| Error::Tensor { source })
}

/// Read a dtype-erased tensor, dispatching on the stored dtype.
///
/// # Examples
///
/// ```
/// use tenferro_hdf5::{hdf5, read_tensor, write_tensor};
/// use tenferro_tensor::{DType, Tensor};
///
/// let dir = tempfile::tempdir().unwrap();
/// let file = hdf5::File::create(dir.path().join("tensor.h5")).unwrap();
/// let group = file.create_group("tensors").unwrap();
/// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0])?;
///
/// write_tensor(&group, "a", &tensor)?;
/// let back = read_tensor(&group, "a")?;
/// assert_eq!(back.dtype(), DType::F32);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// As [`read_typed`]; the dtype is taken from the dataset, so a dtype mismatch
/// cannot occur here.
pub fn read_tensor(group: &Group, name: &str) -> Result<Tensor, Error> {
    read_tensor_with(group, name, &ReadOptions::default())
}

/// Read a dtype-erased tensor under an explicit [`ReadOptions`].
///
/// # Examples
///
/// ```
/// use tenferro_hdf5::{hdf5, read_tensor_with, write_tensor, ReadOptions};
/// use tenferro_tensor::Tensor;
///
/// let dir = tempfile::tempdir().unwrap();
/// let file = hdf5::File::create(dir.path().join("tensor.h5")).unwrap();
/// let group = file.create_group("tensors").unwrap();
/// let tensor = Tensor::from_vec_col_major(vec![2], vec![1_i64, 2])?;
/// write_tensor(&group, "a", &tensor)?;
///
/// let back = read_tensor_with(&group, "a", &ReadOptions::default().max_elements(4))?;
/// assert_eq!(back.shape(), &[2]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// As [`read_tensor`], plus [`Error::ElementLimit`].
pub fn read_tensor_with(group: &Group, name: &str, options: &ReadOptions) -> Result<Tensor, Error> {
    let (dataset, dtype, shape, _) = load(group, name, "read_tensor")?;
    match dtype {
        DType::F32 => erased::<f32>(&dataset, shape, options),
        DType::F64 => erased::<f64>(&dataset, shape, options),
        DType::I32 => erased::<i32>(&dataset, shape, options),
        DType::I64 => erased::<i64>(&dataset, shape, options),
        DType::Bool => erased::<bool>(&dataset, shape, options),
        DType::C32 => erased::<num_complex::Complex32>(&dataset, shape, options),
        DType::C64 => erased::<num_complex::Complex64>(&dataset, shape, options),
        DType::External(_) => Err(Error::UnsupportedDType { dtype }),
    }
}

fn erased<T>(dataset: &Dataset, shape: Vec<usize>, options: &ReadOptions) -> Result<Tensor, Error>
where
    T: TensorScalar + H5Type,
{
    let dtype = T::dtype();
    let tensor = typed_payload::<T, DynRank>(dataset, shape, dtype, options)?;
    Ok(Tensor::from_typed(tensor.into_dynamic()))
}

/// Open a dataset and validate its metadata, read no payload.
fn load(
    group: &Group,
    name: &str,
    op: &'static str,
) -> Result<(Dataset, DType, Vec<usize>, i64), Error> {
    let dataset = group
        .dataset(name)
        .map_err(schema::hdf5_err("group.dataset"))?;
    let (dtype, version) = schema::read_attributes(&dataset, name)?;
    let extents = dataset
        .space()
        .and_then(|space| space.extents())
        .map_err(schema::hdf5_err(op))?;
    let shape = schema::logical_shape(extents, name)?;
    schema::check_datatype(&dataset, dtype, name)?;
    Ok((dataset, dtype, shape, version))
}

/// Read the payload with exactly one allocation of the element type.
fn read_payload<T>(dataset: &Dataset, elements: usize) -> Result<Vec<T>, Error>
where
    T: TensorScalar + H5Type,
{
    if elements == 0 {
        return Ok(Vec::new());
    }
    if TypeId::of::<T>() == TypeId::of::<bool>() {
        // Read the int8 enum members through `i8` so that no unvalidated byte is
        // ever interpreted as a `bool`, then reuse the same allocation.
        let mut raw: Vec<i8> = dataset
            .as_reader()
            .read_raw::<i8>()
            .map_err(schema::hdf5_err("read_typed"))?;
        for (flat_index, &byte) in raw.iter().enumerate() {
            if byte != 0 && byte != 1 {
                return Err(Error::InvalidBool {
                    flat_index,
                    byte: byte as u8,
                });
            }
        }
        let pointer = raw.as_mut_ptr().cast::<T>();
        let len = raw.len();
        let capacity = raw.capacity();
        core::mem::forget(raw);
        // SAFETY: `T` is `bool` here (checked above), so `bool` and `i8` share
        // size and alignment, every byte of the allocation was validated to be
        // 0 or 1 (a valid `bool`), and the allocation is released by the same
        // global allocator with the same layout.
        return Ok(unsafe { Vec::from_raw_parts(pointer, len, capacity) });
    }
    dataset
        .as_reader()
        .read_raw::<T>()
        .map_err(schema::hdf5_err("read_typed"))
}
