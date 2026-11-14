//! The write path: one dataset per tensor, written from the caller's own buffer.

use core::mem::size_of;
use hdf5_rt::{Dataset, Group, H5Type};
use tenferro_tensor::{DType, Representation, Tensor, TensorRank, TensorScalar, TypedTensorView};

use crate::error::Error;
use crate::schema;

/// Write a typed tensor view into `group` under the single link name `name`.
///
/// The payload is written straight from the view's host buffer: nothing is
/// copied, compacted or transferred. Only compact column-major views are
/// accepted, and only host-resident storage.
///
/// # Examples
///
/// ```
/// use tenferro_hdf5::{hdf5, read_typed, write_typed};
/// use tenferro_tensor::{DynRank, Host, TypedTensor};
///
/// let dir = tempfile::tempdir().unwrap();
/// let file = hdf5::File::create(dir.path().join("tensor.h5")).unwrap();
/// let group = file.create_group("tensors").unwrap();
/// let tensor =
///     TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
///
/// write_typed(&group, "a", tensor.as_view())?;
/// let back: TypedTensor<f64, DynRank, Host> = read_typed(&group, "a")?;
/// assert_eq!(back.as_slice(), tensor.as_slice());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::InvalidName`] for a name that is not one ordinary link name,
/// [`Error::RankTooLarge`] above rank 32, [`Error::NotCompact`] for a
/// non-compact view, [`Error::NotHostResident`] for backend storage,
/// [`Error::UnsupportedDType`] for a dtype without a v1 encoding,
/// [`Error::AlreadyExists`] when the name is taken, [`Error::ShapeOverflow`] for
/// an unrepresentable element count, and [`Error::Hdf5`] when the dataset or the
/// payload cannot be written.
pub fn write_typed<T, R, D>(
    group: &Group,
    name: &str,
    view: TypedTensorView<'_, T, R, D>,
) -> Result<Dataset, Error>
where
    T: TensorScalar + H5Type,
    R: TensorRank,
    D: Representation,
{
    const OP: &str = "write_typed";
    schema::validate_name(name)?;
    let dtype = T::dtype();
    let shape = view.shape().to_vec();
    let rank = shape.len();
    if rank > schema::MAX_RANK {
        return Err(Error::RankTooLarge { rank });
    }
    let compact = view
        .is_col_major_contiguous()
        .map_err(|source| Error::Tensor { source })?;
    if !compact {
        return Err(Error::NotCompact {
            shape,
            strides: view.strides().to_vec(),
        });
    }
    let data = view
        .as_slice()
        .map_err(|source| Error::NotHostResident { op: OP, source })?;
    let elements = schema::checked_len(&shape, size_of::<T>())?;
    if group.link_exists(name) {
        return Err(Error::AlreadyExists {
            name: name.to_owned(),
        });
    }
    let dataset = group
        .new_dataset::<T>()
        .shape(schema::file_extents(&shape))
        .create(name)
        .map_err(schema::hdf5_err(OP))?;
    let written = (|| {
        schema::write_attributes(&dataset, dtype)?;
        if elements > 0 {
            dataset
                .as_writer()
                .write_raw(data)
                .map_err(schema::hdf5_err(OP))?;
        }
        Ok(())
    })();
    match written {
        Ok(()) => Ok(dataset),
        Err(error) => {
            let _ = group.unlink(name);
            Err(error)
        }
    }
}

/// Write a dtype-erased tensor, dispatching on its runtime dtype.
///
/// # Examples
///
/// ```
/// use tenferro_hdf5::{hdf5, read_tensor, write_tensor};
/// use tenferro_tensor::Tensor;
///
/// let dir = tempfile::tempdir().unwrap();
/// let file = hdf5::File::create(dir.path().join("tensor.h5")).unwrap();
/// let group = file.create_group("tensors").unwrap();
/// let tensor = Tensor::from_vec_col_major(vec![2], vec![1_i32, 2])?;
///
/// write_tensor(&group, "a", &tensor)?;
/// let back = read_tensor(&group, "a")?;
/// assert_eq!(back.dtype(), tensor.dtype());
/// assert_eq!(back.shape(), tensor.shape());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::UnsupportedDType`] for an `External` payload, plus every
/// error of [`write_typed`].
pub fn write_tensor(group: &Group, name: &str, tensor: &Tensor) -> Result<Dataset, Error> {
    let dtype = tensor.dtype();
    match dtype {
        DType::F32 => write_typed_tensor::<f32>(group, name, tensor),
        DType::F64 => write_typed_tensor::<f64>(group, name, tensor),
        DType::I32 => write_typed_tensor::<i32>(group, name, tensor),
        DType::I64 => write_typed_tensor::<i64>(group, name, tensor),
        DType::Bool => write_typed_tensor::<bool>(group, name, tensor),
        DType::C32 => write_typed_tensor::<num_complex::Complex32>(group, name, tensor),
        DType::C64 => write_typed_tensor::<num_complex::Complex64>(group, name, tensor),
        DType::External(_) => Err(Error::UnsupportedDType { dtype }),
    }
}

fn write_typed_tensor<T>(group: &Group, name: &str, tensor: &Tensor) -> Result<Dataset, Error>
where
    T: TensorScalar + H5Type,
{
    let typed = tensor.as_typed::<T>().ok_or(Error::UnsupportedDType {
        dtype: tensor.dtype(),
    })?;
    write_typed(group, name, typed.as_view())
}
