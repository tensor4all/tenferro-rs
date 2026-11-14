//! Thin HDF5 payload adapter for tenferro tensors.
//!
//! This crate stores ordinary tenferro values in HDF5 datasets: a
//! [`TypedTensor<T, R, Host>`](tenferro_tensor::TypedTensor), the same tensor
//! once it became [`Dynamic`](tenferro_tensor::Dynamic), and the dtype-erased
//! [`Tensor`](tenferro_tensor::Tensor). It is an optional adapter: no tenferro
//! compute crate depends on it, and it never creates an engine, a thread pool or
//! an HDF5 runtime of its own.
//!
//! # What it does not do
//!
//! - **No hidden transfer.** A tensor whose payload lives in a backend buffer is
//!   rejected with [`Error::NotHostResident`]; nothing is downloaded, uploaded,
//!   synchronized or materialized.
//! - **No hidden copies.** A compact column-major view is written straight from
//!   its host buffer, and a read allocates exactly one payload buffer, which the
//!   returned tensor adopts. A non-compact view is rejected with
//!   [`Error::NotCompact`]; compact it explicitly if that is what you want.
//! - **No numeric conversion.** The dtype, shape and values round-trip
//!   unchanged, bit patterns included. There is no `f32` to `f64` conversion and
//!   no integer widening.
//! - **No path policy.** The caller owns the [`hdf5::File`] and [`hdf5::Group`]
//!   handles, and the adapter only touches the one link name it is given.
//!
//! # Backend
//!
//! One backend: [`hdf5-rt`](https://github.com/tensor4all/hdf5-rt), re-exported
//! here as [`hdf5`], so callers always pass handles of the crate this adapter
//! uses. `hdf5-rt` loads `libhdf5` at run time (Linux: the bare `libhdf5.so` and
//! `libhdf5_serial.so` first, so `LD_LIBRARY_PATH` and `ld.so.conf` are honoured;
//! then the usual serial-install paths), needs no library at build time, and
//! panics inside the first HDF5 call if nothing loads. On Linux install it with
//! `apt-get install libhdf5-dev` or use a conda-forge `hdf5`.
//!
//! # Schema (version 1)
//!
//! One dataset per tensor, linked as `name` in the caller's group, with three
//! scalar attributes: `type = "tenferro.Tensor"`, `version = 1`, and `dtype`
//! one of `f32 f64 c32 c64 i32 i64 bool`.
//!
//! The HDF5 dataspace holds the shape **reversed** (rank 0 becomes a scalar
//! dataspace). HDF5 extents are C-order and tenferro storage is column-major, so
//! the reversed extents make the stored payload identical to the column-major
//! buffer, and both directions are copy-free. In h5py the logical tensor is
//! therefore `np.asarray(dataset).T`.
//!
//! | dtype | HDF5 type | h5py reads it as |
//! |---|---|---|
//! | `f32`, `f64` | `H5T_IEEE_F32LE`, `F64LE` | `float32`, `float64` |
//! | `i32`, `i64` | `H5T_STD_I32LE`, `I64LE` | `int32`, `int64` |
//! | `c32`, `c64` | compound `{"r": F, "i": F}` | `complex64`, `complex128` |
//! | `bool` | int8 enum `{FALSE=0, TRUE=1}` | `bool` |
//!
//! [`DType::External`](tenferro_tensor::DType) payloads have no encoding and are
//! rejected with [`Error::UnsupportedDType`]; a Rust `TypeId` is never persisted.
//! Overwriting is the caller's decision: a taken name fails with
//! [`Error::AlreadyExists`], and replacing means unlinking the link first.
//!
//! # Example
//!
//! ```
//! use tenferro_hdf5::{hdf5, read_info, read_typed, write_typed, ReadOptions};
//! use tenferro_tensor::{DynRank, Host, TypedTensor};
//!
//! let dir = tempfile::tempdir()?;
//! let file = hdf5::File::create(dir.path().join("tensors.h5"))?;
//! let group = file.create_group("tensors")?;
//!
//! let tensor: TypedTensor<f64, DynRank, Host> =
//!     TypedTensor::from_host_vec_col_major(vec![2, 3], (0..6).map(f64::from).collect())?;
//! write_typed(&group, "a", tensor.as_view())?;
//!
//! let info = read_info(&group, "a")?;
//! assert_eq!(info.shape, vec![2, 3]);
//!
//! let back: TypedTensor<f64, DynRank, Host> =
//!     tenferro_hdf5::read_typed_with(&group, "a", &ReadOptions::default().max_elements(1 << 20))?;
//! assert_eq!(back.as_slice(), tensor.as_slice());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
#![deny(missing_docs)]

mod error;
mod read;
mod schema;
mod write;

/// The HDF5 binding this adapter is built on.
///
/// Callers open files and groups through this re-export, so their handles are
/// always the ones the adapter expects.
pub use hdf5_rt as hdf5;
pub use hdf5_rt::H5Type;

pub use error::{Error, SchemaViolation};
pub use read::{
    read_info, read_tensor, read_tensor_with, read_typed, read_typed_with, ReadOptions, TensorInfo,
};
pub use write::{write_tensor, write_typed};

/// Value of the `type` attribute that marks a tenferro tensor dataset.
pub const SCHEMA_TYPE: &str = "tenferro.Tensor";

/// Schema version this adapter writes and reads.
pub const SCHEMA_VERSION: i64 = 1;
