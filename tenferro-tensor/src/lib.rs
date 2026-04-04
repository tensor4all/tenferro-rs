//! Dense tensor type with CPU/GPU support.
//!
//! This crate provides [`Tensor<T>`], a multi-dimensional array type composed of
//! shape, strides, and a device-aware [`DataBuffer`]. It supports:
//!
//! - **Zero-copy view operations**: [`Tensor::view`], [`Tensor::permute`],
//!   [`Tensor::broadcast`], [`Tensor::diagonal`], [`Tensor::select`],
//!   [`Tensor::narrow`] modify only metadata (dims/strides)
//! - **Data operations**: [`Tensor::contiguous`] / [`Tensor::into_contiguous`] copy
//!   data into a contiguous layout (the consuming variant avoids allocation when
//!   the tensor is already contiguous); [`Tensor::reshape`] returns a zero-copy
//!   view when possible and otherwise materializes first; [`Tensor::tril`] /
//!   [`Tensor::triu`] extract triangular parts
//! - **Factory functions**: [`Tensor::zeros`], [`Tensor::ones`], [`Tensor::eye`]
//! - **DLPack interop**: [`DataBuffer`] supports both Rust-owned (`Vec<T>`) and
//!   externally-owned memory (e.g., imported via DLPack) with automatic cleanup.
//!
//! # Memory layout
//!
//! tenferro uses **column-major** layout as its internal canonical order.
//! [`MemoryOrder`] is still accepted at import/materialization boundaries
//! (e.g., [`Tensor::from_slice`], [`Tensor::contiguous`]), but shape/view
//! semantics such as [`Tensor::view`] and [`Tensor::reshape`] follow the
//! column-major internal contract.
//!
//! This matches Julia, Fortran, Eigen3's default layout, and BLAS/LAPACK-style
//! linear algebra backends. Row-major ecosystems should normalize at the
//! boundary rather than expecting the tensor core to preserve row-major
//! semantics through view operations.
//!
//! # No strided-rs dependency
//!
//! This crate does **not** depend on `strided-rs`. The strided-rs types
//! (`StridedView`, `StridedViewMut`) are backend implementation details
//! used only in `tenferro-prims`. To pass tensor data to prims backends,
//! use [`DataBuffer::as_slice`] combined with [`Tensor::dims`],
//! [`Tensor::strides`], and [`Tensor::offset`].
//!
//! # Examples
//!
//! ## Creating tensors
//!
//! ```ignore
//! use tenferro_tensor::{MemoryOrder, Tensor};
//! use tenferro_device::LogicalMemorySpace;
//!
//! let a = Tensor::<f64>::zeros(
//!     &[3, 4],
//!     LogicalMemorySpace::MainMemory,
//!     MemoryOrder::ColumnMajor,
//! ).unwrap();
//! let b = Tensor::<f64>::ones(
//!     &[3, 4],
//!     LogicalMemorySpace::MainMemory,
//!     MemoryOrder::RowMajor,
//! ).unwrap();
//!
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let m = Tensor::<f64>::from_slice(&data, &[2, 3], MemoryOrder::ColumnMajor).unwrap();
//! ```
//!
//! ## Transpose, view, and reshape
//!
//! ```ignore
//! let mt = m.permute(&[1, 0]).unwrap();
//! assert_eq!(mt.dims(), &[3, 2]);
//!
//! let flat_view = m.view(&[6]).unwrap();
//! assert_eq!(flat_view.dims(), &[6]);
//!
//! let flat = m.reshape(&[6]).unwrap();
//! assert_eq!(flat.dims(), &[6]);
//! ```
//!
//! ## Broadcasting and materialization
//!
//! ```ignore
//! let col = Tensor::<f64>::ones(
//!     &[3, 1],
//!     LogicalMemorySpace::MainMemory,
//!     MemoryOrder::ColumnMajor,
//! ).unwrap();
//! let expanded = col.broadcast(&[3, 4]).unwrap();
//! let owned = expanded.contiguous(MemoryOrder::ColumnMajor);
//! assert_eq!(owned.dims(), &[3, 4]);
//! ```

pub mod v2;

#[cfg(feature = "cuda")]
mod cuda_runtime;

mod buffer;
mod completion_event;
mod layout;
pub mod structured_tensor;
mod tensor;

pub use buffer::DataBuffer;
pub use completion_event::CompletionEvent;
pub use layout::MemoryOrder;
pub use structured_tensor::StructuredTensor;
pub use tensor::{KeepCountScalar, Tensor};

#[cfg(test)]
mod tests;
