//! Dense tensor type with CPU/GPU support.
//!
//! This crate provides [`Tensor<T>`], a multi-dimensional array type composed of
//! shape, strides, and a device-aware [`DataBuffer`]. It supports:
//!
//! - **Zero-copy view operations**: [`Tensor::permute`], [`Tensor::broadcast`],
//!   [`Tensor::diagonal`], [`Tensor::select`], [`Tensor::narrow`] modify only
//!   metadata (dims/strides)
//! - **Data operations**: [`Tensor::contiguous`] / [`Tensor::into_contiguous`] copy
//!   data into a contiguous layout (the consuming variant avoids allocation when
//!   the tensor is already contiguous); [`Tensor::tril`] / [`Tensor::triu`] extract
//!   triangular parts
//! - **Factory functions**: [`Tensor::zeros`], [`Tensor::ones`], [`Tensor::eye`]
//! - **DLPack interop**: [`DataBuffer`] supports both Rust-owned (`Vec<T>`) and
//!   externally-owned memory (e.g., imported via DLPack) with automatic cleanup.
//!
//! # Memory layout
//!
//! [`Tensor`] stores explicit strides and is not tied to any particular memory
//! order. [`MemoryOrder`] is only used as a parameter when allocating new memory
//! (e.g., [`Tensor::zeros`], [`Tensor::contiguous`]).
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
//! );
//! let b = Tensor::<f64>::ones(
//!     &[3, 4],
//!     LogicalMemorySpace::MainMemory,
//!     MemoryOrder::RowMajor,
//! );
//!
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let m = Tensor::<f64>::from_slice(&data, &[2, 3], MemoryOrder::ColumnMajor).unwrap();
//! ```
//!
//! ## Transpose and reshape
//!
//! ```ignore
//! let mt = m.permute(&[1, 0]).unwrap();
//! assert_eq!(mt.dims(), &[3, 2]);
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
//! );
//! let expanded = col.broadcast(&[3, 4]).unwrap();
//! let owned = expanded.contiguous(MemoryOrder::ColumnMajor);
//! assert_eq!(owned.dims(), &[3, 4]);
//! ```

#[cfg(feature = "cuda")]
mod cuda_runtime;

mod buffer;
mod completion_event;
mod layout;
mod tensor;

pub use buffer::DataBuffer;
pub use completion_event::CompletionEvent;
pub use layout::MemoryOrder;
pub use tensor::Tensor;

#[cfg(test)]
mod tests;
