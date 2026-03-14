//! C-API (FFI) for tenferro.
//!
//! Exposes tensor lifecycle, einsum, SVD (including AD rules), and DLPack
//! interop to host languages such as Julia, Python (JAX, PyTorch), and C/C++.
//!
//! # Design principles
//!
//! - **Opaque pointers**: `TfeTensorF64` is an opaque handle wrapping
//!   `Tensor<f64>`. Host languages never see Rust internals.
//! - **Status codes**: Every function takes a `*mut tfe_status_t` as its
//!   last argument. Rust panics are caught with `catch_unwind` and
//!   converted to `TFE_INTERNAL_ERROR`.
//! - **Stateless AD rules**: Only `rrule` (VJP) and `frule` (JVP) are
//!   exposed. The AD tape / `TrackedValue` / `DualValue` are Rust-internal
//!   and **not** exposed via FFI.
//! - **f64 only** in this POC phase. All functions carry the `_f64` suffix.
//! - **DLPack interop**: Zero-copy tensor exchange via [`DLManagedTensorVersioned`].
//! - **Copy semantics** for convenience functions: `tfe_tensor_f64_from_data`
//!   copies the caller's data into a Rust-owned buffer.
//!
//! # Header generation
//!
//! ```text
//! cbindgen --config cbindgen.toml --crate tenferro-capi --output tenferro.h
//! ```
//!
//! # Examples
//!
//! ```c
//! tfe_status_t status;
//! size_t shape[] = {3, 4};
//! double data[12] = { /* ... */ };
//! tfe_tensor_f64 *a = tfe_tensor_f64_from_data(data, 12, shape, 2, &status);
//! const tfe_tensor_f64 *ops[] = {a, a};
//! tfe_tensor_f64 *c = tfe_einsum_f64("ij,jk->ik", ops, 2, &status);
//! tfe_tensor_f64_release(c);
//! tfe_tensor_f64_release(a);
//! ```

#![allow(clippy::missing_safety_doc)]
#![allow(non_camel_case_types)]

mod dlpack;
mod einsum_api;
mod ffi_utils;
mod handle;
mod status;
mod svd_api;
mod tensor_api;

pub use dlpack::{
    tfe_tensor_f64_from_dlpack, tfe_tensor_f64_to_dlpack, DLDataType, DLDevice,
    DLManagedTensorVersioned, DLPackVersion, DLTensor, DLPACK_FLAG_BITMASK_IS_COPIED,
    DLPACK_FLAG_BITMASK_READ_ONLY, KDLCOMPLEX, KDLCPU, KDLCUDA, KDLCUDA_HOST, KDLCUDA_MANAGED,
    KDLFLOAT, KDLINT, KDLROCM, KDLROCM_HOST,
};
pub use einsum_api::{tfe_einsum_f64, tfe_einsum_frule_f64, tfe_einsum_rrule_f64};
pub use handle::TfeTensorF64;
pub use status::{
    tfe_last_error_message, tfe_status_t, TFE_BUFFER_TOO_SMALL, TFE_INTERNAL_ERROR,
    TFE_INVALID_ARGUMENT, TFE_SHAPE_MISMATCH, TFE_SUCCESS,
};
pub use svd_api::{tfe_svd_f64, tfe_svd_frule_f64, tfe_svd_rrule_f64};
pub use tensor_api::{
    tfe_tensor_f64_clone, tfe_tensor_f64_data, tfe_tensor_f64_from_data, tfe_tensor_f64_len,
    tfe_tensor_f64_ndim, tfe_tensor_f64_release, tfe_tensor_f64_shape, tfe_tensor_f64_zeros,
};

#[cfg(test)]
pub(crate) use handle::{handle_to_ref, tensor_to_handle};

#[cfg(test)]
mod tests;
