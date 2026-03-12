//! C-API (FFI) for tropical semiring tensor operations.
//!
//! Extends [`tenferro_capi`] with tropical einsum functions for three
//! semirings: MaxPlus (⊕=max, ⊗=+), MinPlus (⊕=min, ⊗=+), and
//! MaxMul (⊕=max, ⊗=×).
//!
//! # Design
//!
//! - **Reuses [`tenferro_capi::TfeTensorF64`]** handles from `tenferro-capi`. Since
//!   `MaxPlus<f64>` is `#[repr(transparent)]`, it has the same memory
//!   layout as `f64`. Tropical functions accept f64 tensor handles,
//!   internally wrap data as `MaxPlus<f64>` (or `MinPlus`/`MaxMul`),
//!   perform the tropical einsum, and unwrap the result back to f64.
//! - **No new handle type**: the algebra is selected by the function
//!   name, not the tensor type.
//! - **Naming convention**: `tfe_tropical_einsum_<algebra>_f64`.
//!
//! # Linking
//!
//! This crate produces a separate shared library from `tenferro-capi`.
//! C/Julia/Python consumers load both:
//!
//! ```c
//! // Core tensor operations
//! tfe_tensor_f64 *t = tfe_tensor_f64_from_data(...);
//!
//! // Tropical einsum (from tenferro-tropical-capi)
//! const tfe_tensor_f64 *ops[] = {a, b};
//! tfe_tensor_f64 *c = tfe_tropical_einsum_maxplus_f64("ij,jk->ik", ops, 2, &status);
//! ```
//!
//! # Header generation
//!
//! Generate the extension header from a workspace checkout with:
//!
//! ```text
//! cbindgen \
//!   --config extension/tenferro-tropical-capi/cbindgen.toml \
//!   --crate tenferro-tropical-capi \
//!   --output tenferro_tropical.h
//! ```
//!
//! # Example (C pseudocode)
//!
//! ```c
//! #include "tenferro.h"
//! #include "tenferro_tropical.h"
//!
//! tfe_status_t status;
//! size_t shape[] = {3, 4};
//! double data_a[12] = { /* ... */ };
//! double data_b[12] = { /* ... */ };
//!
//! tfe_tensor_f64 *a = tfe_tensor_f64_from_data(data_a, 12, shape, 2, &status);
//! tfe_tensor_f64 *b = tfe_tensor_f64_from_data(data_b, 12, shape, 2, &status);
//!
//! // MaxPlus tropical einsum: C[i,k] = max_j (A[i,j] + B[j,k])
//! const tfe_tensor_f64 *ops[] = {a, b};
//! tfe_tensor_f64 *c = tfe_tropical_einsum_maxplus_f64("ij,jk->ik", ops, 2, &status);
//! assert(status == TFE_SUCCESS);
//!
//! tfe_tensor_f64_release(c);
//! tfe_tensor_f64_release(b);
//! tfe_tensor_f64_release(a);
//! ```

#![allow(clippy::missing_safety_doc)]
#![allow(non_camel_case_types)]

mod einsum_api;
mod ffi_utils;
mod handle;
mod status;

pub use einsum_api::{
    tfe_tropical_einsum_frule_maxmul_f64, tfe_tropical_einsum_frule_maxplus_f64,
    tfe_tropical_einsum_frule_minplus_f64, tfe_tropical_einsum_maxmul_f64,
    tfe_tropical_einsum_maxplus_f64, tfe_tropical_einsum_minplus_f64,
    tfe_tropical_einsum_rrule_maxmul_f64, tfe_tropical_einsum_rrule_maxplus_f64,
    tfe_tropical_einsum_rrule_minplus_f64,
};
#[cfg(test)]
pub(crate) use handle::{handle_to_ref, tensor_to_handle};

#[cfg(test)]
mod tests;
