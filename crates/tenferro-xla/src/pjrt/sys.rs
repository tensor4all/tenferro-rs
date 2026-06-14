//! Minimal PJRT C API declarations used by the dynamic plugin loader.

/// Opaque PJRT C API table.
#[repr(C)]
#[allow(non_camel_case_types)]
pub(crate) struct PJRT_Api {
    _private: [u8; 0],
}

/// Plugin entry point exported by OpenXLA PJRT plugins.
#[allow(non_camel_case_types)]
pub(crate) type GetPjrtApiFn = unsafe extern "C" fn() -> *const PJRT_Api;
