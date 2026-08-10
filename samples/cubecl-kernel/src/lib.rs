//! Source-blind downstream-style fixture for the CubeCL external-kernel seam
//! (issue #1597).
//!
//! This crate deliberately looks like a downstream kernel provider: it depends
//! on the published crates.io `t4a-cubecl = =0.10.0` package line and on
//! `tenferro-gpu`, then compiles and runs a real `#[cube(launch_unchecked)]`
//! kernel exclusively through the public `cuda::cubecl::Session` API. It must
//! never touch the hidden `cuda::interop` bridge.
//!
//! The fixture proves two properties:
//!
//! 1. A single `t4a-cubecl` runtime type across the dependency graph
//!    (`cargo tree -d`, enforced by the pre-PR gate), so the kernel's generic
//!    `ComputeClient<cuda::CudaRuntime>` unifies with tenferro's.
//! 2. The public seam supports a full upload → launch → sync → download →
//!    assert flow with no raw context handling, no `u64` stream, and no
//!    `set_current_cuda_context` in caller code.

pub mod kernel;
pub mod run;

pub use run::run_scale_check;
