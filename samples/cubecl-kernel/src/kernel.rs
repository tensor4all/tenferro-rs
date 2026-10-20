//! Real `#[cube(launch_unchecked)]` kernels defined outside `tenferro-gpu`.
//!
//! Every kernel here is also the compile-checked source of a snippet in
//! `docs/guides/custom-cuda-kernels.md`, and the device test
//! `tests/scale_kernel_test.rs` launches all of them on CUDA hardware, so the
//! documented typing rules are the rules this pin actually enforces.

use cubecl::cube;
use cubecl::prelude::*;

/// Multiply every element by `factor`.
///
/// Bounds contract: the launcher sizes the domain to exactly `input.len()`
/// elements, so the kernel body is in-bounds for both tensors.
///
/// # Examples
///
/// The kernel is launched through the public CubeCL engine scope; see
/// [`run_scale_check`](crate::run::run_scale_check) for the full end-to-end
/// upload → launch → sync → download flow.
///
/// ```
/// use cubecl::client::Client;
/// use cubecl::prelude::{BufferArg, CubeCount, CubeDim};
/// use cubecl_kernel_sample::kernel::scale;
///
/// fn launch(client: &Client, input: BufferArg, output: BufferArg) {
///     // SAFETY: the kernel reads `input[i]` and writes `output[i]` only for
///     // `i < n` with the domain sized to the argument count; the bindings
///     // are validated resident spans on the caller's runtime.
///     unsafe {
///         scale::launch_unchecked(
///             client,
///             CubeCount::Static(1, 1, 1),
///             CubeDim::new_1d(1),
///             input,
///             output,
///             2.0f32,
///         );
///     }
/// }
/// ```
#[cube(launch_unchecked)]
pub fn scale(input: &[f32], output: &mut [f32], factor: f32) {
    let i = ABSOLUTE_POS;
    output[i] = input[i] * factor;
}

// snippet-start:downstream-cube-index
/// Gather `input[usize::cast_from(indices[i])]` for each `i`.
///
/// `indices` is an `i32` buffer, so its elements are not `usize`; the explicit
/// `usize::cast_from` is what makes them usable as indices, and it accepts a
/// signed source on this pin. `ABSOLUTE_POS` is already `usize`, and an index
/// literal is inferred as `usize`.
///
/// An `i32` buffer is also the only shape a downstream can bind here: tenferro
/// tensors carry `f32`, `f64`, `i32`, `i64`, `bool`, `C32` and `C64` payloads,
/// so there is no `u32`/`u64` tensor to hand `Session::array_arg`.
///
/// Bounds contract: the launcher sizes the domain to `indices.len()`, `output`
/// must be at least `indices.len()` long, and every stored index must be in
/// `0..input.len()`: the cast does not bounds-check, so a negative value becomes
/// a huge unsigned offset rather than an error.
#[cube(launch_unchecked)]
pub fn gather(input: &[f32], indices: &[i32], output: &mut [f32]) {
    let i = ABSOLUTE_POS;
    let src = usize::cast_from(indices[i]);
    output[i] = input[src];
}
// snippet-end:downstream-cube-index

// snippet-start:downstream-cube-shared
/// Reverse one cube-sized tile of `input` through shared memory.
///
/// `UNIT_POS` is a `u32` (as are `CUBE_POS`, `CUBE_DIM` and the other topology
/// constants), so it cannot index a buffer directly: `usize::cast_from` is the
/// conversion. `ABSOLUTE_POS` is already `usize`, and an integer literal is
/// inferred as `usize` where an index is expected, so `count - 1 - unit` needs no
/// suffix and no cast.
///
/// Bounds contract: the launcher runs exactly one cube of `len` units with
/// `len <= 1024` (the shared-memory slice length), `input` holds at least `len`
/// elements, and `output` holds at least `len`.
#[cube(launch_unchecked)]
pub fn reverse_tile(input: &[f32], output: &mut [f32], len: u32) {
    let mut tile = Shared::<[f32]>::new_slice(1024usize);
    let unit = usize::cast_from(UNIT_POS);
    let count = usize::cast_from(len);
    if unit < count {
        tile[unit] = input[unit];
    }
    sync_cube();
    if unit < count {
        output[unit] = tile[count - 1 - unit];
    }
}
// snippet-end:downstream-cube-shared
