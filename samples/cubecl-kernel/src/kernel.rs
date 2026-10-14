//! A real `#[cube(launch_unchecked)]` kernel defined outside `tenferro-gpu`.

use cubecl::cube;
use cubecl::prelude::*;

/// Multiply every element by `factor`.
///
/// Bounds contract: the launcher sizes the domain to exactly `input.len()`
/// elements, so the kernel body is in-bounds for both tensors.
///
/// # Examples
///
/// The kernel is launched through the public `cubecl` session seam; see
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
/// `indices` is a `u32` buffer, so its elements are not `usize`; the explicit
/// `usize::cast_from` is what makes them usable as indices. `ABSOLUTE_POS` is
/// already `usize`, and an index literal is inferred as `usize`.
///
/// Bounds contract: the launcher sizes the domain to `indices.len()` and every
/// stored index must be in bounds for `input`.
#[cube(launch_unchecked)]
pub fn gather(input: &[f32], indices: &[u32], output: &mut [f32]) {
    let i = ABSOLUTE_POS;
    let src = usize::cast_from(indices[i]);
    output[i] = input[src];
}
// snippet-end:downstream-cube-index
