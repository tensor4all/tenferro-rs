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
/// use cubecl::client::ComputeClient;
/// use cubecl::prelude::{ArrayArg, CubeCount, CubeDim};
/// use cubecl_cuda::CudaRuntime as CubeclCudaRuntime;
/// use cubecl_kernel_sample::kernel::scale;
///
/// fn launch(
///     client: &ComputeClient<CubeclCudaRuntime>,
///     input: ArrayArg<CubeclCudaRuntime>,
///     output: ArrayArg<CubeclCudaRuntime>,
/// ) {
///     // SAFETY: the kernel reads `input[i]` and writes `output[i]` only for
///     // `i < n` with the domain sized to the argument count; the bindings
///     // are validated resident spans on the caller's runtime.
///     unsafe {
///         scale::launch_unchecked::<CubeclCudaRuntime>(
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
pub fn scale(input: &Array<f32>, output: &mut Array<f32>, factor: f32) {
    let i = ABSOLUTE_POS;
    output[i] = input[i] * factor;
}
