//! A real `#[cube(launch_unchecked)]` kernel defined outside `tenferro-gpu`.

use cubecl::cube;
use cubecl::prelude::*;

/// Multiply every element by `factor`.
///
/// Bounds contract: the launcher sizes the domain to exactly `input.len()`
/// elements, so the kernel body is in-bounds for both tensors.
#[cube(launch_unchecked)]
pub fn scale(input: &Array<f32>, output: &mut Array<f32>, factor: f32) {
    let i = ABSOLUTE_POS;
    output[i] = input[i] * factor;
}
