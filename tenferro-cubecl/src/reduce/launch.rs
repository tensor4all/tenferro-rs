use cubecl::prelude::*;

use crate::Result;

/// Launch strategy for a single-axis reduction.
///
/// # Examples
///
/// ```
/// use tenferro_cubecl::reduce::ReduceStrategy;
///
/// let strategy = ReduceStrategy::Auto;
/// assert_eq!(format!("{strategy:?}"), "Auto");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceStrategy {
    /// Let the launcher choose the kernel strategy.
    Auto,
    /// Use the single-work-unit placeholder strategy.
    Unit,
}

fn unimplemented_launch() -> Result<()> {
    todo!("implemented in Task 5")
}

/// Launch a floating-point sum reduction.
///
/// # Examples
///
/// ```ignore
/// # use cubecl::prelude::*;
/// # use tenferro_cubecl::reduce::{launch_sum_float, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_cubecl::Result<()> {
/// launch_sum_float::<R, f32>(client, input, output, 0, ReduceStrategy::Auto)
/// # }
/// ```
pub fn launch_sum_float<R: Runtime, F: Float + CubeElement>(
    _client: &ComputeClient<R>,
    _input: TensorBinding<R>,
    _output: TensorBinding<R>,
    _axis: usize,
    _strategy: ReduceStrategy,
) -> Result<()> {
    unimplemented_launch()
}

/// Launch an integer sum reduction.
///
/// # Examples
///
/// ```ignore
/// # use cubecl::prelude::*;
/// # use tenferro_cubecl::reduce::{launch_sum_int, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_cubecl::Result<()> {
/// launch_sum_int::<R, i64>(client, input, output, 0, ReduceStrategy::Auto)
/// # }
/// ```
pub fn launch_sum_int<R: Runtime, I: Int + CubeElement>(
    _client: &ComputeClient<R>,
    _input: TensorBinding<R>,
    _output: TensorBinding<R>,
    _axis: usize,
    _strategy: ReduceStrategy,
) -> Result<()> {
    unimplemented_launch()
}

/// Launch a complex sum reduction.
///
/// # Examples
///
/// ```ignore
/// # use cubecl::prelude::*;
/// # use num_complex::Complex64;
/// # use tenferro_cubecl::reduce::{launch_sum_complex, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_cubecl::Result<()> {
/// launch_sum_complex::<R, Complex64>(client, input, output, 0, ReduceStrategy::Auto)
/// # }
/// ```
pub fn launch_sum_complex<R: Runtime, C: Complex + CubeElement>(
    _client: &ComputeClient<R>,
    _input: TensorBinding<R>,
    _output: TensorBinding<R>,
    _axis: usize,
    _strategy: ReduceStrategy,
) -> Result<()> {
    unimplemented_launch()
}

/// Launch a floating-point product reduction.
///
/// # Examples
///
/// ```ignore
/// # use cubecl::prelude::*;
/// # use tenferro_cubecl::reduce::{launch_prod_float, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_cubecl::Result<()> {
/// launch_prod_float::<R, f64>(client, input, output, 0, ReduceStrategy::Auto)
/// # }
/// ```
pub fn launch_prod_float<R: Runtime, F: Float + CubeElement>(
    _client: &ComputeClient<R>,
    _input: TensorBinding<R>,
    _output: TensorBinding<R>,
    _axis: usize,
    _strategy: ReduceStrategy,
) -> Result<()> {
    unimplemented_launch()
}

/// Launch an integer product reduction.
///
/// # Examples
///
/// ```ignore
/// # use cubecl::prelude::*;
/// # use tenferro_cubecl::reduce::{launch_prod_int, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_cubecl::Result<()> {
/// launch_prod_int::<R, i64>(client, input, output, 0, ReduceStrategy::Auto)
/// # }
/// ```
pub fn launch_prod_int<R: Runtime, I: Int + CubeElement>(
    _client: &ComputeClient<R>,
    _input: TensorBinding<R>,
    _output: TensorBinding<R>,
    _axis: usize,
    _strategy: ReduceStrategy,
) -> Result<()> {
    unimplemented_launch()
}

/// Launch a complex product reduction.
///
/// # Examples
///
/// ```ignore
/// # use cubecl::prelude::*;
/// # use num_complex::Complex32;
/// # use tenferro_cubecl::reduce::{launch_prod_complex, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_cubecl::Result<()> {
/// launch_prod_complex::<R, Complex32>(client, input, output, 0, ReduceStrategy::Auto)
/// # }
/// ```
pub fn launch_prod_complex<R: Runtime, C: Complex + CubeElement>(
    _client: &ComputeClient<R>,
    _input: TensorBinding<R>,
    _output: TensorBinding<R>,
    _axis: usize,
    _strategy: ReduceStrategy,
) -> Result<()> {
    unimplemented_launch()
}

/// Launch a floating-point maximum reduction.
///
/// # Examples
///
/// ```ignore
/// # use cubecl::prelude::*;
/// # use tenferro_cubecl::reduce::{launch_max_float, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_cubecl::Result<()> {
/// launch_max_float::<R, f32>(client, input, output, 0, ReduceStrategy::Auto)
/// # }
/// ```
pub fn launch_max_float<R: Runtime, F: Float + CubeElement>(
    _client: &ComputeClient<R>,
    _input: TensorBinding<R>,
    _output: TensorBinding<R>,
    _axis: usize,
    _strategy: ReduceStrategy,
) -> Result<()> {
    unimplemented_launch()
}

/// Launch a floating-point minimum reduction.
///
/// # Examples
///
/// ```ignore
/// # use cubecl::prelude::*;
/// # use tenferro_cubecl::reduce::{launch_min_float, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_cubecl::Result<()> {
/// launch_min_float::<R, f64>(client, input, output, 0, ReduceStrategy::Auto)
/// # }
/// ```
pub fn launch_min_float<R: Runtime, F: Float + CubeElement>(
    _client: &ComputeClient<R>,
    _input: TensorBinding<R>,
    _output: TensorBinding<R>,
    _axis: usize,
    _strategy: ReduceStrategy,
) -> Result<()> {
    unimplemented_launch()
}
