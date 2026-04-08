# JAX Primitives

**Date:** 2026-04-03
**Status:** Draft
**Parent:** `../README.md`
**Related:** `../spec/primitive-catalog.md`, `stablehlo-primitives.md`, `../spec/backend-contract.md`

---

## I. Purpose

This document records the JAX primitives that are most relevant to tensor
operations and AD design, and gives a short definition of what each primitive
actually does.

The source of truth here is the local checkout at:

- `~/tensor4all/jax`

This is intentionally **not** a catalog of every frontend helper in JAX.
It focuses on the primitive layer that sits below `jax.numpy` and `jax.lax`.

Descriptions below are intentionally short. Exact abstract-eval rules,
dimension-number attributes, and AD registrations should be checked in the
source when needed.

---

## II. What Counts As A Primitive In JAX

At the source level, the main markers are:

- `core.Primitive(...)`
- `standard_primitive(...)`
- `standard_unop(...)`
- `standard_naryop(...)`

And the AD-relevant registrations are attached separately via:

- `ad.deflinear2(...)`
- `ad.defbilinear(...)`
- `ad.primitive_jvps[...] = ...`
- `ad.primitive_transposes[...] = ...`

So a JAX primitive is not just a name. It is a bundle of:

- abstract evaluation / shape rules
- batching rules
- MLIR / StableHLO lowering
- JVP / transpose / linearization behavior

This registry-based organization is specific to JAX's implementation.
In the v2 Rust stack, the corresponding AD behavior is expected to live on each
concrete primitive through `PrimitiveOp::linearize` and
`PrimitiveOp::transpose_rule`, not through a mandatory global registry.

---

## III. Source Files Surveyed

The main source files used for this note are:

- `jax/_src/ad_util.py`
- `jax/_src/lax/lax.py`
- `jax/_src/lax/convolution.py`
- `jax/_src/lax/fft.py`
- `jax/_src/lax/slicing.py`
- `jax/_src/lax/windowed_reductions.py`
- `jax/_src/lax/control_flow/conditionals.py`
- `jax/_src/lax/control_flow/loops.py`
- `jax/_src/lax/control_flow/solves.py`
- `jax/_src/lax/linalg.py`

---

## IV. AD Helper Primitives

From `jax/_src/ad_util.py`:

- `add_any` (`add_jaxvals_p`): adds two tangent/cotangent values in AD space.
  This is a generic accumulation helper, not a tensor kernel.
- `stop_gradient` (`stop_gradient_p`): returns the input value unchanged but
  blocks JVP / VJP propagation through it.
- `zeros_like` (`zeros_like_p`): creates the zero tangent/cotangent
  corresponding to an aval.

---

## V. Core Tensor Primitives In `jax/_src/lax/lax.py`

### Arithmetic and comparison

- `neg_p`: elementwise unary negation.
- `sign_p`: elementwise sign extraction.
- `abs_p`: elementwise absolute value / magnitude.
- `add_p`: elementwise addition.
- `sub_p`: elementwise subtraction.
- `mul_p`: elementwise multiplication.
- `div_p`: elementwise division.
- `rem_p`: elementwise remainder / modulus.
- `max_p`: elementwise maximum.
- `min_p`: elementwise minimum.
- `clamp_p`: elementwise clamp between lower and upper bounds.
- `eq_p`: elementwise equality comparison.
- `ne_p`: elementwise inequality comparison.
- `ge_p`: elementwise greater-or-equal comparison.
- `gt_p`: elementwise greater-than comparison.
- `le_p`: elementwise less-or-equal comparison.
- `lt_p`: elementwise less-than comparison.
- `and_p`: elementwise logical / bitwise AND.
- `or_p`: elementwise logical / bitwise OR.
- `xor_p`: elementwise logical / bitwise XOR.
- `not_p`: elementwise logical / bitwise NOT.
- `shift_left_p`: elementwise bit shift to the left.
- `shift_right_arithmetic_p`: elementwise sign-preserving right shift.
- `shift_right_logical_p`: elementwise zero-filling right shift.
- `population_count_p`: counts set bits in each integer element.
- `clz_p`: counts leading zero bits in each integer element.

### Analytic and complex-valued math

- `exp_p`: elementwise exponential.
- `exp2_p`: elementwise base-2 exponential.
- `log_p`: elementwise natural logarithm.
- `expm1_p`: elementwise `exp(x) - 1`.
- `log1p_p`: elementwise `log(1 + x)`.
- `tanh_p`: elementwise hyperbolic tangent.
- `logistic_p`: elementwise sigmoid / logistic function.
- `sin_p`: elementwise sine.
- `cos_p`: elementwise cosine.
- `tan_p`: elementwise tangent.
- `asin_p`: elementwise arcsine.
- `acos_p`: elementwise arccosine.
- `atan_p`: elementwise arctangent.
- `atan2_p`: elementwise quadrant-aware arctangent of two inputs.
- `sinh_p`: elementwise hyperbolic sine.
- `cosh_p`: elementwise hyperbolic cosine.
- `asinh_p`: elementwise inverse hyperbolic sine.
- `acosh_p`: elementwise inverse hyperbolic cosine.
- `atanh_p`: elementwise inverse hyperbolic tangent.
- `sqrt_p`: elementwise square root.
- `rsqrt_p`: elementwise reciprocal square root.
- `cbrt_p`: elementwise cube root.
- `square_p`: elementwise square.
- `pow_p`: elementwise exponentiation with tensor exponent.
- `integer_pow_p`: exponentiation by a fixed integer exponent.
- `real_p`: extracts the real part of a complex tensor.
- `imag_p`: extracts the imaginary part of a complex tensor.
- `complex_p`: combines real and imaginary tensors into a complex tensor.
- `conj_p`: elementwise complex conjugation.
- `is_finite_p`: elementwise finiteness test.

### Tensor structure, contraction, and conversion

- `dot_general_p`: general tensor contraction with explicit batch and
  contracting dimensions.
- `ragged_dot_general_p`: ragged / grouped variant of generalized contraction.
- `broadcast_in_dim_p`: explicit broadcast that maps input axes into chosen
  output axes.
- `reshape_p`: reshapes a tensor without changing element count.
- `transpose_p`: permutes axes.
- `rev_p`: reverses element order along selected axes.
- `concatenate_p`: concatenates tensors along one axis.
- `pad_p`: pads a tensor with a specified padding value and edge/interior
  padding configuration.
- `squeeze_p`: removes size-1 axes.
- `tile_p`: repeats a tensor along one or more axes.
- `iota_p`: creates an index ramp along a chosen axis.
- `select_n_p`: N-way elementwise selection driven by a boolean or integer
  selector.
- `convert_element_type_p`: converts element dtype with value conversion.
- `bitcast_convert_type_p`: reinterprets bits as another dtype without numeric
  conversion.
- `to_edtype_p`: converts a tensor into an extended-dtype representation.
- `from_edtype_p`: converts an extended-dtype representation back to a regular
  tensor dtype.

### Reductions and ordering

- `reduce_p`: generic reduction over one or more axes using a supplied reducer
  computation.
- `reduce_sum_p`: sum reduction over explicit axes.
- `reduce_prod_p`: product reduction over explicit axes.
- `reduce_max_p`: maximum reduction over explicit axes.
- `reduce_min_p`: minimum reduction over explicit axes.
- `reduce_or_p`: boolean / bitwise OR reduction.
- `reduce_and_p`: boolean / bitwise AND reduction.
- `reduce_xor_p`: boolean / bitwise XOR reduction.
- `reduce_precision_p`: simulates reduced mantissa / exponent precision.
- `sort_p`: sorts values, or tuples of values, along a chosen axis.
- `top_k_p`: returns the top-`k` values and their indices.

### Miscellaneous tensor/runtime helpers in the same file

- `composite_p`: wraps a named composite operation that is defined via a higher
  level decomposition.
- `create_token_p`: creates an effect token.
- `after_all_p`: joins multiple tokens to enforce effect ordering.
- `rng_uniform_p`: generates random values from a uniform distribution.
- `rng_bit_generator_p`: advances an RNG state and returns raw random bits.
- `copy_p`: forces an explicit copy of a tensor value.
- `tie_p`: ties values together to preserve a dependency in lowering /
  scheduling.
- `optimization_barrier_p`: prevents certain compiler optimizations across the
  barrier.

---

## VI. Slicing, Gather, Scatter, And Windowed Reduction Primitives

From `jax/_src/lax/slicing.py`:

- `slice_p`: static slice with compile-time start / limit / stride.
- `dynamic_slice_p`: slice with runtime start indices.
- `dynamic_update_slice_p`: writes an update tensor into an operand at runtime
  start indices.
- `gather_p`: indexed read that gathers slices according to gather dimension
  numbers.
- `scatter_p`: indexed write/update according to scatter dimension numbers and
  an update combiner.
- `scatter_add_p`: indexed additive accumulation.
- `scatter_sub_p`: indexed subtractive accumulation.
- `scatter_mul_p`: indexed multiplicative accumulation.
- `scatter_min_p`: indexed minimum accumulation.
- `scatter_max_p`: indexed maximum accumulation.

From `jax/_src/lax/windowed_reductions.py`:

- `reduce_window_p`: generic sliding-window reduction.
- `reduce_window_sum_p`: sliding-window sum.
- `reduce_window_max_p`: sliding-window maximum.
- `reduce_window_min_p`: sliding-window minimum.
- `select_and_scatter_p`: windowed select followed by scatter of updates.
- `select_and_scatter_add_p`: specialization of select-and-scatter with
  additive accumulation.
- `select_and_gather_add_p`: AD helper used by pooling-style transpose rules;
  gathers selected window values and accumulates them additively.

---

## VII. Control Flow Primitives

From `jax/_src/lax/control_flow/conditionals.py`,
`jax/_src/lax/control_flow/loops.py`, and
`jax/_src/lax/control_flow/solves.py`:

- `cond_p`: branch between alternative jaxprs using a predicate or index.
- `scan_p`: loop primitive with carried state and stacked outputs.
- `while_p`: while-loop primitive with separate condition and body jaxprs.
- `custom_linear_solve` (`linear_solve_p`): linear solve primitive with custom
  forward and transpose semantics.
- `cumsum_p`: cumulative sum along an axis.
- `cumlogsumexp_p`: cumulative `logsumexp` along an axis.
- `cumprod_p`: cumulative product along an axis.
- `cummax_p`: cumulative maximum along an axis.
- `cummin_p`: cumulative minimum along an axis.

Also present as internal support:

- `eval_jaxpr_p`: executes an embedded jaxpr as a primitive call.
- `platform_index_p`: exposes the current platform/device index to the jaxpr.

---

## VIII. Additional Tensor Primitives Outside `lax.py`

From `jax/_src/lax/convolution.py`:

- `conv_general_dilated_p`: general N-D convolution with explicit stride,
  padding, dilation, feature-group, and dimension-number configuration.

From `jax/_src/lax/fft.py`:

- `fft_p`: Fast Fourier Transform primitive with explicit FFT type and lengths.

---

## IX. Structured Linalg Primitives

From `jax/_src/lax/linalg.py`:

- `cholesky_p`: Cholesky factorization of a Hermitian positive-definite matrix.
- `cholesky_update_p`: rank-1 update / downdate of an existing Cholesky factor.
- `eig_p`: general eigenvalue decomposition.
- `eigh_p`: eigenvalue decomposition specialized for Hermitian / symmetric
  matrices.
- `hessenberg_p`: Hessenberg reduction of a square matrix.
- `householder_product_p`: reconstructs an orthogonal / unitary product from
  Householder reflectors.
- `ormqr_p`: multiplies by the `Q` factor represented by Householder data.
- `lu_p`: LU factorization with pivot information.
- `lu_pivots_to_permutation_p`: converts LU pivot indices into an explicit
  permutation.
- `geqrf_p`: QR factorization in LAPACK-style reflector form.
- `geqp3_p`: pivoted QR factorization.
- `qr_p`: QR factorization returning explicit `Q` and `R`.
- `schur_p`: Schur decomposition.
- `svd_p`: singular value decomposition.
- `symmetric_product_p`: symmetric/Hermitian matrix product helper.
- `triangular_solve_p`: solve a linear system with a triangular matrix.
- `tridiagonal_p`: decomposition / helper around tridiagonal structure.
- `tridiagonal_solve_p`: solve a tridiagonal linear system.
