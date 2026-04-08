# StableHLO Primitives

**Date:** 2026-04-03
**Status:** Draft
**Parent:** `../README.md`
**Related:** `../spec/primitive-catalog.md`, `../spec/backend-contract.md`, `jax-primitives.md`

---

## I. Purpose

This document records the current StableHLO operation inventory and gives a
short definition of what each op does.

StableHLO itself uses the word "op", not "primitive". This document uses
"primitive" only to match the naming used elsewhere in `docs/design-v2/`.

The main source is the official specification:

- `https://openxla.org/stablehlo/spec`

Descriptions below are intentionally short. Exact verifier rules, region
signatures, dimension-number attributes, and shape constraints should be
checked in the spec when needed.

---

## II. Current StableHLO Op Inventory

The current spec has **106** op sections after excluding non-op section
headings.

### Elementwise arithmetic, comparison, and math

- `abs`: elementwise absolute value.
- `add`: elementwise addition.
- `and`: elementwise logical / bitwise AND.
- `atan2`: elementwise quadrant-aware arctangent of two inputs.
- `bitcast_convert`: reinterpret bits as another element type without numeric
  conversion.
- `cbrt`: elementwise cube root.
- `ceil`: elementwise round toward positive infinity.
- `clamp`: elementwise clamp between lower and upper bounds.
- `compare`: elementwise comparison with an explicit direction.
- `complex`: combine real and imaginary tensors into a complex tensor.
- `convert`: elementwise numeric type conversion.
- `cosine`: elementwise cosine.
- `count_leading_zeros`: count leading zero bits in each integer element.
- `divide`: elementwise division.
- `exponential`: elementwise `exp(x)`.
- `exponential_minus_one`: elementwise `exp(x) - 1`.
- `floor`: elementwise round toward negative infinity.
- `imag`: extract the imaginary part of a complex tensor.
- `is_finite`: test whether each element is finite.
- `log`: elementwise natural logarithm.
- `log_plus_one`: elementwise `log(1 + x)`.
- `logistic`: elementwise sigmoid.
- `maximum`: elementwise maximum.
- `minimum`: elementwise minimum.
- `multiply`: elementwise multiplication.
- `negate`: elementwise unary negation.
- `not`: elementwise logical / bitwise NOT.
- `or`: elementwise logical / bitwise OR.
- `popcnt`: count set bits in each integer element.
- `power`: elementwise exponentiation.
- `real`: extract the real part of a complex tensor.
- `remainder`: elementwise remainder / modulus.
- `round_nearest_afz`: elementwise round to nearest, with ties away from zero.
- `round_nearest_even`: elementwise round to nearest, with ties to even.
- `rsqrt`: elementwise reciprocal square root.
- `shift_left`: elementwise left bit shift.
- `shift_right_arithmetic`: elementwise sign-preserving right shift.
- `shift_right_logical`: elementwise zero-filling right shift.
- `sign`: elementwise sign extraction.
- `sine`: elementwise sine.
- `sqrt`: elementwise square root.
- `subtract`: elementwise subtraction.
- `tan`: elementwise tangent.
- `tanh`: elementwise hyperbolic tangent.
- `xor`: elementwise logical / bitwise XOR.

### Shape, layout, and tensor construction

- `broadcast_in_dim`: explicit broadcast by mapping input axes into result
  axes.
- `concatenate`: concatenate tensors along one dimension.
- `constant`: embed a literal constant tensor.
- `dynamic_broadcast_in_dim`: broadcast to a runtime-specified output shape.
- `dynamic_iota`: create an index ramp with runtime shape information.
- `dynamic_pad`: pad a tensor using runtime padding sizes.
- `dynamic_reshape`: reshape using a runtime-specified result shape.
- `get_dimension_size`: query the runtime size of one dimension.
- `iota`: create an index ramp along one dimension.
- `optimization_barrier`: block certain compiler rewrites across the barrier.
- `pad`: pad a tensor with a constant padding value.
- `reshape`: change tensor shape without changing element count.
- `reverse`: reverse element order along selected dimensions.
- `transpose`: permute dimensions.

### Slicing, gather/scatter, and indexed updates

- `dynamic_gather`: gather slices using runtime slice-size information.
- `dynamic_slice`: slice using runtime start indices.
- `dynamic_update_slice`: write an update tensor into an operand at runtime
  start indices.
- `gather`: indexed read according to gather dimension numbers.
- `scatter`: indexed write / update according to scatter dimension numbers and
  an update combiner region.
- `select`: choose elementwise between two tensors using a predicate tensor.
- `slice`: static slice with compile-time start / limit / stride.

### Reductions, windows, and ordering

- `reduce`: generic reduction over dimensions using a reducer region.
- `reduce_precision`: simulate reduced floating-point precision.
- `reduce_scatter`: collective reduction followed by sharding / scattering of
  the reduced result.
- `reduce_window`: sliding-window reduction.
- `select_and_scatter`: sliding-window select followed by scattering of source
  values.
- `sort`: sort values, or tuples of values, along one dimension using a
  comparator region.

### Contraction, linalg, and NN-oriented ops

- `batch_norm_grad`: batch-normalization gradient helper.
- `batch_norm_inference`: batch normalization in inference mode.
- `batch_norm_training`: batch normalization in training mode, typically
  returning normalized output plus saved statistics.
- `cholesky`: Cholesky factorization.
- `convolution`: N-D convolution with explicit window, padding, and dimension
  metadata.
- `custom_call`: backend-specific opaque call.
- `dot_general`: general tensor contraction with explicit batch and contracting
  dimensions.
- `dynamic_conv`: convolution variant with runtime-dependent configuration.
- `fft`: Fast Fourier Transform.
- `triangular_solve`: solve a linear system with a triangular matrix.

### Control flow and higher-order ops

- `case`: multi-branch dispatch by integer branch index.
- `if`: two-way branch by boolean predicate.
- `map`: apply a region pointwise / elementwise across one or more tensors.
- `while`: loop with separate condition and body regions.

### Collectives, communication, tokens, and side-effect ordering

- `after_all`: join multiple tokens to enforce side-effect ordering.
- `all_gather`: gather values across replicas / partitions.
- `all_reduce`: reduce values across replicas / partitions.
- `all_to_all`: exchange partitioned slices across replicas / partitions.
- `async_done`: complete / await an asynchronous operation started earlier.
- `async_start`: start an asynchronous operation and return its handle/state.
- `collective_broadcast`: broadcast values collectively across a replica group.
- `collective_permute`: point-to-point collective routing by source/target
  pairs.
- `infeed`: receive data from an external host / device queue.
- `outfeed`: send data to an external host / device queue.
- `partition_id`: return the current partition ID.
- `recv`: point-to-point receive.
- `replica_id`: return the current replica ID.
- `send`: point-to-point send.

### Tuple, RNG, and compatibility-oriented ops

- `get_tuple_element`: extract one field from a tuple value.
- `rng`: legacy random-number generation op.
- `rng_bit_generator`: explicit RNG state transition plus raw random-bit
  generation.
- `tuple`: construct a tuple value.

### Quantization

- `uniform_dequantize`: convert uniformly quantized integers into dequantized
  real values.
- `uniform_quantize`: convert real values into uniformly quantized integers.

---

## III. Deprecated And Legacy Names In The Spec

The spec explicitly calls out these deprecated operations:

- `broadcast`: deprecated broadcast form; use `broadcast_in_dim`.
- `create_token`: deprecated token-creation op.
- `cross-replica-sum`: deprecated collective sum spelling.
- `dot`: deprecated simpler contraction op; `dot_general` is the canonical op.
- `einsum`: deprecated Einstein-summation spelling.
- `torch_index_select`: deprecated compatibility op.
- `unary_einsum`: deprecated unary Einstein-summation spelling.

The spec also says that removal is still being explored for some compatibility
or legacy ops, including:

- `complex`
- `get_tuple_element`
- `map`
- `rng`
- `torch_index_select`
- `tuple`

---

## IV. Alphabetical Appendix

```text
abs
add
after_all
all_gather
all_reduce
all_to_all
and
async_done
async_start
atan2
batch_norm_grad
batch_norm_inference
batch_norm_training
bitcast_convert
broadcast_in_dim
case
cbrt
ceil
cholesky
clamp
collective_broadcast
collective_permute
compare
complex
concatenate
constant
convert
convolution
cosine
count_leading_zeros
custom_call
divide
dot_general
dynamic_broadcast_in_dim
dynamic_conv
dynamic_gather
dynamic_iota
dynamic_pad
dynamic_reshape
dynamic_slice
dynamic_update_slice
exponential
exponential_minus_one
fft
floor
gather
get_dimension_size
get_tuple_element
if
imag
infeed
iota
is_finite
log
log_plus_one
logistic
map
maximum
minimum
multiply
negate
not
optimization_barrier
or
outfeed
pad
partition_id
popcnt
power
real
recv
reduce
reduce_precision
reduce_scatter
reduce_window
remainder
replica_id
reshape
reverse
rng
rng_bit_generator
round_nearest_afz
round_nearest_even
rsqrt
scatter
select
select_and_scatter
send
shift_left
shift_right_arithmetic
shift_right_logical
sign
sine
slice
sort
sqrt
subtract
tan
tanh
transpose
triangular_solve
tuple
uniform_dequantize
uniform_quantize
while
xor
```
