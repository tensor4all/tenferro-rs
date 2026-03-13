# PyTorch Upstream Publish Coverage

Generated from the pinned PyTorch upstream inventory, the mapped DB family surface,
and the checked-in `cases/` tree.

## Upstream Inventory

- AD-relevant linalg upstream variants: 38
- AD-relevant scalar upstream variants: 138
- Mapped publishable success families: 174
- Explicit publishable error families: 2
- Total tracked DB families: 176

## Publishable Family Coverage

Published dtypes are read from the checked-in JSONL files. Missing dtypes indicate
publishable upstream coverage that is not yet materialized in this repository.

| op | family | behavior | supported dtypes | published dtypes | missing publishable dtypes |
| --- | --- | --- | --- | --- | --- |
| __radd__ | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| __rdiv__ | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| __rmod__ | identity | success | float64, float32 | float64, float32 | - |
| __rmul__ | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| __rpow__ | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| __rsub__ | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| abs | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| acos | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| acosh | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| add | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| amax | identity | success | float64, float32 | float64, float32 | - |
| amin | identity | success | float64, float32 | float64, float32 | - |
| angle | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| asin | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| asinh | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| atan | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| atan2 | identity | success | float64, float32 | float64, float32 | - |
| atanh | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| cdouble | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| ceil | identity | success | float64, float32 | float64, float32 | - |
| cholesky | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| cholesky_ex | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| clamp_max | identity | success | float64, float32 | float64, float32 | - |
| clamp_min | identity | success | float64, float32 | float64, float32 | - |
| complex | identity | success | float64, float32 | float64, float32 | - |
| cond | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| conj | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| conj_physical | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| copysign | identity | success | float64, float32 | float64, float32 | - |
| cos | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| cosh | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| cross | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| deg2rad | identity | success | float64, float32 | float64, float32 | - |
| det | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| diagonal | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| digamma | identity | success | float64, float32 | float64, float32 | - |
| div_no_rounding_mode | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| double | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| eig | values_vectors_abs | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| eigh | gauge_ill_defined | error | complex128 | complex128 | - |
| eigh | values_vectors_abs | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| eigvals | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| eigvalsh | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| erf | identity | success | float64, float32 | float64, float32 | - |
| erfc | identity | success | float64, float32 | float64, float32 | - |
| erfinv | identity | success | float64, float32 | float64, float32 | - |
| exp | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| exp2 | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| expm1 | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| fill | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| float_power | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| floor | identity | success | float64, float32 | float64, float32 | - |
| fmax | identity | success | float64, float32 | float64, float32 | - |
| fmin | identity | success | float64, float32 | float64, float32 | - |
| frac | identity | success | float64, float32 | float64, float32 | - |
| frexp | identity | success | float64, float32 | float64, float32 | - |
| householder_product | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| hypot | identity | success | float64, float32 | float64, float32 | - |
| i0 | identity | success | float64, float32 | float64, float32 | - |
| imag | identity | success | complex128, complex64 | complex128, complex64 | - |
| inv | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| inv_ex | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| ldexp | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| lgamma | identity | success | float64, float32 | float64, float32 | - |
| log | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| log10 | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| log1p | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| log2 | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| logaddexp | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| logit | identity | success | float64, float32 | float64, float32 | - |
| lstsq_grad_oriented | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| lu | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| lu_factor | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| lu_factor_ex | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| lu_solve | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| matrix_norm | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| matrix_power | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| max_binary | identity | success | float64, float32 | float64, float32 | - |
| maximum | identity | success | float64, float32 | float64, float32 | - |
| mean | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| min_binary | identity | success | float64, float32 | float64, float32 | - |
| minimum | identity | success | float64, float32 | float64, float32 | - |
| mul | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| multi_dot | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| mvlgamma_mvlgamma_p_1 | identity | success | float64, float32 | float64, float32 | - |
| mvlgamma_mvlgamma_p_3 | identity | success | float64, float32 | float64, float32 | - |
| mvlgamma_mvlgamma_p_5 | identity | success | float64, float32 | float64, float32 | - |
| nan_to_num | identity | success | float64, float32 | float64, float32 | - |
| nanmean | identity | success | float64, float32 | float64, float32 | - |
| nansum | identity | success | float64, float32 | float64, float32 | - |
| neg | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| nn_functional_celu | identity | success | float64, float32 | float64, float32 | - |
| nn_functional_elu | identity | success | float64, float32 | float64, float32 | - |
| nn_functional_hardshrink | identity | success | float64, float32 | float64, float32 | - |
| nn_functional_hardsigmoid | identity | success | float64, float32 | float64, float32 | - |
| nn_functional_hardtanh | identity | success | float64, float32 | float64, float32 | - |
| nn_functional_logsigmoid | identity | success | float64, float32 | float64, float32 | - |
| nn_functional_mish | identity | success | float64, float32 | float64, float32 | - |
| nn_functional_prelu | identity | success | float64, float32 | float64, float32 | - |
| nn_functional_relu | identity | success | float64, float32 | float64, float32 | - |
| nn_functional_relu6 | identity | success | float64, float32 | float64, float32 | - |
| nn_functional_rrelu | identity | success | float64, float32 | float64, float32 | - |
| nn_functional_selu | identity | success | float64, float32 | float64, float32 | - |
| nn_functional_silu | identity | success | float64, float32 | float64, float32 | - |
| nn_functional_softplus | identity | success | float64, float32 | float64, float32 | - |
| nn_functional_softshrink | identity | success | float64, float32 | float64, float32 | - |
| nn_functional_softsign | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| nn_functional_tanhshrink | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| nn_functional_threshold | identity | success | float64, float32 | float64, float32 | - |
| norm | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| pinv | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| pinv_hermitian | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| pinv_singular | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| polar | identity | success | float64, float32 | float64, float32 | - |
| polygamma_polygamma_n_0 | identity | success | float64, float32 | float64, float32 | - |
| polygamma_polygamma_n_1 | identity | success | float64, float32 | float64, float32 | - |
| polygamma_polygamma_n_2 | identity | success | float64, float32 | float64, float32 | - |
| polygamma_polygamma_n_3 | identity | success | float64, float32 | float64, float32 | - |
| polygamma_polygamma_n_4 | identity | success | float64, float32 | float64, float32 | - |
| positive | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| pow | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| prod | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| qr | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| rad2deg | identity | success | float64, float32 | float64, float32 | - |
| real | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| reciprocal | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| round | identity | success | float64, float32 | float64, float32 | - |
| round_decimals_0 | identity | success | float64, float32 | float64, float32 | - |
| round_decimals_3 | identity | success | float64, float32 | float64, float32 | - |
| round_decimals_neg_3 | identity | success | float64, float32 | float64, float32 | - |
| rsqrt | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| rsub | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| sgn | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| sigmoid | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| sign | identity | success | float64, float32 | float64, float32 | - |
| sin | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| sinc | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| sinh | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| slogdet | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| solve | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| solve_ex | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| solve_triangular | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| special_entr | identity | success | float64, float32 | float64, float32 | - |
| special_erfcx | identity | success | float64, float32 | float64, float32 | - |
| special_i0e | identity | success | float64, float32 | float64, float32 | - |
| special_i1 | identity | success | float64, float32 | float64, float32 | - |
| special_i1e | identity | success | float64, float32 | float64, float32 | - |
| special_log_ndtr | identity | success | float64, float32 | float64, float32 | - |
| special_ndtr | identity | success | float64, float32 | float64, float32 | - |
| special_ndtri | identity | success | float64, float32 | float64, float32 | - |
| special_polygamma_special_polygamma_n_0 | identity | success | float64, float32 | float64, float32 | - |
| special_xlog1py | identity | success | float64, float32 | float64, float32 | - |
| sqrt | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| square | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| std | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| std_unbiased | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| sub | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| sum | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| svd | gauge_ill_defined | error | complex128 | complex128 | - |
| svd | s | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| svd | u_abs | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| svd | uvh_product | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| svd | vh_abs | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| svdvals | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| tan | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| tanh | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| tensorinv | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| tensorsolve | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| true_divide | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| trunc | identity | success | float64, float32 | float64, float32 | - |
| vander | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| var | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| var_unbiased | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| vecdot | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| vector_norm | identity | success | float64, complex128, float32, complex64 | float64, complex128, float32, complex64 | - |
| xlogy | identity | success | float64, float32 | float64, float32 | - |

## Missing Publishable Coverage

None.
