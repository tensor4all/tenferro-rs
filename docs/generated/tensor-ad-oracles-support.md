# Tensor AD Oracles Support Coverage

This file is generated from the vendored `third_party/tensor-ad-oracles` subtree and the local oracle replay support registry.

## Summary

- Total published records: 9572
- Supported success records: 1787
- Supported success records with HVP payloads: 1231
- Expected error records: 2
- Unsupported success records: 7783

## Supported

| op | family | observable | sample count |
| --- | --- | --- | ---: |
| cholesky | identity | identity | 48 |
| cholesky_ex | identity | identity | 48 |
| cond | identity | identity | 3 |
| cross | identity | identity | 3 |
| det | identity | identity | 36 |
| eigh | values_vectors_abs | eigh_values_vectors_abs | 32 |
| householder_product | identity | identity | 8 |
| inv | identity | identity | 24 |
| inv_ex | identity | identity | 24 |
| lu_factor | identity | identity | 20 |
| lu_factor_ex | identity | identity | 20 |
| lu_solve | identity | identity | 324 |
| matrix_power | identity | identity | 18 |
| multi_dot | identity | identity | 7 |
| pinv | identity | identity | 72 |
| pinv_hermitian | identity | identity | 8 |
| pinv_singular | identity | identity | 48 |
| qr | identity | identity | 36 |
| slogdet | identity | identity | 36 |
| solve | identity | identity | 24 |
| solve_ex | identity | identity | 24 |
| svd | s | svd_s | 216 |
| svd | u_abs | svd_u_abs | 216 |
| svd | uvh_product | svd_uvh_product | 216 |
| svd | vh_abs | svd_vh_abs | 216 |
| tensorinv | identity | identity | 2 |
| tensorsolve | identity | identity | 4 |
| vander | identity | identity | 10 |
| vecdot | identity | identity | 44 |

## Expected Errors

| op | family | observable | sample count |
| --- | --- | --- | ---: |
| eigh | gauge_ill_defined | eigh_values_vectors_abs | 1 |
| svd | gauge_ill_defined | svd_uvh_product | 1 |

## Unsupported

| op | family | observable | sample count | reason |
| --- | --- | --- | ---: | --- |
| __radd__ | identity | identity | 36 | tenferro replay does not implement this oracle family yet |
| __rdiv__ | identity | identity | 36 | tenferro replay does not implement this oracle family yet |
| __rmod__ | identity | identity | 18 | tenferro replay does not implement this oracle family yet |
| __rmul__ | identity | identity | 36 | tenferro replay does not implement this oracle family yet |
| __rpow__ | identity | identity | 22 | tenferro replay does not implement this oracle family yet |
| __rsub__ | identity | identity | 36 | tenferro replay does not implement this oracle family yet |
| abs | identity | identity | 4 | tenferro replay does not implement this oracle family yet |
| acos | identity | identity | 12 | tenferro replay does not implement this oracle family yet |
| acosh | identity | identity | 12 | tenferro replay does not implement this oracle family yet |
| add | identity | identity | 44 | tenferro replay does not implement this oracle family yet |
| amax | identity | identity | 40 | tenferro replay does not implement this oracle family yet |
| amin | identity | identity | 40 | tenferro replay does not implement this oracle family yet |
| angle | identity | identity | 4 | tenferro replay does not implement this oracle family yet |
| asin | identity | identity | 4 | tenferro replay does not implement this oracle family yet |
| asinh | identity | identity | 4 | tenferro replay does not implement this oracle family yet |
| atan | identity | identity | 4 | tenferro replay does not implement this oracle family yet |
| atan2 | identity | identity | 18 | tenferro replay does not implement this oracle family yet |
| atanh | identity | identity | 4 | tenferro replay does not implement this oracle family yet |
| cdouble | identity | identity | 20 | tenferro replay does not implement this oracle family yet |
| ceil | identity | identity | 2 | tenferro replay does not implement this oracle family yet |
| cholesky | identity | identity | 16 | tenferro replay currently supports this family only for float64/complex64/complex128 |
| cholesky_ex | identity | identity | 16 | tenferro replay currently supports this family only for float64/complex64/complex128 |
| clamp_max | identity | identity | 18 | tenferro replay does not implement this oracle family yet |
| clamp_min | identity | identity | 18 | tenferro replay does not implement this oracle family yet |
| complex | identity | identity | 18 | tenferro replay does not implement this oracle family yet |
| cond | identity | identity | 9 | tenferro replay currently supports this family only for float64 |
| conj | identity | identity | 12 | tenferro replay does not implement this oracle family yet |
| conj_physical | identity | identity | 4 | tenferro replay does not implement this oracle family yet |
| copysign | identity | identity | 20 | tenferro replay does not implement this oracle family yet |
| cos | identity | identity | 12 | tenferro replay does not implement this oracle family yet |
| cosh | identity | identity | 12 | tenferro replay does not implement this oracle family yet |
| cross | identity | identity | 9 | tenferro replay currently supports this family only for float64 |
| deg2rad | identity | identity | 2 | tenferro replay does not implement this oracle family yet |
| diagonal | identity | identity | 60 | tenferro replay does not implement this tensor-construction oracle family yet |
| digamma | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| div_no_rounding_mode | identity | identity | 36 | tenferro replay does not implement this oracle family yet |
| double | identity | identity | 20 | tenferro replay does not implement this oracle family yet |
| eig | values_vectors_abs | eig_values_vectors_abs | 32 | tenferro replay does not implement this spectral/inverse family yet |
| eigvals | identity | identity | 32 | tenferro replay does not implement this scalar-output oracle family yet |
| eigvalsh | identity | identity | 32 | tenferro replay does not implement this scalar-output oracle family yet |
| erf | identity | identity | 2 | tenferro replay does not implement this oracle family yet |
| erfc | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| erfinv | identity | identity | 2 | tenferro replay does not implement this oracle family yet |
| exp | identity | identity | 12 | tenferro replay does not implement this oracle family yet |
| exp2 | identity | identity | 12 | tenferro replay does not implement this oracle family yet |
| expm1 | identity | identity | 4 | tenferro replay does not implement this oracle family yet |
| fill | identity | identity | 12 | tenferro replay does not implement this oracle family yet |
| float_power | identity | identity | 36 | tenferro replay does not implement this oracle family yet |
| floor | identity | identity | 2 | tenferro replay does not implement this oracle family yet |
| fmax | identity | identity | 18 | tenferro replay does not implement this oracle family yet |
| fmin | identity | identity | 18 | tenferro replay does not implement this oracle family yet |
| frac | identity | identity | 2 | tenferro replay does not implement this oracle family yet |
| frexp | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| householder_product | identity | identity | 24 | tenferro replay currently supports this family only for float64 |
| hypot | identity | identity | 18 | tenferro replay does not implement this oracle family yet |
| i0 | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| imag | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| inv | identity | identity | 8 | tenferro replay currently supports this family only for float64/complex64/complex128 |
| inv_ex | identity | identity | 8 | tenferro replay currently supports this family only for float64/complex64/complex128 |
| ldexp | identity | identity | 36 | tenferro replay does not implement this oracle family yet |
| lgamma | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| log | identity | identity | 12 | tenferro replay does not implement this oracle family yet |
| log10 | identity | identity | 12 | tenferro replay does not implement this oracle family yet |
| log1p | identity | identity | 4 | tenferro replay does not implement this oracle family yet |
| log2 | identity | identity | 12 | tenferro replay does not implement this oracle family yet |
| logaddexp | identity | identity | 36 | tenferro replay does not implement this oracle family yet |
| logit | identity | identity | 8 | tenferro replay does not implement this oracle family yet |
| lstsq_grad_oriented | identity | identity | 144 | tenferro replay does not implement this solver/decomposition family yet |
| lu | identity | identity | 80 | tenferro replay does not implement this solver/decomposition family yet |
| lu_factor | identity | identity | 60 | tenferro replay currently supports this family only for float64 |
| lu_factor_ex | identity | identity | 60 | tenferro replay currently supports this family only for float64 |
| lu_solve | identity | identity | 924 | tenferro replay currently supports this family only for float64 |
| matrix_norm | identity | identity | 256 | tenferro replay does not implement this scalar-output oracle family yet |
| matrix_power | identity | identity | 54 | tenferro replay currently supports this family only for float64 |
| max_binary | identity | identity | 18 | tenferro replay does not implement this oracle family yet |
| maximum | identity | identity | 18 | tenferro replay does not implement this oracle family yet |
| mean | identity | identity | 80 | tenferro replay does not implement this oracle family yet |
| min_binary | identity | identity | 18 | tenferro replay does not implement this oracle family yet |
| minimum | identity | identity | 18 | tenferro replay does not implement this oracle family yet |
| mul | identity | identity | 36 | tenferro replay does not implement this oracle family yet |
| multi_dot | identity | identity | 21 | tenferro replay currently supports this family only for float64 |
| mvlgamma_mvlgamma_p_1 | identity | identity | 20 | tenferro replay does not implement this oracle family yet |
| mvlgamma_mvlgamma_p_3 | identity | identity | 20 | tenferro replay does not implement this oracle family yet |
| mvlgamma_mvlgamma_p_5 | identity | identity | 20 | tenferro replay does not implement this oracle family yet |
| nan_to_num | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| nanmean | identity | identity | 70 | tenferro replay does not implement this oracle family yet |
| nansum | identity | identity | 70 | tenferro replay does not implement this oracle family yet |
| neg | identity | identity | 4 | tenferro replay does not implement this oracle family yet |
| nn_functional_celu | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| nn_functional_elu | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| nn_functional_hardshrink | identity | identity | 12 | tenferro replay does not implement this oracle family yet |
| nn_functional_hardsigmoid | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| nn_functional_hardtanh | identity | identity | 10 | tenferro replay does not implement this oracle family yet |
| nn_functional_logsigmoid | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| nn_functional_mish | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| nn_functional_prelu | identity | identity | 50 | tenferro replay does not implement this oracle family yet |
| nn_functional_relu | identity | identity | 8 | tenferro replay does not implement this oracle family yet |
| nn_functional_relu6 | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| nn_functional_rrelu | identity | identity | 10 | tenferro replay does not implement this oracle family yet |
| nn_functional_selu | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| nn_functional_silu | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| nn_functional_softplus | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| nn_functional_softshrink | identity | identity | 10 | tenferro replay does not implement this oracle family yet |
| nn_functional_softsign | identity | identity | 12 | tenferro replay does not implement this oracle family yet |
| nn_functional_tanhshrink | identity | identity | 12 | tenferro replay does not implement this oracle family yet |
| nn_functional_threshold | identity | identity | 8 | tenferro replay does not implement this oracle family yet |
| norm | identity | identity | 408 | tenferro replay does not implement this scalar-output oracle family yet |
| pinv | identity | identity | 24 | tenferro replay currently supports this family only for float64/complex64/complex128 |
| pinv_hermitian | identity | identity | 24 | tenferro replay currently supports this family only for float64 |
| pinv_singular | identity | identity | 144 | tenferro replay currently supports this family only for float64 |
| polar | identity | identity | 18 | tenferro replay does not implement this oracle family yet |
| polygamma_polygamma_n_0 | identity | identity | 20 | tenferro replay does not implement this oracle family yet |
| polygamma_polygamma_n_1 | identity | identity | 20 | tenferro replay does not implement this oracle family yet |
| polygamma_polygamma_n_2 | identity | identity | 20 | tenferro replay does not implement this oracle family yet |
| polygamma_polygamma_n_3 | identity | identity | 20 | tenferro replay does not implement this oracle family yet |
| polygamma_polygamma_n_4 | identity | identity | 20 | tenferro replay does not implement this oracle family yet |
| positive | identity | identity | 4 | tenferro replay does not implement this oracle family yet |
| pow | identity | identity | 36 | tenferro replay does not implement this oracle family yet |
| prod | identity | identity | 156 | tenferro replay does not implement this oracle family yet |
| qr | identity | identity | 108 | tenferro replay currently supports this family only for float64 |
| rad2deg | identity | identity | 2 | tenferro replay does not implement this oracle family yet |
| real | identity | identity | 12 | tenferro replay does not implement this oracle family yet |
| reciprocal | identity | identity | 12 | tenferro replay does not implement this oracle family yet |
| round | identity | identity | 2 | tenferro replay does not implement this oracle family yet |
| round_decimals_0 | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| round_decimals_3 | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| round_decimals_neg_3 | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| rsqrt | identity | identity | 12 | tenferro replay does not implement this oracle family yet |
| rsub | identity | identity | 44 | tenferro replay does not implement this oracle family yet |
| sgn | identity | identity | 4 | tenferro replay does not implement this oracle family yet |
| sigmoid | identity | identity | 12 | tenferro replay does not implement this oracle family yet |
| sign | identity | identity | 2 | tenferro replay does not implement this oracle family yet |
| sin | identity | identity | 4 | tenferro replay does not implement this oracle family yet |
| sinc | identity | identity | 12 | tenferro replay does not implement this oracle family yet |
| sinh | identity | identity | 4 | tenferro replay does not implement this oracle family yet |
| solve | identity | identity | 72 | tenferro replay currently supports this family only for float64 |
| solve_ex | identity | identity | 72 | tenferro replay currently supports this family only for float64 |
| solve_triangular | identity | identity | 1728 | tenferro replay does not implement this solver/decomposition family yet |
| special_entr | identity | identity | 4 | tenferro replay does not implement this oracle family yet |
| special_erfcx | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| special_i0e | identity | identity | 4 | tenferro replay does not implement this oracle family yet |
| special_i1 | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| special_i1e | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| special_log_ndtr | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| special_ndtr | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| special_ndtri | identity | identity | 6 | tenferro replay does not implement this oracle family yet |
| special_polygamma_special_polygamma_n_0 | identity | identity | 20 | tenferro replay does not implement this oracle family yet |
| special_xlog1py | identity | identity | 18 | tenferro replay does not implement this oracle family yet |
| sqrt | identity | identity | 4 | tenferro replay does not implement this oracle family yet |
| square | identity | identity | 12 | tenferro replay does not implement this oracle family yet |
| std | identity | identity | 56 | tenferro replay does not implement this oracle family yet |
| std_unbiased | identity | identity | 8 | tenferro replay does not implement this oracle family yet |
| sub | identity | identity | 44 | tenferro replay does not implement this oracle family yet |
| sum | identity | identity | 80 | tenferro replay does not implement this oracle family yet |
| svdvals | identity | identity | 144 | tenferro replay does not implement this scalar-output oracle family yet |
| tan | identity | identity | 4 | tenferro replay does not implement this oracle family yet |
| tanh | identity | identity | 4 | tenferro replay does not implement this oracle family yet |
| tensorinv | identity | identity | 6 | tenferro replay currently supports this family only for float64 |
| tensorsolve | identity | identity | 12 | tenferro replay currently supports this family only for float64 |
| true_divide | identity | identity | 36 | tenferro replay does not implement this oracle family yet |
| trunc | identity | identity | 2 | tenferro replay does not implement this oracle family yet |
| vander | identity | identity | 30 | tenferro replay currently supports this family only for float64 |
| var | identity | identity | 56 | tenferro replay does not implement this oracle family yet |
| var_unbiased | identity | identity | 8 | tenferro replay does not implement this oracle family yet |
| vecdot | identity | identity | 132 | tenferro replay currently supports this family only for float64 |
| vector_norm | identity | identity | 720 | tenferro replay does not implement this scalar-output oracle family yet |
| xlogy | identity | identity | 18 | tenferro replay does not implement this oracle family yet |
