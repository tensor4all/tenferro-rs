# Tensor AD Oracles Support Coverage

This file is generated from the vendored `third_party/tensor-ad-oracles` subtree and the local oracle replay support registry.

## Summary

- Total published records: 1828
- Supported success records: 867
- Supported success records with HVP payloads: 867
- Expected error records: 2
- Unsupported success records: 959

## Supported

| op | family | observable | sample count |
| --- | --- | --- | ---: |
| cholesky | identity | identity | 16 |
| cholesky_ex | identity | identity | 16 |
| cond | identity | identity | 3 |
| cross | identity | identity | 3 |
| eigh | values_vectors_abs | eigh_values_vectors_abs | 8 |
| householder_product | identity | identity | 8 |
| inv_ex | identity | identity | 8 |
| lu_factor | identity | identity | 20 |
| lu_factor_ex | identity | identity | 20 |
| lu_solve | identity | identity | 324 |
| matrix_power | identity | identity | 18 |
| multi_dot | identity | identity | 7 |
| pinv_hermitian | identity | identity | 8 |
| pinv_singular | identity | identity | 48 |
| qr | identity | identity | 36 |
| solve | identity | identity | 24 |
| solve_ex | identity | identity | 24 |
| svd | s | svd_s | 54 |
| svd | u_abs | svd_u_abs | 54 |
| svd | uvh_product | svd_uvh_product | 54 |
| svd | vh_abs | svd_vh_abs | 54 |
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
| det | identity | identity | 9 | tenferro replay does not implement this scalar-output oracle family yet |
| diagonal | identity | identity | 15 | tenferro replay does not implement this tensor-construction oracle family yet |
| eig | values_vectors_abs | eig_values_vectors_abs | 8 | tenferro replay does not implement this spectral/inverse family yet |
| eigvals | identity | identity | 8 | tenferro replay does not implement this scalar-output oracle family yet |
| eigvalsh | identity | identity | 8 | tenferro replay does not implement this scalar-output oracle family yet |
| inv | identity | identity | 8 | tenferro replay does not implement this spectral/inverse family yet |
| lstsq_grad_oriented | identity | identity | 36 | tenferro replay does not implement this solver/decomposition family yet |
| lu | identity | identity | 20 | tenferro replay does not implement this solver/decomposition family yet |
| matrix_norm | identity | identity | 64 | tenferro replay does not implement this scalar-output oracle family yet |
| norm | identity | identity | 102 | tenferro replay does not implement this scalar-output oracle family yet |
| pinv | identity | identity | 24 | tenferro replay does not implement this spectral/inverse family yet |
| slogdet | identity | identity | 9 | tenferro replay does not implement this scalar-output oracle family yet |
| solve_triangular | identity | identity | 432 | tenferro replay does not implement this solver/decomposition family yet |
| svdvals | identity | identity | 36 | tenferro replay does not implement this scalar-output oracle family yet |
| vector_norm | identity | identity | 180 | tenferro replay does not implement this scalar-output oracle family yet |
