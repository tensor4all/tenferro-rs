# CPU Benchmark Results - 2026-05-23

This is a normal comparison benchmark run, not a publication gate. Times are
mean wall-clock latency in microseconds (`us`); lower is better.

## Run Metadata

- Date: 2026-05-23
- Host CPU: AMD EPYC 7713P 64-Core Processor
- tenferro branch: `codex/cpu-benchmarks-libtorch`
- tenferro HEAD during run: `c9a1a28b` with local working-tree changes
- tidu code during run: local checkout equivalent to tidu-rs PR #25 merged commit `1694275361d2f16abfb3f25ad7941407c88b7c09`
- Rust: `rustc 1.94.1 (e408947bf 2026-03-25)`
- Cargo: `cargo 1.94.1 (29ea6fb6a 2026-03-24)`
- Command: `TENFERRO_MAIN_REPO_DIR=$PWD bash scripts/bench-cpu.sh --kind all --threads 1,2,4 --no-download-libtorch -- --sample-size 10 --warm-up-time 0.2 --measurement-time 0.2`
- Raw log: `target/bench-results/2026-05-23-cpu-all-svd-ad-sizes.log` (not committed)

## SVD Focus

The AD SVD-values benchmark now covers the same square sizes as the primal SVD
benchmark: `4x4`, `8x8`, `16x16`, `32x32`, `64x64`, and `128x128`.

| suite | benchmark | dtype | threads | shape | tenferro_us | torch_us | tenferro/torch | winner |
|---|---|---:|---:|---|---:|---:|---:|---|
| linalg | `f64_svd` | f64 | 1 | `4x4` | 4.661 | 20.170 | 0.23x | tenferro |
| linalg | `f64_svd` | f64 | 1 | `8x8` | 13.383 | 32.586 | 0.41x | tenferro |
| linalg | `f64_svd` | f64 | 1 | `16x16` | 39.607 | 68.808 | 0.58x | tenferro |
| linalg | `f64_svd` | f64 | 1 | `32x32` | 190.525 | 202.919 | 0.94x | tenferro |
| linalg | `f64_svd` | f64 | 1 | `64x64` | 1037.382 | 741.139 | 1.40x | torch |
| linalg | `f64_svd` | f64 | 1 | `128x128` | 3116.023 | 3224.577 | 0.97x | tenferro |
| linalg | `f64_svd` | f64 | 2 | `4x4` | 4.566 | 18.484 | 0.25x | tenferro |
| linalg | `f64_svd` | f64 | 2 | `8x8` | 11.789 | 29.872 | 0.39x | tenferro |
| linalg | `f64_svd` | f64 | 2 | `16x16` | 42.528 | 64.633 | 0.66x | tenferro |
| linalg | `f64_svd` | f64 | 2 | `32x32` | 161.752 | 204.516 | 0.79x | tenferro |
| linalg | `f64_svd` | f64 | 2 | `64x64` | 791.762 | 990.879 | 0.80x | tenferro |
| linalg | `f64_svd` | f64 | 2 | `128x128` | 3637.459 | 3741.419 | 0.97x | tenferro |
| linalg | `f64_svd` | f64 | 4 | `4x4` | 4.663 | 19.449 | 0.24x | tenferro |
| linalg | `f64_svd` | f64 | 4 | `8x8` | 11.800 | 30.323 | 0.39x | tenferro |
| linalg | `f64_svd` | f64 | 4 | `16x16` | 40.146 | 63.297 | 0.63x | tenferro |
| linalg | `f64_svd` | f64 | 4 | `32x32` | 149.570 | 231.305 | 0.65x | tenferro |
| linalg | `f64_svd` | f64 | 4 | `64x64` | 840.349 | 906.113 | 0.93x | tenferro |
| linalg | `f64_svd` | f64 | 4 | `128x128` | 3244.179 | 3628.201 | 0.89x | tenferro |
| ad | `f64_grad_sum_svd_values` | f64 | 1 | `4x4` | 110.679 | 83.703 | 1.32x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 1 | `8x8` | 119.318 | 94.695 | 1.26x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 1 | `16x16` | 146.461 | 133.314 | 1.10x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 1 | `32x32` | 276.862 | 266.650 | 1.04x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 1 | `64x64` | 839.261 | 817.652 | 1.03x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 1 | `128x128` | 3796.990 | 3708.189 | 1.02x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 2 | `4x4` | 108.153 | 74.559 | 1.45x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 2 | `8x8` | 121.471 | 86.604 | 1.40x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 2 | `16x16` | 145.503 | 123.044 | 1.18x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 2 | `32x32` | 364.582 | 293.992 | 1.24x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 2 | `64x64` | 1393.896 | 1086.291 | 1.28x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 2 | `128x128` | 4159.568 | 4359.004 | 0.95x | tenferro |
| ad | `f64_grad_sum_svd_values` | f64 | 4 | `4x4` | 132.926 | 80.191 | 1.66x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 4 | `8x8` | 116.085 | 92.585 | 1.25x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 4 | `16x16` | 161.744 | 133.361 | 1.21x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 4 | `32x32` | 419.578 | 426.326 | 0.98x | tenferro |
| ad | `f64_grad_sum_svd_values` | f64 | 4 | `64x64` | 1053.240 | 1102.792 | 0.96x | tenferro |
| ad | `f64_grad_sum_svd_values` | f64 | 4 | `128x128` | 4051.628 | 4180.504 | 0.97x | tenferro |

## Tenferro Slower Cases

Rows where `tenferro/torch > 1.0`, sorted by ratio.

| suite | benchmark | dtype | threads | shape | tenferro_us | torch_us | tenferro/torch | winner |
|---|---|---:|---:|---|---:|---:|---:|---|
| linalg | `f64_qr` | f64 | 4 | `32x32` | 226.392 | 43.647 | 5.19x | torch |
| linalg | `f64_qr` | f64 | 4 | `64x64` | 1029.538 | 230.502 | 4.47x | torch |
| linalg | `f64_qr` | f64 | 2 | `32x32` | 141.572 | 37.829 | 3.74x | torch |
| linalg | `f64_qr` | f64 | 4 | `128x128` | 2775.310 | 751.851 | 3.69x | torch |
| linalg | `f64_qr` | f64 | 2 | `64x64` | 732.766 | 231.035 | 3.17x | torch |
| linalg | `f64_qr` | f64 | 2 | `128x128` | 1583.197 | 770.130 | 2.06x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 4 | `4x4` | 132.926 | 80.191 | 1.66x | torch |
| matmul | `f64_square` | f64 | 1 | `256x256` | 2973.316 | 1807.628 | 1.64x | torch |
| linalg | `f64_solve_matrix_rhs4` | f64 | 2 | `128x4` | 310.386 | 201.601 | 1.54x | torch |
| linalg | `f64_eigh` | f64 | 4 | `128x128` | 2122.134 | 1394.780 | 1.52x | torch |
| linalg | `f64_eigh` | f64 | 4 | `64x64` | 515.723 | 346.952 | 1.49x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 2 | `4x4` | 108.153 | 74.559 | 1.45x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 2 | `8x8` | 121.471 | 86.604 | 1.40x | torch |
| matmul | `f64_square` | f64 | 4 | `32x32` | 22.259 | 15.894 | 1.40x | torch |
| linalg | `f64_svd` | f64 | 1 | `64x64` | 1037.382 | 741.139 | 1.40x | torch |
| linalg | `f64_eigh` | f64 | 1 | `128x128` | 2201.138 | 1588.106 | 1.39x | torch |
| linalg | `f64_eigh` | f64 | 2 | `128x128` | 2066.426 | 1526.433 | 1.35x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 1 | `4x4` | 110.679 | 83.703 | 1.32x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 2 | `64x64` | 1393.896 | 1086.291 | 1.28x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 1 | `8x8` | 119.318 | 94.695 | 1.26x | torch |
| ad | `f64_grad_sum_matmul` | f64 | 2 | `64x64` | 254.021 | 202.251 | 1.26x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 4 | `8x8` | 116.085 | 92.585 | 1.25x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 2 | `32x32` | 364.582 | 293.992 | 1.24x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 4 | `16x16` | 161.744 | 133.361 | 1.21x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 2 | `16x16` | 145.503 | 123.044 | 1.18x | torch |
| linalg | `f64_solve_matrix_rhs4` | f64 | 4 | `128x4` | 233.449 | 201.381 | 1.16x | torch |
| ad | `f64_grad_sum_matmul` | f64 | 4 | `64x64` | 231.322 | 199.666 | 1.16x | torch |
| linalg | `f64_qr` | f64 | 1 | `32x32` | 45.060 | 39.460 | 1.14x | torch |
| linalg | `f64_eigh` | f64 | 2 | `64x64` | 478.639 | 426.770 | 1.12x | torch |
| ad | `f64_grad_sum_matmul` | f64 | 2 | `16x16` | 60.727 | 54.462 | 1.12x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 1 | `16x16` | 146.461 | 133.314 | 1.10x | torch |
| matmul | `f64_square` | f64 | 2 | `32x32` | 15.237 | 14.024 | 1.09x | torch |
| linalg | `f64_qr` | f64 | 2 | `16x16` | 17.576 | 16.340 | 1.08x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 1 | `32x32` | 276.862 | 266.650 | 1.04x | torch |
| linalg | `f64_qr` | f64 | 1 | `64x64` | 157.708 | 152.176 | 1.04x | torch |
| ad | `f64_grad_sum_matmul` | f64 | 2 | `4x4` | 49.490 | 48.083 | 1.03x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 1 | `64x64` | 839.261 | 817.652 | 1.03x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 1 | `128x128` | 3796.990 | 3708.189 | 1.02x | torch |
| ad | `f64_grad_sum_matmul` | f64 | 4 | `4x4` | 53.243 | 52.372 | 1.02x | torch |

## Full Paired Results

| suite | benchmark | dtype | threads | shape | tenferro_us | torch_us | tenferro/torch | winner |
|---|---|---:|---:|---|---:|---:|---:|---|
| matmul | `f64_square` | f64 | 1 | `2x2` | 1.199 | 9.707 | 0.12x | tenferro |
| matmul | `f64_square` | f64 | 1 | `4x4` | 1.576 | 8.985 | 0.18x | tenferro |
| matmul | `f64_square` | f64 | 1 | `8x8` | 1.281 | 7.637 | 0.17x | tenferro |
| matmul | `f64_square` | f64 | 1 | `16x16` | 1.617 | 8.386 | 0.19x | tenferro |
| matmul | `f64_square` | f64 | 1 | `32x32` | 4.109 | 16.519 | 0.25x | tenferro |
| matmul | `f64_square` | f64 | 1 | `128x128` | 203.068 | 340.270 | 0.60x | tenferro |
| matmul | `f64_square` | f64 | 1 | `256x256` | 2973.316 | 1807.628 | 1.64x | torch |
| matmul | `f64_square` | f64 | 1 | `512x512` | 11128.600 | 14721.812 | 0.76x | tenferro |
| matmul | `f64_square` | f64 | 2 | `2x2` | 1.254 | 8.571 | 0.15x | tenferro |
| matmul | `f64_square` | f64 | 2 | `4x4` | 1.206 | 7.947 | 0.15x | tenferro |
| matmul | `f64_square` | f64 | 2 | `8x8` | 1.250 | 6.744 | 0.19x | tenferro |
| matmul | `f64_square` | f64 | 2 | `16x16` | 2.646 | 8.846 | 0.30x | tenferro |
| matmul | `f64_square` | f64 | 2 | `32x32` | 15.237 | 14.024 | 1.09x | torch |
| matmul | `f64_square` | f64 | 2 | `128x128` | 118.188 | 277.935 | 0.43x | tenferro |
| matmul | `f64_square` | f64 | 2 | `256x256` | 705.562 | 1982.694 | 0.36x | tenferro |
| matmul | `f64_square` | f64 | 2 | `512x512` | 5282.173 | 7718.604 | 0.68x | tenferro |
| matmul | `f64_square` | f64 | 4 | `2x2` | 1.192 | 5.112 | 0.23x | tenferro |
| matmul | `f64_square` | f64 | 4 | `4x4` | 1.211 | 4.109 | 0.29x | tenferro |
| matmul | `f64_square` | f64 | 4 | `8x8` | 1.272 | 4.306 | 0.30x | tenferro |
| matmul | `f64_square` | f64 | 4 | `16x16` | 1.737 | 5.642 | 0.31x | tenferro |
| matmul | `f64_square` | f64 | 4 | `32x32` | 22.259 | 15.894 | 1.40x | torch |
| matmul | `f64_square` | f64 | 4 | `128x128` | 82.358 | 116.945 | 0.70x | tenferro |
| matmul | `f64_square` | f64 | 4 | `256x256` | 475.553 | 1012.908 | 0.47x | tenferro |
| matmul | `f64_square` | f64 | 4 | `512x512` | 2823.277 | 4768.127 | 0.59x | tenferro |
| matmul | `c64_square` | c64 | 1 | `2x2` | 1.155 | 15.012 | 0.08x | tenferro |
| matmul | `c64_square` | c64 | 1 | `4x4` | 1.208 | 12.634 | 0.10x | tenferro |
| matmul | `c64_square` | c64 | 1 | `8x8` | 1.370 | 12.769 | 0.11x | tenferro |
| matmul | `c64_square` | c64 | 1 | `16x16` | 2.671 | 16.730 | 0.16x | tenferro |
| matmul | `c64_square` | c64 | 1 | `32x32` | 12.982 | 39.960 | 0.32x | tenferro |
| matmul | `c64_square` | c64 | 2 | `2x2` | 1.254 | 13.140 | 0.10x | tenferro |
| matmul | `c64_square` | c64 | 2 | `4x4` | 1.219 | 11.706 | 0.10x | tenferro |
| matmul | `c64_square` | c64 | 2 | `8x8` | 1.409 | 11.890 | 0.12x | tenferro |
| matmul | `c64_square` | c64 | 2 | `16x16` | 3.707 | 14.664 | 0.25x | tenferro |
| matmul | `c64_square` | c64 | 2 | `32x32` | 15.602 | 35.431 | 0.44x | tenferro |
| matmul | `c64_square` | c64 | 4 | `2x2` | 1.165 | 7.362 | 0.16x | tenferro |
| matmul | `c64_square` | c64 | 4 | `4x4` | 1.369 | 6.803 | 0.20x | tenferro |
| matmul | `c64_square` | c64 | 4 | `8x8` | 1.408 | 7.735 | 0.18x | tenferro |
| matmul | `c64_square` | c64 | 4 | `16x16` | 4.599 | 11.402 | 0.40x | tenferro |
| matmul | `c64_square` | c64 | 4 | `32x32` | 9.637 | 35.448 | 0.27x | tenferro |
| linalg | `f64_svd` | f64 | 1 | `4x4` | 4.661 | 20.170 | 0.23x | tenferro |
| linalg | `f64_svd` | f64 | 1 | `8x8` | 13.383 | 32.586 | 0.41x | tenferro |
| linalg | `f64_svd` | f64 | 1 | `16x16` | 39.607 | 68.808 | 0.58x | tenferro |
| linalg | `f64_svd` | f64 | 1 | `32x32` | 190.525 | 202.919 | 0.94x | tenferro |
| linalg | `f64_svd` | f64 | 1 | `64x64` | 1037.382 | 741.139 | 1.40x | torch |
| linalg | `f64_svd` | f64 | 1 | `128x128` | 3116.023 | 3224.577 | 0.97x | tenferro |
| linalg | `f64_svd` | f64 | 2 | `4x4` | 4.566 | 18.484 | 0.25x | tenferro |
| linalg | `f64_svd` | f64 | 2 | `8x8` | 11.789 | 29.872 | 0.39x | tenferro |
| linalg | `f64_svd` | f64 | 2 | `16x16` | 42.528 | 64.633 | 0.66x | tenferro |
| linalg | `f64_svd` | f64 | 2 | `32x32` | 161.752 | 204.516 | 0.79x | tenferro |
| linalg | `f64_svd` | f64 | 2 | `64x64` | 791.762 | 990.879 | 0.80x | tenferro |
| linalg | `f64_svd` | f64 | 2 | `128x128` | 3637.459 | 3741.419 | 0.97x | tenferro |
| linalg | `f64_svd` | f64 | 4 | `4x4` | 4.663 | 19.449 | 0.24x | tenferro |
| linalg | `f64_svd` | f64 | 4 | `8x8` | 11.800 | 30.323 | 0.39x | tenferro |
| linalg | `f64_svd` | f64 | 4 | `16x16` | 40.146 | 63.297 | 0.63x | tenferro |
| linalg | `f64_svd` | f64 | 4 | `32x32` | 149.570 | 231.305 | 0.65x | tenferro |
| linalg | `f64_svd` | f64 | 4 | `64x64` | 840.349 | 906.113 | 0.93x | tenferro |
| linalg | `f64_svd` | f64 | 4 | `128x128` | 3244.179 | 3628.201 | 0.89x | tenferro |
| linalg | `f64_qr` | f64 | 1 | `4x4` | 2.009 | 10.157 | 0.20x | tenferro |
| linalg | `f64_qr` | f64 | 1 | `8x8` | 3.575 | 10.908 | 0.33x | tenferro |
| linalg | `f64_qr` | f64 | 1 | `16x16` | 9.884 | 16.043 | 0.62x | tenferro |
| linalg | `f64_qr` | f64 | 1 | `32x32` | 45.060 | 39.460 | 1.14x | torch |
| linalg | `f64_qr` | f64 | 1 | `64x64` | 157.708 | 152.176 | 1.04x | torch |
| linalg | `f64_qr` | f64 | 1 | `128x128` | 579.450 | 759.769 | 0.76x | tenferro |
| linalg | `f64_qr` | f64 | 2 | `4x4` | 2.569 | 10.749 | 0.24x | tenferro |
| linalg | `f64_qr` | f64 | 2 | `8x8` | 3.301 | 11.303 | 0.29x | tenferro |
| linalg | `f64_qr` | f64 | 2 | `16x16` | 17.576 | 16.340 | 1.08x | torch |
| linalg | `f64_qr` | f64 | 2 | `32x32` | 141.572 | 37.829 | 3.74x | torch |
| linalg | `f64_qr` | f64 | 2 | `64x64` | 732.766 | 231.035 | 3.17x | torch |
| linalg | `f64_qr` | f64 | 2 | `128x128` | 1583.197 | 770.130 | 2.06x | torch |
| linalg | `f64_qr` | f64 | 4 | `4x4` | 1.870 | 12.968 | 0.14x | tenferro |
| linalg | `f64_qr` | f64 | 4 | `8x8` | 3.012 | 14.420 | 0.21x | tenferro |
| linalg | `f64_qr` | f64 | 4 | `16x16` | 9.091 | 18.705 | 0.49x | tenferro |
| linalg | `f64_qr` | f64 | 4 | `32x32` | 226.392 | 43.647 | 5.19x | torch |
| linalg | `f64_qr` | f64 | 4 | `64x64` | 1029.538 | 230.502 | 4.47x | torch |
| linalg | `f64_qr` | f64 | 4 | `128x128` | 2775.310 | 751.851 | 3.69x | torch |
| linalg | `f64_eigh` | f64 | 1 | `4x4` | 3.153 | 13.852 | 0.23x | tenferro |
| linalg | `f64_eigh` | f64 | 1 | `8x8` | 7.345 | 18.818 | 0.39x | tenferro |
| linalg | `f64_eigh` | f64 | 1 | `16x16` | 23.127 | 34.511 | 0.67x | tenferro |
| linalg | `f64_eigh` | f64 | 1 | `32x32` | 88.806 | 105.748 | 0.84x | tenferro |
| linalg | `f64_eigh` | f64 | 1 | `64x64` | 344.060 | 364.886 | 0.94x | tenferro |
| linalg | `f64_eigh` | f64 | 1 | `128x128` | 2201.138 | 1588.106 | 1.39x | torch |
| linalg | `f64_eigh` | f64 | 2 | `4x4` | 3.211 | 12.516 | 0.26x | tenferro |
| linalg | `f64_eigh` | f64 | 2 | `8x8` | 7.032 | 17.128 | 0.41x | tenferro |
| linalg | `f64_eigh` | f64 | 2 | `16x16` | 26.233 | 32.678 | 0.80x | tenferro |
| linalg | `f64_eigh` | f64 | 2 | `32x32` | 88.984 | 118.558 | 0.75x | tenferro |
| linalg | `f64_eigh` | f64 | 2 | `64x64` | 478.639 | 426.770 | 1.12x | torch |
| linalg | `f64_eigh` | f64 | 2 | `128x128` | 2066.426 | 1526.433 | 1.35x | torch |
| linalg | `f64_eigh` | f64 | 4 | `4x4` | 3.083 | 12.821 | 0.24x | tenferro |
| linalg | `f64_eigh` | f64 | 4 | `8x8` | 6.912 | 16.959 | 0.41x | tenferro |
| linalg | `f64_eigh` | f64 | 4 | `16x16` | 22.281 | 31.451 | 0.71x | tenferro |
| linalg | `f64_eigh` | f64 | 4 | `32x32` | 82.132 | 120.621 | 0.68x | tenferro |
| linalg | `f64_eigh` | f64 | 4 | `64x64` | 515.723 | 346.952 | 1.49x | torch |
| linalg | `f64_eigh` | f64 | 4 | `128x128` | 2122.134 | 1394.780 | 1.52x | torch |
| linalg | `f64_solve_column_rhs1` | f64 | 1 | `4x1` | 1.253 | 23.431 | 0.05x | tenferro |
| linalg | `f64_solve_column_rhs1` | f64 | 1 | `8x1` | 1.588 | 23.435 | 0.07x | tenferro |
| linalg | `f64_solve_column_rhs1` | f64 | 1 | `16x1` | 3.858 | 26.128 | 0.15x | tenferro |
| linalg | `f64_solve_column_rhs1` | f64 | 1 | `32x1` | 8.490 | 35.631 | 0.24x | tenferro |
| linalg | `f64_solve_column_rhs1` | f64 | 2 | `4x1` | 1.301 | 20.956 | 0.06x | tenferro |
| linalg | `f64_solve_column_rhs1` | f64 | 2 | `8x1` | 1.608 | 21.534 | 0.07x | tenferro |
| linalg | `f64_solve_column_rhs1` | f64 | 2 | `16x1` | 3.573 | 23.317 | 0.15x | tenferro |
| linalg | `f64_solve_column_rhs1` | f64 | 2 | `32x1` | 7.655 | 34.099 | 0.22x | tenferro |
| linalg | `f64_solve_column_rhs1` | f64 | 4 | `4x1` | 1.182 | 21.176 | 0.06x | tenferro |
| linalg | `f64_solve_column_rhs1` | f64 | 4 | `8x1` | 1.486 | 21.486 | 0.07x | tenferro |
| linalg | `f64_solve_column_rhs1` | f64 | 4 | `16x1` | 3.223 | 24.149 | 0.13x | tenferro |
| linalg | `f64_solve_column_rhs1` | f64 | 4 | `32x1` | 7.102 | 31.889 | 0.22x | tenferro |
| linalg | `f64_solve_matrix_rhs4` | f64 | 1 | `4x4` | 2.775 | 25.083 | 0.11x | tenferro |
| linalg | `f64_solve_matrix_rhs4` | f64 | 1 | `8x4` | 2.290 | 26.156 | 0.09x | tenferro |
| linalg | `f64_solve_matrix_rhs4` | f64 | 1 | `16x4` | 3.103 | 29.389 | 0.11x | tenferro |
| linalg | `f64_solve_matrix_rhs4` | f64 | 1 | `32x4` | 8.289 | 40.142 | 0.21x | tenferro |
| linalg | `f64_solve_matrix_rhs4` | f64 | 1 | `64x4` | 27.302 | 65.672 | 0.42x | tenferro |
| linalg | `f64_solve_matrix_rhs4` | f64 | 1 | `128x4` | 161.697 | 170.990 | 0.95x | tenferro |
| linalg | `f64_solve_matrix_rhs4` | f64 | 2 | `4x4` | 1.329 | 22.142 | 0.06x | tenferro |
| linalg | `f64_solve_matrix_rhs4` | f64 | 2 | `8x4` | 1.749 | 23.633 | 0.07x | tenferro |
| linalg | `f64_solve_matrix_rhs4` | f64 | 2 | `16x4` | 3.558 | 26.928 | 0.13x | tenferro |
| linalg | `f64_solve_matrix_rhs4` | f64 | 2 | `32x4` | 7.174 | 41.873 | 0.17x | tenferro |
| linalg | `f64_solve_matrix_rhs4` | f64 | 2 | `64x4` | 49.779 | 80.343 | 0.62x | tenferro |
| linalg | `f64_solve_matrix_rhs4` | f64 | 2 | `128x4` | 310.386 | 201.601 | 1.54x | torch |
| linalg | `f64_solve_matrix_rhs4` | f64 | 4 | `4x4` | 1.177 | 22.744 | 0.05x | tenferro |
| linalg | `f64_solve_matrix_rhs4` | f64 | 4 | `8x4` | 1.580 | 23.666 | 0.07x | tenferro |
| linalg | `f64_solve_matrix_rhs4` | f64 | 4 | `16x4` | 3.146 | 26.604 | 0.12x | tenferro |
| linalg | `f64_solve_matrix_rhs4` | f64 | 4 | `32x4` | 7.210 | 45.892 | 0.16x | tenferro |
| linalg | `f64_solve_matrix_rhs4` | f64 | 4 | `64x4` | 66.865 | 77.154 | 0.87x | tenferro |
| linalg | `f64_solve_matrix_rhs4` | f64 | 4 | `128x4` | 233.449 | 201.381 | 1.16x | torch |
| linalg | `c64_eigh` | c64 | 1 | `4x4` | 2.891 | 14.581 | 0.20x | tenferro |
| linalg | `c64_eigh` | c64 | 1 | `8x8` | 8.208 | 20.035 | 0.41x | tenferro |
| linalg | `c64_eigh` | c64 | 1 | `16x16` | 24.561 | 43.253 | 0.57x | tenferro |
| linalg | `c64_eigh` | c64 | 1 | `32x32` | 112.024 | 149.170 | 0.75x | tenferro |
| linalg | `c64_eigh` | c64 | 2 | `4x4` | 3.171 | 13.097 | 0.24x | tenferro |
| linalg | `c64_eigh` | c64 | 2 | `8x8` | 7.987 | 18.141 | 0.44x | tenferro |
| linalg | `c64_eigh` | c64 | 2 | `16x16` | 25.842 | 41.690 | 0.62x | tenferro |
| linalg | `c64_eigh` | c64 | 2 | `32x32` | 114.903 | 158.444 | 0.73x | tenferro |
| linalg | `c64_eigh` | c64 | 4 | `4x4` | 3.217 | 13.505 | 0.24x | tenferro |
| linalg | `c64_eigh` | c64 | 4 | `8x8` | 7.541 | 18.311 | 0.41x | tenferro |
| linalg | `c64_eigh` | c64 | 4 | `16x16` | 25.038 | 39.728 | 0.63x | tenferro |
| linalg | `c64_eigh` | c64 | 4 | `32x32` | 113.260 | 155.682 | 0.73x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 1 | `2x2x16` | 4.656 | 19.640 | 0.24x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 1 | `2x2x64` | 5.142 | 22.054 | 0.23x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 1 | `2x2x256` | 12.760 | 26.282 | 0.49x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 1 | `4x4x16` | 3.255 | 21.380 | 0.15x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 1 | `4x4x64` | 6.558 | 27.481 | 0.24x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 1 | `4x4x256` | 31.671 | 50.638 | 0.63x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 1 | `8x8x16` | 8.522 | 107.073 | 0.08x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 1 | `8x8x64` | 9.903 | 364.840 | 0.03x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 1 | `8x8x256` | 40.896 | 1406.283 | 0.03x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 1 | `16x16x16` | 9.689 | 141.774 | 0.07x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 1 | `16x16x64` | 35.161 | 498.093 | 0.07x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 1 | `16x16x256` | 111.427 | 1950.126 | 0.06x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 2 | `2x2x16` | 3.004 | 19.684 | 0.15x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 2 | `2x2x64` | 5.072 | 20.896 | 0.24x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 2 | `2x2x256` | 12.478 | 24.924 | 0.50x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 2 | `4x4x16` | 3.319 | 21.205 | 0.16x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 2 | `4x4x64` | 5.863 | 25.191 | 0.23x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 2 | `4x4x256` | 16.926 | 47.033 | 0.36x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 2 | `8x8x16` | 4.023 | 103.589 | 0.04x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 2 | `8x8x64` | 8.717 | 335.369 | 0.03x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 2 | `8x8x256` | 27.583 | 1488.444 | 0.02x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 2 | `16x16x16` | 9.223 | 133.165 | 0.07x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 2 | `16x16x64` | 32.316 | 453.731 | 0.07x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 2 | `16x16x256` | 108.969 | 1860.461 | 0.06x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 4 | `2x2x16` | 2.985 | 19.231 | 0.16x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 4 | `2x2x64` | 7.220 | 20.941 | 0.34x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 4 | `2x2x256` | 14.531 | 23.133 | 0.63x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 4 | `4x4x16` | 3.309 | 20.784 | 0.16x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 4 | `4x4x64` | 6.292 | 25.828 | 0.24x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 4 | `4x4x256` | 19.661 | 44.584 | 0.44x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 4 | `8x8x16` | 5.455 | 102.489 | 0.05x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 4 | `8x8x64` | 9.712 | 346.916 | 0.03x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 4 | `8x8x256` | 27.386 | 1280.667 | 0.02x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 4 | `16x16x16` | 9.324 | 134.258 | 0.07x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 4 | `16x16x64` | 36.371 | 460.721 | 0.08x | tenferro |
| batched_einsum_rightmost_batch | `f64_ikb_knb_to_inb` | f64 | 4 | `16x16x256` | 102.220 | 2113.694 | 0.05x | tenferro |
| einsum_patterns | `f64_binary_ij_jk_to_ik` | f64 | 1 | `64x64` | 21.514 | 49.824 | 0.43x | tenferro |
| einsum_patterns | `f64_binary_ij_jk_to_ik` | f64 | 2 | `64x64` | 40.110 | 66.289 | 0.61x | tenferro |
| einsum_patterns | `f64_binary_ij_jk_to_ik` | f64 | 4 | `64x64` | 36.210 | 56.845 | 0.64x | tenferro |
| einsum_patterns | `f64_chain_ij_jk_kl_to_il` | f64 | 1 | `64x64` | 49.646 | 87.487 | 0.57x | tenferro |
| einsum_patterns | `f64_chain_ij_jk_kl_to_il` | f64 | 2 | `64x64` | 81.315 | 125.273 | 0.65x | tenferro |
| einsum_patterns | `f64_chain_ij_jk_kl_to_il` | f64 | 4 | `64x64` | 100.516 | 106.778 | 0.94x | tenferro |
| einsum_patterns | `f64_multiedge_ijk_jkl_to_il` | f64 | 1 | `8x16x8__16x8x8` | 3.149 | 17.545 | 0.18x | tenferro |
| einsum_patterns | `f64_multiedge_ijk_jkl_to_il` | f64 | 2 | `8x16x8__16x8x8` | 3.319 | 21.011 | 0.16x | tenferro |
| einsum_patterns | `f64_multiedge_ijk_jkl_to_il` | f64 | 4 | `8x16x8__16x8x8` | 4.539 | 22.159 | 0.20x | tenferro |
| einsum_patterns | `c64_binary_ij_jk_to_ik` | c64 | 1 | `32x32` | 16.275 | 43.074 | 0.38x | tenferro |
| einsum_patterns | `c64_binary_ij_jk_to_ik` | c64 | 2 | `32x32` | 10.837 | 50.269 | 0.22x | tenferro |
| einsum_patterns | `c64_binary_ij_jk_to_ik` | c64 | 4 | `32x32` | 19.448 | 53.476 | 0.36x | tenferro |
| ad | `f64_grad_sum_svd_values` | f64 | 1 | `4x4` | 110.679 | 83.703 | 1.32x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 1 | `8x8` | 119.318 | 94.695 | 1.26x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 1 | `16x16` | 146.461 | 133.314 | 1.10x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 1 | `32x32` | 276.862 | 266.650 | 1.04x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 1 | `64x64` | 839.261 | 817.652 | 1.03x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 1 | `128x128` | 3796.990 | 3708.189 | 1.02x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 2 | `4x4` | 108.153 | 74.559 | 1.45x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 2 | `8x8` | 121.471 | 86.604 | 1.40x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 2 | `16x16` | 145.503 | 123.044 | 1.18x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 2 | `32x32` | 364.582 | 293.992 | 1.24x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 2 | `64x64` | 1393.896 | 1086.291 | 1.28x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 2 | `128x128` | 4159.568 | 4359.004 | 0.95x | tenferro |
| ad | `f64_grad_sum_svd_values` | f64 | 4 | `4x4` | 132.926 | 80.191 | 1.66x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 4 | `8x8` | 116.085 | 92.585 | 1.25x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 4 | `16x16` | 161.744 | 133.361 | 1.21x | torch |
| ad | `f64_grad_sum_svd_values` | f64 | 4 | `32x32` | 419.578 | 426.326 | 0.98x | tenferro |
| ad | `f64_grad_sum_svd_values` | f64 | 4 | `64x64` | 1053.240 | 1102.792 | 0.96x | tenferro |
| ad | `f64_grad_sum_svd_values` | f64 | 4 | `128x128` | 4051.628 | 4180.504 | 0.97x | tenferro |
| ad | `f64_grad_sum_matmul` | f64 | 1 | `4x4` | 52.688 | 58.551 | 0.90x | tenferro |
| ad | `f64_grad_sum_matmul` | f64 | 1 | `16x16` | 59.181 | 61.138 | 0.97x | tenferro |
| ad | `f64_grad_sum_matmul` | f64 | 1 | `64x64` | 132.613 | 146.561 | 0.90x | tenferro |
| ad | `f64_grad_sum_matmul` | f64 | 2 | `4x4` | 49.490 | 48.083 | 1.03x | torch |
| ad | `f64_grad_sum_matmul` | f64 | 2 | `16x16` | 60.727 | 54.462 | 1.12x | torch |
| ad | `f64_grad_sum_matmul` | f64 | 2 | `64x64` | 254.021 | 202.251 | 1.26x | torch |
| ad | `f64_grad_sum_matmul` | f64 | 4 | `4x4` | 53.243 | 52.372 | 1.02x | torch |
| ad | `f64_grad_sum_matmul` | f64 | 4 | `16x16` | 56.500 | 59.314 | 0.95x | tenferro |
| ad | `f64_grad_sum_matmul` | f64 | 4 | `64x64` | 231.322 | 199.666 | 1.16x | torch |
| ad | `f64_grad_sum_solve` | f64 | 1 | `4x1` | 57.191 | 107.013 | 0.53x | tenferro |
| ad | `f64_grad_sum_solve` | f64 | 1 | `16x1` | 69.958 | 111.985 | 0.62x | tenferro |
| ad | `f64_grad_sum_solve` | f64 | 2 | `4x1` | 82.525 | 94.779 | 0.87x | tenferro |
| ad | `f64_grad_sum_solve` | f64 | 2 | `16x1` | 101.421 | 103.210 | 0.98x | tenferro |
| ad | `f64_grad_sum_solve` | f64 | 4 | `4x1` | 99.574 | 101.767 | 0.98x | tenferro |
| ad | `f64_grad_sum_solve` | f64 | 4 | `16x1` | 99.551 | 105.708 | 0.94x | tenferro |

## Tenferro-Only Results

These are decomposition measurements without a Torch C++ counterpart in this
benchmark file.

| suite | benchmark | dtype | threads | shape | mean_us |
|---|---|---:|---:|---|---:|
| matmul | `c64_square` | c64 | 1 | `128x128` | 635.341 |
| matmul | `c64_square` | c64 | 2 | `128x128` | 295.996 |
| matmul | `c64_square` | c64 | 4 | `128x128` | 177.806 |
| linalg | `f64_solve_column_rhs1` | f64 | 1 | `64x1` | 23.176 |
| linalg | `f64_solve_column_rhs1` | f64 | 1 | `128x1` | 119.972 |
| linalg | `f64_solve_column_rhs1` | f64 | 2 | `64x1` | 73.785 |
| linalg | `f64_solve_column_rhs1` | f64 | 2 | `128x1` | 262.443 |
| linalg | `f64_solve_column_rhs1` | f64 | 4 | `64x1` | 76.915 |
| linalg | `f64_solve_column_rhs1` | f64 | 4 | `128x1` | 241.144 |
| ad | `f64_backward_only_reduce_sum` | f64 | 1 | `64x64` | 16.919 |
| ad | `f64_backward_only_reduce_sum` | f64 | 2 | `64x64` | 16.812 |
| ad | `f64_backward_only_reduce_sum` | f64 | 4 | `64x64` | 17.059 |
| ad | `f64_backward_only_sum_matmul` | f64 | 1 | `64x64` | 94.843 |
| ad | `f64_backward_only_sum_matmul` | f64 | 2 | `64x64` | 162.973 |
| ad | `f64_backward_only_sum_matmul` | f64 | 4 | `64x64` | 149.265 |
| ad | `f64_forward_matmul_sum_tracked` | f64 | 1 | `64x64` | 41.368 |
| ad | `f64_forward_matmul_sum_tracked` | f64 | 2 | `64x64` | 68.622 |
| ad | `f64_forward_matmul_sum_tracked` | f64 | 4 | `64x64` | 55.194 |
| ad | `f64_forward_matmul_sum_untracked` | f64 | 1 | `64x64` | 29.351 |
| ad | `f64_forward_matmul_sum_untracked` | f64 | 2 | `64x64` | 44.449 |
| ad | `f64_forward_matmul_sum_untracked` | f64 | 4 | `64x64` | 38.852 |
| ad | `f64_manual_grad_sum_matmul_math` | f64 | 1 | `64x64` | 47.680 |
| ad | `f64_manual_grad_sum_matmul_math` | f64 | 2 | `64x64` | 56.383 |
| ad | `f64_manual_grad_sum_matmul_math` | f64 | 4 | `64x64` | 60.032 |

## Torch-Only Results

None.
