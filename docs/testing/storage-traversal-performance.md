# Storage traversal performance

```json
{
  "schema": "tenferro.storage-traversal-performance.v1",
  "candidate_commit": "506e22dd9138585787723abe9dc20e05c8da0ade",
  "benchmark_path": "crates/tenferro-tensor/benches/element_access.rs",
  "baseline_obligation": "p1-element-access-baseline",
  "baseline_report": "docs/testing/storage-element-access-baseline.json",
  "baseline_measured_commit": "da7b36e699f9f4731dec08de6a4e1ca93f20cd6f",
  "command": "cargo bench --locked -p tenferro-tensor --bench element_access -- --warm-up-time 2 --measurement-time 5 --sample-size 100 --noplot",
  "environment": {
    "rustc_version": "rustc 1.97.1 (8bab26f4f 2026-07-14)",
    "cargo_version": "cargo 1.97.1 (c980f4866 2026-06-30)",
    "target": "x86_64-unknown-linux-gnu",
    "architecture": "x86_64",
    "os": "Linux 6.8.0-101-generic",
    "cpu_model": "AMD EPYC 7713P 64-Core Processor",
    "cpu_affinity": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42,
      43,
      44,
      45,
      46,
      47,
      48,
      49,
      50,
      51,
      52,
      53,
      54,
      55,
      56,
      57,
      58,
      59,
      60,
      61,
      62,
      63
    ],
    "thread_environment": {
      "RAYON_NUM_THREADS": "1",
      "OMP_NUM_THREADS": "1",
      "OPENBLAS_NUM_THREADS": "1",
      "MKL_NUM_THREADS": "1",
      "VECLIB_MAXIMUM_THREADS": "1"
    }
  },
  "sample_size": 100,
  "warm_up_seconds": 2.0,
  "measurement_seconds": 5.0,
  "medians_ns": {
    "contiguous_read": {
      "id": "linear_iteration/col_major/as_slice_iter",
      "estimate_ns": 54791.223115274785,
      "lower_bound_ns": 54471.234651507526,
      "upper_bound_ns": 55178.01350502327,
      "standard_error_ns": 181.1635041928464
    },
    "contiguous_write": {
      "id": "linear_iteration/col_major/tensor_iter_mut",
      "estimate_ns": 59340.648322516245,
      "lower_bound_ns": 57217.232416407795,
      "upper_bound_ns": 61639.805817926055,
      "standard_error_ns": 1132.1570225504292
    },
    "dynamic_contiguous": {
      "id": "linear_iteration/col_major/dynamic_tensor_iter",
      "estimate_ns": 54279.45595093962,
      "lower_bound_ns": 53931.85732392471,
      "upper_bound_ns": 54688.3338366264,
      "standard_error_ns": 194.16579638919907
    },
    "fixed_rank": {
      "id": "rank_fixed/2d/col_major/get2/4096",
      "estimate_ns": 31318.118530341148,
      "lower_bound_ns": 30972.926304470624,
      "upper_bound_ns": 31691.08446323086,
      "standard_error_ns": 183.85495829878073
    },
    "strided": {
      "id": "strided_traversal/rectangular_transpose/logical_order_get/3840",
      "estimate_ns": 14809.47845198695,
      "lower_bound_ns": 14731.783106747373,
      "upper_bound_ns": 14900.715755683097,
      "standard_error_ns": 43.25854253301044
    },
    "empty": {
      "id": "linear_iteration/col_major/empty",
      "estimate_ns": 0.5516940584489786,
      "lower_bound_ns": 0.5482323916586962,
      "upper_bound_ns": 0.5556784635494348,
      "standard_error_ns": 0.0019097551646253643
    }
  },
  "comparisons": {
    "contiguous_read": {
      "baseline_ns": 54986.53687097039,
      "current_ns": 54791.223115274785,
      "ratio": 0.9964479713251642,
      "limit": 1.1
    },
    "dynamic_contiguous": {
      "baseline_ns": 54426.40404333462,
      "current_ns": 54279.45595093962,
      "ratio": 0.9973000587678362,
      "limit": 1.15
    },
    "fixed_rank": {
      "baseline_ns": 28555.716496630084,
      "current_ns": 31318.118530341148,
      "ratio": 1.0967372691922144,
      "limit": 1.15
    },
    "strided": {
      "baseline_ns": 14849.827750481152,
      "current_ns": 14809.47845198695,
      "ratio": 0.9972828440051842,
      "limit": 1.15
    }
  },
  "setup_measurements": {
    "prepare_map_bind_dispatch": "not part of element_access benchmark; covered by storage preparation contracts"
  },
  "result": "pass",
  "reason": "comparable traversal cases satisfy the documented limits; new write/empty cases are recorded without a historical baseline"
}
```
