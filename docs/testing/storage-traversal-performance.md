# Storage traversal performance

```json
{
  "schema": "tenferro.storage-traversal-performance.v1",
  "candidate_commit": "e8c7d3019df316424de12dc6d69b3f67bce87e14",
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
      "estimate_ns": 59026.225575458084,
      "lower_bound_ns": 58351.51102772824,
      "upper_bound_ns": 59702.98827959148,
      "standard_error_ns": 345.51034563948775
    },
    "contiguous_write": {
      "id": "linear_iteration/col_major/tensor_iter_mut",
      "estimate_ns": 58555.60213489787,
      "lower_bound_ns": 57734.880386698685,
      "upper_bound_ns": 59388.67594857676,
      "standard_error_ns": 422.6171080676672
    },
    "dynamic_contiguous": {
      "id": "linear_iteration/col_major/dynamic_tensor_iter",
      "estimate_ns": 59242.59266978542,
      "lower_bound_ns": 58655.72653587723,
      "upper_bound_ns": 59829.001448781906,
      "standard_error_ns": 299.96625187914856
    },
    "fixed_rank": {
      "id": "rank_fixed/2d/col_major/get2/4096",
      "estimate_ns": 30879.31415115523,
      "lower_bound_ns": 30541.770327103302,
      "upper_bound_ns": 31212.96113800486,
      "standard_error_ns": 171.41969028593266
    },
    "strided": {
      "id": "strided_traversal/rectangular_transpose/logical_order_get/3840",
      "estimate_ns": 16159.613766825232,
      "lower_bound_ns": 15988.841080795422,
      "upper_bound_ns": 16330.077963313262,
      "standard_error_ns": 86.81650348612419
    },
    "empty": {
      "id": "linear_iteration/col_major/empty",
      "estimate_ns": 0.6000441444893193,
      "lower_bound_ns": 0.5937724722106618,
      "upper_bound_ns": 0.6062836212720286,
      "standard_error_ns": 0.0031939285899349693
    }
  },
  "comparisons": {
    "contiguous_read": {
      "baseline_ns": 54986.53687097039,
      "current_ns": 59026.225575458084,
      "ratio": 1.0734668690622777,
      "limit": 1.1
    },
    "dynamic_contiguous": {
      "baseline_ns": 54426.40404333462,
      "current_ns": 59242.59266978542,
      "ratio": 1.0884899289436083,
      "limit": 1.15
    },
    "fixed_rank": {
      "baseline_ns": 28555.716496630084,
      "current_ns": 30879.31415115523,
      "ratio": 1.0813706654777637,
      "limit": 1.15
    },
    "strided": {
      "baseline_ns": 14849.827750481152,
      "current_ns": 16159.613766825232,
      "ratio": 1.0882021016238146,
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
