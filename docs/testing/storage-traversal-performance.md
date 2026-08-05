# Storage traversal performance

```json
{
  "schema": "tenferro.storage-traversal-performance.v1",
  "candidate_commit": "71222c11ce54af81e553574c2977feaeda39c14c",
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
      "estimate_ns": 54467.185751637706,
      "lower_bound_ns": 54245.460346899,
      "upper_bound_ns": 54716.88154039882,
      "standard_error_ns": 120.78488613399762
    },
    "contiguous_write": {
      "id": "linear_iteration/col_major/tensor_iter_mut",
      "estimate_ns": 55915.8596099555,
      "lower_bound_ns": 55369.24152594092,
      "upper_bound_ns": 56523.6592592655,
      "standard_error_ns": 295.07365077326676
    },
    "dynamic_contiguous": {
      "id": "linear_iteration/col_major/dynamic_tensor_iter",
      "estimate_ns": 55175.21157468861,
      "lower_bound_ns": 54845.4201643725,
      "upper_bound_ns": 55546.04813493421,
      "standard_error_ns": 179.56729022275917
    },
    "fixed_rank": {
      "id": "rank_fixed/2d/col_major/get2/4096",
      "estimate_ns": 31073.275416778433,
      "lower_bound_ns": 30793.344821987044,
      "upper_bound_ns": 31378.77858792707,
      "standard_error_ns": 149.8542425511762
    },
    "strided": {
      "id": "strided_traversal/rectangular_transpose/logical_order_get/3840",
      "estimate_ns": 15089.423082702686,
      "lower_bound_ns": 14984.114328721169,
      "upper_bound_ns": 15209.397486645481,
      "standard_error_ns": 57.776492559073155
    },
    "empty": {
      "id": "linear_iteration/col_major/empty",
      "estimate_ns": 0.5642676406772166,
      "lower_bound_ns": 0.5597979750680867,
      "upper_bound_ns": 0.5691413886770718,
      "standard_error_ns": 0.0023922762395502347
    }
  },
  "comparisons": {
    "contiguous_read": {
      "baseline_ns": 54986.53687097039,
      "current_ns": 54467.185751637706,
      "ratio": 0.9905549403747435,
      "limit": 1.1
    },
    "dynamic_contiguous": {
      "baseline_ns": 54426.40404333462,
      "current_ns": 55175.21157468861,
      "ratio": 1.0137581665464759,
      "limit": 1.15
    },
    "fixed_rank": {
      "baseline_ns": 28555.716496630084,
      "current_ns": 31073.275416778433,
      "ratio": 1.0881630450577366,
      "limit": 1.15
    },
    "strided": {
      "baseline_ns": 14849.827750481152,
      "current_ns": 15089.423082702686,
      "ratio": 1.0161345529555903,
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
