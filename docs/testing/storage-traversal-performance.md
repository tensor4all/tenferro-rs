# Storage traversal performance

```json
{
  "schema": "tenferro.storage-traversal-performance.v1",
  "candidate_commit": "402c962c61543f1477e3e3e0ade2c293b9d05ad4",
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
      "estimate_ns": 54846.56127470774,
      "lower_bound_ns": 54514.55887597067,
      "upper_bound_ns": 55240.21149230169,
      "standard_error_ns": 186.10167325857273
    },
    "contiguous_write": {
      "id": "linear_iteration/col_major/tensor_iter_mut",
      "estimate_ns": 59113.063622304595,
      "lower_bound_ns": 57094.60430252396,
      "upper_bound_ns": 61302.0076460078,
      "standard_error_ns": 1073.655387821805
    },
    "dynamic_contiguous": {
      "id": "linear_iteration/col_major/dynamic_tensor_iter",
      "estimate_ns": 54254.06540186487,
      "lower_bound_ns": 54057.45094402752,
      "upper_bound_ns": 54493.34307805127,
      "standard_error_ns": 111.84920441218866
    },
    "fixed_rank": {
      "id": "rank_fixed/2d/col_major/get2/4096",
      "estimate_ns": 30619.94353344338,
      "lower_bound_ns": 30452.43092127707,
      "upper_bound_ns": 30810.885490537705,
      "standard_error_ns": 91.71017189429922
    },
    "strided": {
      "id": "strided_traversal/rectangular_transpose/logical_order_get/3840",
      "estimate_ns": 14855.636463682342,
      "lower_bound_ns": 14774.407783681874,
      "upper_bound_ns": 14947.45154553069,
      "standard_error_ns": 44.37531890039062
    },
    "empty": {
      "id": "linear_iteration/col_major/empty",
      "estimate_ns": 0.5516471592200741,
      "lower_bound_ns": 0.5484793691564211,
      "upper_bound_ns": 0.555388511180491,
      "standard_error_ns": 0.0017691733979960233
    }
  },
  "comparisons": {
    "contiguous_read": {
      "baseline_ns": 54986.53687097039,
      "current_ns": 54846.56127470774,
      "ratio": 0.9974543660279768,
      "limit": 1.1
    },
    "dynamic_contiguous": {
      "baseline_ns": 54426.40404333462,
      "current_ns": 54254.06540186487,
      "ratio": 0.9968335471633853,
      "limit": 1.15
    },
    "fixed_rank": {
      "baseline_ns": 28555.716496630084,
      "current_ns": 30619.94353344338,
      "ratio": 1.072287698929106,
      "limit": 1.15
    },
    "strided": {
      "baseline_ns": 14849.827750481152,
      "current_ns": 14855.636463682342,
      "ratio": 1.0003911636753497,
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
