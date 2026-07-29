# Elementwise Into Defaults

## Summary

Fixed #1504 by routing the six existing elementwise `*_read_into` methods
through one `TensorElementwise::elementwise_read_into` hook. Equal-shape,
equal-dtype host operations use strided-kernel's compile-free erased one-shot
map/zip entry points. Promotion, broadcasting, and backend storage preserve
the previous allocate-plus-copy fallback.

`CpuBackend` and `CpuExecSession` override only the shared hook. Both enter the
owned CPU execution domain and map its thread budget to an explicit strided
`ExecContext`; neither uses `ExecContext::ambient()`. Dispatch wrappers such
as `EagerBackend` forward the hook to the selected backend so they retain that
backend's execution context.

## Dependency

The workspace strided-rs pin advances from
`06c825829e399dd769504e6ce10dbca8f07b5e15` to the merged W1 commit
`649772c8402e5fe95335366326b6623f9a4f5b0a`. `tenferro-tensor` depends only on
the dtype-erased `strided-kernel` API.

## Safety And Semantics

- The hook validates destination/input storage disjointness before fallback or
  mutation. Shared backend allocations return a typed invalid-argument error.
- The strided one-shot boundary validates raw host buffer overlap before
  forming input references.
- Caller-owned strided output views retain their backing storage; tests mutate
  the backing storage after the operation returns.
- Tensor trait defaults use `ExecContext::serial()`. CPU implementations
  replace it with the runtime-owned bounded context.
- #1502 remains separate. A later caller-owned linear-solve destination should
  reuse this hook pattern rather than add another default execution boundary.

## Benchmark Evidence

Focused public API rows were measured sequentially on the same 64-core Linux
host with `cpu-faer`, `PUBLICATION_GATE_PROFILE=full`, 15 measured runs, three
warmups, and the same release target cache. The exact allocating counterparts
were measured in the same process configuration.

| row | before t1 ms | after t1 ms | before t4 ms | after t4 ms |
| --- | ---: | ---: | ---: | ---: |
| `add_into` | 220.698 | 53.079 | 79.108 | 16.033 |
| `sub_into` | 223.976 | 61.362 | 81.765 | 17.414 |
| `mul_into` | 227.839 | 94.802 | 83.214 | 25.662 |
| `div_into` | 220.185 | 84.867 | 74.667 | 23.396 |
| `neg_into` | 208.688 | 39.227 | 75.061 | 11.307 |
| `conj_into` | 209.528 | 25.892 | 74.583 | 11.953 |

All six `*_into` rows now beat their allocating counterparts at t1 and t4.
The allocating rows changed between about -8.5% and +4.5%; every change
remains far below the +20% stop-the-line gate. The final formal
cross-framework gate remains W10.

Raw focused results:

- baseline:
  `tenferro-benchmark/data/w2-baseline-output-reuse-20260729.csv`
- candidate:
  `tenferro-benchmark/data/w2-candidate-final-output-reuse-20260729.csv`

## Verification

```bash
python3 scripts/ci/run_profile.py fmt
cargo test -p tenferro-tensor
cargo test -p tenferro-cpu --test integration
cargo test -p tenferro-ad --lib eager_backend
```

Before PR creation, this work also runs `scripts/check-pr-fast.sh` and
`scripts/repository-rules-review.py` under the repository routing policy.
