# RunPod full CI cost reduction (#2002)

## Decisions

- Record stage costs and cache hits from the trusted execution workflow after
  mandatory deletion. Preserve failed workload conclusions and unknown cache
  outputs; report an estimate rather than a provider invoice. Telemetry must
  never delay or block pod deletion.
- Investigate hosted runtime preparation, slimmer CUDA images and GPU selection
  using the complete frozen CUDA/PJRT/tutorial workload. Experimental benchmark
  workflows do not alter production until complete-workload evidence supports
  promotion. Keep failed pilots and their expense visible.

## Verification conclusions and constraints

- Original production observation 37783818267: A40 at $0.49/hour, 577.430 paid
  seconds, estimated $0.07859; all 285 CUDA cases, three PJRT cases and tutorial
  passed. This single observation motivates work; it does not prove savings.
- Native CUDA PyTorch image diagnostic 37788536318 passed the same full workload
  but cost $0.08658 for 636.124 seconds. Do not promote this larger image.
- Slim-image experiments are ongoing in tenferro-benchmark branch
  experiment/runpod-ci-cost. Primary paired cost confirmation has not started.
- The focused CI helper gate passed; pagination, failure preservation, cache
  unknowns, exact stage reconciliation and nonblocking invalid metadata are
  covered. No Rust numerical or device-execution semantics are changed by the
  telemetry helper.

- Need measurement for the host SVD checker: production run 37783818267 spent
  83.584s and 83.900s in the tall/wide full-SVD-above-1024 cases, about 45%
  of its CUDA test duration. Their host oracle calculates both orientations of
  every Gram entry with per-element Complex64 iterator arithmetic. Threshold
  shape and all acceptance identities must remain; do not sample columns or
  remove the row-unitarity checks.
- Candidate direction: compute each unordered column pair once (Hermitian Gram
  symmetry), use contiguous column slices and scalar real/imaginary dot sums.
  Retain the existing norm tolerance and both U/Vt left/right identities.
  CPU tests compare with the complete original complex Gram reference and
  reject changed entries and nonfinite matrices. Full paid comparison still
  uses all 285 CUDA cases, three PJRT cases and tutorial, serial, profile ci.
  Baseline source is 3f10f960af05efa4fc5705aac3dafbac1ae2269e; freeze the
  committed candidate and matching workflow configuration before any candidate
  measurement. Three complete alternating pairs, median cost >=20% reduction,
  every pair nonregressing and within-arm max/min <=1.5 remain the primary
  acceptance gates. No speedup is inferred from the static change alone.

- The extracted checker passed two CPU integration tests in the ordinary test
  profile and the same source module's tests compiled with rustc -O against the
  existing release num-complex dependency. This checks numerical parity in
  optimized code; it is not a runtime speed measurement or a full release GPU
  test. The code-change PR gate, including CI-parity clippy, formatting and
  293 CI helper tests, passed. Numerical library code and GPU kernels are
  unchanged; only test-oracle arithmetic and reporting are changed.
