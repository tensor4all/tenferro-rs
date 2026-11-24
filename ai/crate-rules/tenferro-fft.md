# tenferro-fft crate rules

Template and policy: [`README.md`](README.md).

### Adding, renaming, or removing a test is not done until the GPU partition names it

- **Applies to:** every test of this crate that is compiled into the GPU
  gate's archive, CPU-only tests included. The archive is built with (abridged)
  `cargo nextest archive -p tenferro-gpu -p tenferro-ad -p tenferro-linalg -p tenferro-fft -p tenferro-einsum --features cuda,tenferro-einsum/autodiff`
  (`.github/workflows/runpod-gpu-test.yml`, "Build CUDA test archive"), so
  tests behind `#![cfg(feature = "autodiff")]` here (`tests/eager_allocations.rs`,
  `tests/eager_in_place.rs`, `tests/real_fft_ad.rs`) are not in it and must
  not get a line: the gate also fails on `absent` rows. Every test that is in
  it needs exactly one line in `scripts/ci/cuda_test_partition.tsv`:
  `lane<TAB>binary-id<TAB>test-name`. The binary id is `tenferro-fft` for
  unit tests in `src/` and `tenferro-fft::<file stem>` for
  `tests/<file stem>.rs`; the lane is `host` for a test that never opens a
  device. The same holds for the other four crates in that archive, and
  `tenferro-xla` tests (`--features pjrt`) go in
  `scripts/ci/pjrt_test_partition.tsv`.
- **Check:** `scripts/ci/gpu_test_partition.py` compares the archive with the
  partition and refuses to run on any difference, with
  `GPU test inventory changed; audit test bodies/helpers and update the partition. Unclassified: [...]; absent: [...]`.
  It runs only in the GPU gate, after the archive build, so catch it before
  pushing: for each archived test you added, renamed or removed in this crate,
  check its line. The line the incident below was missing would be checked with
  `grep -P '\ttenferro-fft::cpu_reuse\tunrecognized_fft_provider_is_an_error$' scripts/ci/cuda_test_partition.tsv`.
  Classify by reading the test body and its helpers, as the file header asks,
  not by the test name.
- **Incident:**
  [tenferro-rs#2677](https://github.com/tensor4all/tenferro-rs/pull/2677#issuecomment-7130806179)
  -- the RunPod gate stopped before provisioning because the archive held a new
  test the partition did not name:
  `Unclassified: [('tenferro-fft::cpu_reuse', 'unrecognized_fft_provider_is_an_error')]`.
