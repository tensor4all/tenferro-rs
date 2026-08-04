# Frozen storage hardware matrix

Unavailable hardware is a structured skip, never a pass.

```json
{
  "schema": "tenferro.storage-hardware-matrix.v1",
  "candidate_commit": "653a6449c6f40aff2e6b2a6407b124cedcff76b5",
  "required_lanes": [
    "cpu",
    "cuda2",
    "webgpu",
    "metal",
    "cuda-ad"
  ],
  "required_mode": false,
  "status": "structured-skip",
  "environment": {
    "host": "x86_64 AMD EPYC 7713P 64-Core Processor; Linux 6.8.0-101-generic",
    "python": "3.12.11"
  },
  "lanes": [
    {
      "lane": "cpu",
      "status": "pass",
      "command": "cargo test -p tenferro-tensor --test storage_public_api",
      "environment": "x86_64 AMD EPYC 7713P 64-Core Processor; Linux 6.8.0-101-generic",
      "device_facts": "host CPU (see environment)",
      "test_count": 3,
      "passed": 3,
      "failed": 0,
      "ignored": 0,
      "duration_seconds": 0.159,
      "output_tail": "    Finished `test` profile [unoptimized] target(s) in 0.12s\n     Running tests/storage_public_api.rs (target/debug/deps/storage_public_api-eff000170450a05f)\n\nrunning 3 tests\ntest dtype_erased_views_have_explicit_duplicate_boundaries ... ok\ntest canonical_owner_view_and_mutable_view_surface_is_available ... ok\ntest provider_exports_and_transfer_defaults_are_normalized ... ok\n\ntest result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n\n",
      "evidence": "crates/tenferro-tensor/tests/storage_public_api.rs",
      "skip_reason": null
    },
    {
      "lane": "cuda2",
      "status": "pass",
      "command": "cargo test -p tenferro-gpu --features cuda --test storage_provider_cuda -- --nocapture",
      "environment": "x86_64 AMD EPYC 7713P 64-Core Processor; Linux 6.8.0-101-generic",
      "device_facts": "NVIDIA CUDA device(s), queried by the provider test",
      "test_count": 4,
      "passed": 4,
      "failed": 0,
      "ignored": 0,
      "duration_seconds": 9.936,
      "output_tail": "   Compiling tenferro-tensor v0.2.0 (/home/shinaoka/tensor4all/tenferro-rs/.worktrees/issue-1558-task2-corrections/crates/tenferro-tensor)\n   Compiling tenferro-internal-ops v0.2.0 (/home/shinaoka/tensor4all/tenferro-rs/.worktrees/issue-1558-task2-corrections/crates/tenferro-internal-ops)\n   Compiling tenferro-internal-cpu-kernels v0.2.0 (/home/shinaoka/tensor4all/tenferro-rs/.worktrees/issue-1558-task2-corrections/crates/tenferro-internal-cpu-kernels)\n   Compiling tenferro-runtime v0.2.0 (/home/shinaoka/tensor4all/tenferro-rs/.worktrees/issue-1558-task2-corrections/crates/tenferro-runtime)\n   Compiling tenferro-cpu v0.2.0 (/home/shinaoka/tensor4all/tenferro-rs/.worktrees/issue-1558-task2-corrections/crates/tenferro-cpu)\n   Compiling tenferro-gpu v0.2.0 (/home/shinaoka/tensor4all/tenferro-rs/.worktrees/issue-1558-task2-corrections/crates/tenferro-gpu)\n    Finished `test` profile [unoptimized] target(s) in 9.18s\n     Running tests/storage_provider_cuda.rs (target/debug/deps/storage_provider_cuda-ac608892ec09a7a1)\n\nrunning 4 tests\ntest cuda_provider_does_not_expose_safe_unscoped_raw_access ... ok\ntest cuda_prepared_state_is_consumed_by_the_exact_binding_without_host_mapping ... ok\ntest cuda_tensor_view_keeps_the_single_root_identity ... ok\ntest cuda_duplicate_is_explicit_same_placement_allocation ... ok\n\ntest result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.56s\n\n",
      "evidence": "crates/tenferro-gpu/tests/storage_provider_cuda.rs",
      "skip_reason": null
    },
    {
      "lane": "webgpu",
      "status": "pass",
      "command": "cargo test -p tenferro-gpu --features webgpu --test storage_provider_webgpu -- --nocapture",
      "environment": "x86_64 AMD EPYC 7713P 64-Core Processor; Linux 6.8.0-101-generic",
      "device_facts": "wgpu adapter, queried by WebGpuRuntime::new_default",
      "test_count": 3,
      "passed": 3,
      "failed": 0,
      "ignored": 0,
      "duration_seconds": 8.424,
      "output_tail": "   Compiling tenferro-tensor v0.2.0 (/home/shinaoka/tensor4all/tenferro-rs/.worktrees/issue-1558-task2-corrections/crates/tenferro-tensor)\n   Compiling tenferro-internal-ops v0.2.0 (/home/shinaoka/tensor4all/tenferro-rs/.worktrees/issue-1558-task2-corrections/crates/tenferro-internal-ops)\n   Compiling tenferro-internal-cpu-kernels v0.2.0 (/home/shinaoka/tensor4all/tenferro-rs/.worktrees/issue-1558-task2-corrections/crates/tenferro-internal-cpu-kernels)\n   Compiling tenferro-runtime v0.2.0 (/home/shinaoka/tensor4all/tenferro-rs/.worktrees/issue-1558-task2-corrections/crates/tenferro-runtime)\n   Compiling tenferro-cpu v0.2.0 (/home/shinaoka/tensor4all/tenferro-rs/.worktrees/issue-1558-task2-corrections/crates/tenferro-cpu)\n   Compiling tenferro-gpu v0.2.0 (/home/shinaoka/tensor4all/tenferro-rs/.worktrees/issue-1558-task2-corrections/crates/tenferro-gpu)\n    Finished `test` profile [unoptimized] target(s) in 8.08s\n     Running tests/storage_provider_webgpu.rs (target/debug/deps/storage_provider_webgpu-ae7c5e648e7360b1)\n\nrunning 3 tests\ntest empty_upload_keeps_a_zero_logical_root_span ... ok\ntest device_local_host_mapping_is_rejected_without_an_implicit_download ... ok\ntest uploaded_storage_is_root_owned_and_prepares_once_at_the_descriptor_boundary ... ok\n\ntest result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.24s\n\n",
      "evidence": "crates/tenferro-gpu/tests/storage_provider_webgpu.rs",
      "skip_reason": null
    },
    {
      "lane": "metal",
      "status": "skip",
      "command": "cargo test -p tenferro-gpu --features webgpu --test integration -- apple --nocapture",
      "environment": "x86_64 AMD EPYC 7713P 64-Core Processor; Linux 6.8.0-101-generic",
      "device_facts": "Apple Metal device, required on macOS",
      "test_count": 0,
      "passed": 0,
      "failed": 0,
      "ignored": 0,
      "duration_seconds": 2.547,
      "output_tail": "   Compiling tenferro-gpu v0.2.0 (/home/shinaoka/tensor4all/tenferro-rs/.worktrees/issue-1558-task2-corrections/crates/tenferro-gpu)\n    Finished `test` profile [unoptimized] target(s) in 2.48s\n     Running tests/integration.rs (target/debug/deps/integration-0c8f9ea7d611b078)\n\nrunning 0 tests\n\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 87 filtered out; finished in 0.00s\n\n",
      "evidence": "crates/tenferro-gpu/tests/integration/apple_context.rs",
      "skip_reason": "no tests ran for this platform or no provider device was available"
    },
    {
      "lane": "cuda-ad",
      "status": "pass",
      "command": "cargo test -p tenferro-ad --features cuda --test integration -- gpu_ad_tests --nocapture",
      "environment": "x86_64 AMD EPYC 7713P 64-Core Processor; Linux 6.8.0-101-generic",
      "device_facts": "NVIDIA CUDA device used by AD integration tests",
      "test_count": 2,
      "passed": 2,
      "failed": 0,
      "ignored": 0,
      "duration_seconds": 10.766,
      "output_tail": "   Compiling tenferro-internal-ops v0.2.0 (/home/shinaoka/tensor4all/tenferro-rs/.worktrees/issue-1558-task2-corrections/crates/tenferro-internal-ops)\n   Compiling tenferro-runtime v0.2.0 (/home/shinaoka/tensor4all/tenferro-rs/.worktrees/issue-1558-task2-corrections/crates/tenferro-runtime)\n   Compiling tenferro-cpu v0.2.0 (/home/shinaoka/tensor4all/tenferro-rs/.worktrees/issue-1558-task2-corrections/crates/tenferro-cpu)\n   Compiling tenferro-gpu v0.2.0 (/home/shinaoka/tensor4all/tenferro-rs/.worktrees/issue-1558-task2-corrections/crates/tenferro-gpu)\n   Compiling tenferro-ad v0.2.0 (/home/shinaoka/tensor4all/tenferro-rs/.worktrees/issue-1558-task2-corrections/crates/tenferro-ad)\n    Finished `test` profile [unoptimized] target(s) in 9.84s\n     Running tests/integration.rs (target/debug/deps/integration-dfa12a7a1c7db902)\n\nrunning 2 tests\ntest gpu_ad_tests::test_gpu_eager_backward_smoke ... ok\ntest gpu_ad_tests::test_gpu_matmul_vjp ... ok\n\ntest result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 333 filtered out; finished in 0.69s\n\n",
      "evidence": "crates/tenferro-ad/tests/integration/gpu_ad_tests.rs",
      "skip_reason": null
    }
  ],
  "evidence_paths": [
    "crates/tenferro-ad/tests/integration/gpu_ad_tests.rs",
    "crates/tenferro-gpu/tests/integration/apple_context.rs",
    "crates/tenferro-gpu/tests/storage_provider_cuda.rs",
    "crates/tenferro-gpu/tests/storage_provider_webgpu.rs",
    "crates/tenferro-tensor/tests/storage_public_api.rs"
  ]
}
```
