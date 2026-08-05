# Frozen storage hardware matrix

Unavailable hardware is a structured skip, never a pass.

```json
{
  "schema": "tenferro.storage-hardware-matrix.v1",
  "candidate_commit": "652b5c45f753f04425d71541b387acedc39cfa04",
  "required_lanes": [
    "cpu",
    "cuda2",
    "webgpu",
    "metal",
    "cuda-ad"
  ],
  "required_mode": true,
  "complete": true,
  "status": "pass",
  "environment": {
    "hosts": [
      "arm64 Apple M5 Max; Darwin 25.5.0",
      "x86_64 AMD EPYC 7713P 64-Core Processor; Linux 6.8.0-101-generic"
    ]
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
      "duration_seconds": 0.17,
      "output_tail": "    Finished `test` profile [unoptimized] target(s) in 0.13s\n     Running tests/storage_public_api.rs (target/debug/deps/storage_public_api-e3637038f40271a2)\n\nrunning 3 tests\ntest dtype_erased_views_have_explicit_duplicate_boundaries ... ok\ntest canonical_owner_view_and_mutable_view_surface_is_available ... ok\ntest provider_exports_and_transfer_defaults_are_normalized ... ok\n\ntest result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n\n",
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
      "duration_seconds": 0.917,
      "output_tail": "    Finished `test` profile [unoptimized] target(s) in 0.19s\n     Running tests/storage_provider_cuda.rs (target/debug/deps/storage_provider_cuda-69ba1cd28c6a5de7)\n\nrunning 4 tests\ntest cuda_provider_does_not_expose_safe_unscoped_raw_access ... ok\ntest cuda_tensor_view_keeps_the_single_root_identity ... ok\ntest cuda_prepared_state_is_consumed_by_the_exact_binding_without_host_mapping ... ok\ntest cuda_duplicate_is_explicit_same_placement_allocation ... ok\n\ntest result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.56s\n\n",
      "evidence": "crates/tenferro-gpu/tests/storage_provider_cuda.rs",
      "skip_reason": null
    },
    {
      "lane": "webgpu",
      "status": "pass",
      "command": "cargo test -p tenferro-gpu --features webgpu --test storage_provider_webgpu -- --nocapture",
      "environment": "arm64 Apple M5 Max; Darwin 25.5.0",
      "device_facts": "wgpu adapter, queried by WebGpuRuntime::new_default",
      "test_count": 3,
      "passed": 3,
      "failed": 0,
      "ignored": 0,
      "duration_seconds": 1.652,
      "output_tail": "   Compiling tenferro-gpu v0.2.0 (/Users/hiroshi/projects/tensor4all/tenferro-rs-1617-mac/crates/tenferro-gpu)\n    Finished `test` profile [unoptimized] target(s) in 0.90s\n     Running tests/storage_provider_webgpu.rs (target/debug/deps/storage_provider_webgpu-ef93ec8fc3a1246d)\n\nrunning 3 tests\ntest empty_upload_keeps_a_zero_logical_root_span ... ok\ntest device_local_host_mapping_is_rejected_without_an_implicit_download ... ok\ntest uploaded_storage_is_root_owned_and_prepares_once_at_the_descriptor_boundary ... ok\n\ntest result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.03s\n\n",
      "evidence": "crates/tenferro-gpu/tests/storage_provider_webgpu.rs",
      "skip_reason": null
    },
    {
      "lane": "metal",
      "status": "pass",
      "command": "cargo test -p tenferro-gpu --features webgpu --test integration -- apple --nocapture",
      "environment": "arm64 Apple M5 Max; Darwin 25.5.0",
      "device_facts": "Apple Metal device, required on macOS",
      "test_count": 4,
      "passed": 4,
      "failed": 0,
      "ignored": 0,
      "duration_seconds": 64.464,
      "output_tail": "ling tenferro-internal-cpu-kernels v0.2.0 (/Users/hiroshi/projects/tensor4all/tenferro-rs-1617-mac/crates/tenferro-internal-cpu-kernels)\n   Compiling nano-gemm v0.2.2\n   Compiling trybuild v1.0.120\n   Compiling gemm-f64 v0.19.0\n   Compiling gemm-c64 v0.19.0\n   Compiling gemm-f32 v0.19.0\n   Compiling gemm-c32 v0.19.0\n   Compiling gemm v0.19.0\n   Compiling faer v0.24.4\n   Compiling tenferro-runtime v0.2.0 (/Users/hiroshi/projects/tensor4all/tenferro-rs-1617-mac/crates/tenferro-runtime)\n   Compiling t4a-cubecl-core v0.10.0 (https://github.com/tensor4all/cubecl.git?rev=1c88bb6f1a47ffb11755e05048b7828a743f53e1#1c88bb6f)\n   Compiling t4a-cubecl-std v0.10.0 (https://github.com/tensor4all/cubecl.git?rev=1c88bb6f1a47ffb11755e05048b7828a743f53e1#1c88bb6f)\n   Compiling t4a-cubek-std v0.2.0 (https://github.com/tensor4all/cubek.git?rev=5535bda85c68b1d286f1e4660fa15f7b0eb5cf04#5535bda8)\n   Compiling cubek-fft v0.2.0 (https://github.com/tensor4all/cubek.git?rev=5535bda85c68b1d286f1e4660fa15f7b0eb5cf04#5535bda8)\n   Compiling t4a-cubek-matmul v0.2.0 (https://github.com/tensor4all/cubek.git?rev=5535bda85c68b1d286f1e4660fa15f7b0eb5cf04#5535bda8)\n   Compiling tenferro-cpu v0.2.0 (/Users/hiroshi/projects/tensor4all/tenferro-rs-1617-mac/crates/tenferro-cpu)\n   Compiling tenferro-gpu v0.2.0 (/Users/hiroshi/projects/tensor4all/tenferro-rs-1617-mac/crates/tenferro-gpu)\n    Finished `test` profile [unoptimized] target(s) in 1m 01s\n     Running tests/integration.rs (target/debug/deps/integration-fe506db7fc7f79d9)\n\nrunning 4 tests\ntest apple_context::cpu_domain_allocator_produces_write_only_managed_outputs_without_transfers ... ok\ntest apple_context::independent_contexts_reject_foreign_managed_allocations ... ok\ntest apple_context::managed_upload_maps_without_post_creation_transfers_and_keeps_identity ... ok\ntest apple_context::metal_output_stays_in_the_context_domain_without_host_transfers ... ok\n\ntest result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 87 filtered out; finished in 0.12s\n\n",
      "evidence": "crates/tenferro-gpu/tests/integration/apple_context.rs",
      "skip_reason": null
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
      "duration_seconds": 1.056,
      "output_tail": "    Finished `test` profile [unoptimized] target(s) in 0.18s\n     Running tests/integration.rs (target/debug/deps/integration-9bf778a674b2eb73)\n\nrunning 2 tests\ntest gpu_ad_tests::test_gpu_eager_backward_smoke ... ok\ntest gpu_ad_tests::test_gpu_matmul_vjp ... ok\n\ntest result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 333 filtered out; finished in 0.67s\n\n",
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
