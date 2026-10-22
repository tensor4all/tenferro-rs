# CPU residuals of #2233: decode_proj, activations, 4T softmax, transpose

Follow-up to `perf-cpu-softmax-fused-2228.md` and `perf-cpu-realfft-2230.md`.
Each residual turned out to be a mis-routing somewhere in the stack rather than
a kernel that is slow by itself, so three of the four fixes live in the owning
upstream crates (tcontract, strided-rs, elem-simd-math) and this repository
only routes to them.

## Decisions

- **decode_proj (4.2x vs direct faer).** `tcontract::Plan` served every
  *fresh* (uninitialized) output of a Faer-strategy plan through its packed
  driver, because faer's Replace path may form references over D before it
  stores. `BinaryPlan::new` always sets `fresh_output`, and every allocating
  `dot_general` in the CPU engine (`contraction::dot_fresh`) uses the fresh
  contract, so each eager contraction that faer should have run went to the
  packed driver; for M=8, K=N=1024 that was 2.4 ms against faer's 0.57 ms.
  tcontract now zero-fills a fresh D in one output pass (raw-pointer stores,
  the same pass `beta`/`OutputOnly` already used) and runs faer; the prepared
  packed alternative is gone. `PlanConfig::fresh_output` stays as an accepted
  no-op so cpueinsum's call site compiles. Rejected: zeroing the pooled output
  in tenferro and calling the initialized route, which would have kept the
  dead packed alternative alive in the owner.
- **structural.transpose (2x vs PyTorch).** The HPTT kernel of strided-perm
  runs the 4096x4096 f64 transpose in 32 ms, faster than PyTorch's 63 ms.
  tenferro's 119 ms came from `strided_basic::copy_into_uninit`, which
  reinterpreted the *source* as `MaybeUninit<T>` to meet the uninit
  destination type; strided-perm selects its native float kernels by element
  type, so `MaybeUninit<f64>` fell to the generic element path. strided-perm
  gained `copy_into_uninit` / `copy_into_uninit_par` (dispatch on `T`, raw
  stores into the `MaybeUninit` destination) and strided-basic routes its
  uninit copy through them under the same execution policy. Rejected: a
  `BLOCK_MEMORY_SIZE` sweep, which the issue proposed; the kernel was never
  the problem.
- **activations (gelu 3.45x, sigmoid 1.81x, silu 1.98x).** The previous fused
  hook was scalar and lost to the vectorized composite. The fused kernels now
  live in elem-simd-math (`f32::sigmoid` and friends), one vector pass each
  composed from its tier kernels the way PyTorch's `Vectorized<float>`
  activations are; `tenferro-cpu/src/activation.rs` owns only the tensor
  boundary (dtype dispatch, pooled output, the shared parallel split). The
  `f32` `gelu` uses the Abramowitz–Stegun erf that PyTorch's CPU GELU uses
  (absolute bound `3e-7 * |x|` plus 3 ULP; `f64` keeps the 1.0 ULP erf). This
  is the one accuracy-class change of the batch and is documented in the
  kernel; the composite's own `1 + erf` cancellation already bounds the
  negative tail the same way.
- **softmax 4T len64 (2.54x) and the "launch overhead".** The fused hook was
  already at PyTorch speed (31.7 us vs 44 us at 1T, 15 us vs 15 us at 4T); the
  benchmark's extra 40 us was the fresh pages of outputs it retains, which the
  `vec![0; n]` output could not recycle. The hook now writes into a
  `PooledUninitOutput`, so retained outputs return to the pool and the
  steady-state batches reuse them. Rejected: a worker cap (measured earlier)
  and any change to the parallel split, which was not the cost.
- Not changed: rfft/irfft vs MKL (needs a real FFT library, not a routing
  fix), the host-dependent typed add / 2x2 rows, and gather.

## Verification conclusions and constraints

Same host as the earlier worklogs (Ryzen AI 9 HX 470, measured on cores 0 /
0-3), `cpu/perf_issues` harness built without the `kache` rustc wrapper and
with the four upstream crates patched to their fix branches; PyTorch 2.12 CPU
on the same cores.

| case | before | after | PyTorch | ratio |
|---|---:|---:|---:|---:|
| sigmoid 1024x64 f32, 1T / 4T | 171.7 / 184.7 us | 44.5 / 19.0 us | 49.4 / 27.0 us | 0.90x / 0.70x |
| silu | 181.3 / 204.0 | 46.8 / 20.5 | 49.9 / 27.8 | 0.94x / 0.74x |
| softplus | 192.4 / 183.9 | 160.1 / 48.9 | 174.7 / 89.9 | 0.92x / 0.54x |
| gelu | 333.1 / 253.2 | 65.1 / 24.5 | 61.0 / 23.9 | 1.07x / 1.03x |
| gelu_tanh | 306.4 / 304.0 | 188.6 / 55.6 | 214.1 / 57.4 | 0.88x / 0.97x |
| softmax len64 b8 eager, 1T / 4T | 76.3 / 61.9 | 32.2 / 16.3 | 44.4 / 14.7 | 0.73x / 1.11x |
| decode_proj len8 out1024, 1T | 2423.9 | 577.8 (faer direct 581.5) | 555.1 | 1.04x |
| decode_proj len8 out4096 / len64 out1024 / len512 out1024 | 9780 / 3881 / 19286 | 2390 / 2210 / 17798 | faer direct 2330 / 2134 / 17105 | — |
| transpose 4096x4096 f64 (mwe), 1T / 4T | 119 / — ms | 70 / 21 ms | 63 ms | 1.11x / 0.33x |

- The remaining transpose gap at 1T is the first-touch cost of a fresh 128 MB
  output (the raw kernel is 32 ms; allocation plus kernel measured 78 ms);
  the pool's default retained capacity (100 MiB) is below one such output, so
  the benchmark's retained outputs are never recycled. That is a pool policy,
  not a kernel issue, and is left as is.
- Accuracy of the fused kernels was checked against an mpmath oracle on 2^18
  samples per kernel and dtype; the crate's MPFR suites could not run on this
  host (no `m4` to build `gmp-mpfr-sys`), so they run in that crate's CI.
- The upstream fixes are released (elem-simd-math 0.1.1, strided-perm 0.6.1,
  strided-basic 0.6.4, tcontract 0.4.3, cpueinsum 0.4.3). With them the CPU
  stack's exact pins became caret requirements at the patch each API needs
  and `Cargo.lock` is committed: Cargo already unifies one compatible line to
  one version, so the exact pins bought nothing the lock does not, while they
  forced a coordinated release for every upstream patch. The GPU fork pins
  (`t4a-cubecl*`, `t4a-cubek*`) are deliberately left exact.
