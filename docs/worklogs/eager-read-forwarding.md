# Eager read forwarding

2026-09-17; AMD EPYC7713P, explicit 1T provider-matched diagnostics.

The saved provider-matched multiply profile attributes ~5.77M instructions to
input materialization (~14% of the former eager total, before the strided SIMD
fix). `exec_standard_op_on_tensor_reads_in_session` unconditionally copies views
before calling `_read` methods which already accept them. CPU multiply has direct
view dispatch. Default trait `_read` methods actually reject borrowed views;
they do not promise a copy fallback. Production CPU sessions override the read
hooks, and CUDA adapters own their explicit device-local layout handling. Owned-only operations already use `concrete_tensor_read(s)` in
their individual arms. This is meaningful avoidable work, not a helper-only
microbenchmark hypothesis.

Remove only the blanket eager conversion. Preserve dtype promotion and its
necessary conversion copies, owned-only operation boundaries, placement checks,
extension registration, borrowed lifetimes, and backend/session ownership. Do
not change AD rules, storage ownership, or introduce unsafe code. Test with the
existing recording backend, non-contiguous and empty inputs, mixed dtypes and a
backend whose default read hook is unsupported (propagate that error, do not
hide it with a copy); retain eager/AD regression coverage. The initial test
assumption that the default exp_read materializes was disproved by execution
and corrected after inspecting the trait's read_tensor contract. Callers are
internal eager dispatch over the closed CPU/CUDA backend selection, not a new
public generic-backend compatibility promise.

Predeclared instruction protocol: baseline tenferro 57bee76, fixed strided
17e05ff, benchmark 976c4b9 source in the existing isolated benchmark worktree.
Record candidate commit before execution. Use the same OpenBLAS Docker image,
Rust release+debuginfo build, CPU16, explicit and verified 1T. Complete cases:
bin_elementwise_mul_2048x2048, bin_matmul_1024, and
lm_batch_likelihood_sentence_3_12d. Three independent baseline/candidate N1/N3
pairs, one fixed warmup; use median paired percentage changes. Primary multiply
Ir reduction >=20%; each control must regress <=1%. Retain all samples and
inspect GEMM self-cost invariance. Any incomplete run, thread/affinity mismatch,
or numerical failure invalidates the complete experiment. Host workloads and
CPU16's L3 domain are recorded; contention permits instruction diagnostics only,
never native-time promotion. Quiet-host native validation remains separate.

Functional verification: the new recording test fails on the baseline (two
copies, expected zero) and passes after forwarding. 91 unit tests, 354 functional
integration tests and 174 doctests passed initially. The first UI wrapper run
failed: trybuild lost command-line dependency patches, and the host compiler
produced two formatting-only snapshot differences. The later Docker Rust1.98.1
CPU/AD gate passed all five UI fixtures using temporary manifest patches and a
test-only OpenBLAS linker; see canonical-copy-dispatch.md. The manifest was
restored byte-for-byte; no snapshots were changed. This UI blocker is resolved.

## Collected evidence and next bottleneck

The three matched 1T instruction pairs pass the predeclared gates: median paired
reductions are 52.18% for multiply, 0.28% for GEMM1024, and 1.81% for LM. Vendor
GEMM self-Ir is unchanged. Benchmark evidence commit `d823ab9` stores the raw
profiles, protocol, parser, source/binary provenance and failure logs under
`result/amd-cpu/eager-read-forwarding/`. Docker release library tests also pass
(91); focused release Memcheck reports zero errors with leak checking disabled.
These are instruction findings, not native speedup claims.

The candidate LM N3 caller profile records 408 canonical operand materializations
and 1.429 billion inclusive instructions through `materialize_canonical_operand`
for the whole invocation (not one contraction). Its uninitialized copy leaf is
`structural::typed_copy_into_uninit`, which currently uses generic `map_into`.
Remaining memset cost must not all be attributed to tenferro allocation: the
largest named caller is OpenBLAS `dgemm_beta_HASWELL`.

Before replacing canonical copies, preserve the custom GEMM provider's supported
layout/ownership fallback and rhs-error pool reclamation. Existing strided
`CopyPlan::execute_uninit` is reusable but its fused replay is currently serial;
the centralized permutation-copy dispatcher is also serial. Blindly substituting
either for the current parallel-capable map would change the 4T policy. Kernel
selection needs bounded-thread validation and quiet-host measurements, rather
than choosing a cache-sensitive traversal from instruction counts alone. No
compact-operand borrowing change was retained. The subsequent shared permutation
copy change preserves bounded-parallel map dispatch; see canonical-copy-dispatch.md.

The maintainer subsequently requested PRs and merging. Strided PR259 is merged,
and the ordinary git pin now selects that commit. No package publication was
requested or performed. Native measurements remain inconclusive; 4T is deferred.
