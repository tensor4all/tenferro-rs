# RunPod reporting reliability and bounded CUDA test concurrency

The previous bootstrap comparison lost its final cost/placement record when the
pre-delete API read hit a DNS timeout. The Pod was deleted successfully. Preserve
only non-secret reporting fields in a small hosted-job output before GPU execution,
and let the post-delete reporter fall back to that snapshot. Keep deletion before
checkout/reporting and retain the existing bounded best-effort final read. Never
invent a start timestamp or price when both records are unavailable.

The current CUDA workload dominates paid lifetime: the preceding seven complete
pairs spent about 227 of 315 seconds in CUDA tests. This establishes the need to
investigate test scheduling; it is not a claim about numerical kernel speed.
The helper currently hardcodes one nextest process, so changing its environment
alone cannot enable parallel tests. Add an explicit bounded 1/2-process argument,
keep the default serial until the experiment passes, and leave PJRT serial.
Nextest runs each test in its own process. The reviewed cache/session tests use
process-owned state and small fixtures; test identity, assertions, timeout and
archive partition validation stay unchanged. Observe device memory to catch
resource contention, rather than assuming process isolation limits VRAM use.

Use the current 304 CUDA cases plus three PJRT cases and tutorial. The successful
artifact source run 38056837195 tested eb4d13fdfbd66541a6dfdc28f04d1cee0604f365;
its crate sources, Cargo.lock and test partitions are identical to main
 e20f1fb9e3591135c830126d641645c3ef914e1f. Freeze those artifacts and exact case
identities in the benchmark repository. Both arms use the metadata fix, official
runner and Ubuntu 24.04. The only treatment is CUDA nextest concurrency 1 vs 2.

Predeclare the complete paired experiment in tenferro-benchmark before paid runs.
Keep all failures and cold pulls, validate the full workload and deletion, and
measure both paid lifetime and CUDA-stage duration. No production concurrency
change is promoted from a partial or inconclusive comparison. The previous
bootstrap campaign remains closed and INCONCLUSIVE; this is a different candidate
against the current implementation.
