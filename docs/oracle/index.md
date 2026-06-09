# Oracle Replay

AD correctness is validated by the [tensor-ad-oracles](../../third_party/tensor-ad-oracles/)
database — a collection of PyTorch-generated reference values for forward, VJP,
JVP, and HVP computations across 171 op families.

## How it works

Oracle replay was historically implemented as an integration-test harness under
the old root facade crate. The root facade has been removed; the oracle support
matrix remains useful for tracking AD coverage, while any new replay harness
should live in the crate that owns the behavior under test. Each test case:

1. Reads a JSONL record containing input tensors, op parameters, and PyTorch reference outputs
2. Executes the same operation through tenferro
3. Compares results within tolerance

## Coverage

See [Oracle Coverage Status](./tensor-ad-oracles-support.md) for the current
per-op support matrix (auto-generated).

## CI policy

Oracle replay has two CI tiers:

1. **PR sentinel tier**: run on a standard GitHub-hosted Linux runner. This tier
   uses a fixed sentinel set plus an affected-op subset derived from the PR
   diff. It must stay small enough for normal pull-request turnaround and must
   not require larger Linux or GPU runners.

2. **Post-merge full tier**: run after merge to `main` on standard
   GitHub-hosted Linux runners. This tier shards the full supported oracle
   matrix across ordinary Linux jobs. Unsupported rows in
   [Oracle Coverage Status](./tensor-ad-oracles-support.md) remain tracking
   data, not required CI gates, until the owning crate implements the replay
   adapter for that family.

The PR sentinel set should be chosen by these criteria:

- include at least one record for every contract the PR changes, such as dtype,
  shape, batching, AD direction, or complex convention;
- prefer cases that have both Torch reference payloads and finite-difference
  checks;
- include real and complex dtype representatives when a rule branches on
  complex behavior;
- include boundary cases that previously failed or that differ across CPU
  providers;
- keep the set deterministic and stable so a PR failure points to a contract
  regression rather than oracle sampling variance.

The current root-facade oracle harness has been removed. Until a crate-local
replay harness is restored, CI cannot claim full oracle replay coverage; focused
regression tests and this support matrix are the active gates.

## Links

- Test source: historical root-facade oracle replay harness, now removed
- Oracle database: `third_party/tensor-ad-oracles/`
- Design spec: [Oracle Replay Design](../superpowers/specs/2026-04-06-oracle-replay-design.md)
