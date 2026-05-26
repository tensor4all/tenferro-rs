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

## Links

- Test source: historical root-facade oracle replay harness, now removed
- Oracle database: `third_party/tensor-ad-oracles/`
- Design spec: [Oracle Replay Design](../superpowers/specs/2026-04-06-oracle-replay-design.md)
