# Oracle Replay

AD correctness is validated by the [tensor-ad-oracles](../../third_party/tensor-ad-oracles/)
database — a collection of PyTorch-generated reference values for forward, VJP,
JVP, and HVP computations across 171 op families.

## How it works

Oracle replay tests live in `tenferro/tests/oracle_replay/`. Each test case:

1. Reads a JSONL record containing input tensors, op parameters, and PyTorch reference outputs
2. Executes the same operation through tenferro
3. Compares results within tolerance

## Coverage

See [Oracle Coverage Status](./tensor-ad-oracles-support.md) for the current
per-op support matrix (auto-generated).

## Links

- Test source: `tenferro/tests/oracle_replay/`
- Oracle database: `third_party/tensor-ad-oracles/`
- Design spec: [Oracle Replay Design](../superpowers/specs/2026-04-06-oracle-replay-design.md)
