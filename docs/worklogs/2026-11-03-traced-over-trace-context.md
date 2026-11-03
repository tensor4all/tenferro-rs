
### Completion (doctests, tests, gates)

- Doctests: `cargo test --workspace --doc` is green (2035 examples). Every example
  now follows the context API; hidden doctest lines use external crate paths, and
  `rank`/`dtype` are read through accessors.
- Workspace tests: 5340 passed / 5 failed on this host, down from 195 failed. All
  five are host-dependent and fail identically on `origin/main` here (see the PR
  table): #2422 (operand-copy and plan-reuse assertions), #2430 (CPU linalg
  allocation ceilings on macOS arm64), #2431 (static replay last-ULP on macOS
  arm64).
- Cross-context combinations in tests are now the typed `InvalidArgument` they
  are by design; the test helpers take the caller's context instead of creating
  their own (`metadata_views`, `runtime_execution`, `oracle_replay`).
- Values from two contexts cannot be combined, so `GraphCompiler::compile_many`
  refuses an empty output list (no context to snapshot) and a zero-output program
  is built with `SemanticProgramBuilder`.
- Two recorded structural baselines moved by one instruction and were re-recorded
  with the measurement attached: the eigvalsh VJP program (14 instructions
  instead of 13, one extra `Transpose`) and the 2x2 QR VJP census (21 commands
  instead of 20). The numerical contracts stay unchanged and the semantic
  assertions of both tests (`DotGeneral == 1`, `Mul == 1`, `Extension <= 2`,
  central finite differences) still hold; the extra transpose is tracked as
  #2429.
- A symbolic-extent contract moved with the typed op layer: the AD seed of an
  externally defined scalar must carry that scalar's canonical identity, so a
  preset `f64` cotangent for a `df64` loss is a typed seed-dtype refusal rather
  than a rule-domain error.
- Dead code left by the deletions (checkpoint graph nodes, the input-descriptor
  chain, `register_metadata_or_runtime_state`, `validate_shape_guards`, the
  per-tensor metadata helpers) and every unused import are removed; builder
  capabilities that only the crate's own tests use are `#[cfg(test)]`.
