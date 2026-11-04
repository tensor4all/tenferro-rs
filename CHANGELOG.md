# Changelog

All notable changes to the `tenferro-*` crates are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Breaking (minor): traced values are owned by a `TraceContext`.** This is a
  minor release, not a patch: `TracedTensor` no longer carries its own graph, so
  context-free constructors, `TracedTensor::graph()` and the public
  `id`/`rank`/`dtype`/`val` fields are gone. See
  [`docs/design/traced-over-trace-context.md`](docs/design/traced-over-trace-context.md)
  §3.9 for the complete list and
  [`docs/guides/custom-operations.md`](docs/guides/custom-operations.md#migrating-a-context-free-constructor)
  for the one-line shape of the migration.
  - Extension authors: `TracedTensor::input_concrete_shape(dtype, shape)` becomes
    `<ctx>.input_concrete_shape(dtype, shape)`, and a traced value can only be
    combined with values from the same context.
  - `GraphCompiler::compile_many` requires at least one output; a zero-output
    program is built with `SemanticProgramBuilder::finish(&[])`.
  - AD seeds of an externally defined scalar must carry that scalar's canonical
    identity, which the context constructors now supply.
