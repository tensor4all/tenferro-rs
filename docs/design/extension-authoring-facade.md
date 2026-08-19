# Extension Authoring Facade

## Status

Accepted design for issue #1708. Implementation requires an independent design-review verdict before code changes.

## Goal

An out-of-tree crate can define, trace, register, and execute an extension without depending directly on any `tenferro-internal-*` crate or naming generated runtime implementation types.

## Public boundary

`tenferro_runtime::extension` is the stable authoring facade. It re-exports only the existing author-facing contracts needed to implement an operation and install its generated runtime:

- `ExtensionOp`, `ExtensionShapeContext`, `ExtensionFamilyId`, and `SymDim`;
- `ExtensionEffect`, `ExtensionEffectAccess`, and `ExtensionEffectDeclaration`;
- `ExtensionAlias` and `ExtensionAliasDeclaration`;
- `ExtensionModule`, `ExtensionModuleId`, and `ExtensionModuleRegistrar`;
- `ExtensionExecutionContext` and the existing cache types;
- `define_extension_runtime`.

The implementation remains in the existing internal crates. Generated engine, planning-config, prepared-operation, and executor types remain private to the invoking crate. Existing root-level runtime exports remain source-compatible.

The runtime crate may depend on the existing workspace proc-macro crate because that crate emits paths but does not depend on the runtime. No new package or third-party dependency is introduced.

## Generated runtime contract

`define_extension_runtime!` remains the sole adapter. Its required execution callback is `execute_reads`, which receives borrowed `TensorRead` inputs. The legacy parsed `execute` argument is accepted when present for existing callers but is no longer required or documented because generated code does not use it.

Each extension family invokes the macro inside its own ordinary Rust module. This lets multiple families coexist in one crate while retaining the existing `extension_module` constructor name and avoiding new naming syntax or exposed generated implementation types.

Runtime ownership remains explicit: the application builds a `Runtime`, installs the returned `ExtensionModule`, compiles the traced graph, and executes it. There is no registry, discovery, implicit backend, transfer, or fallback.

## Documentation and executable fixture

`docs/guides/custom-operations.md` uses only public direct crates and the current `ExtensionModule` plus `execute_reads` path. It removes the deleted `ExtensionOp::host_reference` recommendation.

A workspace-external fixture crate under `tests/fixtures/extension-authoring` is excluded from workspace membership and compiled/run by a focused repository test script. It depends only on deliberate public crates (`tenferro-runtime`, `tenferro-tensor`, and `tenferro-cpu`), defines two families in separate modules through the facade macro, traces and executes one family, and exercises typed missing-module, wrong-family, and wrong-context failures where those states are reachable through the public API. The fixture is the executable source of truth for the guide; copied guide snippets must use the repository snippet-sync mechanism or the guide must point directly to the fixture.

## Non-goals

- No global extension registry or backend discovery.
- No host-reference compatibility layer or second execution framework.
- No public generated engine, planning, prepared-operation, or executor types.
- No new extension DSL beyond making the unused legacy `execute` argument optional.
- No implicit device transfer or CPU fallback.

## Verification

- Run the external fixture and its typed failure assertions.
- Run proc-macro tests, including omission of `execute` and paired session callback validation.
- Run `tenferro-runtime` tests and doctests.
- Run documentation snippet/site checks and public error-doc checks through the local PR gate.
- Confirm packaged `tenferro-runtime` resolves the re-exported proc macro and contains no accidental internal API documentation.
- Review modified-file coverage against the repository 90% target.
