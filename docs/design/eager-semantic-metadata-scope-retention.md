# Eager Semantic Metadata-Scope Retention

Status: implemented for issue [#1700](https://github.com/tensor4all/tenferro-rs/issues/1700).

## Problem

Eager AD records active operations into a raw semantic trace and defers whole-graph metadata analysis until the first derivative request. Mixed-dtype promotion and exact-shape concatenate recording may insert ordinary `TracedTensor` helper nodes (`Convert` or `Reshape`) into that raw trace. Those helper nodes own scoped metadata registrations for the external parent value they consume.

`record_semantic_eager_outputs` currently constructs the next raw carrier with an empty metadata-scope list. When a helper tensor is a local temporary, its scope is dropped after recording even though the raw output graph still references the helper's output. Deferred compilation then fails with `missing input metadata`.

The minimal reproducer is a tracked `F64` chain containing a temporary untracked exponent, followed by a mixed `F64 * C64` operation. The promotion cast consumes the non-leaf raw trace, and `backward()` fails after the cast temporary has been dropped.

## Required behavior

- A raw eager semantic output must retain every metadata scope needed by its final semantic inputs.
- Retention must work after promotion casts and concatenate exactification reshapes.
- Shared scope histories and shared chain nodes must remain pointer-deduplicated; scope propagation must be persistent and must not materialize or scan whole histories per eager operation.
- The eager primal path, dtype promotion, active-edge pruning, public traced behavior, and deferred analysis timing remain unchanged.
- No tensor is materialized and no AD graph is detached.

## Design

Keep `MetadataScopeChain` private to `tenferro-runtime` and extend the existing runtime-owned raw-append seam with a hidden carrier-construction helper. The helper receives the final `&TracedTensor` inputs and output metadata, appends the raw operation, merges private runtime state (input maps, leaf metadata, roots, and metadata-scope chains), and returns output `TracedTensor` carriers. Existing public hidden `RawAppend` and `TracedTensorParts` layouts remain unchanged.

`MetadataScopeChain::merge` returns the empty chain for no parents, clones the existing chain for one parent, and creates one parent-only node for multiple inputs. Materialization tracks both scope pointers and chain-node pointers so diamond-shaped histories visit each shared node once. Ordinary traced helpers continue adding one new scope above inherited chains; raw eager operations add no scope.

After promotion and concatenate exactification select the final `semantic_inputs`, `record_semantic_eager_outputs` delegates carrier construction to the runtime helper. The helper computes one persistent merged chain and clones that token into every output carrier. No scope vector is materialized per eager operation, and no new public transfer type crosses the crate boundary. The carrier only keeps scopes already required by helper graph boundaries alive until deferred analysis consumes the complete graph.

No public tensor API, existing public hidden struct layout, operation semantics, backend contract, or dependency changes.

## Rejected alternatives

- **Keep temporary constants alive in downstream callers.** This leaks an internal RAII requirement into users and does not fix composed eager graphs.
- **Materialize or detach before mixed operations.** This destroys AD connectivity.
- **Replace promotion casts with a second raw-op construction path.** That duplicates traced graph assembly and leaves exactification reshapes with the same lifetime problem.
- **Flatten scope vectors directly in `tenferro-ad` or at `TracedTensorParts`.** This either permits duplicate histories or scans and clones an accumulating history per eager operation, producing quadratic graph-construction work.
- **Replace `TracedTensorParts::metadata_scopes` with a transfer token.** Despite `#[doc(hidden)]`, this is a public struct-literal field and changing it would be source-breaking. The runtime-owned construction helper keeps the persistent type private instead.

## Verification

Add an eager promotion integration regression whose temporary exponent, intermediate values, and complex factor all leave scope before differentiation. Verify:

1. scalar `backward()` succeeds and records the expected real gradient;
2. functional VJP with a complex cotangent returns the expected real projection;
3. functional JVP returns the expected complex directional derivative.

Add a concatenate regression where an exactified temporary constant and its intermediate values are dropped before a downstream operation and differentiation. Add runtime-owned chain tests proving the single-parent fast path, pointer deduplication, and linear node visits for shared histories. Add a raw-carrier construction test proving one private merged chain is shared across multiple outputs and retains helper metadata after the helper tensor is dropped.

Run the focused regressions, the `tenferro-ad` and `tenferro-runtime` test suites, the repository fast PR gate, and the deterministic repository-rules review. Hosted CI remains responsible for the full backend and coverage matrices.
