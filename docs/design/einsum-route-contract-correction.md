# Einsum route-contract correction before benchmark expansion

Status: Flash design review completed before implementation. Initial review
approved the source corrections subject to clarifying successful prepare evidence;
the clarified data-only design received **Correct-to-merge** with no findings.
Source inspected: tenferro-rs aa68a0d7703b8f38700f9f3234e8f06ed757fb05.
This is a data/documentation correction, not a new API or performance outcome.

## Evidence and problem

The canonical export contains only einsum.einsum.ordinary.concrete and
 einsum.einsum.prepared.traced. Neither is an EagerTensor route. A concrete
ConcreteEinsumPlan execution must not be called traced merely to satisfy binding.

The ordinary.concrete case currently cites an EagerTensor integration test.
The prepared.traced case cites a generic binary planning test, not an executing
compiled graph. Concrete/eager surface ownership currently overstates
execute_einsum_extension as the universal owner.

Actual source boundaries:
- concrete.rs::ConcreteEinsumPlan::prepare(inputs, subscripts) parses notation;
  prepare_subscripts_internal captures input specs and calls plan_subscripts.
- execute(inputs, session) validates specs then calls eager_einsum_exec; its
  BackendSession is supplied by its caller, not opened by ExtensionEngine here.
- eager_ad.rs::einsum_subscripts_with_broadcast dispatches first to direct binary
  dot_general, then untracked whole-program / expanded standard operations, with
  EinsumExtensionOp only as fallback. Rank2 matrix multiplication therefore does
  not isolate extension-owner overhead.

## Bounded change

Keep schema v1, existing IDs and phases. Add exactly three execution/setup route
contracts under the existing einsum selector:
1. einsum.einsum.ordinary.eager — eager / execution, real EagerTensor entry.
2. einsum.einsum.prepare.concrete — concrete / setup, public plan creation only.
3. einsum.einsum.prepared.concrete — concrete / execution, existing plan execution
   with dtype/shape revalidation and caller-owned session; no replanning.

Expected export count184 from181. No private parse/revalidation probe may use these
whole-public-operation IDs as its primary component contract. No aliases, new
registry, runtime code, dependencies, caches, or public methods.

Correct the two existing test references and new references using real numerical
or public-error tests, whose bodies must be read before editing:
- concrete_tests.rs::public_tensor_einsum_ext_executes_dtype_erased_inputs;
- concrete_tests.rs::public_typed_tensor_einsum_ext_preserves_complex_dtype as
  additional C64 evidence, not a substitute for dtype-erased route evidence;
- concrete_tests.rs::concrete_einsum_plan_executes_without_replanning_contract;
- concrete_tests.rs::concrete_einsum_plan_rejects_shape_and_dtype_mismatches;
- tests/integration/eager_tensor.rs::eager_tensor_einsum_matmul_primal_matches_expected_values;
- tests/integration/traced_correctness.rs::explicit_path_matmul_executes_numerically.
The prepare case is a successful public setup route, not an invalid-prepare
scenario. Cite concrete_einsum_plan_executes_without_replanning_contract: it
invokes public prepare, unwraps successful construction, executes that plan and
asserts numerical results. This proves success-path setup, not prepare-error
coverage. No cited test currently asserts a public prepare error; do not claim
otherwise. Retain existing execution-mismatch/error evidence separately. No new
Rust test or runtime code is authorized in this correction; broader
error-path measurement coverage remains a separate #95/#1760 requirement.

Correct concrete/eager source ownership and prose to these real dispatch paths.
For concrete admission explicitly describe the borrowed caller session; do not
replace one false session owner with another. Use actual existing source anchors.
For eager list the dispatcher/branches and distinguish direct standard operations
from extension fallback; do not state all einsum calls use the fallback.
Retain follow-up dispositions: source classification is not measured evidence.
Do not silently reclassify unrelated families or claim their final audit complete.

## Verification and rollout

Design and full deliverable require user-selected Flash verdicts. Generate export
with the maintained checker; verify exact184 ID set, unchanged181 IDs, all
surface/phase matches, preserved unrelated metadata, and mutation-test freshness.
Update the existing Python inventory mutation-test expectation from the old
181-row set to the exact184-row set, explicitly asserting the three new IDs and
retention of previous IDs. Update both the count assertion and the ordinary.eager
operation assertion in test_current_inventory_is_exhaustive: eager routes now
include add and einsum rather than add alone. This is the sole authorized
test-code change: no
generator behavior or Rust tests change. The parent reproduced the stale-count
failure in test_current_inventory_is_exhaustive before authorizing this update.
Run repository data/CI-helper rules and fast gates, focused referenced Rust tests where
needed, docs/format gates and all required hosted CI. Preserve user work in a new
worktree; no benchmark or existing probe-worktree edits in this change.

Merge library correction first; verify merged revision and checks, then update
benchmark's owned library pin and revalidate existing24 cases before adding new
producers. Benchmark expansion and its independent design/review remain separate.
No timers, performance assertions or parent-completion claim belong to this task.
