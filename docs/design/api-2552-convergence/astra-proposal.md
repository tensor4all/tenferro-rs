## 1. The decision.

**Add one independent `MaterializationPolicy` argument to execution calls, rather than introducing an execution-request wrapper.** The argument applies to that invocation only; `Default::default()` means `Allow`. Keep existing resource ownership, operation names, semantic parameters and result types in this change. This establishes one materialization convention across the three families without pretending to finish their separate typed-view/API consolidation. Existing entry points change in place: there are no `_no_copy` siblings and no second numerical implementation. The following are proposed signatures, using the existing types and receivers shown in the supplied material:

```rust
// Defined once in tenferro_tensor; shared by operation families.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MaterializationPolicy {
    #[default]
    Allow,
    Forbid,
}

// tenferro_linalg: existing free function, one added argument.
pub fn svd_with_exec<T: LinalgScalar>(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
    full: bool,
    materialization: MaterializationPolicy,
) -> tenferro_tensor::Result<(
    TypedTensor<T>,
    TypedTensor<T::RealScalar>,
    TypedTensor<T>,
)>;

// Existing concrete einsum trait: one added argument.
fn einsum(
    &self,
    subscripts: &str,
    session: &mut dyn Engine,
    materialization: MaterializationPolicy,
) -> tenferro_einsum::Result<Tensor>;

// FftExecutor: retains its cache ownership and mutable receiver.
pub fn fft(
    &mut self,
    input: &Tensor,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
    session: &mut dyn Engine,
    materialization: MaterializationPolicy,
) -> tenferro_tensor::Result<Tensor>;
```

```rust
let (u, s, vh) =
    svd_with_exec(exec, buffers, &view, false, Default::default())?;

let result =
    svd_with_exec(exec, buffers, &view, false, MaterializationPolicy::Forbid);

let contracted =
    operands.einsum("ij,jk->ik", engine, MaterializationPolicy::Forbid)?;

let transformed =
    fft_executor.fft(&a, None, -1, norm, engine, Default::default())?;
```

## 2. Where the materialization policy lives

For immediate execution, the policy is a **by-value execution argument**, after the existing arguments. Public wrappers must forward it unchanged, not substitute `Allow`. A multi-input operation has one policy covering every input.

For traced execution, it belongs to **runtime preparation**, not graph construction. Preparation records the policy as an immutable prepared-plan constraint. Include it in physical-plan cache identity; validate runtime layouts before executing a prepared plan when preparation did not know them. There is no execution-time setter that weakens or strengthens an already-prepared plan.

**NEED: the existing runtime preparation signature, options definition, and extension execution interface.** The material does not provide these. Extend that existing mechanism; do not invent a second preparation API or claim that `PrepareOptions` already has a particular shape.

| Placement | Decision |
|---|---|
| Per-call request wrapper | Not selected: the supplied interfaces do not establish a common resource bundle. Wrapping `(Exec, BufferPool)`, `dyn Engine` and an FFT cache owner would introduce another design decision. |
| Independent policy argument | Selected: one shared enum, no ownership changes, no conversion traits, no hidden defaults. |
| Existing execution context | Use its internal propagation mechanism if available, but never persistent engine configuration. Runtime preparation is the appropriate lifetime for traced execution. |

Rust has no omitted default arguments: callers explicitly write `Default::default()` or `Allow`. That small repetition is preferable here to adding a wrapper solely to avoid passing one enum.

**Policy boundary**

- Prefer direct execution whenever the selected provider supports the input descriptor.
- Otherwise, `Allow` permits same-placement, same-dtype input-layout normalization; `Forbid` returns a typed materialization-required error.
- Validate all inputs before normalizing any input, writing outputs or submitting numerical work. The error identifies the operation and offending operand, and recommends permitting materialization or explicitly normalizing it.
- Neither value authorizes device transfers, dtype conversion, backend fallback or changed mathematical semantics.
- Output allocation, intrinsic workspace and supported kernel packing remain allowed. They are not evidence that input materialization occurred.
- `Forbid` is provider-dependent, not a portability guarantee.

**Relationship to R1–R7**

| Rule | Treatment |
|---|---|
| **R1: one fact, one expression** | One enum and one provider admission decision. Callers do not duplicate provider/layout predicates. |
| **R2: one entry shape; `_into` primitive** | No new operation spelling or numerical path. This change does **not** repair the existing three entry shapes or establish `_into` primacy; claiming otherwise would be false. That remains a separate consolidation decision. |
| **R3: never canonicalize implicitly** | **Must be amended.** Replace the absolute prohibition with policy-authorized, observable input-layout normalization. Preserve direct provider-supported views and the prohibition on implicit transfers/conversions. |
| **R4: erasure only at FFI** | No new erasure. Existing `Tensor`/`dyn Engine` boundaries remain; this patch does not claim to make them R4-compliant. |
| **R5: no foreign public types** | The added enum is tenferro-owned. Existing public `Exec` exposure remains unresolved; packaging it in a request would not solve version identity. |
| **R6: checked public transitions** | Refusal is a typed error, not a panic or an invariant comment. |
| **R7: encode what types can carry** | A closed enum carries the execution constraint; provider-dependent layout admission remains a checked runtime decision. |

Only R3 changes meaning. R2/R4/R5 are acknowledged existing consolidation work, not silently weakened rules.

## 3. One convention for all three families

“Eager entry” below means immediate execution as in the supplied three examples, not an assertion that these are all AD eager APIs. Current shapes come from the issue/thread; omitted signatures are not reconstructed.

| Family | Row | Today | Proposed |
|---|---|---|---|
| linalg | eager entry | `svd_with_exec(exec, buffers, view, full)`; typed borrowed input | Same entry and input, with final `materialization` argument |
| linalg | traced entry | `TraceContextLinalgExt` and traced methods reported by the audit | Graph-building signatures unchanged; policy supplied at preparation |
| linalg | exec parameter | Explicit `&mut Exec<'_>` | Unchanged; policy does not own or replace it |
| linalg | buffer ownership | Explicit `&mut BufferPool` | Unchanged; normalization uses the existing resource path |
| linalg | options | `full: bool` on the illustrated entry; semantic option structs elsewhere | Unchanged; no policy field in semantic options |
| linalg | materialization policy | Illustrated borrowed-`Exec` path rejects negative strides; read routes reportedly gather | Same policy contract at every executing entry; all routes forward it to admission |
| einsum | eager entry | Tensor trait method taking `subscripts` and `session` | Same method and receiver, with final `materialization` argument |
| einsum | traced entry | `einsum(&mut self, inputs: &[TraceValue], subscripts: &str) -> Result<TraceValue>` | Unchanged; no policy attribute on the graph node |
| einsum | exec parameter | `&mut dyn Engine` | Unchanged; policy is independent |
| einsum | buffer ownership | No explicit pool in the illustrated signature | No ownership change; **NEED: actual scratch owner and admission boundary** |
| einsum | options | Subscripts plus notation/optimization variants described in the thread | Unchanged; execution constraint is not an optimization option |
| einsum | materialization policy | No policy argument in the supplied signatures | One argument covers all operands; preflight all before materializing any |
| fft | eager entry | `FftExecutor::fft(&mut self, &Tensor, n, axis, norm, session)` | Same method, with final `materialization` argument |
| fft | traced entry | `TraceContextFftExt` / `TracedTensorFftExt` reported by the audit | Graph-building signatures unchanged; policy supplied at preparation |
| fft | exec parameter | Mutable executor receiver **and** `&mut dyn Engine` | Both retained; the cache owner is not conflated with execution resources |
| fft | buffer ownership | Executor owns plans; lane buffers are reported; no explicit public pool | Ownership unchanged; **NEED: lane-buffer owner and asynchronous retention path** |
| fft | options | Positional `n`, `axis`, `norm` | Unchanged; no new `FftOptions` required |
| fft | materialization policy | No policy argument; CPU lane copying reported | Independent argument governs layout adaptation, not every lane-buffer write |

The common convention is **execution constraint as an explicit value, semantic graph without that value**. This deliberately does not advertise identical receivers, tensor types or resource ownership.

AD eager execution must forward the value for work executed during that invocation without recording it as an AD semantic attribute. A later backward invocation supplies its own execution constraint.

**NEED: exact eager-session execution signatures and their policy propagation boundary.** They must be covered before implementation is considered complete.

## 4. Extensions follow the same convention

A fourth family follows the FFT example: retain its existing semantic entry and resource/cache owner, accept the shared policy at immediate execution, and receive the prepared constraint during traced execution. It does not get an engine setter, family-specific policy enum or separate no-copy API.

Its provider performs this admission sequence:

1. Validate operation, dtype, placement and input descriptors.
2. Determine whether each input is directly consumable.
3. For each unsupported descriptor, determine whether same-placement, same-dtype normalization would make it consumable.
4. If normalization is necessary under `Forbid`, refuse before execution side effects.
5. Under `Allow`, normalize each affected operand at most once at that admission boundary, then use the existing numerical path.
6. If normalization cannot make the operation supported, refuse under **both** values.

Changing a temporary’s name to “workspace” does not exempt a layout conversion. Classification follows its purpose:

| Copy | Classification |
|---|---|
| Constructing a supported input layout solely because the original descriptor cannot be consumed | Input materialization, even when stored in scratch |
| FFT lane buffer required for execution on directly supported inputs as well | Intrinsic workspace |
| Additional whole-input FFT gather to satisfy layout admission | Input materialization |
| Kernel packing that directly accepts the original strides | Kernel packing |
| Separate dense permutation solely to make an einsum operand admissible to the selected contraction path | Input materialization |

**NEED: verify these classifications against the actual FFT lane and einsum lowering paths.** The audit’s descriptions alone do not prove which category each current copy belongs to. The provider contract and instrumented tests must name the actual steps.

The input policy grants **no new permission** to stage destinations or resolve unsupported input/output aliasing. Preserve existing write contracts. Overlapping read-only inputs are not automatically invalid; write aliasing requires the operation’s existing validation.

For asynchronous execution, materialized storage must remain alive until its consumer completes, using the existing retention mechanism—not the lifetime of the policy argument. No synchronization is added merely to inspect the enum.

## 5. What is explicitly out of scope

- **Canonical typed-view receiver consolidation:** replacing owned/read/typed facades is a separate public-interface decision.
- **New `_into` operations or changed overwrite guarantees:** input policy does not settle output allocation or failure semantics.
- **Uniform execution/resource wrapper:** no demonstrated ownership problem justifies replacing the current resource parameters.
- **Options structs, boolean removal, naming and argument-order cleanup:** these change semantic presentation independently of materialization.
- **Moving FFT caches into engines:** cache ownership need not change to enforce an invocation constraint.
- **Foreign-type insulation and erasure removal:** neither is solved by wrapping the existing types.
- **Cargo features, CUDA visibility, capability gates and error-family consolidation:** independent audit findings, not prerequisites for this contract.
- **New provider layout support or numerical algorithms:** admit existing capabilities accurately; do not expand them to make `Forbid` portable.
- **Public profiling infrastructure:** use test instrumentation or an existing opt-in facility; no mandatory hot-path counter.

## 6. Migration and proof

Callers append `Default::default()` to preserve default-Allow intent, or `MaterializationPolicy::Forbid` to assert that layout normalization is forbidden. Existing borrowed-`Exec` callers relying on negative-stride refusal must choose `Forbid`; selecting the new default deliberately changes that behavior. Callers do not change resource ownership, destructure different results or branch on providers.

Traced graph construction remains unchanged. Runtime preparation receives the constraint; changing it requires preparation under the new constraint. No compatibility overloads or `_no_copy` functions are introduced.

**Assertions introducing behavior unavailable in the supplied API:**

1. **Default:** `MaterializationPolicy::default() == Allow`, and the three proposed call shapes compile.
2. **Borrowed linalg normalization:** a supported negative-stride input succeeds under `Allow`, with one normalization; under `Forbid`, it yields a typed operand-specific refusal with zero numerical submissions.
3. **No state leakage:** `Forbid` followed by default-Allow on the same engine behaves exactly like two independently configured invocations.
4. **Multi-input atomic admission:** if the last operand requires forbidden materialization, no earlier operand has been materialized and no destination has changed.
5. **Prepared-plan enforcement:** an Allow plan containing normalization cannot be obtained as a Forbid cache hit or executed as a Forbid plan.

**Regression assertions protecting existing capabilities:**

- Directly supported transpose, positive-stride and zero-stride views have zero input materializations under either policy; parameterize expectations by provider.
- Owned and borrowed routes reach the same admission decision for identical metadata.
- FFT lane workspace and einsum kernel packing remain allowed where their verified classification permits them; additional layout normalization is counted separately.
- Empty inputs do not bypass applicable layout validation; invalid rank-0 inputs retain their operation-specific errors.
- Neither policy changes dtype, placement or complex-valued results.
- Unsupported writes and aliases retain their existing errors; policy rejection leaves destinations unchanged.
- Cold and warm runs count materializations separately from allocations: a recycled temporary is still a materialization.
- CUDA temporaries survive asynchronous consumption and are not recycled prematurely.

Tests above are proposed assertions, not claimed reproductions.

The implementation must update R3, `docs/spec/api-conventions.md`, and the affected storage/backend contracts alongside the code. Shared general policy belongs once in [`tensor4all-agent-rules/rules/common/repository.md`](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/repository.md), indexed through [`rules/index.md`](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md); local rules should link rather than duplicate it. Linked changes must record exact shared-rules revision and tenferro implementation commit, with a checked coordination gate before completion.

The documented inconsistency being resolved is R3/shared “no hidden layout conversions” versus the reported existing gather paths. The other R2/R4/R5 inconsistencies remain explicitly deferred—not declared fixed.

No source changes, compilation, GPU tests or benchmarks were performed. The marked `NEED` items are implementation blockers; this document is not approval to ship or completion of #2552.
