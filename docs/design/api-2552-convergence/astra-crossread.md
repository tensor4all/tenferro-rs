## 1. **Where the two designs actually differ.**

1. **Carrier:** A adds a trailing argument; B replaces the execution-resource argument—a preference unless the reported prohibition on extra parameters is binding, in which case A violates it.
2. **Prepared execution:** A binds policy at preparation and separates cache identity; B checks it at execution—A unnecessarily duplicates policy-independent plans, while B’s prepare-time Boolean alone misses runtime-dependent normalization.
3. **Admission boundary:** B checks immediately before each gather; that can allow earlier operands to be materialized before a later refusal—a real defect unless admission happens for all operands first.
4. **Copy classification:** B exempts copies performed for every layout; that can exempt an unconditional dense-layout conversion—a real defect because frequency does not establish purpose.
5. **Proof:** B treats successful `Forbid` execution and bit-identical outputs as evidence; neither proves the gate covers every conversion, and nondeterministic kernels undermine bitwise equality—a real verification defect.
6. **Rules compliance:** B declares R2 satisfied by forwarding the request; forwarding does not establish `_into` primacy—a real overclaim, not additional implementation scope.
7. **Public footprint:** A introduces only an enum; B adds constructors and reborrowing—the additional surface is a trade-off, not an ownership defect, because the wrapper need not absorb buffers or caches.

## 2. **What you concede.**

- **Use B’s request carrier:** replacing the existing execution parameter preserves arity without changing resource ownership; A’s objection incorrectly assumed the wrapper needed to bundle every resource.
- **Keep explicit constructors:** concrete request parameters avoid the implicit-reborrowing and coercion problems of an `impl Into<ExecutionRequest>` compatibility layer.
- **Make policy a gate, not a selector:** provider and numerical-path selection must not change merely because the caller chooses `Forbid`.
- **Share physical plans across policies:** if selection is policy-independent and execution validates admission, A’s mandatory policy cache key and immutable prepared-policy binding are unnecessary.
- **Use B’s concrete error payload:** operation name and operand index are sufficient; explanatory remediation belongs in its display text, not another public diagnostic type.

## 3. **What you insist on.**

**Admission must precede every operand’s adaptation.** Consider two operands: operand 0 needs normalization, while operand 1 has an unsupported placement. An implementation that normalizes operand 0 before validating operand 1 has already performed work before an inevitable refusal. Validate all operands first, then check the policy, then normalize.

**A prepared-plan Boolean is insufficient when layouts vary.** Prepare with a compact input and reuse with a negative-stride input of the same shape. A stored `contains_materialization = false` cannot authorize execution if the runtime path would now gather. Either existing layout guards reject that reuse or admission evaluates the actual runtime descriptor before execution.

**Workspace is determined by purpose, not prevalence.** A provider that always calls `to_contiguous()` still performs input-layout normalization, even if it also copies already-contiguous inputs. Conversely, an FFT lane buffer required by its numerical algorithm can remain intrinsic workspace. Renaming or making a conversion unconditional cannot exempt it.

**A request does not prove correct forwarding.** Its mutable borrow prevents simultaneous conflicting resource access; it does not prevent a wrapper from extracting the engine and constructing a fresh default-Allow request. Forwarding and enforcement still require tests.

**Reject generic implicit conversion, not generic resource storage.** This four-line example isolates the move problem:

```rust
fn take<T>(_: T) {}
fn main() {
    let mut x = 0; let r = &mut x; take(r); take(r);
}
```

Expected result: E0382, use of moved value. Changing the first line to `fn take(_: &mut i32) {}` should compile because the calls implicitly reborrow. This supports explicit request construction; it does not establish that a request is inherently safer than a separate enum.

**Test execution structure separately from numerical results.** A broken implementation that ignores `Forbid` can return `Ok` while copying. Instrument normalization boundaries in tests. Check equivalent numerical results using the operation’s existing comparison contract, not a new universal bitwise-determinism promise.

## 4. **The merged design.**

### Public surface

Introduce exactly one policy enum and one borrowing request type, in the existing crate that owns `Engine`. Do not introduce a new crate. The request replaces the existing execution-resource parameter in place; separate buffers and FFT cache ownership remain unchanged.

```rust
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MaterializationPolicy {
    #[default]
    Allow,
    Forbid,
}

pub struct ExecutionRequest<'a, R: ?Sized = dyn Engine + 'a> {
    resources: &'a mut R,
    materialization: MaterializationPolicy,
}

impl<'a> ExecutionRequest<'a> {
    pub fn new(engine: &'a mut dyn Engine) -> Self {
        Self::over(engine)
    }
}

impl<'a, R: ?Sized> ExecutionRequest<'a, R> {
    pub fn over(resources: &'a mut R) -> Self {
        Self {
            resources,
            materialization: MaterializationPolicy::Allow,
        }
    }

    pub fn with_materialization(
        mut self,
        policy: MaterializationPolicy,
    ) -> Self {
        self.materialization = policy;
        self
    }

    pub fn materialization(&self) -> MaterializationPolicy {
        self.materialization
    }

    pub fn resources(&mut self) -> &mut R {
        &mut *self.resources
    }

    pub fn reborrow(&mut self) -> ExecutionRequest<'_, R> {
        ExecutionRequest {
            resources: &mut *self.resources,
            materialization: self.materialization,
        }
    }
}
```

The request is neither `Copy` nor `Clone`. It allocates nothing and owns no engine, buffer pool or plan cache. There are no conversion traits, engine setters, additional policy types or compatibility entry points.

Add this variant to the existing `tenferro_tensor::Error`, not a new error enum:

```rust
InputMaterializationForbidden {
    op: &'static str,
    input: usize,
}
```

`input` is the zero-based operand index. Display text identifies the operation and operand and recommends allowing materialization or explicitly normalizing the input. Other error layers preserve this typed cause through their existing wrapping mechanism.

### The three signatures

```rust
// tenferro_linalg: existing free function.
pub fn svd_with_exec<T: LinalgScalar>(
    exec: ExecutionRequest<'_, Exec<'_>>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
    full: bool,
) -> tenferro_tensor::Result<(
    TypedTensor<T>,
    TypedTensor<T::RealScalar>,
    TypedTensor<T>,
)>;

// Inside the existing concrete einsum extension trait.
fn einsum(
    &self,
    subscripts: &str,
    session: ExecutionRequest<'_>,
) -> tenferro_einsum::Result<Tensor>;

// Inside the existing FftExecutor implementation.
pub fn fft(
    &mut self,
    input: &Tensor,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
    session: ExecutionRequest<'_>,
) -> tenferro_tensor::Result<Tensor>;
```

One caller example per family:

```rust
let (u, s, vh) = svd_with_exec(
    ExecutionRequest::over(exec),
    buffers,
    &view,
    false,
)?;

let contracted = operands.einsum(
    "ij,jk->ik",
    ExecutionRequest::new(engine)
        .with_materialization(MaterializationPolicy::Forbid),
)?;

let transformed = fft_executor.fft(
    &a,
    None,
    -1,
    norm,
    ExecutionRequest::new(engine),
)?;
```

Apply the same replacement to all public immediate-execution entries in these families that currently take `&mut dyn Engine` or the illustrated borrowed `Exec`. Preserve every other parameter, receiver, name and result type.

Wrappers forward the request unchanged; helpers making multiple calls use `request.reborrow()`. They must not reconstruct a default-Allow request. Existing allocating and `_into` forms share their existing implementation path.

### Execution contract

One request’s policy covers every input of that invocation.

1. **Validate first.** Check semantic arguments, dtype, placement, descriptors and existing output/alias requirements for all operands. Determine provider support without numerical execution. Preserve existing error precedence; unsupported dtype, placement or operation must not become a materialization error.
2. **Select independently of policy.** Use the same provider and numerical path for both values. Where that path directly consumes the descriptor, use it without normalization. `Forbid` never triggers a different provider, algorithm or fallback.
3. **Determine adaptation needs for all operands.** Input materialization means constructing a replacement input layout to make an otherwise inadmissible descriptor consumable. Storage location, allocation reuse and the name “workspace” do not change that classification.
4. **Apply the gate.** If an operand needs that adaptation, `Forbid` returns `InputMaterializationForbidden` before any normalization, destination write or numerical submission. When several operands need it, report the lowest operand index. Perform admission before zero-size shortcuts.
5. **Execute under Allow.** Normalize only the affected operands, at most once per operand at that operation’s admission boundary, preserving dtype and placement. Then use the existing numerical implementation.
6. **Retain asynchronous storage.** Keep normalized inputs alive through consumer completion using the existing retention mechanism. The request’s lifetime is not the lifetime of submitted work.

Output allocation, intrinsic numerical workspace and packing performed by kernels that directly accept the original descriptor remain permitted. A separate dense permutation or gather used solely for input admission is materialization.

Neither policy authorizes device transfers, dtype conversion, output staging, new alias handling or backend fallback. Existing write contracts remain unchanged. `Forbid` is provider/path-dependent, not a portability or allocation-free guarantee.

Admission refusal is side-effect-free in the stated sense; this does not introduce transactional rollback for failures occurring after numerical execution begins. FFT cache statistics are not the normative definition of that guarantee.

### Traced execution and AD

Graph-building signatures and semantic graph attributes remain unchanged.

At the existing engine-bound graph-execution boundary, replace the execution-resource argument with the same request type. Policy applies to that execution, not permanently to the graph or engine. Do not introduce another preparation API or attach policy to semantic options.

Physical planning is policy-independent, so **do not add policy to physical-plan cache identity**. Every execution, including a cache hit, must validate materialization requirements against the actual runtime descriptors before numerical work begins:

- Fixed normalization steps remain identifiable in the prepared plan.
- Existing layout guards or descriptor-based admission checks cover runtime-dependent normalization.
- A prepare-time Boolean alone is insufficient when runtime layouts can differ.
- An incompatible cached plan follows existing layout-invalidation/repreparation behavior, identically under both policies; it never bypasses admission.

AD eager work forwards the current request. A later backward invocation receives its own request; policy is not recorded as a differentiable semantic attribute.

### Required proof and migration

Add focused tests covering:

- Default Allow and compilation of all three call shapes, including request reborrowing.
- A supported negative-stride linalg input: Allow normalizes once; Forbid refuses before numerical submission.
- All-input preflight, including a later invalid operand, with no earlier normalization or destination modification.
- Directly supported layouts: zero normalization under either policy.
- Verified FFT workspace and einsum packing classifications.
- A shared prepared-plan cache hit with changed runtime layout: Forbid cannot silently normalize.
- No policy leakage between successive calls on the same engine.
- Policy-independent unsupported-operation errors, existing empty-input ordering and asynchronous temporary retention where applicable.

Use test-only instrumentation or existing instrumentation; add no public counters.

Existing borrowed-`Exec` callers that require the former negative-stride refusal must explicitly choose `Forbid`. Default Allow deliberately changes that behavior.

Amend R3 to permit policy-authorized, same-placement, same-dtype input-layout normalization while retaining the prohibition on implicit transfers and conversions. Update `docs/spec/api-conventions.md`, affected storage/backend contracts, and the shared “no hidden layout conversions” rule in `tensor4all-agent-rules`; record the coordinated revisions.

The supplied material reports existing gathers despite that prohibition: this is the documentation/behavior inconsistency being resolved. Existing R2/R4/R5 issues remain unresolved; this change neither establishes `_into` primacy nor removes existing erasure or foreign public types.

**Scope is closed:** no receiver consolidation, new `_into` operations, resource bundling, naming cleanup, semantic options redesign, provider capability expansion, new numerical algorithms or profiling API.

## 5. **Open facts.**

- NEED: The original issue’s exact parameter-count constraint and withdrawn proposals, to verify that this request carrier satisfies the owner’s constraints.
- NEED: The crate owning `Engine` and its dependency relationship with the three families and `tenferro_tensor`, confirming that the stated type placement creates no dependency cycle.
- NEED: The complete immediate, eager-AD, backward and extension execution interfaces that must forward the request.
- NEED: The existing graph prepare/run signatures, cache layout guards and representation of normalization steps, including whether runtime-dependent requirements can be validated before numerical execution.
- NEED: Actual FFT lane-buffer, einsum lowering and linalg gather paths, to classify copies by purpose rather than audit shorthand.
- NEED: Existing error wrapping, scratch ownership, asynchronous retention mechanisms and test instrumentation.
- NEED: The exact R3/shared-rule text and coordinated repository revisions.

No repository implementation, compilation or runtime tests were performed; the Rust diagnostic above is predicted, not observed. The listed integration facts remain blockers to implementation readiness.
