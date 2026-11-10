収束文書は依頼どおり英語で書きました。リポジトリは参照しておらず、コンパイルも実行していません。主な結論は、運び手については A(独立引数)に譲り、traced プランのキャッシュキーと「gate であって selector ではない」点は B を維持する、というものです。

---

# Round 2 convergence (author of Proposal B)

## 1. Where the two designs actually differ

1. **Carrier of the policy.** A uses a final `MaterializationPolicy` argument; B uses an `ExecutionRequest<'a, R>` wrapper.
   - Break in B: every helper that makes two calls needs an explicit `.reborrow()`, the generic `R` plus `over` is an open-ended bundle, and `BufferPool` still sits outside the "request". This is a real defect.
   - Break in A: one more positional argument and a rote `Allow` at each call site. This is a preference, unless the issue's point 6 is a hard constraint (see section 5).
2. **Policy in the physical-plan cache key.** A includes it; B does not. Break in A: two cache entries hold identical plans, and the run-time layout check is still needed. This is a real but small defect (redundancy).
3. **Where a traced run receives the policy.** A fixes it at preparation; B supplies it at the run. Break in A: one prepared plan cannot serve an `Allow` run and a `Forbid` run. Whether this is a real cost depends on the prepare/run API (section 5).
4. **Gate versus selector.** B states that the provider picks the same path under both values. A does not exclude a provider picking a different kernel under `Forbid`. This is real underspecification in A, because results could then depend on the policy.
5. **Error shape.** A says "a typed error"; B gives the variant. A is incomplete here; we do not disagree.
6. **Home of the enum.** A puts it in `tenferro_tensor`; B left a `NEED`. The defect is B's: the `NEED` was unnecessary.
7. **Multi-operand atomicity.** A says to validate every operand before normalizing any; B only says "before any side effect". B is underspecified, and this is real.
8. **Asynchronous retention of normalized storage.** A covers it; B omits it. This is a real omission in B.
9. **Definition of input materialization.** A uses a purpose table with two `NEED`s; B uses one counterfactual sentence. This is a preference, since both classify every listed copy the same way.
10. **R2 claim and source audit.** B said "R2 holds" and asserted that no public function takes `&mut dyn Engine`. A says the change does not repair entry shapes. B overstated.
11. **Counting assertions.** A's tests count normalizations ("exactly one", "counted separately from allocations"); B defers any counter. This is a scope preference, but A's tests cannot be written without instrumentation that neither design ships.

## 2. What I concede

- **Difference 1, the carrier.** The plain argument is the smaller design: one `Copy` enum, with no lifetime, no generic parameter and no `reborrow`. My own objection to `impl Into` (it loses the implicit reborrow) applies equally to my wrapper.
  - Under A, `fn f(s: &mut dyn Engine, p: MaterializationPolicy) { a.svd(s, p)?; b.svd(s, p)?; }` compiles unchanged.
  - Under B, the same helper needs `r.reborrow()` on every call but the last.
- **Difference 1, forwarding.** The one advantage I claimed for the wrapper was that sugar cannot drop the policy. The compiler already enforces that for a plain argument:

  ```rust
  #![deny(unused_variables)]
  fn svd(a: &Tensor, s: &mut dyn Engine, materialization: MaterializationPolicy) -> Result<Svd> {
      svd_into(a, s, Default::default())   // drops the caller's policy
  }
  ```

  If this is rejected with "unused variable: `materialization`", forwarding is compiler-enforced without a wrapper. That holds wherever CI denies warnings (section 5).
- **Difference 6.** The enum belongs in `tenferro_tensor`, because the error variant already has to live there and all three families return its types.
- **Difference 7.** "Validate all operands, then normalize" must be a stated rule; without it an einsum can gather operand 0 and then refuse on operand 2.
- **Difference 8.** Normalized storage must outlive its asynchronous consumer through the existing retention path; my design said nothing about this.
- **Difference 10.** This change adds one convention and repairs none of the three entry shapes. I withdraw "R2 holds" and the `&mut dyn Engine` source audit.

## 3. What I insist on

- **Difference 2: the policy is not part of the plan cache key.**
  - Counterexample: prepare graph G on compact inputs under `Allow`, then under `Forbid`. A stores two entries with the same plan.
  - Where layouts are known at preparation, "this plan contains an input-normalization step" is a function of (graph, layouts), which the key already holds.
  - Where layouts are not known, A itself requires a run-time check. The key component therefore decides nothing in either case.
  - The plan records the fact, and the run compares it with the policy. A's own assertion 5 passes unchanged.
- **Difference 3: the policy is an argument of the call that submits numerical work.** This is the same rule as for immediate execution, so there is one convention and not two. If the repository's only traced entry is a combined prepare-and-run, the argument goes there and nothing else changes.
- **Difference 4: gate, not selector.**
  - Counterexample: a provider that under `Forbid` switches to a strided kernel returns results that differ in the last bits from `Allow`. It also needs a second numerical path, which A forbids elsewhere.
  - Deciding test: for every input where `Forbid` returns `Ok`, the output is bit-identical to `Allow`.
- **Difference 5: the error is one concrete variant with `op` and `input` index.** A's "identifies the operation and offending operand" is not implementable until the fields are named.
- **Difference 9: one sentence defines the gated copy** (rule 3 below). A's table rows are consequences of it, including "renaming a temporary to workspace exempts nothing". The definition itself then needs no `NEED`; only the facts about what FFT and einsum currently do remain open.
- **Difference 11: no counter in this change.** `Forbid` returning `Ok` already proves zero normalizations. "Exactly one under `Allow`" is replaced by "result equals the result on a compacted copy".

## 4. The merged design

**New items, both in `tenferro_tensor`:**

```rust
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MaterializationPolicy {
    #[default]
    Allow,
    Forbid,
}

// New variant of the existing #[non_exhaustive] tenferro_tensor::Error.
// `input` is the zero-based operand index.  Display text names `op` and `input` and says:
// "pass MaterializationPolicy::Allow or compact the input first".
InputMaterializationForbidden { op: &'static str, input: usize }
```

**Signature rule.** Every public entry in linalg, einsum and fft that executes numerical work gains exactly one parameter, `materialization: MaterializationPolicy`, in final position. This covers the six `*_with_exec` functions, the `TensorLinalgExt` methods, `einsum` and `einsum_with`, and every `FftExecutor` transform. Nothing else in any signature changes. Graph-building (traced) entries do not change.

```rust
// tenferro_linalg
pub fn svd_with_exec<T: LinalgScalar>(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
    full: bool,
    materialization: MaterializationPolicy,
) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<T::RealScalar>, TypedTensor<T>)>;

// tenferro_einsum (concrete trait method)
fn einsum(
    &self,
    subscripts: &str,
    session: &mut dyn Engine,
    materialization: MaterializationPolicy,
) -> tenferro_einsum::Result<Tensor>;

// tenferro_fft
impl FftExecutor {
    pub fn fft(
        &mut self,
        input: &Tensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        session: &mut dyn Engine,
        materialization: MaterializationPolicy,
    ) -> tenferro_tensor::Result<Tensor>;
}
```

**Callers:**

```rust
let (u, s, vh) = svd_with_exec(exec, buffers, &view, false, MaterializationPolicy::Allow)?;
let c = a.einsum("ij,jk->ik", &mut engine, MaterializationPolicy::Forbid)?;
let y = fft_executor.fft(&x, None, -1, norm, &mut engine, MaterializationPolicy::Allow)?;
```

**Rules:**

1. **Lifetime.** The value applies to that call only. `Engine`, `Exec`, `BufferPool`, `FftExecutor` and every `*Options` struct gain no field, method or setter that mentions it.
2. **Forwarding.** An entry that calls another executing entry passes its `materialization` through unchanged.
3. **Definition.** A copy is input materialization if and only if it would not occur had the same operand arrived, at the same placement and dtype, in a layout the selected provider consumes directly. Copies made for every layout (output allocation, intrinsic workspace, kernel packing) are never gated.
4. **Gate, not selector.** The provider selects the same path under both values. `Forbid` only turns a required input materialization into `InputMaterializationForbidden`.
5. **Admission order**, all before any allocation, write or submission:
   1. Provider, dtype and placement checks, with their existing errors, identical under both values.
   2. The layout gate over all operands; under `Forbid` the first offending index is reported.
   3. The zero-size shortcut.
   4. Under `Allow`, normalization of each affected operand once.
6. **No other permissions.** Neither value permits a device transfer, a dtype conversion, a backend fallback, destination staging or alias resolution.
7. **Asynchronous execution.** Normalized storage is kept alive by the existing retention mechanism until its consumer completes.
8. **Traced execution.**
   - The graph and the plan cache key do not contain the policy.
   - A physical plan records whether it contains an input-materialization step. Where layouts are unknown at preparation, this is determined at run.
   - The call that runs the plan takes `materialization` as its final argument and applies rule 5 before executing.
9. **Extensions.** A further family follows the signature rule and rules 1–8, and adds no policy type of its own.
10. **Behaviour change.** The six `*_with_exec` functions normalize a negative-stride input under `Allow`, where they refuse today. `Forbid` reproduces the refusal.
11. **Documents landing with the code.**
    - R3 becomes: "Never change placement or dtype implicitly. Same-placement, same-dtype input-layout normalization is done only by the provider, only under the invocation's `MaterializationPolicy`; `Forbid` asserts its absence."
    - `docs/spec/api-conventions.md` is updated to match.
    - The shared "no hidden layout conversions" rule in `tensor4all-agent-rules` is amended in a linked PR.

**Proof (tests):**

1. A negative-stride input under `Forbid` returns `InputMaterializationForbidden { input: 0, .. }`, and `FftExecutor::cache_stats()` is unchanged by the refused call.
2. The same input under `Allow` returns `Ok`, equal to the result on a compacted copy.
3. Wherever `Forbid` returns `Ok`, the output is bit-identical to `Allow`.
4. Directly consumable views (transposed, positive-stride, zero-stride, per provider) return `Ok` under `Forbid`.
5. A `Forbid` refusal followed by `Allow` on the same engine and input returns `Ok`.
6. An einsum whose last operand is the offender reports that index and leaves every destination unchanged.
7. An unsupported dtype or placement gives the same error under both values.
8. An empty negative-stride input is refused under `Forbid`.
9. One prepared plan runs under `Allow` and is refused under `Forbid`, with an unchanged cache key.
10. Source audit: every public executing entry in the three families has exactly one `MaterializationPolicy` parameter, in final position; no `*Options` struct and no `Engine` item mentions the type.

## 5. Open facts

- `NEED:` the exact wording of the issue's point 6. If it forbids adding a parameter to execution signatures as a hard constraint, concession 1 reverses and the carrier is B's `ExecutionRequest`; rules 1–11 are otherwise unaffected.
- `NEED:` whether CI builds the workspace with warnings denied (`unused_variables`). If it does not, rule 2 needs a lint or a test per forwarding entry.
- `NEED:` the traced prepare and run entry names and signatures, whether they are one call or two, and whether layouts are known at preparation.
- `NEED:` the eager AD session's execution entries, and whether a backward pass is a separate invocation that takes its own `materialization`.
- `NEED:` how `tenferro_einsum::Error` carries a `tenferro_tensor::Error`, so the new variant reaches einsum callers without a second variant.
- `NEED:` whether the CPU FFT compacts the whole input before lane iteration for some layouts (gated), or only fills per-lane buffers for every layout (not gated).
- `NEED:` which einsum lowering step permutes an operand into a dense temporary only for non-compact layouts, and the single point where that happens.
- `NEED:` the name of the existing asynchronous retention mechanism for CUDA temporaries referred to in rule 7.
- `NEED:` the list of six withdrawn proposals, to confirm that a final policy argument is not among them.
