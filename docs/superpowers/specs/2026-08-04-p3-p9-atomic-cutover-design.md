# P3+P9 Atomic Host Ownership, Submission, and AD Retention Design

Date: 2026-08-04

Status: design complete; implementation reconciliation required before P8/P10 closure

Authority: #1555, #1559, #1565, `docs/design/storage-ownership-contracts.md`
(G2/G3/G4/G7), and `scripts/storage-ownership-contracts.toml`

## Scope

P3 and P9 are one atomic ownership cutover. The public host tensor family,
runtime submission, result aliasing, and AD/checkpoint retention must use one
allocation-group ownership model. This specification reconciles the earlier
P3/P9 design with the final G2/G3/G7 signatures and records current
transitional paths that must not survive into P8/P10 or the P13-A candidate.
It does not change code or ledger state.

The design preserves Rust aliasing and lifetime soundness, provider retirement,
numerical correctness, and explicit copy/transfer behavior. It does not add a
compatibility owner, global liveness registry, cancellation protocol,
quarantine/poison/retry state machine, cryptographic evidence, or repeated
static descriptor validation.

## Ownership model

### Tensor owners and views

`TypedTensor<T, R>` and dtype-erased `Tensor` are move-only owners. Each owner
contains one `AllocationGroup` and one local `DescriptorSlot`; it has no
parallel `StorageBuffer`, provider handle, or second tensor owner. Immutable and
mutable views borrow that group. `as_view()` and `as_view_mut()` preserve static
rank and are allocation-, refcount-, layout-clone-, provider-resolution-, and
synchronization-free.

`duplicate()` is the explicit same-placement copy boundary and creates a fresh
allocation identity. A consuming descriptor-only operation moves the existing
group and owner. No operation silently duplicates, materializes, uploads, or
downloads.

### One group table

```rust
pub struct AllocationGroup {
    allocations: Vec<Option<OwnedStorage>>,
    descriptors: Vec<Option<DescriptorRecord>>,
}
```

There is no `tensor_owners: Vec<Option<Tensor>>` or another owner table.
Building a group from tensors consumes each tensor, moves its allocation slots
and descriptor records into the destination group, and remaps only local slot
indices. Repeated logical bindings copy `DescriptorSlot` lookup metadata, not
owners. A slot never authorizes access independently of a group borrow.

## Detached submission

The normative public shape is:

```rust
pub struct ExecutionInputs {
    group: AllocationGroup,
    bindings: Box<[DescriptorSlot]>,
}

pub fn submit(
    &self,
    program: &CompiledGraph,
    inputs: ExecutionInputs,
) -> Result<ExecutionHandle, SubmitRejected>;

impl ExecutionHandle {
    pub fn wait(self) -> ExecutionOutcome;
}

pub struct SubmitRejected {
    cause: SubmitError,
    inputs: ExecutionInputs,
}

impl SubmitRejected {
    pub fn into_parts(self) -> (SubmitError, ExecutionInputs);
}

pub enum ExecutionOutcome {
    Completed(ExecutionBundle),
    RetiredFailed {
        cause: ExecutionError,
        inputs: ExecutionInputs,
    },
    CompletionUnproven {
        cause: CompletionError,
        diagnostic_keys: Box<[DiagnosticKey]>,
    },
}
```

### Admission boundary

Validation, graph preparation, provider preparation that cannot enqueue, and
worker creation occur before admission. Worker-spawn failure is a
`SubmitRejected`; it returns the exact input package. An implementation using a
per-submit thread keeps the package in an unstarted in-flight record and takes
it back if spawn fails. It must not have an ownerless `WorkerSpawn` variant.

Admission is the first provider call that may enqueue work. Once that call is
reached, the worker/reaper exclusively owns the group, prepared bindings,
events, root holds, and provider contexts. Immediate post-boundary errors enter
`Draining`; they never return an owner directly.

A detached worker or provider panic is observed at the existing worker/thread
or FFI boundary and also enters `Draining`. Panic catching is an error-reporting
boundary, not an ownership proof and not a provider-destructor recovery
mechanism.

### Terminal ownership

- `Completed` is published only after every admitted event domain has retired.
- `RetiredFailed` returns the exact inputs only after retirement is proven.
- `CompletionUnproven` returns diagnostics and no owner. A provider-private
  record permanently retains the package, bindings, event, roots, and provider
  context because safe release is unproven.
- Dropping `ExecutionHandle` detaches observation. It does not cancel work,
  alter terminal state, or release resources early.

There is no public retry or owner-recovery API for `CompletionUnproven`.

## Alias-safe results

```rust
pub struct ExecutionBundle {
    group: AllocationGroup,
    outputs: Box<[ExecutionOutput]>,
}

pub enum ExecutionOutput {
    Tensor(DescriptorSlot),
    Metadata(OutputMetadata),
}

pub enum OutputRef<'a> {
    Tensor(TensorView<'a>),
    Metadata(&'a OutputMetadata),
}

pub enum OutputExtractError {
    InvalidOutput,
    MetadataOutput,
    Extract(ExtractError),
}

impl ExecutionBundle {
    pub fn output(&self, output: usize)
        -> Result<OutputRef<'_>, OutputAccessError>;

    pub fn into_output(self, output: usize)
        -> Result<Tensor, (Self, OutputExtractError)>;
}
```

Identity, metadata-only, repeated, and duplicate outputs are descriptor slots
in one group. `output()` returns a borrow. `into_output()` consumes the whole
bundle and delegates to G2 consuming extraction. Success moves one existing
owner while all repeated aliases, remaining descriptors, and the output map
disappear together. Failure returns the exact unchanged bundle. No extracted
boolean, tombstone generation, or `Completed(Vec<Tensor>)` representation is
permitted.

## Synchronous borrowed execution

```rust
pub struct ScopedReadInputs<'env> {
    bindings: Box<[ScopedReadBinding<'env>]>,
}

pub struct ScopedReadBinding<'env> {
    tensor: TensorView<'env>,
}

pub fn execute_scoped_read_only<'env>(
    &self,
    program: &CompiledGraph,
    inputs: ScopedReadInputs<'env>,
) -> Result<ScopedExecutionOutcome<'env>, ScopedSubmitRejected<'env>>;

pub enum ScopedExecutionOutcome<'env> {
    Completed(ScopedExecutionBundle<'env>),
    RetiredFailed {
        cause: ExecutionError,
        inputs: ScopedReadInputs<'env>,
    },
}

pub struct ScopedExecutionBundle<'env> {
    owned: AllocationGroup,
    outputs: Box<[ScopedOutput<'env>]>,
}

pub enum ScopedOutput<'env> {
    Borrowed(TensorView<'env>),
    Owned(DescriptorSlot),
    Metadata(OutputMetadata),
}
```

This API is immutable-input-only. It is supported only by providers that prove
all work retired before return or unwind. Host/CPU may support it. CUDA,
WebGPU, Metal, and any asynchronous provider reject before admission. Borrowed
identity outputs remain bounded by `'env`; fresh outputs live in `owned`.
Consuming extraction succeeds only for an `Owned` output and otherwise returns
the unchanged scoped bundle. Soundness does not depend on catch-unwind or a
`Drop` callback running.

## AD and checkpoint retention

### Direct retention container

```rust
struct RetentionContainer {
    group: AllocationGroup,
}

struct AdValueRecord {
    container: Arc<RetentionContainer>,
    slot: DescriptorSlot,
}

#[derive(Clone)]
pub struct EagerTensor {
    record: Arc<AdValueRecord>,
}
```

The container owns each physical allocation once. `EagerTensor::clone()`
clones a read-only descriptor-record handle. It cannot produce a mutable view,
owner projection, or write capability. A tape/checkpoint retains the same kind
of direct record; no external key-to-owner registry participates in lifetime.
`ValueKey` is only local lookup metadata.

The public `Arc<Tensor>` surface is replaced by:

```rust
impl EagerTensor {
    pub fn value(&self) -> Result<ValueGuard<'_>>;
    pub fn duplicate_value(&self) -> Result<Tensor>;
    pub fn into_value(self) -> Result<Tensor, IntoValueError<Self>>;
}

pub enum IntoValueError<H> {
    NotUnique(H),
    Extract { value: H, error: ExtractError },
}
```

`ValueGuard` borrows a descriptor view and requests G1 host access only when
host bytes are requested. If an eager value still represents pending
computation, `value()` is the documented evaluation/synchronization boundary
and records any fresh kernel output allocation as `OperationOutput`; it never
copies an already concrete retained value merely to expose the guard.
`duplicate_value` records an explicit copy and fresh destination allocation.
`into_value` first attempts structural `Arc` unwrapping. This decides only whether a zero-copy move is possible; it is not
write authority. Non-unique failure returns the usable handle and does not call
G2 extraction. A later G2 failure reconstructs and returns the same handle.

`materialized() -> Arc<Tensor>`, `materialized_arc()`, and
`Arc<OnceLock<Arc<Tensor>>>` caches are absent.

### Gradients

Backward returns a move-only `Gradients` bundle backed by one allocation group:

```rust
impl Gradients {
    pub fn grad(&self, key: &ValueKey) -> Option<TensorView<'_>>;
    pub fn take_grad(
        &mut self,
        key: &ValueKey,
    ) -> Result<Option<Tensor>, ExtractError>;
}
```

`Ok(None)` means no gradient. Successful extraction moves one owner. An
`ExtractError` leaves the bundle unchanged. Accumulated gradient state retains
descriptor records/containers, not `Arc<Tensor>` owners.

### Checkpoints and reinterpretation

Only checkpoint boundary inputs/outputs enter the retained group. Interior
values are omitted and drop after forward use unless separately retained.
Backward recomputation creates fresh owners classified as
`CheckpointRecomputeOutput`.

A retained real/complex alias is another descriptor in the same container.
Mutable reinterpretation requires exclusive ownership and is unavailable while
another handle, tape, checkpoint, or execution record retains the container.
The only alternative is an explicit duplicate.

### Copy and allocation accounting

Retention, handle clone, alias descriptor creation, and checkpoint boundary
recording emit neither copy nor allocation events. Copy reasons are limited to
`ExplicitDuplicate` and `Transfer`; allocation reasons distinguish operation
outputs, checkpoint recomputation, explicit duplicate destinations, and
transfer destinations. CPU and one real asynchronous accelerator must observe
zero retention-attributable events.

## Required removals

The reconciled implementation physically removes:

- `AllocationGroup::tensor_owners` and `tensor_refs` as a parallel owner path;
- cloneable `TensorValue { Arc<TensorOwnerRecord> }` ownership;
- `Arc<Tensor>` materialization and gradient APIs/caches;
- `ProgramBindings` and traced/checkpoint maps that retain `Arc<Tensor>`;
- `ExecutionOutcome::Completed(Vec<Tensor>)`;
- ownerless pre-enqueue worker-spawn errors;
- `GradSlot = Arc<Mutex<Option<Arc<Tensor>>>>`;
- borrow-and-shallow-clone runtime submission and any P3/P9 adapter.

Read-only graph/program metadata may remain shared, but it cannot own a second
tensor or authorize mutation.

## State summary

| Transition | Owner visible to caller | Provider work | Result |
|---|---:|---:|---|
| detached rejection before admission | yes, exact inputs | no enqueue possible | `SubmitRejected` |
| admitted/running/draining | no | possible or active | worker/reaper retains all resources |
| retirement proven, success | yes, bundle | retired | `Completed` |
| retirement proven, failure/panic | yes, exact inputs | retired | `RetiredFailed` |
| retirement unproven | no | unknown | `CompletionUnproven`; private permanent retention |
| scoped asynchronous provider | borrowed inputs unchanged | none | pre-admission unsupported |
| scoped synchronous provider | borrow remains in call | retired before return | completed or retired-failed scoped bundle |

## Verification ownership

P9 acceptance covers exact pre-admission recovery, terminal outcomes, handle
detachment, repeated/duplicate outputs, consuming extraction, borrowed-provider
eligibility, no borrowed work across return/unwind, compile-fail host-guard
exclusion, zero-copy AD/checkpoint retention, and source/API inventory of the
removed paths. The ledger command remains:

```text
cargo test -p tenferro-tensor --test storage_compile_contract
```

Runtime and AD integration suites provide the behavioral proof that the compile
artifact alone cannot express. Hardware AD and asynchronous retirement evidence
belongs to P11. This design does not itself activate or modify a ledger row.
