# CubeCL 0.11 fusion codegen: IR seam probe

Status: design for the probe; implementation not started. This is the last
blocking item of the 0.11 migration (`kernels/reduce` is already clean; the
crate sits at 61 errors, 23 of them here). Independently reviewed on
`openai/gpt-6.1-sol` (two rounds); the review corrected the guard mapping, the
conditional-body and NaN operations, the insertion recipe and the acceptance
criteria. Those corrections are folded in below.

## Problem

`crates/tenferro-gpu/src/cubecl/fusion/codegen.rs` builds a kernel definition by
hand from the 0.10 IR (`Instruction::new(op, out)`, `ManagedVariable`,
`Operator::{Arithmetic, Comparison, Select, Clamp, IndexOperator,
IndexAssignOperator}`, `Metadata::Length`, `Variable::builtin(Builtin::…)`).
All of those types were removed in 0.11 and replaced by the Pliron-dialect IR, so
the whole file is a compile error, not a mechanical rename.

`fusion/launch.rs` already implements the 0.11 `CubeKernel` shape: `id()`
(plan + cube_dim + address_type), `address_type() -> ElemType`, `define() ->
KernelDefinition`. Only `build_kernel_definition()` has to produce a definition
in the 0.11 form.

## Verified 0.11 seam (read from the combined upstream port worktree)

Insertion idiom: `Op::new(ctx, …)` then, for an operation that produces an SSA
result, `scope.register_with_result(&op) -> Value`
(`cubecl-ir/src/scope.rs:561`; used by
`cubecl-core/src/frontend/container/buffer_len.rs:7`). `Scope::register(&dyn Op)
-> ()` (`scope.rs:551`) is for terminators, stores and other resultless
operations — the `register(…) -> Value` form in the first draft of this note was
wrong. Scalars: `KernelBuilder::scalar(ElemType) -> usize`; buffers/tensors:
`KernelBuilder::{buffer,tensor}(TypeHandle) -> Value`; `inplace(position) -> Value`;
`build() -> KernelDefinition` (`cubecl-core/src/compute/builder.rs`).
`BufferLenOp` takes the *buffer binding index* from that registration order, not a
buffer `Value` (`cubecl-ir/src/dialect/general.rs:308`).

Conditional body: `IfOp::new(ctx, cond)`, then a child scope per branch via
`scope.child(OpInserter::new_at_block_end(block))`, populate, `terminate_yield()`,
and `scope.register(&if_op)` — the shape at
`cubecl-core/src/frontend/branch.rs:106–117`. The 0.10 `Branch::If` /
argumentless `scope.child()` pair no longer exists.

Every operation the fusion plan can emit exists as a public dialect op:

| tenferro `ElementwiseFusionOp` / kernel need | 0.11 op |
|---|---|
| Add / Multiply / Divide | `math.f_add` `FAddOp`, `math.f_mul` `FMulOp`, `math.f_div` `FDivOp` |
| Negate | `math.f_neg` `FNegOp` |
| Abs | `math.f_abs` `FAbsOp`; complex `CAbsOp` |
| Conj | `math.c_conj` `CConjOp` |
| Maximum / Minimum | `cmp.f_max` `FMaxOp`, `cmp.f_min` `FMinOp`; NaN-propagating `FMaxNanOp`, `FMinNanOp` |
| Clamp | `FClampOp` (NaN-propagating `FClampNanOp`) |
| Exp, Log, Expm1, Log1p | `ExpOp`, `LogOp`, `Expm1Op`, `Log1pOp` |
| Sin, Cos, Tanh | `SinOp`, `CosOp`, `TanhOp` |
| Sqrt, Rsqrt | `SqrtOp`, `RsqrtOp` |
| Pow | `math.powf` `PowfOp` |
| Erf | `math.erf` `ErfOp` |
| bounds guard | `cmp.u_less_than` `ULessThanOp` — the position and the length are `usize`, so the integer form (`cubecl-core/src/frontend/operation/cmp.rs:397`); `FLessThanOp` would lower to `FCmp` (`cubecl-llvm/src/shared/to_llvm/cmp.rs:30,54,68`) |
| conditional body (`Branch::If`) | `branch::IfOp` + `branch::YieldOp` (`cubecl-ir/src/dialect/branch.rs:252,63`) |
| `Comparison::IsNan` (codegen.rs:180,182) | `math.is_nan` `IsNanOp` (`cubecl-ir/src/dialect/math.rs:207`) |
| select | `general::SelectOp` |
| index / load / store | `memory::{IndexOp, LoadOp, StoreOp}` |
| buffer length (`Metadata::Length`) | `general::BufferLenOp` |
| absolute position (`Builtin::AbsolutePos`) | `general::ReadBuiltinOp` with `BuiltinAttr(Builtin::AbsolutePos)` |
| casts | `general::CastOp` |

No fusion op needs an upstream addition. This is the finding that makes the
probe worth running before the full port.

## Probe

One isolated f32 kernel, hand-built through the public seam, no shim and no
compat layer:

- two inputs `a`, `b`, two outputs `out0`, `out1` (multi-output is a live
  requirement of the region work), all `f32` buffers registered through
  `KernelBuilder::buffer`, fed with position-varying non-zero data;
- body: bounds guard `ABSOLUTE_POS < len`; `out0 = a + b`, `out1 = -(a + b)`.
  The two outputs must be *different expressions*: writing the same value to both
  cannot catch an emitter that always routes through the first output value id or
  ignores the plan's per-output selection (`codegen.rs:92–97`,
  `crates/tenferro-tensor/src/backend.rs:1311`);
- `CubeKernel::id()` keeps the existing plan/type/cube/address identity, so a
  repeated launch with the same plan must reuse the compiled kernel and a
  changed plan/address type must not;
- launched through the existing `fusion/launch.rs` path, not a new one.

Acceptance:

1. `build_kernel_definition()` returns a definition accepted by the CUDA backend
   and the generated kernel compiles (C++/NVVM path).
2. CUDA numerical result equals the CPU reference for f32, including the
   write to both outputs.
3. The bounds guard is preserved *and* the valid remainder is written: launch
   `n = 257` with the 256-lane configuration (two cubes). Check the 257th element
   is written and that `n+1`-onward is untouched (IR inspection and/or an
   OOB-sensitive check); downloading the first `n` elements alone cannot show
   whether surplus lanes wrote past the logical output.
4. Cache identity, precisely: a warm re-launch of the same plan reuses the
   resident kernel; a changed plan and a changed address type each get a distinct
   structural id; the same plan at another runtime length gives correct results
   with an unchanged id (length stays runtime metadata). A cold launch may
   legitimately perform zero new NVRTC compilations (persistent server cache and
   source-hash reuse: `cubecl-server/src/compiler.rs:221–235`,
   `cubecl-cuda/src/compute/context.rs:161–200,335–349`), so "one compile" is not
   the assertion.

The complex path is a second step after the f32 probe passes: complex fusion
supports only Add/Multiply/Negate/Conj/Divide (`classify.rs:118–127`), which map
to the same `FAddOp`/`FMulOp`/`FNegOp`/`FDivOp`/`CConjOp`; there is no complex
`Abs` to port.

## Risks / open items

- Ordinary arithmetics are settled by source and need no probe step: the scalar
  unary/binary ops infer the result type from their operands and take no separate
  vector-size argument (`cubecl-ir/src/dialect/base.rs:198–217,238–257`).
- `nan_propagating_max`/`nan_propagating_min` stay a full-port decision, not
  probe work. `FMaxNanOp`/`FMinNanOp` are NaN-propagating but their contract
  leaves opposite-zero signs and NaN payload/sign unspecified
  (`cubecl-ir/src/dialect/cmp.rs:253–287`); tenferro's helpers delegate non-NaN
  values to plain `max`/`min` and the existing test only covers NaN/finite pairs
  (`kernels/helpers.rs:41–62`, `cubecl/tests/elementwise_tests.rs:1732–1740`), so
  bitwise equivalence is *not* established by the current tests. Decide with a
  test during the full port, not by assumption here.
- No definition-side read/write authority is required: `StoreOp` declares its
  pointer-write effect and visibility analysis infers the attributes
  (`cubecl-ir/src/dialect/memory.rs:433–452`,
  `cubecl-opt/src/passes/annotate_buffer_visibility.rs:35–45`).
  `KernelLauncher::declare_io` is the pre-compilation classification used when
  compiled analysis is unavailable; calling it in the existing launcher keeps
  inputs from being classified writable on a compile failure
  (`cubecl-core/src/compute/launcher.rs:112–122,230`).

## Delivery state

The `path` dependencies on the local port trees in this worktree's `Cargo.toml`
are staging so the migration can be built against the proposed upstream shape.
They are not the deliverable: the final state is a git pin to upstream with the
ports merged, and no extra patch section beyond what upstream then carries.

## Non-goals

## Non-goals

- No fork shim, no compat layer, no re-introduction of a hand-built 0.10 IR.
- No change to the fusion plan language or to `classify.rs`'s op support set.
