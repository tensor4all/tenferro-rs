# CubeCL 0.11 fusion codegen: IR seam probe

Status: design for the probe; implementation not started. This is the last
blocking item of the 0.11 migration (`kernels/reduce` is already clean; the
crate sits at 61 errors, 23 of them here).

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

Insertion idiom: `Op::new(ctx, …)` then `scope.register(&op) -> Value`
(`cubecl-core/src/frontend/container/buffer_len.rs:7`,
`cubecl-core/src/frontend/cmma.rs:903`). Scalars: `KernelBuilder::scalar(ElemType)
-> usize`; buffers/tensors: `KernelBuilder::{buffer,tensor}(TypeHandle) -> Value`;
`inplace(position) -> Value`; `build() -> KernelDefinition`
(`cubecl-core/src/compute/builder.rs`).

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
| bounds guard | `cmp.f_less_than` `FLessThanOp` |
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
  `KernelBuilder::buffer`;
- body: bounds guard `ABSOLUTE_POS < len`; `out_i = -((a+b))`, i.e. load, add,
  negate, store into both outputs;
- `CubeKernel::id()` keeps the existing plan/type/cube/address identity, so a
  repeated launch with the same plan must reuse the compiled kernel and a
  changed plan/address type must not;
- launched through the existing `fusion/launch.rs` path, not a new one.

Acceptance:

1. `build_kernel_definition()` returns a definition accepted by the CUDA backend
   and the generated kernel compiles (C++/NVVM path).
2. CUDA numerical result equals the CPU reference for f32, including the
   write to both outputs.
3. The bounds guard is preserved: with `n` not a multiple of the cube size, the
   tail elements are not written.
4. Cache identity: same plan+address type ⇒ one compile; changed plan ⇒ new id.

The complex path (`CConjOp`/`CAbsOp`, `ComplexCore`) is a second step after the
f32 probe passes; it reuses the same helpers.

## Risks / open items to settle during the probe

- Whether the arithmetic op constructors take the same operand shape as 0.10
  (e.g. an explicit vector size), i.e. whether the emitted `Value`s and
  `TypeHandle`s line up without an extra cast.
- Whether `nan_propagating_max/min` (tenferro helpers with signed-zero and NaN
  semantics fixed by tests) can be expressed as `FMaxNanOp`/`FMinNanOp` with
  identical semantics, or must stay an explicit compare+select sequence. The
  latter is acceptable and must be chosen by test, not by assumption.
- Whether the definition needs `declare_io`-style read/write authority for the
  two outputs, which the 0.11 write path enforces.

## Non-goals

- No fork shim, no compat layer, no re-introduction of a hand-built 0.10 IR.
- No change to the fusion plan language or to `classify.rs`'s op support set.
