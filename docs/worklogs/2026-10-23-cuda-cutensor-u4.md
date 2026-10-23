# U4, part 3: cuTENSOR contraction with shape-keyed plans

Date: 2026-10-23. Issue: [tenferro-rs#2208](https://github.com/tensor4all/tenferro-rs/issues/2208)
unit U4, vendor operations (cuBLAS landed in #2255; cuSOLVER and cuFFT remain).

## What landed

- `cuda_next::Cutensor`: one cuTENSOR handle per runtime, a `d = alpha * a * b + beta * c`
  contraction over caller-declared modes, extents and strides, executed on the runtime's
  own stream.
- **Shape-keyed plans.** A plan and its workspace depend only on the descriptor signature
  (dtype + every operand's modes, extents and strides), so the cache is keyed by exactly
  that. The first call with a signature pays descriptor creation, the workspace estimate
  and the plan build (`cold_plans()`); each later call with the same signature reuses both
  (`warm_hits()`) and runs no cuTENSOR planning at all. This is the vendor half of U3's
  "zero compile, zero planning on the warm path" invariant.
- The entry points are bound from the library file this crate found, not from a fixed list
  of sonames: `TENFERRO_CUTENSOR_PATH` first, then
  `/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2`. This is required in practice,
  because the distribution installs cuTENSOR in a versioned subdirectory that no plain
  `libcutensor.so.N` lookup finds, and it keeps the search order consistent with the
  existing cuTENSOR loader.

## Three ABI facts this version enforces

1. `cutensorCreateContraction` rejects a **null C descriptor**; a contraction without an
   addend passes the destination as C.
2. `cutensorContract` likewise rejects a **null C pointer**, so the same call passes the
   destination pointer for C. `beta` is zero, which cuTENSOR documents as "C is not read",
   so the destination's prior contents cannot influence the result.
3. `CUTENSOR_COMPUTE_DESC_*` are exported as data whose **pointee** is the descriptor
   (pointers, so `d + beta * c` uses one more indirection than the symbol's address).
   Reading the symbol's address directly yields `CUTENSOR_STATUS_NOT_INITIALIZED` from
   `cutensorCreateContraction`, which is a confusing symptom for a descriptor mistake.

Also: this header has `CUTENSOR_WORKSPACE_DEFAULT`/`MIN`/`MAX` and no
`CUTENSOR_WORKSPACE_RECOMMENDED`; `DEFAULT` is the recommended-estimate mode.

## Evidence

`tests/seam_contraction.rs` on the local A100 (driver 580.126.09, CUDA 12.6), against host
column-major references:

- `a_matrix_product_as_a_contraction_matches_the_host` — a 23x31 times 31x17 f64 product
  expressed as modes `(i,j)(j,k) -> (i,k)`, exact equality, `cold_plans() == 1`.
- `a_repeated_signature_reuses_the_plan` — three identical calls: one plan, two warm hits,
  one cached plan, and the value is right.
- `a_permuted_destination_mode_matches_the_host` — the destination declared with modes
  `(k, i)` and strides `(1, n)`, i.e. the transposed output layout, matched element by
  element.
- `disagreeing_operand_lists_are_refused` — an operand whose mode, extent and stride lists
  disagree is a typed refusal before any library call.

## Not done (rest of U4)

cuSOLVER linalg and cuFFT on the runtime's stream, each with its own handle and plan cache.
