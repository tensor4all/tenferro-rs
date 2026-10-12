# Rule: the fork line carries only submittable changes

Anything that cannot be submitted upstream as-is (or after a small reshape) is
not kept as a fork-only patch; for each held item, either reshape it into a
submittable change or remove the need from the consumer.

- **log1p**: drop the unsubmittable polynomial; prepare the native lowering in
  `cubecl-cpp` plus the `lower_unary_math_arith!` gate shape in `cubecl-llvm`
  (as `ExpOp`/`LogOp`), with a polyfill only where no native exists.
- **bool**: upstream has no portable bool-buffer representation, so the
  consumer stops using `&[bool]` buffers and carries bool tensors as `u8`
  (1 byte, bit-identical) with kernels writing 0/1; no CubeCL change needed,
  and the `SizedType for BoolType` hunk leaves the fork line.
