# Rule: the fork line carries only submittable changes

Anything that cannot be submitted upstream as-is (or after a small reshape) is
not kept as a fork-only patch; for each held item, either reshape it into a
submittable change or remove the need from the consumer.

- **log1p**: drop the unsubmittable polynomial; prepare the native lowering in
  `cubecl-cpp` plus the `lower_unary_math_arith!` gate shape in `cubecl-llvm`
  (as `ExpOp`/`LogOp`), with a polyfill only where no native exists.
- **bool**: upstream has no portable bool-buffer representation (WGSL
  `array<bool>`, no SPIR-V physical bool buffer, packed bool vectors). Two routes:
  **A** keep `SizedType for BoolType` and add an explicit rejection where no
  storage representation exists (submittable, smaller); **B** carry bool tensors
  as `u8` in the consumer and adapt its 65 boolean sites (no CubeCL change, larger
  refactor). Choice open.
