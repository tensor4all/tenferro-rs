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

## Route A for bool: state and design (choice made)

**Part 1 done** — `tensor4all/cubecl` `feat/tenferro-upstream-bool-buffers`
`5d0fa26b5`: `SizedType for BoolType` through the existing `sized!` macro (one
byte, the C++/LLVM native size), a comment naming the storage limitation, and an
inline unit test (`cargo test -p cubecl-ir --lib`: 34 passed / 0 failed,
`cargo fmt --all --check` clean).

**Part 2, the refusal (design).** Nothing enforces `TypeUsage::Buffer` today: the
feature tables only *declare* it (`register_type_usage`) and the only readers are
hand-written checks in runtime tests. Both backends that cannot store a bool
currently claim it:

- `crates/cubecl-wgpu/src/backend/wgsl.rs:76-91` registers `ElemType::Bool` with
  `TypeUsage::all()`
- `crates/cubecl-wgpu/src/backend/vulkan.rs:528-547` does the same

So part 2 is:

1. Narrow those two registrations so bool keeps its value uses but not
   `TypeUsage::Buffer` (WGSL has no host-shareable `bool` buffer, and the Vulkan
   path has no physical bool storage either). `cubecl-cuda`, `cubecl-cpu` and
   `cubecl-cpp` keep registering bool with `Buffer`, and they are correct: C++
   prints `bool`, LLVM uses byte-sized `i1`.
2. Enforce the claim where both the element types and the device features are
   available — kernel compilation, which sees the IR's buffer globals and the
   device properties. A shared helper (e.g. in `cubecl-ir`) walks the buffer
   element types and refuses a type the device cannot store, called from the wgpu
   compiler (`crates/cubecl-wgpu/src/compiler/`) and the SPIR-V compiler
   (`crates/cubecl-spirv/src/compiler.rs`), both of which already return a
   result. That turns the tables from documentation into a check for every type,
   not just bool.
3. Tests: the part-1 unit test for the size; a feature-table test that the wgpu
   backends do not claim `Buffer` for bool; a compile-refusal test for a kernel
   with a `&[bool]` buffer on those targets; and a positive runtime test that a
   `&[bool]` kernel works on the backends that can store it (CUDA/CPU).

This becomes PR candidate 4, after the arithmetic series (1) and the log1p native
lowering (3), and like them it needs the gpt-6.1-sol review then approval.
