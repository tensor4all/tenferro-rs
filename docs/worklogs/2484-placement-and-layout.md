# One placement answer, one layout legality rule (#2484, rows 1 and 4)

## Decisions

- **Host memory is a property of the payload, decided once.** `is_host_memory()`
  is the only host/device predicate: a plain host owner, a host slice and a
  CPU-resident allocation group answer true, a backend buffer answers false, and
  `Placement::memory_kind` is never consulted. `TensorRead::try_host`,
  `try_mut_host`, the CPU provider's operand admission
  (`structural::require_host_memory`) and linalg's borrowed entries all reduce to
  it, so the three former judgements — the `Representation` type axis, the
  placement label, and the storage variant — now have one runtime answer.
- **Chosen behaviour for the audit's reproducer: reject the label that creates
  it.** `TypedTensor::set_placement` returns `Result` and refuses
  `MemoryKind::Device` or `Managed` on a plain host payload, so "host tensor
  relabelled device-resident" — the input on which the CPU path said no and
  linalg said yes — cannot be built. Rejected alternatives: (a) accept the state
  in both paths (the device path cannot pick such a tensor up at all — it keys on
  the backend family and allocation, which a host payload does not have — so
  accepting would only hide a label that no consumer can act on); (b) keep the
  label check in the CPU provider and drop it in linalg, which leaves two
  judgements and no rule for which one is right.
- **The opposite direction is deliberately still legal.** A backend payload is
  an opaque `BackendAllocation`, so the tensor crate cannot decide whether the
  handle is device memory or host-mapped, and the runtime labels such a payload
  with the caller's storage class (its own tests build a backend buffer under a
  host label). Refusing that would make a routing key unrepresentable without
  removing any alias risk, because the payload — not the label — decides every
  host admission. This is why `from_buffer_col_major` validates only the
  host-payload direction.
- **Layout legality is strided-rs' injectivity analysis.** `strided-view 0.7.1`
  already owns `is_injective_layout`, the predicate that guards its own mutable
  kernels; `tenferro-tensor-core` now depends on it and
  `TensorLayout::validate_mutable_no_overlap` is a thin wrapper. The local
  stride-span proof and the 4096-element exact-enumeration ceiling are deleted,
  so a legal layout is no longer refused for being large.

## Verification conclusions and constraints

- The two new integration tests pin both halves: one tensor through a CPU entry
  and a linalg entry gives one verdict (accepted together for a host payload
  whatever its label, refused together with the same error kind for a backend
  payload), and the contradictory relabel is refused where it is written.
  Restoring either the retired label rule in the CPU view admission or the
  unfallible relabel makes one of them fail.
- Cost of dropping the ceiling: the decision is allocation-free for every
  separated, two-axis and difference-search-decidable layout (no dependence on
  element count — a 1 G-element compact layout and an 18 M-element two-axis
  interleaved layout each decide in about 40 ns). The exact fallback is bounded
  at `2^24` block elements with at most 8 bytes per element, and is reachable
  only for a three-or-more-axis interleaved block whose difference search is out
  of budget. Worst measured cost is high rank, not large element count: rank 12
  costs about 250 ns against 145 ns for the retired sorted-span proof, because
  the shared analysis reduces axes by repeated selection (O(rank^2)) instead of
  sorting once.
- Limit of the shared analysis: an interleaved block of three or more axes whose
  difference search exceeds its budget *and* whose element count exceeds `2^24`
  is reported non-injective, i.e. conservatively refused. That residual is
  strided-rs' documented contract, not a tenferro-local choice.
