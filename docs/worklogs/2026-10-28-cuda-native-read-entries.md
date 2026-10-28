# CUDA read entries consume a view's own layout (#2223 item 1)

## Decisions

- One crate-private boundary, `CudaEngine::read_region`, yields what the catalogue's
  planners take: the allocation, the dtype, the operand's own `Layout`, that allocation's
  length in elements and the logical shape. It applies `device_read`'s acceptance (healthy
  engine, device operand of this runtime, no host operand) before any planner sees the
  operand, so an empty or no-op input is validated exactly like a live one. `device_read`
  stays for the entries that genuinely need a compact copy.
- Each overridden `*_read` entry owns the operation and its owned twin forwards with
  `TensorRead::from_tensor(&t)`, so an operation has one body. `gather_read` lives on
  `Engine` (that is where the trait declares it) and `TensorIndexing::gather` forwards to it.
- `reverse_read` validates the named axes first, negates the remaining axes' strides of the
  operand's own layout with checked arithmetic and moves the offset to each far end, then
  materializes once; no staging copy is inserted first. An empty region skips the geometry
  arithmetic, and a singleton axis keeps its geometry because reversing it is the identity.
- `scatter_read` opens with a strided-to-strided copy of the operand region into the fresh
  output. That copy is the operation's required output write, so it is not operator staging;
  the additive pass then reads `updates` through its own layout.
- `concatenate` resolves and validates every input region before allocating the output or
  preparing a dispatcher, then copies each input straight into the output's slab for it (the
  output's compact strides over the input's extents, offset by the running axis offset times the
  output's stride on the concatenating axis). The old read default already staged a borrowed view
  into a compact tensor, so the native path replaces that staging with a direct copy rather than
  removing an additional compaction. `ConcatenateCopy` keeps only its general-path caller at
  rank above `MAX_RANK`.
- `embed_diagonal` keeps its zero-filled output but replaces the compact-source entry with a
  strided-to-strided copy into a diagonal *view* of the output: input axis `i` reads output
  axis `i + (i >= axis_b)`, and `axis_a` reads the inserted axis as well, so its stride is
  the sum of the two output strides. The compact-source kernel remains the general-path
  caller for a wider rank.
- Rank ceiling: a view wider than `t4a_cuda_kernels::MAX_RANK` has no by-value layout
  argument, so its entry keeps one staging copy and the catalogue's device-geometry path.
  This is marked `// ponytail:` at each site; removing it needs a general-path
  strided-to-strided copy (slice, pad, dynamic-slice, gather, scatter and both diagonal
  entries) which is a separate change.
- Not in this unit, by scope: `tril`/`triu` (compact-only kernel), `dynamic_update_slice` (no
  kernel) and the `gather`/`scatter` **index** operand (#1930).

## Verification conclusions and constraints

- `CudaEngine::staged_reads` counts operand materializations only: the `TensorRead::View`
  branch of `device_read`. `tril_read` on a view and `to_contiguous_read` each move it by one,
  which is the counter's negative control.
- The no-staging unit test drives slice, dynamic-slice, pad, reverse, extract/embed diagonal,
  gather, scatter and concatenate on a strided view and requires the counter not to move;
  before this change each view operand cost one staged read (two for scatter).
- Value tests compare slice, dynamic slice, pad, reverse, gather's operand, scatter's operand
  and update window, concatenate, and both diagonal entries on transposed, offset and
  negative-stride f64 views. Complex dtype covers slice, reverse and both diagonal entries;
  empty operands cover slice, reverse, pad, concatenate, scatter and both diagonal entries.
  Dynamic slice and gather are f64-only and use live operands. The owned twins are exercised
  by the pre-existing suites.
- Device evidence: `tenferro-gpu` lib (6 tests), `seam_structural` (27), `seam_indexing` (23),
  `seam_launch` (4), `tenferro-linalg --test integration` (64) and `tenferro-ad` (14) pass on
  the A100. The CUDA test partition in `scripts/ci/cuda_test_partition.tsv` matches the
  nextest inventory (1235 tests).
- The staging removal is asserted by counters, not timed. No benchmark row was added: the
  change removes a device copy per view operand and does not alter the kernels' own work.
