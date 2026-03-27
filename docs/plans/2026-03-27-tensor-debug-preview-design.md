# tenferro::Tensor Debug Preview Design

## Goal

Improve `tenferro::Tensor` `Debug` output so small tensors show actual values,
while large or awkward-to-materialize tensors stay bounded and readable.

## Selected Approach

Use a metadata-preserving `Debug` format with a bounded `preview` field.

- Keep the existing semantic metadata visible:
  - `scalar_type`
  - `dims`
  - `axis_classes`
  - `mode`
  - `is_dense`
  - `is_diag`
- Add `preview` for small tensors only.
- Preview uses logical values, not compressed payload values.
  - This keeps `diag([3, 4])` aligned with the tensor the user sees, not the
    internal payload representation.
- Preview remains bounded:
  - only materialize when the logical element count is small
  - avoid previewing tensors that are not already in main memory
  - show an omission marker instead of forcing an expensive or surprising path

## Output Contract

`format!("{tensor:?}")` should behave as follows.

### Small tensors on main memory

Show metadata plus a logical-value preview.

- Dense tensors: preview the dense logical values
- Structured tensors: preview logical values after bounded materialization to a
  dense logical tensor

### Large tensors

Show metadata plus a bounded omission marker instead of attempting a full
preview.

### Non-main-memory tensors

Show metadata plus an omission marker instead of forcing host transfer for
debugging.

### Preview formatting

Format preview values in logical axis order, not raw storage order. Nested
formatting is preferred when it stays small enough to read naturally.

## Implementation Shape

Keep the `Tensor` enum in `dyn_ad_tensor/mod.rs`, but move the `Debug`
formatting details into a focused helper module so the policy stays isolated
from the runtime tensor API surface.

The helper should:

1. Decide whether preview is allowed
2. Materialize a dense logical snapshot only when the preview policy permits it
3. Format small logical tensors into a bounded debug preview
4. Fall back to an explicit omission/unavailable marker when preview is not
   appropriate

## Testing Strategy

Cover the user-facing `tenferro` facade contract in integration tests.

- Small dense tensor `Debug` output includes values
- Small structured tensor `Debug` output includes logical values, not payload
  values
- Large tensor `Debug` output stays bounded
- Existing semantic metadata remains visible

Update the `Tensor` rustdoc example to show the new `Debug` behavior.
