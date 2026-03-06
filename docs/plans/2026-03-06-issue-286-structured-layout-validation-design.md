# Issue 286 Design: Reject Mixed Structured Layouts in Dynamic Tensor Merge Paths

## Summary

`DynAdTensor::axpby` and `DynAdTensor::compose_complex` currently allow
non-equivalent structured layouts to merge when their compressed payload shapes
happen to match. This silently reinterprets one operand under the other
operand's logical layout, which is invalid.

## Problem

The shared helper `merge_add_ad_tensors` reconstructs the merged result with the
left-hand logical layout after adding compressed payload tensors. Today it does
not verify that:

- `logical_dims` match
- `axis_classes` match

As a result, cases like `diag(2x2) + dense vec(2)` can succeed even though the
operands do not live in the same structured tensor space.

## Decision

For issue #286, mismatched structured layouts will be rejected explicitly.

The validation boundary will be `merge_add_ad_tensors`, not individual public
methods. `DynAdTensor::axpby` and `DynAdTensor::compose_complex` will continue to
route through that helper, but the helper will now fail before combining
payloads when layout metadata differs.

This keeps the rule centralized and prevents future callers from reintroducing
the same bug by forgetting to validate layouts at the call site.

## Behavior Changes

- `merge_add_ad_tensors(lhs, rhs)` returns `Error::InvalidAdTensor` when
  `lhs.logical_dims() != rhs.logical_dims()`
- `merge_add_ad_tensors(lhs, rhs)` returns `Error::InvalidAdTensor` when
  `lhs.axis_classes() != rhs.axis_classes()`
- the same validation applies to tangent merging because tangents flow through
  the same helper
- layout-compatible cases continue to work unchanged

## Non-Goals

- no implicit layout alignment
- no dense materialization fallback for mismatched layouts
- no new public alignment API

## Testing

Add regression coverage for:

- `axpby(diag(2x2), dense vec(2)) -> Error::InvalidAdTensor`
- `compose_complex(diag(2x2), dense vec(2)) -> Error::InvalidAdTensor`
- same `logical_dims` but different `axis_classes` -> `Error::InvalidAdTensor`
- layout-compatible merge paths still succeed

## Rationale

Rejecting mismatched layouts is the narrowest correct fix for this issue. It
preserves explicit semantics, avoids hidden densification, and removes the
invalid assumption that compressed payload shape equality implies structured
layout compatibility.
