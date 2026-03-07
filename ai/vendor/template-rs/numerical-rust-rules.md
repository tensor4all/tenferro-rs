# Numerical Rust Rules

## Testing

- Use small deterministic inputs for correctness tests.
- Prefer hard-coded data or seeded RNGs over unseeded randomness.
- Feature-gate expensive tests.
- For local trial-and-error loops, `cargo test --release --workspace` is preferred when it materially reduces iteration time.

## Generic Test Patterns

- Share test logic across scalar types with generic helpers or macros instead of duplicating bodies.
- Centralize approximate equality helpers so tolerance policy lives in one place.
- Parameterize over algorithms when the validation logic is the same.

## Performance Habits

- Avoid duplicate typed implementations when generic code or a macro is enough.
- Avoid allocating dense temporary buffers when the operation can work over strided or borrowed views.
- Avoid zero-filling buffers that are immediately overwritten.
- Avoid repeated index multiplication inside hot loops when incremental offsets suffice.
- Avoid allocations inside hot loops.
- Precompute plans and reusable metadata outside execution loops.

## Numerical API Design

- Keep examples small enough to read in rustdoc.
- Prefer APIs that make shape and layout expectations explicit.
- Expose helpers for common validation rather than repeating ad hoc checks across crates.
