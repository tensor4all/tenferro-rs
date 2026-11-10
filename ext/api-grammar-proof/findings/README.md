# Prototype findings

One file per prototype (p1.md … p6.md): what was built, what passed, what the
grammar could not express, and what was not verified.

Toolchain: `cargo +nightly` (rustc 1.93) with
`RUSTFLAGS="-Zcrate-attr=feature(atomic_try_update)"` and `--ignore-rust-version`,
because rustc 1.96 is not installed on the machine that ran them.
