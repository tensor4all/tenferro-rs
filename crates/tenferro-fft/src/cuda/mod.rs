// INVARIANT: Task 1 defines descriptor and error ownership before the later
// CUDA loader and execution tasks consume these crate-private declarations.
#![allow(dead_code)]

mod descriptor;
mod error;

#[cfg(test)]
mod tests;
