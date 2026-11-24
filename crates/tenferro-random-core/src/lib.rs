//! Counter-based random number algorithms and sample transforms for tenferro.
//!
//! A sample is a pure function of `(algorithm, key, element index)`: there is
//! no generator state, so the result does not depend on the thread count,
//! the launch configuration, the device or the storage layout. This crate is
//! the scalar Rust reference that tenferro's CPU kernels call and its CUDA
//! kernels reproduce bit for bit; it has no dependencies. The design is
//! [tensor4all-meta `docs/design/random.md`](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/design/random.md).
//!
//! # Keys
//!
//! A key is two `u32` words (tenferro stores it as an `I32[2]` tensor). A key
//! never changes; new randomness comes from deriving keys:
//!
//! - [`CounterRng::seed`]: `[seed >> 32, seed & 0xFFFF_FFFF]` for every
//!   algorithm.
//! - [`CounterRng::split`]: `n` independent keys.
//! - [`CounterRng::fold_in`]: a key mixed with a 32-bit value.
//!
//! # Algorithms
//!
//! An algorithm implements [`CounterRng`]; [`RngAlgorithm`] names the
//! built-in ones. [`Philox4x32_10`] (`"philox4x32-10"`, the default) and
//! [`Threefry2x32_20`] (`"threefry2x32-20"`) both match the Random123
//! known-answer vectors, and the Threefry split and fold-in match JAX.
//!
//! Each algorithm places the 64-bit block counter `c` in its own counter words,
//! following its reference: Philox uses `[c & 0xFFFF_FFFF, c >> 32, 0, 0]` (low
//! word first, words 2 and 3 free for key derivation), Threefry uses
//! `[c >> 32, c & 0xFFFF_FFFF]` (high word first, as JAX does).
//!
//! # Samples
//!
//! The **word stream** of a key is the concatenation of its blocks at
//! counters `0, 1, 2, ...` ([`fill_words`]). Element `i` of a tensor (its
//! logical column-major index) is computed from a fixed slice of that
//! stream, with `w = 1` for `f32` and `w = 2` for `f64`:
//!
//! | Function | Words of element `i` | Transform |
//! | --- | --- | --- |
//! | [`bits`] | `i` | the word |
//! | [`uniform`], [`uniform_range`] | `w i .. w i + w` | mantissa fill on `[0, 1)`; `lo + (hi - lo) u` |
//! | [`uniform_complex`] | `2w i .. 2w i + 2w` | two uniforms `[re, im]` |
//! | [`normal`] | `2w d .. 2w d + 2w`, `d = i / 2` | Box–Muller; `i` even takes the cosine, odd the sine |
//! | [`normal_complex`] | `2w i .. 2w i + 2w` | Box–Muller with `E\|z\|^2 = 1` |
//! | [`integer`] | `2i .. 2i + 2` | 64-bit multiply, no rejection |
//!
//! Every transform reads a fixed number of words, so a sample never depends
//! on another. The normal transforms use this crate's own [`log`] and
//! [`sincospi`], built only from correctly rounded operations, so normal
//! samples are bit-identical across backends too. As in JAX, drawing `n`
//! samples at once need not equal `n` draws of one sample.
//!
//! # Examples
//!
//! ```
//! use tenferro_random_core::{normal, uniform, CounterRng, Philox4x32_10};
//!
//! let key = Philox4x32_10::seed(2026);
//! let mut keys = [[0u32; 2]; 2];
//! Philox4x32_10::split(key, 2, &mut keys);
//!
//! // A 2 x 3 column-major tensor of uniforms, and one of normals.
//! let u: Vec<f32> = (0..6).map(|i| uniform::<Philox4x32_10, f32>(keys[0], i)).collect();
//! let z: Vec<f64> = (0..6).map(|i| normal::<Philox4x32_10, f64>(keys[1], i)).collect();
//! assert!(u.iter().all(|x| (0.0..1.0).contains(x)));
//! assert!(z.iter().all(|x| x.is_finite()));
//!
//! // The same key gives the same samples.
//! assert_eq!(u[4], uniform::<Philox4x32_10, f32>(keys[0], 4));
//! ```

mod algorithm;
mod math;
mod philox;
mod sample;
mod threefry;

pub use algorithm::{CounterRng, RngAlgorithm, MAX_WORDS_PER_BLOCK};
pub use math::{log, sincospi};
pub use philox::Philox4x32_10;
pub(crate) use sample::sealed;
pub use sample::{
    bits, fill_words, integer, normal, normal_complex, uniform, uniform_complex, uniform_range,
    RandomFloat, RandomInt,
};
pub use threefry::Threefry2x32_20;
