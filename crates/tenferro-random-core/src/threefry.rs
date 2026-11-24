//! Threefry2x32-20 (Salmon, Moraes, Dror and Shaw, SC'11), written from the
//! paper's definition with the Threefish/Skein key-schedule parity constant
//! and the paper's 2x32 rotation amounts. The output is checked against the
//! Random123 `kat_vectors` file, and the counter layout of block, split and
//! fold-in against JAX's `threefry_2x32`, in `tests/threefry.rs`.

use crate::CounterRng;

/// The 2x32 rotation amounts, applied in rounds `r % 8`.
const ROTATIONS: [u32; 8] = [13, 15, 26, 6, 17, 29, 16, 24];
/// The Skein key-schedule parity constant for 32-bit words.
const KS_PARITY: u32 = 0x1BD1_1BDA;

/// The Threefry2x32-20 generator: two 32-bit counter words, a 64-bit key and
/// twenty rounds, giving two `u32` words per block (`"threefry2x32-20"`).
///
/// The layout follows JAX's `threefry_2x32` with
/// `jax_threefry_partitionable` (the JAX default since 0.5.0), so the block,
/// split and fold-in values equal JAX's:
///
/// - [`CounterRng::block`] at counter `c` encrypts `[c >> 32, c & 0xFFFF_FFFF]`.
/// - [`CounterRng::split`] key `i` is the block at counter `i`.
/// - [`CounterRng::fold_in`] with `data` is the block at counter `data`.
///
/// Key derivation therefore shares the counter space of the samples, as in
/// JAX: a key that is split or folded must not also be sampled from directly.
/// [`Philox4x32_10`](crate::Philox4x32_10) has a separate key domain.
///
/// # Examples
///
/// ```
/// use tenferro_random_core::{CounterRng, Threefry2x32_20};
///
/// let key = Threefry2x32_20::seed(0);
/// let mut keys = [[0u32; 2]; 2];
/// Threefry2x32_20::split(key, 2, &mut keys);
/// assert_eq!(keys[1], Threefry2x32_20::fold_in(key, 1));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Threefry2x32_20;

impl Threefry2x32_20 {
    /// The raw Random123 `threefry2x32_R(20, counter, key)` block function.
    ///
    /// [`CounterRng::block`] is this function with the counter
    /// `[c >> 32, c & 0xFFFF_FFFF]` of a 64-bit block counter `c`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_random_core::Threefry2x32_20;
    ///
    /// // A Random123 known-answer vector.
    /// assert_eq!(
    ///     Threefry2x32_20::raw_block([0; 2], [0; 2]),
    ///     [0x6b20_0159, 0x99ba_4efe],
    /// );
    /// ```
    #[must_use]
    pub fn raw_block(counter: [u32; 2], key: [u32; 2]) -> [u32; 2] {
        let ks = [key[0], key[1], KS_PARITY ^ key[0] ^ key[1]];
        let mut x0 = counter[0].wrapping_add(ks[0]);
        let mut x1 = counter[1].wrapping_add(ks[1]);
        for round in 0..20 {
            x0 = x0.wrapping_add(x1);
            x1 = x1.rotate_left(ROTATIONS[round % 8]);
            x1 ^= x0;
            if round % 4 == 3 {
                let injection = round / 4 + 1;
                x0 = x0.wrapping_add(ks[injection % 3]);
                x1 = x1
                    .wrapping_add(ks[(injection + 1) % 3])
                    .wrapping_add(injection as u32);
            }
        }
        [x0, x1]
    }
}

impl CounterRng for Threefry2x32_20 {
    const NAME: &'static str = "threefry2x32-20";
    const WORDS_PER_BLOCK: usize = 2;

    fn block(key: [u32; 2], counter: u64, out: &mut [u32]) {
        out.copy_from_slice(&Self::raw_block(
            [(counter >> 32) as u32, counter as u32],
            key,
        ));
    }

    fn split(key: [u32; 2], n: usize, out: &mut [[u32; 2]]) {
        assert_eq!(out.len(), n, "split: `out` must hold exactly `n` keys");
        for (index, slot) in out.iter_mut().enumerate() {
            let index = index as u64;
            *slot = Self::raw_block([(index >> 32) as u32, index as u32], key);
        }
    }

    fn fold_in(key: [u32; 2], data: u32) -> [u32; 2] {
        Self::raw_block([0, data], key)
    }
}
