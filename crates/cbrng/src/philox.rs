//! Philox4x32-10 (Salmon, Moraes, Dror and Shaw, "Parallel random numbers: as
//! easy as 1, 2, 3", SC'11), written from the paper's definition. Its
//! multipliers and Weyl constants are the paper's; the output is checked
//! against the Random123 `kat_vectors` file in `tests/philox.rs`.

use crate::CounterRng;

const M0: u32 = 0xD251_1F53;
const M1: u32 = 0xCD9E_8D57;
const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;

/// Counter word 3 of the blocks [`Philox4x32_10::split`] draws from.
const SPLIT_DOMAIN: u32 = 1;
/// Counter word 3 of the blocks [`Philox4x32_10::fold_in`] draws from.
const FOLD_IN_DOMAIN: u32 = 2;

/// The Philox4x32-10 generator: four 32-bit counter words, a 64-bit key and
/// ten rounds, giving four `u32` words per block. It is tenferro's default
/// algorithm (`"philox4x32-10"`).
///
/// [`CounterRng::block`] puts the 64-bit block counter in counter words 0
/// (low half) and 1 (high half) and leaves words 2 and 3 zero. Key derivation
/// uses word 3: [`CounterRng::split`] sets it to 1 with the key index in
/// words 0–1, and [`CounterRng::fold_in`] sets it to 2 with the data in
/// word 0, so a derived key never comes from a block that also produces
/// samples.
///
/// # Examples
///
/// ```
/// use cbrng::{CounterRng, Philox4x32_10};
///
/// let key = Philox4x32_10::seed(42);
/// let mut block = [0u32; 4];
/// Philox4x32_10::block(key, 0, &mut block);
/// // The same key and counter always give the same words.
/// let mut again = [0u32; 4];
/// Philox4x32_10::block(key, 0, &mut again);
/// assert_eq!(block, again);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Philox4x32_10;

impl Philox4x32_10 {
    /// The raw Random123 `philox4x32_R(10, counter, key)` block function, for
    /// a full 128-bit counter.
    ///
    /// [`CounterRng::block`] is this function with the counter
    /// `[lo, hi, 0, 0]` of a 64-bit block counter.
    ///
    /// # Examples
    ///
    /// ```
    /// use cbrng::Philox4x32_10;
    ///
    /// // A Random123 known-answer vector.
    /// assert_eq!(
    ///     Philox4x32_10::raw_block([0; 4], [0; 2]),
    ///     [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8],
    /// );
    /// ```
    #[must_use]
    pub fn raw_block(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
        let mut ctr = counter;
        let mut key = key;
        for round in 0..10 {
            if round > 0 {
                key = [key[0].wrapping_add(W0), key[1].wrapping_add(W1)];
            }
            let p0 = u64::from(M0) * u64::from(ctr[0]);
            let p1 = u64::from(M1) * u64::from(ctr[2]);
            let (hi0, lo0) = ((p0 >> 32) as u32, p0 as u32);
            let (hi1, lo1) = ((p1 >> 32) as u32, p1 as u32);
            ctr = [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0];
        }
        ctr
    }
}

impl CounterRng for Philox4x32_10 {
    const NAME: &'static str = "philox4x32-10";
    const WORDS_PER_BLOCK: usize = 4;

    fn block(key: [u32; 2], counter: u64, out: &mut [u32]) {
        out.copy_from_slice(&Self::raw_block(
            [counter as u32, (counter >> 32) as u32, 0, 0],
            key,
        ));
    }

    fn split(key: [u32; 2], n: usize, out: &mut [[u32; 2]]) {
        assert_eq!(out.len(), n, "split: `out` must hold exactly `n` keys");
        for (index, slot) in out.iter_mut().enumerate() {
            let index = index as u64;
            let words = Self::raw_block([index as u32, (index >> 32) as u32, 0, SPLIT_DOMAIN], key);
            *slot = [words[0], words[1]];
        }
    }

    fn fold_in(key: [u32; 2], data: u32) -> [u32; 2] {
        let words = Self::raw_block([data, 0, 0, FOLD_IN_DOMAIN], key);
        [words[0], words[1]]
    }
}
