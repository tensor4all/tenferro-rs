//! The counter-based generator trait and the algorithm attribute.

use core::fmt;

use crate::{Philox4x32_10, Threefry2x32_20};

/// The largest [`CounterRng::WORDS_PER_BLOCK`] an algorithm may declare.
///
/// The sample functions keep one block on the stack; using an algorithm with a
/// larger block in them is a compile-time error.
///
/// # Examples
///
/// ```
/// use tenferro_random_core::{CounterRng, Philox4x32_10, MAX_WORDS_PER_BLOCK};
///
/// assert!(Philox4x32_10::WORDS_PER_BLOCK <= MAX_WORDS_PER_BLOCK);
/// ```
pub const MAX_WORDS_PER_BLOCK: usize = 16;

/// A counter-based generator: [`block`](CounterRng::block) is a pure function
/// of the key and the counter.
///
/// An algorithm is a zero-sized type implementing this trait. The sample
/// functions of this crate ([`uniform`](crate::uniform),
/// [`normal`](crate::normal), [`integer`](crate::integer), ...) are generic
/// over it, so adding an algorithm changes no sample function and no
/// operation. The output of a released algorithm is frozen: a change of
/// output is a new algorithm with a new [`NAME`](CounterRng::NAME).
///
/// A key is two `u32` words. tenferro stores it as an `I32[2]` tensor and
/// reads the two elements as `u32` bit patterns.
///
/// # Examples
///
/// Code generic over the algorithm works with every built-in one:
///
/// ```
/// use tenferro_random_core::{CounterRng, Philox4x32_10, Threefry2x32_20};
///
/// fn first_word<R: CounterRng>(seed: u64) -> u32 {
///     let mut block = [0u32; 4];
///     R::block(R::seed(seed), 0, &mut block[..R::WORDS_PER_BLOCK]);
///     block[0]
/// }
///
/// assert_ne!(
///     first_word::<Philox4x32_10>(1),
///     first_word::<Threefry2x32_20>(1),
/// );
/// ```
pub trait CounterRng: Copy + Send + Sync + 'static {
    /// Stable name; part of every random operation's identity.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_random_core::{CounterRng, Philox4x32_10};
    ///
    /// assert_eq!(Philox4x32_10::NAME, "philox4x32-10");
    /// ```
    const NAME: &'static str;

    /// `u32` words produced per block, between 1 and
    /// [`MAX_WORDS_PER_BLOCK`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_random_core::{CounterRng, Threefry2x32_20};
    ///
    /// assert_eq!(Threefry2x32_20::WORDS_PER_BLOCK, 2);
    /// ```
    const WORDS_PER_BLOCK: usize;

    /// Writes the block at `counter` for `key` into `out`.
    ///
    /// # Panics
    ///
    /// Panics if `out.len()` is not
    /// [`WORDS_PER_BLOCK`](CounterRng::WORDS_PER_BLOCK).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_random_core::{CounterRng, Philox4x32_10};
    ///
    /// let mut out = [0u32; 4];
    /// Philox4x32_10::block([0, 0], 0, &mut out);
    /// assert_eq!(out, Philox4x32_10::raw_block([0; 4], [0; 2]));
    /// ```
    fn block(key: [u32; 2], counter: u64, out: &mut [u32]);

    /// The key of a 64-bit seed: `[seed >> 32, seed & 0xFFFF_FFFF]`.
    ///
    /// The rule is the same for every algorithm (it matches JAX's
    /// `threefry_seed` for a 64-bit seed); implementations keep this default.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_random_core::{CounterRng, Philox4x32_10};
    ///
    /// assert_eq!(
    ///     Philox4x32_10::seed(0x0000_0001_0000_0002),
    ///     [1, 2],
    /// );
    /// ```
    fn seed(seed: u64) -> [u32; 2] {
        [(seed >> 32) as u32, seed as u32]
    }

    /// Writes `n` keys derived from `key` into `out`; `out[i]` depends only
    /// on `key` and `i`, so a longer split extends a shorter one.
    ///
    /// # Panics
    ///
    /// Panics if `out.len()` is not `n`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_random_core::{CounterRng, Philox4x32_10};
    ///
    /// let key = Philox4x32_10::seed(7);
    /// let mut keys = [[0u32; 2]; 3];
    /// Philox4x32_10::split(key, 3, &mut keys);
    /// assert_ne!(keys[0], keys[1]);
    /// assert_ne!(keys[0], key);
    /// ```
    fn split(key: [u32; 2], n: usize, out: &mut [[u32; 2]]);

    /// Derives a key from `key` and a 32-bit value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_random_core::{CounterRng, Philox4x32_10};
    ///
    /// let key = Philox4x32_10::seed(7);
    /// let step_3 = Philox4x32_10::fold_in(key, 3);
    /// assert_eq!(step_3, Philox4x32_10::fold_in(key, 3));
    /// assert_ne!(step_3, Philox4x32_10::fold_in(key, 4));
    /// ```
    fn fold_in(key: [u32; 2], data: u32) -> [u32; 2];
}

/// The algorithm attribute a random operation carries: one variant per
/// built-in [`CounterRng`], identified by its [`NAME`](CounterRng::NAME).
///
/// The default is [`Philox4x32_10`].
///
/// # Examples
///
/// ```
/// use tenferro_random_core::RngAlgorithm;
///
/// let algorithm = RngAlgorithm::default();
/// assert_eq!(algorithm, RngAlgorithm::Philox4x32_10);
/// assert_eq!(RngAlgorithm::from_name(algorithm.name()), Some(algorithm));
/// assert_eq!(algorithm.to_string(), "philox4x32-10");
/// ```
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RngAlgorithm {
    /// [`Philox4x32_10`], `"philox4x32-10"`.
    #[default]
    Philox4x32_10,
    /// [`Threefry2x32_20`], `"threefry2x32-20"`.
    Threefry2x32_20,
}

impl RngAlgorithm {
    /// Every built-in algorithm.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_random_core::RngAlgorithm;
    ///
    /// assert!(RngAlgorithm::ALL.contains(&RngAlgorithm::Threefry2x32_20));
    /// ```
    pub const ALL: [Self; 2] = [Self::Philox4x32_10, Self::Threefry2x32_20];

    /// The algorithm's stable name.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_random_core::RngAlgorithm;
    ///
    /// assert_eq!(RngAlgorithm::Threefry2x32_20.name(), "threefry2x32-20");
    /// ```
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Philox4x32_10 => Philox4x32_10::NAME,
            Self::Threefry2x32_20 => Threefry2x32_20::NAME,
        }
    }

    /// The algorithm with this name, or `None` for an unknown name.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_random_core::RngAlgorithm;
    ///
    /// assert_eq!(
    ///     RngAlgorithm::from_name("threefry2x32-20"),
    ///     Some(RngAlgorithm::Threefry2x32_20),
    /// );
    /// assert_eq!(RngAlgorithm::from_name("mt19937"), None);
    /// ```
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|algorithm| algorithm.name() == name)
    }

    /// `u32` words per block of the algorithm.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_random_core::RngAlgorithm;
    ///
    /// assert_eq!(RngAlgorithm::Philox4x32_10.words_per_block(), 4);
    /// ```
    #[must_use]
    pub const fn words_per_block(self) -> usize {
        match self {
            Self::Philox4x32_10 => Philox4x32_10::WORDS_PER_BLOCK,
            Self::Threefry2x32_20 => Threefry2x32_20::WORDS_PER_BLOCK,
        }
    }
}

impl fmt::Display for RngAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
