//! The word stream and the bits-to-sample transforms (design §4.2).

use crate::math::{log_f32, log_f64, sincospi_f32, sincospi_f64};
use crate::{CounterRng, MAX_WORDS_PER_BLOCK};

pub(crate) mod sealed {
    /// The private half of [`RandomFloat`](crate::RandomFloat).
    pub trait Float:
        Copy
        + PartialOrd
        + core::ops::Add<Output = Self>
        + core::ops::Sub<Output = Self>
        + core::ops::Mul<Output = Self>
    {
        /// `u32` words of one uniform sample.
        const WORDS: usize;
        const ZERO: Self;
        const ONE: Self;
        const TWO: Self;
        /// A uniform sample on `[0, 1)` from `WORDS` words (mantissa fill).
        fn unit(words: &[u32]) -> Self;
        fn log(x: Self) -> Self;
        fn sincospi(x: Self) -> (Self, Self);
        fn sqrt(self) -> Self;
        fn next_down(self) -> Self;
    }

    /// The private half of [`RandomInt`](crate::RandomInt).
    pub trait Int: Copy {
        /// `lo + floor(x (hi - lo) / 2^64)`, or `lo` when `hi <= lo`.
        fn offset(lo: Self, hi: Self, x: u64) -> Self;
    }
}

/// A floating-point sample type: `f32` or `f64`.
///
/// The trait is sealed; complex samples are returned as `[re, im]` pairs of
/// this type.
///
/// # Examples
///
/// ```
/// use cbrng::{uniform, Philox4x32_10, RandomFloat};
///
/// fn mean<T: RandomFloat + Into<f64>>(key: [u32; 2]) -> f64 {
///     (0..8).map(|i| uniform::<Philox4x32_10, T>(key, i).into()).sum::<f64>() / 8.0
/// }
///
/// let m = mean::<f32>([0, 1]);
/// assert!((0.0..1.0).contains(&m));
/// ```
pub trait RandomFloat: sealed::Float {}

/// An integer sample type: `i32` or `i64`.
///
/// # Examples
///
/// ```
/// use cbrng::{integer, Threefry2x32_20, RandomInt};
///
/// fn die<T: RandomInt + From<i32> + PartialOrd>(key: [u32; 2], i: u64) -> T {
///     integer::<Threefry2x32_20, T>(key, i, T::from(1), T::from(7))
/// }
///
/// let roll: i64 = die([3, 4], 0);
/// assert!((1..7).contains(&roll));
/// ```
pub trait RandomInt: sealed::Int {}

impl sealed::Float for f32 {
    const WORDS: usize = 1;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;

    fn unit(words: &[u32]) -> Self {
        f32::from_bits(0x3F80_0000 | (words[0] >> 9)) - 1.0
    }

    fn log(x: Self) -> Self {
        log_f32(x)
    }

    fn sincospi(x: Self) -> (Self, Self) {
        sincospi_f32(x)
    }

    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }

    fn next_down(self) -> Self {
        f32::next_down(self)
    }
}

impl sealed::Float for f64 {
    const WORDS: usize = 2;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;

    fn unit(words: &[u32]) -> Self {
        f64::from_bits(0x3FF0_0000_0000_0000 | (word_pair(words) >> 12)) - 1.0
    }

    fn log(x: Self) -> Self {
        log_f64(x)
    }

    fn sincospi(x: Self) -> (Self, Self) {
        sincospi_f64(x)
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    fn next_down(self) -> Self {
        f64::next_down(self)
    }
}

impl RandomFloat for f32 {}
impl RandomFloat for f64 {}

impl sealed::Int for i32 {
    fn offset(lo: Self, hi: Self, x: u64) -> Self {
        if hi <= lo {
            return lo;
        }
        let range = (i64::from(hi) - i64::from(lo)) as u64;
        let offset = ((u128::from(x) * u128::from(range)) >> 64) as i64;
        (i64::from(lo) + offset) as i32
    }
}

impl sealed::Int for i64 {
    fn offset(lo: Self, hi: Self, x: u64) -> Self {
        if hi <= lo {
            return lo;
        }
        let range = hi.wrapping_sub(lo) as u64;
        let offset = ((u128::from(x) * u128::from(range)) >> 64) as u64;
        lo.wrapping_add(offset as i64)
    }
}

impl RandomInt for i32 {}
impl RandomInt for i64 {}

/// The 64-bit value of two words, the first word high.
fn word_pair(words: &[u32]) -> u64 {
    (u64::from(words[0]) << 32) | u64::from(words[1])
}

/// Writes words `first_word, first_word + 1, ...` of the word stream of
/// `key` into `out`.
///
/// The word stream of a key is the concatenation of the blocks at counters
/// `0, 1, 2, ...`: word `w` is word `w % WORDS_PER_BLOCK` of the block at
/// counter `w / WORDS_PER_BLOCK`. Every sample function reads it (see the
/// [crate documentation](crate) for which words a sample uses), and
/// [`bits`] is one word of it.
///
/// # Examples
///
/// ```
/// use cbrng::{bits, fill_words, CounterRng, Philox4x32_10};
///
/// let key = Philox4x32_10::seed(5);
/// let mut words = [0u32; 6];
/// fill_words::<Philox4x32_10>(key, 3, &mut words);
/// // Word 3 is the last word of block 0, words 4..8 are block 1.
/// let mut block_1 = [0u32; 4];
/// Philox4x32_10::block(key, 1, &mut block_1);
/// assert_eq!(&words[1..5], &block_1);
/// assert_eq!(words[0], bits::<Philox4x32_10>(key, 3));
/// ```
pub fn fill_words<R: CounterRng>(key: [u32; 2], first_word: u64, out: &mut [u32]) {
    const {
        assert!(
            R::WORDS_PER_BLOCK >= 1 && R::WORDS_PER_BLOCK <= MAX_WORDS_PER_BLOCK,
            "CounterRng::WORDS_PER_BLOCK must be in 1..=MAX_WORDS_PER_BLOCK",
        );
    }
    let words_per_block = R::WORDS_PER_BLOCK as u64;
    let mut buffer = [0u32; MAX_WORDS_PER_BLOCK];
    let block = &mut buffer[..R::WORDS_PER_BLOCK];
    let mut counter = first_word / words_per_block;
    let mut offset = (first_word % words_per_block) as usize;
    let mut written = 0;
    while written < out.len() {
        R::block(key, counter, block);
        let take = (R::WORDS_PER_BLOCK - offset).min(out.len() - written);
        out[written..written + take].copy_from_slice(&block[offset..offset + take]);
        written += take;
        offset = 0;
        counter = counter.wrapping_add(1);
    }
}

/// Reads the `words.len()` words of draw `draw`.
fn draw_words<R: CounterRng>(key: [u32; 2], draw: u64, words: &mut [u32]) {
    fill_words::<R>(key, draw.wrapping_mul(words.len() as u64), words);
}

/// Raw bits: word `index` of the word stream.
///
/// # Examples
///
/// ```
/// use cbrng::{bits, CounterRng, Threefry2x32_20};
///
/// let key = Threefry2x32_20::seed(0);
/// let block = Threefry2x32_20::raw_block([0, 0], key);
/// assert_eq!(bits::<Threefry2x32_20>(key, 0), block[0]);
/// assert_eq!(bits::<Threefry2x32_20>(key, 1), block[1]);
/// ```
#[must_use]
pub fn bits<R: CounterRng>(key: [u32; 2], index: u64) -> u32 {
    let mut word = [0u32];
    draw_words::<R>(key, index, &mut word);
    word[0]
}

/// A uniform sample on `[0, 1)`: element `index` of a uniform tensor.
///
/// `f32` uses one word and keeps its top 23 bits, `f64` uses two words (the
/// first high) and keeps the top 52 bits of the 64; the bits fill the
/// mantissa of a number in `[1, 2)` and 1 is subtracted, exactly. Every value
/// is a multiple of `2^-23` or `2^-52`.
///
/// # Examples
///
/// ```
/// use cbrng::{uniform, CounterRng, Philox4x32_10};
///
/// let key = Philox4x32_10::seed(1);
/// let u: f64 = uniform::<Philox4x32_10, f64>(key, 0);
/// assert!((0.0..1.0).contains(&u));
/// assert_eq!(u, uniform::<Philox4x32_10, f64>(key, 0));
/// ```
#[must_use]
pub fn uniform<R: CounterRng, T: RandomFloat>(key: [u32; 2], index: u64) -> T {
    let mut buffer = [0u32; 2];
    let words = &mut buffer[..T::WORDS];
    draw_words::<R>(key, index, words);
    T::unit(words)
}

/// A uniform sample on `[lo, hi)`: `lo + (hi - lo) u` with `u` from
/// [`uniform`], replaced by the largest value below `hi` when rounding
/// reaches `hi`.
///
/// The bounds are expected finite with `lo < hi` and `hi - lo` finite; the
/// operations that call this validate them. Other bounds give an unspecified
/// value (NaN bounds give NaN).
///
/// # Examples
///
/// ```
/// use cbrng::{uniform_range, CounterRng, Philox4x32_10};
///
/// let key = Philox4x32_10::seed(1);
/// let x: f32 = uniform_range::<Philox4x32_10, f32>(key, 3, -2.0, 2.0);
/// assert!((-2.0..2.0).contains(&x));
/// ```
#[must_use]
pub fn uniform_range<R: CounterRng, T: RandomFloat>(key: [u32; 2], index: u64, lo: T, hi: T) -> T {
    let x = lo + (hi - lo) * uniform::<R, T>(key, index);
    if x >= hi {
        hi.next_down()
    } else {
        x
    }
}

/// A complex uniform sample `[re, im]`, each part on `[0, 1)`.
///
/// Element `index` uses words `2wi .. 2wi + 2w` of the stream, where `w` is
/// the word count of one real uniform of `T` (1 for `f32`, 2 for `f64`): the
/// first `w` words give the real part, the next `w` the imaginary part.
///
/// # Examples
///
/// ```
/// use cbrng::{uniform_complex, CounterRng, Philox4x32_10};
///
/// let key = Philox4x32_10::seed(2);
/// let [re, im] = uniform_complex::<Philox4x32_10, f32>(key, 0);
/// assert!((0.0..1.0).contains(&re) && (0.0..1.0).contains(&im));
/// ```
#[must_use]
pub fn uniform_complex<R: CounterRng, T: RandomFloat>(key: [u32; 2], index: u64) -> [T; 2] {
    let mut buffer = [0u32; 4];
    let words = &mut buffer[..2 * T::WORDS];
    draw_words::<R>(key, index, words);
    let (re, im) = words.split_at(T::WORDS);
    [T::unit(re), T::unit(im)]
}

/// The Box–Muller pair of `draw`: `(log(1 - a), sincospi(2 b))` for the two
/// uniforms `a` and `b` of words `2wd .. 2wd + 2w`.
fn box_muller_parts<R: CounterRng, T: RandomFloat>(key: [u32; 2], draw: u64) -> (T, (T, T)) {
    let mut buffer = [0u32; 4];
    let words = &mut buffer[..2 * T::WORDS];
    draw_words::<R>(key, draw, words);
    let (a, b) = words.split_at(T::WORDS);
    let (a, b) = (T::unit(a), T::unit(b));
    // `1 - a` is exact and lies in (0, 1]; `b + b` is exact and lies in [0, 2).
    (T::log(T::ONE - a), T::sincospi(b + b))
}

/// A standard normal sample, by the Box–Muller transform.
///
/// Elements `2d` and `2d + 1` share draw `d`, the two uniforms `a` and `b` of
/// words `2wd .. 2wd + 2w` (`w` as in [`uniform_complex`]). With
/// `r = sqrt(0 - 2 log(1 - a))` and `(s, c) = sincospi(2b)`, element `2d` is
/// `r c` and element `2d + 1` is `r s`. [`log`](crate::log) and
/// [`sincospi`](crate::sincospi) are tenferro's own, so the sample is the
/// same bits on every backend. The largest magnitude is
/// `sqrt(2 log 2^23)` (about 5.64) for `f32` and `sqrt(2 log 2^52)` (about
/// 8.49) for `f64`.
///
/// # Examples
///
/// ```
/// use cbrng::{normal, CounterRng, Philox4x32_10};
///
/// let key = Philox4x32_10::seed(3);
/// let z: f64 = normal::<Philox4x32_10, f64>(key, 0);
/// assert!(z.is_finite() && z.abs() < 8.5);
/// ```
#[must_use]
pub fn normal<R: CounterRng, T: RandomFloat>(key: [u32; 2], index: u64) -> T {
    let (log_u, (sin, cos)) = box_muller_parts::<R, T>(key, index / 2);
    let radius = (T::ZERO - T::TWO * log_u).sqrt();
    if index.is_multiple_of(2) {
        radius * cos
    } else {
        radius * sin
    }
}

/// A standard complex normal sample `[re, im]` with `E|z|^2 = 1`: the real
/// and imaginary parts are independent N(0, 1/2), as in JAX and PyTorch.
///
/// Element `index` is draw `index` of [`normal`] with radius
/// `sqrt(0 - log(1 - a))`: `[r c, r s]`.
///
/// # Examples
///
/// ```
/// use cbrng::{normal_complex, CounterRng, Threefry2x32_20};
///
/// let key = Threefry2x32_20::seed(3);
/// let [re, im] = normal_complex::<Threefry2x32_20, f32>(key, 5);
/// assert!(re.is_finite() && im.is_finite());
/// ```
#[must_use]
pub fn normal_complex<R: CounterRng, T: RandomFloat>(key: [u32; 2], index: u64) -> [T; 2] {
    let (log_u, (sin, cos)) = box_muller_parts::<R, T>(key, index);
    let radius = (T::ZERO - log_u).sqrt();
    [radius * cos, radius * sin]
}

/// An integer sample on `[lo, hi)`, by a 64-bit multiply without rejection.
///
/// Element `index` reads words `2i` and `2i + 1` as `x = w0 << 32 | w1` and
/// returns `lo + floor(x (hi - lo) / 2^64)`. Without rejection the result is
/// a pure function of the index; the price is a bias of at most
/// `(hi - lo) / 2^64` in the probability of any value. `hi <= lo` returns
/// `lo` (the operations that call this reject such bounds).
///
/// # Examples
///
/// ```
/// use cbrng::{integer, CounterRng, Philox4x32_10};
///
/// let key = Philox4x32_10::seed(4);
/// let k: i32 = integer::<Philox4x32_10, i32>(key, 0, -3, 3);
/// assert!((-3..3).contains(&k));
/// assert_eq!(integer::<Philox4x32_10, i64>(key, 0, 9, 10), 9);
/// ```
#[must_use]
pub fn integer<R: CounterRng, T: RandomInt>(key: [u32; 2], index: u64, lo: T, hi: T) -> T {
    let mut words = [0u32; 2];
    draw_words::<R>(key, index, &mut words);
    T::offset(lo, hi, word_pair(&words))
}
