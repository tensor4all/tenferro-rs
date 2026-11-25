//! Sample known answers for both algorithms with the key `seed(2026)`.
//!
//! The exact transforms (bits, uniforms, ranges, integers) are compared with
//! the exact values of `reference/samples.py`, an independent implementation
//! of the crate's layout. Normal samples are frozen bits (the released
//! output), each checked to be within 1 ulp of the exact Box–Muller value
//! that the same script computes with mpmath and rounds once.

mod common;

use cbrng::{
    bits, fill_words, integer, normal, normal_complex, uniform, uniform_complex, uniform_range,
    CounterRng, Philox4x32_10, Threefry2x32_20,
};
use common::{ulps_f32, ulps_f64};

struct Expected {
    bits: [u32; 8],
    uniform_f32: [u32; 8],
    uniform_range_f32: [u32; 8],
    uniform_f64: [u64; 8],
    uniform_range_f64: [u64; 8],
    normal_f32: [u32; 8],
    normal_f32_mpmath: [u32; 8],
    normal_complex_f32: [[u32; 2]; 4],
    normal_complex_f32_mpmath: [[u32; 2]; 4],
    normal_f64: [u64; 8],
    normal_f64_mpmath: [u64; 8],
    normal_complex_f64: [[u64; 2]; 4],
    normal_complex_f64_mpmath: [[u64; 2]; 4],
    integer_i32_small: [i32; 8],
    integer_i32_full: [i32; 8],
    integer_i64: [i64; 8],
}

const PHILOX: Expected = Expected {
    bits: [
        0xac83c808, 0x3222c6d5, 0x60e42337, 0xbafdc912, 0x044e5ce9, 0x452f4a73, 0xe0f36b89,
        0x011538dd,
    ],
    uniform_f32: [
        0x3f2c83c8, 0x3e488b18, 0x3ec1c844, 0x3f3afdc8, 0x3c89cb80, 0x3e8a5e94, 0x3f60f36a,
        0x3b8a9c00,
    ],
    uniform_range_f32: [
        0x3faf4974, 0xbf82a911, 0xbddc5ab0, 0x3fd37a74, 0xbff53c1a, 0xbf26138e, 0x40193044,
        0xbffd4af4,
    ],
    uniform_f64: [
        0x3fe5907901064458,
        0x3fd83908cdeebf70,
        0x3f913973a514bd00,
        0x3fec1e6d712022a6,
        0x3fef8c4b26eb0062,
        0x3fe49cc7dd993790,
        0x3fec6f3ac7355500,
        0x3fd5c72a9987b104,
    ],
    uniform_range_f64: [
        0x3ff5e92e828faadc,
        0xbfbb8b4fe9590b40,
        0xbffea782f71a613c,
        0x40032608cd682b50,
        0x40076f5df0a5c07a,
        0x3ff387f3a9ff0ae8,
        0x40038b097902aa40,
        0xbfd31c2b00598aec,
    ],
    normal_f32: [
        0x3effce95, 0x3fb4a11f, 0xbdf4ec1f, 0xbf77c9d3, 0xbcbf7ed7, 0x3e3b16f9, 0x40036a07,
        0x3d5f9655,
    ],
    normal_f32_mpmath: [
        0x3effce95, 0x3fb4a11f, 0xbdf4ec20, 0xbf77c9d4, 0xbcbf7ed7, 0x3e3b16f9, 0x40036a07,
        0x3d5f9654,
    ],
    normal_complex_f32: [
        [0x3eb4e201, 0x3f7f72d2],
        [0xbdad2faf, 0xbf2f367d],
        [0xbc876863, 0x3e044adc],
        [0x3fb9d903, 0x3d1e199c],
    ],
    normal_complex_f32_mpmath: [
        [0x3eb4e202, 0x3f7f72d2],
        [0xbdad2faf, 0xbf2f367d],
        [0xbc876862, 0x3e044adc],
        [0x3fb9d903, 0x3d1e199c],
    ],
    normal_f64: [
        0xbff14d921c5bb7bb,
        0x3ff08feaf7d590f9,
        0x3fc10e4dc094f281,
        0xbfc0471e305cab14,
        0xbffcd2bb1d4ba8f1,
        0xc0025f7e67d7b136,
        0xbff202b2b93414ea,
        0x3ffc4501f97f1a1b,
    ],
    normal_f64_mpmath: [
        0xbff14d921c5bb7ba,
        0x3ff08feaf7d590f9,
        0x3fc10e4dc094f281,
        0xbfc0471e305cab14,
        0xbffcd2bb1d4ba8f2,
        0xc0025f7e67d7b136,
        0xbff202b2b93414ea,
        0x3ffc4501f97f1a1c,
    ],
    normal_complex_f64: [
        [0xbfe8785be8f4a5b0, 0x3fe76c2639bd7c89],
        [0x3fb81ee2ccd53d9d, 0xbfb70531d6f3f948],
        [0xbff4618cf884adb9, 0xbff9fbbea38a2285],
        [0xbfe9788307e8f68d, 0x3ff3fd5654acfa58],
    ],
    normal_complex_f64_mpmath: [
        [0xbfe8785be8f4a5af, 0x3fe76c2639bd7c89],
        [0x3fb81ee2ccd53d9d, 0xbfb70531d6f3f948],
        [0xbff4618cf884adb9, 0xbff9fbbea38a2285],
        [0xbfe9788307e8f68d, 0x3ff3fd5654acfa59],
    ],
    integer_i32_small: [1, -2, -5, 3, 4, 1, 3, -2],
    integer_i32_full: [
        746833927,
        -521919689,
        -2075239191,
        1626565512,
        2086820150,
        619069164,
        1668929080,
        -685987227,
    ],
    integer_i64: [
        1603813648572179306,
        -1120813996128148343,
        -4456542227780885191,
        3493022841577315438,
        4481412151327689391,
        1329440908363317304,
        3583997912550736927,
        -1473146350313796640,
    ],
};

const THREEFRY: Expected = Expected {
    bits: [
        0xe1e32c41, 0x07245dc9, 0x8dee15f1, 0x42b8c27f, 0xe83f4516, 0x2eaa7f83, 0x9914276a,
        0x0cc9ed69,
    ],
    uniform_f32: [
        0x3f61e32c, 0x3ce48b80, 0x3f0dee14, 0x3e857184, 0x3f683f44, 0x3e3aa9f8, 0x3f191426,
        0x3d4c9ec0,
    ],
    uniform_range_f32: [
        0x401a5bf8, 0xbfee251a, 0x3f45a664, 0xbf326436, 0x40224f14, 0xbf8b55c5, 0x3f7d64c0,
        0xbfe00732,
    ],
    uniform_f64: [
        0x3fec3c658820e48a,
        0x3fe1bdc2be285718,
        0x3fed07e8a2c5d54e,
        0x3fe32284ed41993c,
        0x3fe8b00f113f45be,
        0x3fe8a3d4ccc5a57a,
        0x3fe1502eb127e7f2,
        0x3fed7821879a1d1e,
    ],
    uniform_range_f64: [
        0x40034b7eea291dac,
        0x3fe8b4cdb6c9b378,
        0x400449e2cb774aa2,
        0x3fefac98a247fe2c,
        0x3ffdb825ab1e2e5c,
        0x3ffd9993ffee1db0,
        0x3fe690e975c787b8,
        0x4004d629e980a466,
    ],
    normal_f32: [
        0x400261e1, 0x3eb8bc51, 0xbdadd2b5, 0x3fa263d1, 0x3f6664cf, 0x3ffe3bd0, 0x3fa45a9c,
        0x3ed56824,
    ],
    normal_f32_mpmath: [
        0x400261e1, 0x3eb8bc52, 0xbdadd2b6, 0x3fa263d1, 0x3f6664cf, 0x3ffe3bd0, 0x3fa45a9c,
        0x3ed56824,
    ],
    normal_complex_f32: [
        [0x3fb86373, 0x3e82a0b8],
        [0xbd75d2ad, 0x3f65a76d],
        [0x3f22e9bb, 0x3fb3c535],
        [0x3f686e7c, 0x3e96e6c2],
    ],
    normal_complex_f32_mpmath: [
        [0x3fb86373, 0x3e82a0b8],
        [0xbd75d2ae, 0x3f65a76d],
        [0x3f22e9bb, 0x3fb3c535],
        [0x3f686e7c, 0x3e96e6c2],
    ],
    normal_f64: [
        0xbfff2fd97ff10823,
        0xbfe6326ad865d826,
        0xbffc7c7368cefeaa,
        0xbff425082f091961,
        0x3fcb90511f7d1b12,
        0xbffb46707848e7b8,
        0x3ff18db07747ae67,
        0xbfe309d226792f0e,
    ],
    normal_f64_mpmath: [
        0xbfff2fd97ff10823,
        0xbfe6326ad865d826,
        0xbffc7c7368cefeaa,
        0xbff425082f091961,
        0x3fcb90511f7d1b12,
        0xbffb46707848e7b9,
        0x3ff18db07747ae67,
        0xbfe309d226792f0e,
    ],
    normal_complex_f64: [
        [0xbff60d6f279bd7ed, 0xbfdf6426d61108dc],
        [0xbff4248a9ad22290, 0xbfec7d25013ff3c4],
        [0x3fc37d91cc1d9a76, 0xbff349548833d6c3],
        [0x3fe8d309504cd1be, 0xbfdaec9fa5589948],
    ],
    normal_complex_f64_mpmath: [
        [0xbff60d6f279bd7ed, 0xbfdf6426d61108dc],
        [0xbff4248a9ad22290, 0xbfec7d25013ff3c3],
        [0x3fc37d91cc1d9a76, 0xbff349548833d6c3],
        [0x3fe8d309504cd1be, 0xbfdaec9fa5589948],
    ],
    integer_i32_small: [3, 0, 4, 0, 2, 2, 0, 4],
    integer_i32_full: [
        1642277952, 233706992, 1748976917, 420751209, 1166047369, 1159636581, 176256392, 1807813691,
    ],
    integer_i64: [
        3526765049598324452,
        501881946450452799,
        3755899332525899713,
        903556343458494132,
        2504067659819579923,
        2490300597846535460,
        378507722353527129,
        3882250343952972726,
    ],
};

fn check_samples<R: CounterRng>(expected: &Expected) {
    let key = R::seed(2026);
    let name = R::NAME;
    for i in 0..8u64 {
        let at = i as usize;
        assert_eq!(bits::<R>(key, i), expected.bits[at], "{name} bits {i}");
        assert_eq!(
            uniform::<R, f32>(key, i).to_bits(),
            expected.uniform_f32[at],
            "{name} uniform f32 {i}"
        );
        assert_eq!(
            uniform::<R, f64>(key, i).to_bits(),
            expected.uniform_f64[at],
            "{name} uniform f64 {i}"
        );
        assert_eq!(
            uniform_range::<R, f32>(key, i, -2.0, 3.0).to_bits(),
            expected.uniform_range_f32[at],
            "{name} uniform_range f32 {i}"
        );
        assert_eq!(
            uniform_range::<R, f64>(key, i, -2.0, 3.0).to_bits(),
            expected.uniform_range_f64[at],
            "{name} uniform_range f64 {i}"
        );
        assert_eq!(
            integer::<R, i32>(key, i, -5, 5),
            expected.integer_i32_small[at],
            "{name} i32 {i}"
        );
        assert_eq!(
            integer::<R, i32>(key, i, i32::MIN, i32::MAX),
            expected.integer_i32_full[at],
            "{name} i32 full range {i}"
        );
        assert_eq!(
            integer::<R, i64>(key, i, -(1 << 62), 1 << 62),
            expected.integer_i64[at],
            "{name} i64 {i}"
        );

        let z32 = normal::<R, f32>(key, i).to_bits();
        assert_eq!(z32, expected.normal_f32[at], "{name} normal f32 {i}");
        assert!(
            ulps_f32(z32, expected.normal_f32_mpmath[at]) <= 1,
            "{name} normal f32 {i} vs mpmath"
        );
        let z64 = normal::<R, f64>(key, i).to_bits();
        assert_eq!(z64, expected.normal_f64[at], "{name} normal f64 {i}");
        assert!(
            ulps_f64(z64, expected.normal_f64_mpmath[at]) <= 1,
            "{name} normal f64 {i} vs mpmath"
        );
    }
    for i in 0..4u64 {
        let at = i as usize;
        // The complex uniform is the real uniforms of two consecutive elements.
        let [re, im] = uniform_complex::<R, f32>(key, i);
        assert_eq!(
            [re.to_bits(), im.to_bits()],
            [
                expected.uniform_f32[2 * at],
                expected.uniform_f32[2 * at + 1]
            ]
        );
        let [re, im] = uniform_complex::<R, f64>(key, i);
        assert_eq!(
            [re.to_bits(), im.to_bits()],
            [
                expected.uniform_f64[2 * at],
                expected.uniform_f64[2 * at + 1]
            ]
        );

        let z = normal_complex::<R, f32>(key, i).map(f32::to_bits);
        assert_eq!(
            z, expected.normal_complex_f32[at],
            "{name} normal complex f32 {i}"
        );
        for part in 0..2 {
            assert!(ulps_f32(z[part], expected.normal_complex_f32_mpmath[at][part]) <= 1);
        }
        let z = normal_complex::<R, f64>(key, i).map(f64::to_bits);
        assert_eq!(
            z, expected.normal_complex_f64[at],
            "{name} normal complex f64 {i}"
        );
        for part in 0..2 {
            assert!(ulps_f64(z[part], expected.normal_complex_f64_mpmath[at][part]) <= 1);
        }
    }
}

#[test]
fn philox_samples_match_known_answers() {
    check_samples::<Philox4x32_10>(&PHILOX);
}

#[test]
fn threefry_samples_match_known_answers() {
    check_samples::<Threefry2x32_20>(&THREEFRY);
}

#[test]
fn word_stream_is_the_concatenation_of_blocks_at_any_offset() {
    fn check<R: CounterRng>() {
        let key = R::seed(77);
        let mut reference = [0u32; 40];
        for (counter, chunk) in reference.chunks_mut(R::WORDS_PER_BLOCK).enumerate() {
            R::block(key, counter as u64, chunk);
        }
        for first in 0..9usize {
            for len in [0usize, 1, 3, 7, 16] {
                let mut window = vec![0u32; len];
                fill_words::<R>(key, first as u64, &mut window);
                assert_eq!(
                    window,
                    reference[first..first + len],
                    "{} {first}+{len}",
                    R::NAME
                );
            }
        }
        // Counters past 2^32 use the high counter word.
        let mut high = [0u32; 1];
        let first = (1u64 << 32) * R::WORDS_PER_BLOCK as u64;
        fill_words::<R>(key, first, &mut high);
        let mut block = vec![0u32; R::WORDS_PER_BLOCK];
        R::block(key, 1 << 32, &mut block);
        assert_eq!(high[0], block[0]);
        assert_ne!(high[0], reference[0]);
    }
    check::<Philox4x32_10>();
    check::<Threefry2x32_20>();
}

#[test]
fn samples_are_pure_functions_of_the_index() {
    // Element i does not depend on which other elements are drawn or in which
    // order: evaluate the same indices forwards, backwards and strided.
    let key = Philox4x32_10::seed(31);
    let forward: Vec<f64> = (0..64)
        .map(|i| normal::<Philox4x32_10, f64>(key, i))
        .collect();
    let backward: Vec<f64> = (0..64)
        .rev()
        .map(|i| normal::<Philox4x32_10, f64>(key, i))
        .collect();
    assert!(forward.iter().eq(backward.iter().rev()));
    for i in (0..64).step_by(5) {
        assert_eq!(
            forward[i as usize].to_bits(),
            normal::<Philox4x32_10, f64>(key, i).to_bits()
        );
    }
    // A different key gives different samples.
    let other = Philox4x32_10::fold_in(key, 1);
    assert!((0..64).all(|i| normal::<Philox4x32_10, f64>(other, i) != forward[i as usize]));
}

#[test]
fn normal_elements_2d_and_2d_plus_1_share_one_box_muller_draw() {
    // The complex draw d has radius sqrt(-log(1 - a)), the real pair
    // sqrt(-2 log(1 - a)); both use the same angle, so the ratios agree.
    let key = Threefry2x32_20::seed(5);
    for d in 0..8u64 {
        let re = normal::<Threefry2x32_20, f64>(key, 2 * d);
        let im = normal::<Threefry2x32_20, f64>(key, 2 * d + 1);
        let [cre, cim] = normal_complex::<Threefry2x32_20, f64>(key, d);
        let scale = core::f64::consts::SQRT_2;
        assert!((re - scale * cre).abs() <= 4.0 * f64::EPSILON * re.abs().max(1.0));
        assert!((im - scale * cim).abs() <= 4.0 * f64::EPSILON * im.abs().max(1.0));
    }
}

#[test]
fn uniform_range_stays_below_hi_when_rounding_reaches_it() {
    // hi - lo is one ulp of 1, so lo + (hi - lo) u rounds to hi for u > 1/2;
    // the result is then the largest value below hi, which is lo.
    let key = Philox4x32_10::seed(8);
    let (lo, hi) = (1.0f32, 1.0f32.next_up());
    let mut reached_hi = 0;
    for i in 0..64 {
        let u = uniform::<Philox4x32_10, f32>(key, i);
        if lo + (hi - lo) * u == hi {
            reached_hi += 1;
        }
        assert_eq!(uniform_range::<Philox4x32_10, f32>(key, i, lo, hi), lo);
    }
    assert!(reached_hi > 0, "no draw exercised the clamp");
    let (lo, hi) = (1.0f64, 1.0f64.next_up());
    for i in 0..64 {
        assert_eq!(uniform_range::<Philox4x32_10, f64>(key, i, lo, hi), lo);
    }
}

#[test]
fn integer_bounds_edge_cases() {
    let key = Threefry2x32_20::seed(3);
    for i in 0..16 {
        assert_eq!(integer::<Threefry2x32_20, i32>(key, i, 7, 8), 7);
        assert_eq!(integer::<Threefry2x32_20, i64>(key, i, -1, 0), -1);
        // hi <= lo returns lo.
        assert_eq!(integer::<Threefry2x32_20, i32>(key, i, 4, 4), 4);
        assert_eq!(integer::<Threefry2x32_20, i64>(key, i, 4, -9), 4);
        // The widest i64 range.
        let x = integer::<Threefry2x32_20, i64>(key, i, i64::MIN, i64::MAX);
        assert!(x < i64::MAX);
    }
    // With range 2^63 the result is lo + (x >> 1) for the 64-bit word pair x.
    let x = (u64::from(bits::<Threefry2x32_20>(key, 0)) << 32)
        | u64::from(bits::<Threefry2x32_20>(key, 1));
    assert_eq!(
        integer::<Threefry2x32_20, i64>(key, 0, 0, i64::MAX),
        ((u128::from(x) * u128::from(i64::MAX as u64)) >> 64) as i64
    );
    assert_eq!(
        integer::<Threefry2x32_20, i64>(key, 0, i64::MIN, 0),
        i64::MIN + (x >> 1) as i64
    );
}
