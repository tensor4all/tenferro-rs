//! Seed, split and fold-in properties shared by every algorithm, and the
//! algorithm attribute.

use cbrng::{CounterRng, Philox4x32_10, RngAlgorithm, Threefry2x32_20};

/// The checks every algorithm passes through the same generic code: the trait
/// is the only thing they share.
fn check_key_derivation<R: CounterRng>() {
    // Seed: [seed >> 32, seed & 0xFFFF_FFFF].
    assert_eq!(R::seed(0), [0, 0]);
    assert_eq!(R::seed(0xffff_ffff), [0, 0xffff_ffff]);
    assert_eq!(R::seed(0x1_0000_0000), [1, 0]);
    assert_eq!(R::seed(u64::MAX), [0xffff_ffff, 0xffff_ffff]);

    let key = R::seed(0x0bad_cafe_f00d);
    let original = key;

    // Determinism, and a longer split extends a shorter one.
    let mut short = [[0u32; 2]; 3];
    let mut long = [[0u32; 2]; 8];
    R::split(key, 3, &mut short);
    R::split(key, 8, &mut long);
    assert_eq!(short, long[..3]);
    let mut again = [[0u32; 2]; 8];
    R::split(key, 8, &mut again);
    assert_eq!(long, again);

    // Split keys differ from each other and from the parent.
    for (i, a) in long.iter().enumerate() {
        assert_ne!(*a, key, "{}: split key {i} equals its parent", R::NAME);
        for b in &long[i + 1..] {
            assert_ne!(a, b, "{}: two split keys are equal", R::NAME);
        }
    }
    // A different parent gives different children.
    let mut other = [[0u32; 2]; 8];
    R::split(R::seed(0x0bad_cafe_f00e), 8, &mut other);
    assert!(long.iter().zip(&other).all(|(a, b)| a != b));

    // Fold-in is deterministic and depends on the data and the key.
    assert_eq!(R::fold_in(key, 5), R::fold_in(key, 5));
    assert_ne!(R::fold_in(key, 5), R::fold_in(key, 6));
    assert_ne!(R::fold_in(key, 5), R::fold_in(R::seed(1), 5));
    assert_ne!(R::fold_in(key, 0), key);

    // An empty split writes nothing.
    R::split(key, 0, &mut []);

    // Keys are values: deriving never changes the parent.
    assert_eq!(key, original);
}

#[test]
fn key_derivation_properties_hold_for_every_builtin_algorithm() {
    check_key_derivation::<Philox4x32_10>();
    check_key_derivation::<Threefry2x32_20>();
}

#[test]
fn algorithms_with_the_same_key_give_different_blocks() {
    let key = [0x1234_5678, 0x9abc_def0];
    let mut philox = [0u32; 4];
    let mut threefry = [0u32; 2];
    Philox4x32_10::block(key, 0, &mut philox);
    Threefry2x32_20::block(key, 0, &mut threefry);
    assert_ne!(philox[..2], threefry);
}

#[test]
#[should_panic(expected = "split: `out` must hold exactly `n` keys")]
fn philox_split_rejects_an_output_of_the_wrong_length() {
    Philox4x32_10::split([0, 1], 3, &mut [[0; 2]; 2]);
}

#[test]
#[should_panic(expected = "split: `out` must hold exactly `n` keys")]
fn threefry_split_rejects_an_output_of_the_wrong_length() {
    Threefry2x32_20::split([0, 1], 1, &mut [[0; 2]; 2]);
}

#[test]
#[should_panic]
fn block_rejects_an_output_of_the_wrong_length() {
    Philox4x32_10::block([0, 1], 0, &mut [0; 2]);
}

#[test]
fn rng_algorithm_names_the_builtin_algorithms() {
    assert_eq!(RngAlgorithm::default(), RngAlgorithm::Philox4x32_10);
    assert_eq!(RngAlgorithm::Philox4x32_10.name(), "philox4x32-10");
    assert_eq!(RngAlgorithm::Threefry2x32_20.name(), "threefry2x32-20");
    assert_eq!(RngAlgorithm::Philox4x32_10.words_per_block(), 4);
    assert_eq!(RngAlgorithm::Threefry2x32_20.words_per_block(), 2);
    for algorithm in RngAlgorithm::ALL {
        assert_eq!(RngAlgorithm::from_name(algorithm.name()), Some(algorithm));
        assert_eq!(algorithm.to_string(), algorithm.name());
    }
    assert_eq!(RngAlgorithm::from_name("Philox4x32-10"), None);
    assert_eq!(RngAlgorithm::from_name(""), None);
}
