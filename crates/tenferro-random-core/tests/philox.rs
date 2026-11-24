//! Philox4x32-10 known answers: the Random123 vectors for the block function,
//! and the crate's counter layout for block, split and fold-in.
//!
//! The Random123 vectors are the `philox4x32 10` lines of
//! `tests/kat_vectors` in DEShawResearch/random123 (counter, key, output).
//! The split and fold-in values are from `reference/samples.py`, which
//! implements the layout independently.

use tenferro_random_core::{bits, CounterRng, Philox4x32_10};

#[test]
fn philox_raw_block_matches_random123_kat_vectors() {
    let vectors: [([u32; 4], [u32; 2], [u32; 4]); 3] = [
        (
            [0x0000_0000; 4],
            [0x0000_0000; 2],
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8],
        ),
        (
            [0xffff_ffff; 4],
            [0xffff_ffff; 2],
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd],
        ),
        (
            [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
            [0xa409_3822, 0x299f_31d0],
            [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1],
        ),
    ];
    for (counter, key, expected) in vectors {
        assert_eq!(
            Philox4x32_10::raw_block(counter, key),
            expected,
            "counter {counter:08x?}, key {key:08x?}"
        );
    }
}

#[test]
fn philox_block_puts_the_64_bit_counter_in_words_0_and_1() {
    let key = [0xa409_3822, 0x299f_31d0];
    for counter in [
        0u64,
        1,
        0xffff_ffff,
        0x1_0000_0000,
        0x0123_4567_89ab_cdef,
        u64::MAX,
    ] {
        let mut out = [0u32; 4];
        Philox4x32_10::block(key, counter, &mut out);
        assert_eq!(
            out,
            Philox4x32_10::raw_block([counter as u32, (counter >> 32) as u32, 0, 0], key),
            "counter {counter:#x}"
        );
    }
}

#[test]
fn philox_split_and_fold_in_match_the_reference_layout() {
    let key = Philox4x32_10::seed(2026);
    assert_eq!(key, [0, 2026]);

    let mut keys = [[0u32; 2]; 3];
    Philox4x32_10::split(key, 3, &mut keys);
    assert_eq!(
        keys,
        [
            [0x31d3_f1e5, 0x2f0f_8841],
            [0x6c4c_e44e, 0x410f_241a],
            [0xdee0_796d, 0xc1b1_12fe],
        ]
    );
    // Key i is words 0-1 of the block at counter [i, 0, 0, 1].
    for (index, derived) in keys.iter().enumerate() {
        let words = Philox4x32_10::raw_block([index as u32, 0, 0, 1], key);
        assert_eq!(*derived, [words[0], words[1]]);
    }

    assert_eq!(Philox4x32_10::fold_in(key, 0), [0xdcae_4c78, 0x8667_bd93]);
    assert_eq!(Philox4x32_10::fold_in(key, 1), [0x48b4_de43, 0x0331_e96b]);
    assert_eq!(
        Philox4x32_10::fold_in(key, 0xdead_beef),
        [0xfe2c_0c9e, 0x4221_2b9a]
    );
    let words = Philox4x32_10::raw_block([0xdead_beef, 0, 0, 2], key);
    assert_eq!(
        Philox4x32_10::fold_in(key, 0xdead_beef),
        [words[0], words[1]]
    );
}

#[test]
fn philox_derived_keys_do_not_reuse_sample_blocks() {
    // Samples read counter word 3 = 0, split word 3 = 1, fold-in word 3 = 2,
    // so split key i, fold-in key i and sample block i all differ.
    let key = Philox4x32_10::seed(9);
    let mut keys = [[0u32; 2]; 16];
    Philox4x32_10::split(key, 16, &mut keys);
    for (index, derived) in keys.iter().enumerate() {
        let index = index as u64;
        let sample_words = [
            bits::<Philox4x32_10>(key, 4 * index),
            bits::<Philox4x32_10>(key, 4 * index + 1),
        ];
        assert_ne!(
            *derived, sample_words,
            "split key {index} equals sample block {index}"
        );
        assert_ne!(*derived, Philox4x32_10::fold_in(key, index as u32));
    }
}
