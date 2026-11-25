//! Threefry2x32-20 known answers: the Random123 vectors for the block
//! function and JAX's block, seed, split and fold-in values.
//!
//! The Random123 vectors are the `threefry2x32 20` lines of
//! `tests/kat_vectors` in DEShawResearch/random123. The JAX values are printed
//! by `reference/jax_threefry.py` (JAX 0.12.1, partitionable Threefry, the
//! JAX default since 0.5.0).

use cbrng::{bits, CounterRng, Threefry2x32_20};

#[test]
fn threefry_raw_block_matches_random123_kat_vectors() {
    let vectors: [([u32; 2], [u32; 2], [u32; 2]); 3] = [
        (
            [0x0000_0000; 2],
            [0x0000_0000; 2],
            [0x6b20_0159, 0x99ba_4efe],
        ),
        (
            [0xffff_ffff; 2],
            [0xffff_ffff; 2],
            [0x1cb9_96fc, 0xbb00_2be7],
        ),
        (
            [0x243f_6a88, 0x85a3_08d3],
            [0x1319_8a2e, 0x0370_7344],
            [0xc492_3a9c, 0x483d_f7a0],
        ),
    ];
    for (counter, key, expected) in vectors {
        assert_eq!(
            Threefry2x32_20::raw_block(counter, key),
            expected,
            "counter {counter:08x?}, key {key:08x?}"
        );
    }
}

#[test]
fn threefry_block_matches_jax_threefry_2x32() {
    // (key, counter) -> jax.extend.random.threefry_2x32(key, counter)
    let vectors: [([u32; 2], [u32; 2], [u32; 2]); 3] = [
        (
            [0x1319_8a2e, 0x0370_7344],
            [0x243f_6a88, 0x85a3_08d3],
            [0xc492_3a9c, 0x483d_f7a0],
        ),
        ([0, 7], [0, 5], [0xd591_7e25, 0x7415_e54e]),
        (
            [0xdead_beef, 0x0123_4567],
            [0xffff_ffff, 0x0000_0001],
            [0x0c8f_26c9, 0xe6d6_c435],
        ),
    ];
    for (key, counter, expected) in vectors {
        assert_eq!(Threefry2x32_20::raw_block(counter, key), expected);
        // CounterRng::block reads the 64-bit counter as [high, low].
        let mut out = [0u32; 2];
        Threefry2x32_20::block(
            key,
            (u64::from(counter[0]) << 32) | u64::from(counter[1]),
            &mut out,
        );
        assert_eq!(out, expected);
    }
}

#[test]
fn threefry_seed_split_and_fold_in_match_jax() {
    // jax.random.PRNGKey(jnp.uint64(seed)) with jax_enable_x64.
    assert_eq!(Threefry2x32_20::seed(0), [0, 0]);
    assert_eq!(Threefry2x32_20::seed(42), [0, 42]);
    assert_eq!(
        Threefry2x32_20::seed(0x0123_4567_89ab_cdef),
        [0x0123_4567, 0x89ab_cdef]
    );

    let key = Threefry2x32_20::seed(42);
    let mut keys = [[0u32; 2]; 4];
    Threefry2x32_20::split(key, 4, &mut keys);
    // jax.random.split(PRNGKey(42), 4)
    assert_eq!(
        keys,
        [
            [0x6d3e_048f, 0x1022_172d],
            [0x03d7_b32d, 0xadd0_83f4],
            [0x92fb_20ea, 0x0f38_d913],
            [0xbad5_6946, 0x354b_a891],
        ]
    );
    // jax.random.fold_in(PRNGKey(42), data)
    assert_eq!(Threefry2x32_20::fold_in(key, 0), [0x6d3e_048f, 0x1022_172d]);
    assert_eq!(Threefry2x32_20::fold_in(key, 1), [0x03d7_b32d, 0xadd0_83f4]);
    assert_eq!(Threefry2x32_20::fold_in(key, 7), [0x97d0_552f, 0x51bf_719f]);
    assert_eq!(
        Threefry2x32_20::fold_in(key, 0xffff_ffff),
        [0x8ef6_c48d, 0xe7eb_97ca]
    );
}

#[test]
fn threefry_split_fold_in_and_samples_share_the_counter_space_as_in_jax() {
    // Documented consequence of the JAX layout: split key i, fold_in(i) and
    // the sample block i are the same words.
    let key = Threefry2x32_20::seed(2026);
    let mut keys = [[0u32; 2]; 3];
    Threefry2x32_20::split(key, 3, &mut keys);
    for (index, derived) in keys.iter().enumerate() {
        assert_eq!(*derived, Threefry2x32_20::fold_in(key, index as u32));
        let word = 2 * index as u64;
        assert_eq!(
            *derived,
            [
                bits::<Threefry2x32_20>(key, word),
                bits::<Threefry2x32_20>(key, word + 1)
            ]
        );
    }
    assert_eq!(
        keys,
        [
            [0xe1e3_2c41, 0x0724_5dc9],
            [0x8dee_15f1, 0x42b8_c27f],
            [0xe83f_4516, 0x2eaa_7f83],
        ]
    );
    assert_eq!(
        Threefry2x32_20::fold_in(key, 0xdead_beef),
        [0x98ed_6c0d, 0x67c7_ed8c]
    );
}
