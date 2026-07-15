use super::*;

#[test]
fn numeric_regions_reject_all_six_alias_pairs_without_touching_sentinels() {
    let separate = [
        ByteRegion { start: 0, end: 16 },
        ByteRegion { start: 32, end: 48 },
        ByteRegion { start: 64, end: 72 },
        ByteRegion {
            start: 96,
            end: 104,
        },
    ];
    for (expected, lhs, rhs) in [
        ("input and U", 0, 1),
        ("input and S", 0, 2),
        ("input and Vt", 0, 3),
        ("U and S", 1, 2),
        ("U and Vt", 1, 3),
        ("S and Vt", 2, 3),
    ] {
        let mut regions = separate;
        regions[rhs] = regions[lhs];
        let sentinels = [11_u8, 22, 33];
        let error = validate_non_aliasing(
            Some(regions[0]),
            Some(regions[1]),
            Some(regions[2]),
            Some(regions[3]),
        )
        .unwrap_err();
        assert!(error.to_string().contains(expected));
        assert_eq!(sentinels, [11, 22, 33]);
    }
}
