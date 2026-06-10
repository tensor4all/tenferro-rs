use super::adjoint_lu_solve_flags;

#[test]
fn adjoint_lu_solve_flags_preserve_mixed_complex_adjoint_cases() {
    let cases = [
        ((false, false), (true, true)),
        ((true, false), (false, true)),
        ((false, true), (true, false)),
        ((true, true), (false, false)),
    ];

    for ((transpose_a, conjugate_a), expected) in cases {
        assert_eq!(
            adjoint_lu_solve_flags(transpose_a, conjugate_a),
            expected,
            "adjoint flags for transpose_a={transpose_a}, conjugate_a={conjugate_a}"
        );
    }
}
