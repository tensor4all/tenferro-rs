use num_complex::Complex64;
use tenferro_api_grammar_proof::p2_einsum_dest::{
    c64, Dest, EinsumInputs, EinsumOptions, EinsumSpec,
};
use tenferro_api_grammar_proof::policy::MaterializationPolicy::{Allow, Forbid};
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{
    parse_einsum_notation, ConcreteEinsumPlan, EinsumNotation, EinsumOptimize, EinsumSubscripts,
};
use tenferro_tensor::{DotGeneralAccumulation, Tensor, TensorRead, TensorWrite};

type C = Complex64;

fn z(re: f64, im: f64) -> C {
    C::new(re, im)
}

/// Column-major `rows x cols` matrix of deterministic, non-symmetric complex values.
fn mat(rows: usize, cols: usize, seed: f64) -> (Tensor, Vec<C>) {
    let mut data = Vec::with_capacity(rows * cols);
    for j in 0..cols {
        for i in 0..rows {
            let re = seed + (i as f64) * 0.7 - (j as f64) * 0.3 + (i * j) as f64 * 0.11;
            let im = 0.5 * seed - (i as f64) * 0.2 + (j as f64) * 0.9;
            data.push(z(re, im));
        }
    }
    (
        Tensor::from_vec_col_major(vec![rows, cols], data.clone()).unwrap(),
        data,
    )
}

/// Independent reference: C[i,k] = sum_j op(A)[i,j] * op(B)[j,k], column-major.
fn ref_mm(a: &[C], m: usize, n: usize, b: &[C], p: usize, ca: bool, cb: bool) -> Vec<C> {
    let mut out = vec![z(0.0, 0.0); m * p];
    for k in 0..p {
        for i in 0..m {
            let mut s = z(0.0, 0.0);
            for j in 0..n {
                let x = if ca {
                    a[i + m * j].conj()
                } else {
                    a[i + m * j]
                };
                let y = if cb {
                    b[j + n * k].conj()
                } else {
                    b[j + n * k]
                };
                s += x * y;
            }
            out[i + m * k] = s;
        }
    }
    out
}

macro_rules! reads_of {
    ($($t:expr),+) => { [$(TensorRead::from_tensor(&$t)),+] };
}

fn close(actual: &[C], expected: &[C]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).norm() <= 1e-10 * (1.0 + e.norm()),
            "element {i}: got {a}, expected {e}"
        );
    }
}

fn out_tensor(m: usize, p: usize, fill: C) -> Tensor {
    Tensor::from_vec_col_major(vec![m, p], vec![fill; m * p]).unwrap()
}

fn vals(t: &Tensor) -> Vec<C> {
    t.as_slice::<C>().unwrap().to_vec()
}

struct Fixture {
    a: Tensor,
    av: Vec<C>,
    b: Tensor,
    bv: Vec<C>,
    c: Tensor,
    cv: Vec<C>,
}

// A: 3x4, B: 4x2, C: 2x5 ; binary -> 3x2 ; ternary -> 3x5
fn fx() -> Fixture {
    let (a, av) = mat(3, 4, 1.0);
    let (b, bv) = mat(4, 2, -0.5);
    let (c, cv) = mat(2, 5, 0.25);
    Fixture {
        a,
        av,
        b,
        bv,
        c,
        cv,
    }
}

fn ref_binary(f: &Fixture, ca: bool, cb: bool) -> Vec<C> {
    ref_mm(&f.av, 3, 4, &f.bv, 2, ca, cb)
}
fn ref_ternary(f: &Fixture) -> Vec<C> {
    let ab = ref_binary(f, false, false);
    ref_mm(&ab, 3, 2, &f.cv, 5, false, false)
}

#[test]
fn binary_overwrite_all_spec_forms() {
    let f = fx();
    let mut eng = CpuEngine::serial();
    let r = reads_of!(f.a, f.b);
    let expected = ref_binary(&f, false, false);

    // text
    let out = r.einsum("ij,jk->ik", &mut eng).unwrap();
    assert_eq!(out.shape(), &[3, 2]);
    close(&vals(&out), &expected);

    // parsed integer subscripts
    let subs = EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    close(&vals(&r.einsum(&subs, &mut eng).unwrap()), &expected);

    // rank-unresolved notation
    let notation: EinsumNotation = parse_einsum_notation("ij,jk->ik").unwrap();
    close(&vals(&r.einsum(&notation, &mut eng).unwrap()), &expected);

    // owned index lists
    let spec = EinsumSpec::indices(&[&[0, 1], &[1, 2]], &[0, 2]);
    close(&vals(&r.einsum(spec, &mut eng).unwrap()), &expected);

    // into + Overwrite over a garbage-filled destination
    let mut dst = out_tensor(3, 2, z(7.0, -7.0));
    r.einsum_into(
        "ij,jk->ik",
        &EinsumOptions::default(),
        &mut eng,
        Allow,
        Dest::overwrite(TensorWrite::from_tensor(&mut dst)),
    )
    .unwrap();
    close(&vals(&dst), &expected);
}

#[test]
fn ternary_overwrite_value_and_into() {
    let f = fx();
    let mut eng = CpuEngine::serial();
    let r = reads_of!(f.a, f.b, f.c);
    let expected = ref_ternary(&f);

    let out = r.einsum("ij,jk,kl->il", &mut eng).unwrap();
    assert_eq!(out.shape(), &[3, 5]);
    close(&vals(&out), &expected);

    let mut dst = out_tensor(3, 5, z(9.0, 9.0));
    r.einsum_into(
        "ij,jk,kl->il",
        &EinsumOptions::default(),
        &mut eng,
        Allow,
        Dest::overwrite(TensorWrite::from_tensor(&mut dst)),
    )
    .unwrap();
    close(&vals(&dst), &expected);
}

#[test]
fn accumulate_binary_and_ternary() {
    let f = fx();
    let mut eng = CpuEngine::serial();
    let old = z(1.5, -2.0);

    let r2 = reads_of!(f.a, f.b);
    let mut dst = out_tensor(3, 2, old);
    r2.einsum_into(
        "ij,jk->ik",
        &EinsumOptions::default(),
        &mut eng,
        Allow,
        Dest::accumulate(TensorWrite::from_tensor(&mut dst)),
    )
    .unwrap();
    let expected: Vec<C> = ref_binary(&f, false, false)
        .iter()
        .map(|v| v + old)
        .collect();
    close(&vals(&dst), &expected);

    let r3 = reads_of!(f.a, f.b, f.c);
    let mut dst = out_tensor(3, 5, old);
    r3.einsum_into(
        "ij,jk,kl->il",
        &EinsumOptions::default(),
        &mut eng,
        Allow,
        Dest::accumulate(TensorWrite::from_tensor(&mut dst)),
    )
    .unwrap();
    let expected: Vec<C> = ref_ternary(&f).iter().map(|v| v + old).collect();
    close(&vals(&dst), &expected);
}

#[test]
fn scaled_alpha_beta_binary_and_ternary() {
    let f = fx();
    let mut eng = CpuEngine::serial();
    let old = z(1.5, -2.0);
    let alpha = z(0.5, 2.0);
    let beta = z(-1.0, 0.25);

    let r2 = reads_of!(f.a, f.b);
    let mut dst = out_tensor(3, 2, old);
    r2.einsum_into(
        "ij,jk->ik",
        &EinsumOptions::default(),
        &mut eng,
        Allow,
        Dest::scaled(
            TensorWrite::from_tensor(&mut dst),
            c64(alpha.re, alpha.im),
            c64(beta.re, beta.im),
        ),
    )
    .unwrap();
    let expected: Vec<C> = ref_binary(&f, false, false)
        .iter()
        .map(|v| alpha * v + beta * old)
        .collect();
    close(&vals(&dst), &expected);

    let r3 = reads_of!(f.a, f.b, f.c);
    let mut dst = out_tensor(3, 5, old);
    r3.einsum_into(
        "ij,jk,kl->il",
        &EinsumOptions::default(),
        &mut eng,
        Allow,
        Dest::scaled(
            TensorWrite::from_tensor(&mut dst),
            c64(alpha.re, alpha.im),
            c64(beta.re, beta.im),
        ),
    )
    .unwrap();
    let expected: Vec<C> = ref_ternary(&f)
        .iter()
        .map(|v| alpha * v + beta * old)
        .collect();
    close(&vals(&dst), &expected);
}

#[test]
fn beta_zero_never_reads_old_output() {
    let f = fx();
    let mut eng = CpuEngine::serial();
    let nan = z(f64::NAN, f64::NAN);
    let alpha = z(2.0, -1.0);

    for beta_zero in [c64(0.0, 0.0), c64(-0.0, 0.0), c64(0.0, -0.0)] {
        // binary, scaled with beta == 0
        let r2 = reads_of!(f.a, f.b);
        let mut dst = out_tensor(3, 2, nan);
        r2.einsum_into(
            "ij,jk->ik",
            &EinsumOptions::default(),
            &mut eng,
            Allow,
            Dest::scaled(
                TensorWrite::from_tensor(&mut dst),
                c64(alpha.re, alpha.im),
                beta_zero,
            ),
        )
        .unwrap();
        let exp: Vec<C> = ref_binary(&f, false, false)
            .iter()
            .map(|v| alpha * v)
            .collect();
        assert!(vals(&dst)
            .iter()
            .all(|v| v.re.is_finite() && v.im.is_finite()));
        close(&vals(&dst), &exp);

        // ternary, scaled with beta == 0
        let r3 = reads_of!(f.a, f.b, f.c);
        let mut dst = out_tensor(3, 5, nan);
        r3.einsum_into(
            "ij,jk,kl->il",
            &EinsumOptions::default(),
            &mut eng,
            Allow,
            Dest::scaled(
                TensorWrite::from_tensor(&mut dst),
                c64(alpha.re, alpha.im),
                beta_zero,
            ),
        )
        .unwrap();
        let exp: Vec<C> = ref_ternary(&f).iter().map(|v| alpha * v).collect();
        close(&vals(&dst), &exp);
    }

    // Overwrite over NaN: binary and ternary
    let r2 = reads_of!(f.a, f.b);
    let mut dst = out_tensor(3, 2, nan);
    r2.einsum_into(
        "ij,jk->ik",
        &EinsumOptions::default(),
        &mut eng,
        Allow,
        Dest::overwrite(TensorWrite::from_tensor(&mut dst)),
    )
    .unwrap();
    close(&vals(&dst), &ref_binary(&f, false, false));
    let r3 = reads_of!(f.a, f.b, f.c);
    let mut dst = out_tensor(3, 5, nan);
    r3.einsum_into(
        "ij,jk,kl->il",
        &EinsumOptions::default(),
        &mut eng,
        Allow,
        Dest::overwrite(TensorWrite::from_tensor(&mut dst)),
    )
    .unwrap();
    close(&vals(&dst), &ref_ternary(&f));

    // Control: beta != 0 over NaN DOES propagate NaN (it is a read-modify-write).
    let mut dst = out_tensor(3, 2, nan);
    reads_of!(f.a, f.b)
        .einsum_into(
            "ij,jk->ik",
            &EinsumOptions::default(),
            &mut eng,
            Allow,
            Dest::accumulate(TensorWrite::from_tensor(&mut dst)),
        )
        .unwrap();
    assert!(vals(&dst).iter().all(|v| v.re.is_nan() || v.im.is_nan()));
}

#[test]
fn bad_specs_return_errors() {
    let f = fx();
    let mut eng = CpuEngine::serial();
    let r2 = reads_of!(f.a, f.b);
    let r3 = reads_of!(f.a, f.b, f.c);

    for text in [
        "ij,jk-ik",
        "ij,jk->ikz",
        "ij->ij",
        "ijj,jk->ik",
        "",
        "ij,jk->ik->",
        "i-j,jk->ik",
    ] {
        assert!(
            r2.einsum(text, &mut eng).is_err(),
            "{text:?} must be an error"
        );
    }
    // wrong number of inputs for the spec
    assert!(r3.einsum("ij,jk->ik", &mut eng).is_err());
    // inconsistent dimension for label j (4 vs 5)
    let (b5, _) = mat(5, 2, 0.1);
    let bad = reads_of!(f.a, b5);
    assert!(bad.einsum("ij,jk->ik", &mut eng).is_err());
    // integer-label spec with wrong arity
    let subs = EinsumSubscripts::new(&[&[0, 1]], &[0]);
    assert!(r2.einsum(&subs, &mut eng).is_err());
    // notation with ellipsis on rank-2 input is legal; wrong count is not
    let n = EinsumNotation::parse("...,...->...").unwrap();
    assert!(r3.einsum(&n, &mut eng).is_err());
    // empty input slice
    let none: [TensorRead<'_>; 0] = [];
    assert!(none.einsum("->", &mut eng).is_err());

    // The error must be a typed tenferro_tensor::Error that keeps the einsum cause.
    let err = r2.einsum("ij,jk-ik", &mut eng).unwrap_err();
    eprintln!("bad-spec error: {err:?} kind={:?}", err.kind());
    let mut src = std::error::Error::source(&err);
    let mut chain = 0;
    while let Some(s) = src {
        eprintln!("  source[{chain}]: {s}");
        chain += 1;
        src = s.source();
    }
}

#[test]
fn forbid_is_typed_unsupported_and_has_no_side_effect() {
    let f = fx();
    let mut eng = CpuEngine::serial();
    let r2 = reads_of!(f.a, f.b);
    let mut dst = out_tensor(3, 2, z(4.0, 4.0));
    let err = r2
        .einsum_into(
            "ij,jk->ik",
            &EinsumOptions::default(),
            &mut eng,
            Forbid,
            Dest::overwrite(TensorWrite::from_tensor(&mut dst)),
        )
        .unwrap_err();
    assert!(
        matches!(err, tenferro_tensor::Error::Unsupported { .. }),
        "{err:?}"
    );
    assert!(vals(&dst).iter().all(|v| *v == z(4.0, 4.0)));
    let err = r2
        .einsum_with("ij,jk->ik", &EinsumOptions::default(), &mut eng, Forbid)
        .unwrap_err();
    assert!(matches!(err, tenferro_tensor::Error::Unsupported { .. }));
}

#[test]
fn optimize_other_than_auto_is_rejected_by_concrete_layer() {
    let f = fx();
    let mut eng = CpuEngine::serial();
    let r3 = reads_of!(f.a, f.b, f.c);
    for opt in [EinsumOptimize::False] {
        let opts = EinsumOptions::default().with_optimize(opt);
        let err = r3
            .einsum_with("ij,jk,kl->il", &opts, &mut eng, Allow)
            .unwrap_err();
        assert!(
            matches!(err, tenferro_tensor::Error::Unsupported { .. }),
            "{err:?}"
        );
    }
}

// ---- conjugation -------------------------------------------------------------------

fn conj_all(v: &[C]) -> Vec<C> {
    v.iter().map(|x| x.conj()).collect()
}

#[test]
fn conj_binary_exact_output_is_honoured() {
    let f = fx();
    let mut eng = CpuEngine::serial();
    let r2 = reads_of!(f.a, f.b);
    for (ca, cb) in [(true, false), (false, true), (true, true)] {
        let mut dst = out_tensor(3, 2, z(f64::NAN, f64::NAN));
        r2.einsum_into(
            "ij,jk->ik",
            &EinsumOptions::default().with_conj(vec![ca, cb]),
            &mut eng,
            Allow,
            Dest::overwrite(TensorWrite::from_tensor(&mut dst)),
        )
        .unwrap();
        close(&vals(&dst), &ref_binary(&f, ca, cb));
    }
    // and with accumulation + a swapped operand order in the text (kj,ij->ik needs swap)
    let mut dst = out_tensor(3, 2, z(1.0, 1.0));
    r2.einsum_into(
        "ij,jk->ik",
        &EinsumOptions::default().with_conj(vec![true, false]),
        &mut eng,
        Allow,
        Dest::accumulate(TensorWrite::from_tensor(&mut dst)),
    )
    .unwrap();
    let exp: Vec<C> = ref_binary(&f, true, false)
        .iter()
        .map(|v| v + z(1.0, 1.0))
        .collect();
    close(&vals(&dst), &exp);
}

#[test]
fn conj_ternary_current_behaviour() {
    let f = fx();
    let mut eng = CpuEngine::serial();
    let r3 = reads_of!(f.a, f.b, f.c);
    let honoured = {
        // reference with A conjugated
        let ab = ref_mm(&conj_all(&f.av), 3, 4, &f.bv, 2, false, false);
        ref_mm(&ab, 3, 2, &f.cv, 5, false, false)
    };
    let unconj = ref_ternary(&f);

    // Through the adapter (-> execute_read_into_accum -> CPU N-ary branch).
    let mut dst = out_tensor(3, 5, z(0.0, 0.0));
    r3.einsum_into(
        "ij,jk,kl->il",
        &EinsumOptions::default().with_conj(vec![true, false, false]),
        &mut eng,
        Allow,
        Dest::accumulate(TensorWrite::from_tensor(&mut dst)),
    )
    .unwrap();
    let got = vals(&dst);
    let matches_conj = got
        .iter()
        .zip(&honoured)
        .all(|(a, e)| (a - e).norm() < 1e-9);
    let matches_plain = got.iter().zip(&unconj).all(|(a, e)| (a - e).norm() < 1e-9);
    eprintln!("ternary conj_lhs: honoured={matches_conj} silently_dropped={matches_plain}");
    // PINNED CURRENT BEHAVIOUR: the request succeeds (Ok) and conj is silently dropped.
    assert!(
        matches_plain && !matches_conj,
        "ternary conj behaviour changed; update findings/p2.md"
    );

    // Same through the raw plan API, to show the adapter is not the cause.
    let plan = ConcreteEinsumPlan::prepare_read(&r3, "ij,jk,kl->il").unwrap();
    let mut accum = DotGeneralAccumulation::overwrite(tenferro_tensor::DType::C64).unwrap();
    accum.lhs_conj = true;
    accum.rhs_conj = true;
    let mut dst = out_tensor(3, 5, z(0.0, 0.0));
    plan.execute_read_into_accum(&r3, &mut eng, accum, TensorWrite::from_tensor(&mut dst))
        .unwrap();
    let got = vals(&dst);
    assert!(got.iter().zip(&unconj).all(|(a, e)| (a - e).norm() < 1e-9));

    // Conjugating the 3rd operand cannot even be named.
    let err = r3
        .einsum_into(
            "ij,jk,kl->il",
            &EinsumOptions::default().with_conj(vec![false, false, true]),
            &mut eng,
            Allow,
            Dest::overwrite(TensorWrite::from_tensor(&mut dst)),
        )
        .unwrap_err();
    assert!(matches!(err, tenferro_tensor::Error::Unsupported { .. }));
}

#[test]
fn conj_binary_non_exact_output_current_behaviour() {
    // ij,jk->ki is a binary contraction whose output order is not the dot_general
    // order, so ConcreteEinsumPlan has no binary_dot shortcut for it.
    let f = fx();
    let mut eng = CpuEngine::serial();
    let r2 = reads_of!(f.a, f.b);
    let mut dst = out_tensor(2, 3, z(0.0, 0.0));
    r2.einsum_into(
        "ij,jk->ki",
        &EinsumOptions::default().with_conj(vec![true, false]),
        &mut eng,
        Allow,
        Dest::overwrite(TensorWrite::from_tensor(&mut dst)),
    )
    .unwrap();
    // reference (transposed): out[k,i] = sum_j conj(A[i,j]) B[j,k]
    let c_ik = ref_binary(&f, true, false);
    let c_plain = ref_binary(&f, false, false);
    let transpose = |v: &[C]| {
        let mut t = vec![z(0.0, 0.0); 6];
        for i in 0..3 {
            for k in 0..2 {
                t[k + 2 * i] = v[i + 3 * k];
            }
        }
        t
    };
    let got = vals(&dst);
    let honoured = got
        .iter()
        .zip(transpose(&c_ik))
        .all(|(a, e)| (a - e).norm() < 1e-9);
    let dropped = got
        .iter()
        .zip(transpose(&c_plain))
        .all(|(a, e)| (a - e).norm() < 1e-9);
    eprintln!("binary ij,jk->ki conj_lhs: honoured={honoured} dropped={dropped}");
    // PINNED: the transposed-output binary case honours conj (binary_dot plan handles it).
    assert!(honoured && !dropped);
}

#[test]
fn conj_binary_without_dot_plan_current_behaviour() {
    // "ii,ij->j" repeats a label inside one operand, so no BinaryDotPlan exists and
    // execution falls into eager_einsum_exec_read_into_accum's generic tail.
    let (sq, sqv) = mat(3, 3, 0.3);
    let (m, mv) = mat(3, 2, -1.1);
    let mut eng = CpuEngine::serial();
    let r2 = reads_of!(sq, m);
    let mut dst = Tensor::from_vec_col_major(vec![2], vec![z(0.0, 0.0); 2]).unwrap();
    r2.einsum_into(
        "ii,ij->j",
        &EinsumOptions::default().with_conj(vec![true, false]),
        &mut eng,
        Allow,
        Dest::overwrite(TensorWrite::from_tensor(&mut dst)),
    )
    .unwrap();
    let tr = |conj: bool| -> Vec<C> {
        (0..2)
            .map(|j| {
                (0..3)
                    .map(|i| {
                        let d = sqv[i + 3 * i];
                        (if conj { d.conj() } else { d }) * mv[i + 3 * j]
                    })
                    .sum()
            })
            .collect()
    };
    let got = vals(&dst);
    let honoured = got.iter().zip(tr(true)).all(|(a, e)| (a - e).norm() < 1e-9);
    let dropped = got
        .iter()
        .zip(tr(false))
        .all(|(a, e)| (a - e).norm() < 1e-9);
    eprintln!("binary ii,ij->j conj_lhs: honoured={honoured} dropped={dropped}");
    // PINNED: without a BinaryDotPlan the conj request succeeds and is silently dropped.
    assert!(
        dropped && !honoured,
        "behaviour changed; update findings/p2.md"
    );
}

#[test]
fn conj_on_value_returning_einsum_is_not_expressible() {
    let f = fx();
    let mut eng = CpuEngine::serial();
    let r2 = reads_of!(f.a, f.b);
    let err = r2
        .einsum_with(
            "ij,jk->ik",
            &EinsumOptions::default().with_conj(vec![true, false]),
            &mut eng,
            Allow,
        )
        .unwrap_err();
    assert!(matches!(err, tenferro_tensor::Error::Unsupported { .. }));
}
