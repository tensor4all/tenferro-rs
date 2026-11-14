//! The four guards of the generated grammar (#2552 §4.4) and the generated
//! documentation check (§4.5).

use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_tensor::{
    AsView, Dest, ErrorKind, StridedSliceSpec, Tensor, TensorRead, TensorScalar, TypedTensor,
};

use super::cases;
use super::support::*;
use crate::api::registry::{api_docs_sync, API_LAYERS, API_OPS};
use crate::{
    CholeskyOptions, LinalgOps, MatOp, Svd, SvdGauge, SvdOptions, SvdShape, TriangularSolveOptions,
    Uplo,
};

// ------------------------------------------------------------ 1. forwarding

#[test]
fn every_registry_entry_has_a_forwarding_case() {
    let cases = cases::all();
    for op in API_OPS {
        for layer in op.layers {
            if *layer == "eager" && cfg!(not(feature = "autodiff")) {
                continue;
            }
            assert!(
                cases.iter().any(|c| c.op == op.name && c.layer == *layer),
                "no forwarding case for {}.{}",
                layer,
                op.name
            );
        }
    }
    for c in &cases {
        let op = API_OPS
            .iter()
            .find(|op| op.name == c.op)
            .unwrap_or_else(|| panic!("case for unknown op {}", c.op));
        assert!(
            op.layers.contains(&c.layer),
            "{} is not declared on {}",
            c.op,
            c.layer
        );
    }
    assert!(API_LAYERS.iter().any(|(name, _)| *name == "concrete"));
}

#[test]
fn op_and_op_with_default_options_agree_bit_for_bit() {
    let mut failures = Vec::new();
    for case in cases::all() {
        for (dtype, run) in ["f64", "c64"].iter().zip(case.run) {
            match run() {
                Ok((plain, with)) if plain == with => {}
                Ok(_) => {
                    failures.push(format!("{}.{} ({dtype}): bits differ", case.layer, case.op))
                }
                Err(e) => failures.push(format!("{}.{} ({dtype}): {e}", case.layer, case.op)),
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

// ---------------------------------------------------------- 2. input shapes

/// A 3 x 3 matrix whose transposed and reversed views are again the inputs
/// of the same computation once their logical elements are copied out.
fn base<T: Sample>() -> TestResult<TypedTensor<T>> {
    let entries = [
        (4.0, 0.0),
        (1.0, 0.5),
        (0.5, -1.0),
        (2.0, 1.0),
        (5.0, 0.0),
        (1.0, 0.0),
        (0.0, 0.5),
        (1.5, 0.0),
        (6.0, 0.25),
    ];
    Ok(TypedTensor::from_vec_col_major(
        vec![3, 3],
        entries.iter().map(|&(re, im)| T::mk(re, im)).collect(),
    )?)
}

/// The logical elements of `view` in a fresh compact tensor.
fn compact_copy<T: Sample>(view: &tenferro_tensor::TypedTensorView<'_, T>) -> TestResult<Tensor> {
    let (m, n) = (view.shape()[0], view.shape()[1]);
    let mut data = Vec::with_capacity(m * n);
    for col in 0..n {
        for row in 0..m {
            data.push(*view.get(&[row, col]).ok_or("index out of range")?);
        }
    }
    Ok(Tensor::from_typed::<T>(TypedTensor::from_vec_col_major(
        vec![m, n],
        data,
    )?))
}

/// Owned tensor, compact borrowed view, transposed view, negative-stride view
/// and typed tensor all return the result of the owned call on the same
/// logical matrix. The provider reads the strided views in place and gathers
/// the negative-stride one; every route runs the same kernel, so the results
/// agree bit for bit.
fn input_kinds_agree<T: Sample>() -> TestResult<()> {
    let typed_base = base::<T>()?;
    let mut e = CpuEngine::serial();
    let owned = Tensor::from_typed::<T>(typed_base.duplicate()?);

    let check =
        |label: &str, logical: &Tensor, got: Svd<Tensor>, e: &mut CpuEngine| -> TestResult<()> {
            let want = logical.svd(e)?;
            assert_eq!(
                got.bits()?,
                want.bits()?,
                "svd of the {label} input differs"
            );
            Ok(())
        };
    let check_solve =
        |label: &str, logical: &Tensor, got: Tensor, e: &mut CpuEngine| -> TestResult<()> {
            let b = tensor::<T>(rhs3::<T>())?;
            let want = logical.solve(&b, e)?;
            assert_eq!(
                got.bits()?,
                want.bits()?,
                "solve of the {label} input differs"
            );
            Ok(())
        };
    let b = tensor::<T>(rhs3::<T>())?;

    // owned
    check("owned", &owned, owned.svd(&mut e)?, &mut e)?;
    check_solve("owned", &owned, owned.solve(&b, &mut e)?, &mut e)?;

    // compact borrowed view
    let read = owned.as_read();
    check("compact view", &owned, read.svd(&mut e)?, &mut e)?;
    check_solve(
        "compact view",
        &owned,
        read.solve(&b.as_read(), &mut e)?,
        &mut e,
    )?;

    // transposed and negative-stride views
    let views = [
        ("transposed", typed_base.as_view().transpose_view([1, 0])?),
        (
            "negative-stride",
            typed_base
                .as_view()
                .slice_view(&[StridedSliceSpec::reverse(), StridedSliceSpec::reverse()])?,
        ),
    ];
    for (label, view) in views {
        let logical = compact_copy(&view)?;
        let r = TensorRead::from_view(T::tensor_view(view.clone()));
        check(label, &logical, r.svd(&mut e)?, &mut e)?;
        check_solve(label, &logical, r.solve(&b.as_read(), &mut e)?, &mut e)?;
        // The typed view returns typed results with the same bits.
        let typed = view.svd(&mut e)?;
        let erased = Svd {
            u: Tensor::from_typed::<T>(typed.u),
            s: Tensor::from_typed::<T::Real>(typed.s),
            vh: Tensor::from_typed::<T>(typed.vh),
        };
        check(&format!("typed {label}"), &logical, erased, &mut e)?;
    }

    // typed tensor
    let typed_svd = typed_base.svd(&mut e)?;
    let erased = Svd {
        u: Tensor::from_typed::<T>(typed_svd.u),
        s: Tensor::from_typed::<T::Real>(typed_svd.s),
        vh: Tensor::from_typed::<T>(typed_svd.vh),
    };
    check("typed", &owned, erased, &mut e)?;
    let typed_b = typed::<T>(rhs3::<T>())?;
    let x = typed_base.solve(&typed_b, &mut e)?;
    check_solve("typed", &owned, Tensor::from_typed::<T>(x), &mut e)?;
    Ok(())
}

#[test]
fn every_input_kind_returns_the_owned_result() -> TestResult<()> {
    input_kinds_agree::<f64>()?;
    input_kinds_agree::<Complex64>()
}

// ------------------------------------------------------------------ 3. _into

fn into_ops() -> Vec<&'static str> {
    API_OPS
        .iter()
        .filter(|op| !op.into.is_empty())
        .map(|op| op.name)
        .collect()
}

#[test]
fn only_solve_declares_into_and_only_overwrite() {
    assert_eq!(into_ops(), vec!["solve"]);
    let solve = API_OPS
        .iter()
        .find(|op| op.name == "solve")
        .map(|op| op.into);
    assert_eq!(solve, Some(&["overwrite"][..]));
}

#[test]
fn solve_into_overwrite_never_reads_the_destination() -> TestResult<()> {
    let mut e = CpuEngine::serial();
    let a = typed::<f64>(sq::<f64>())?;
    let b = typed::<f64>(rhs::<f64>())?;
    let want = a.solve(&b, &mut e)?;
    for poison in [f64::NAN, -0.0, f64::INFINITY] {
        let mut out = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], vec![poison; 2])?;
        a.solve_into(&b, &mut e, Dest::overwrite(&mut out))?;
        let got: Vec<u64> = out.host_data()?.iter().map(|x| x.to_bits()).collect();
        let exp: Vec<u64> = want.host_data()?.iter().map(|x| x.to_bits()).collect();
        assert_eq!(got, exp, "poison {poison} leaked into the overwrite result");
    }
    // The erased spelling writes the same bits.
    let (ea, eb) = (
        Tensor::from_typed(a.duplicate()?),
        Tensor::from_typed(b.duplicate()?),
    );
    let mut out = Tensor::from_typed(TypedTensor::<f64>::from_vec_col_major(
        vec![2, 1],
        vec![f64::NAN; 2],
    )?);
    ea.solve_into(&eb, &mut e, Dest::overwrite(&mut out))?;
    assert_eq!(out.bits()?, Tensor::from_typed(want).bits()?);
    Ok(())
}

// ----------------------------------------- 4. undeclared modes and omissions

#[test]
fn undeclared_into_modes_are_unsupported_before_any_write() -> TestResult<()> {
    let mut e = CpuEngine::serial();
    let a = typed::<f64>(sq::<f64>())?;
    let b = typed::<f64>(rhs::<f64>())?;
    let sentinel = vec![7.5_f64, -3.25];
    type MakeDest = fn(&mut TypedTensor<f64>) -> Dest<tenferro_tensor::TypedTensorWrite<'_, f64>>;
    let modes: [(&str, MakeDest); 2] = [
        ("accumulate", |out| Dest::accumulate(out)),
        ("scaled", |out| Dest::scaled(out, 2.0, 0.0)),
    ];
    for (label, mk) in modes {
        let mut out = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], sentinel.clone())?;
        let err = a.solve_into(&b, &mut e, mk(&mut out)).expect_err(label);
        assert_eq!(err.kind(), ErrorKind::Unsupported, "{label}: {err}");
        assert_eq!(
            out.host_data()?,
            &sentinel[..],
            "{label} touched the destination"
        );
    }
    Ok(())
}

#[test]
fn declared_omissions_are_unsupported() -> TestResult<()> {
    let mut e = CpuEngine::serial();
    let a = tensor::<f64>(tall::<f64>())?;
    // `SvdShape::Full` with a non-default gauge.
    let opts = SvdOptions::default()
        .shape(SvdShape::Full)
        .gauge(SvdGauge::CanonicalPivot);
    let err = a.svd_with(&opts, &mut e).expect_err("full + gauge");
    assert_eq!(err.kind(), ErrorKind::Unsupported, "{err}");
    // `CholeskyOptions::uplo(Upper)`.
    let spd = tensor::<f64>(sq::<f64>())?;
    let err = spd
        .cholesky_with(&CholeskyOptions::default().uplo(Uplo::Upper), &mut e)
        .expect_err("upper cholesky");
    assert_eq!(err.kind(), ErrorKind::Unsupported, "{err}");
    // `MatOp::Adjoint` on a complex matrix.
    let c = tensor::<Complex64>(sq::<Complex64>())?;
    let cb = tensor::<Complex64>(rhs::<Complex64>())?;
    let err = c
        .triangular_solve_with(
            &cb,
            &TriangularSolveOptions::default().op(MatOp::Adjoint),
            &mut e,
        )
        .expect_err("complex adjoint");
    assert_eq!(err.kind(), ErrorKind::Unsupported, "{err}");
    Ok(())
}

// --------------------------------------------------------- generated docs

#[test]
fn api_docs_are_current() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/spec/generated/api-ops.md"
    );
    let current = std::fs::read_to_string(path).expect("read docs/spec/generated/api-ops.md");
    let expected = api_docs_sync(&current);
    if std::env::var_os("UPDATE_API_DOCS").is_some() {
        std::fs::write(path, &expected).expect("write docs/spec/generated/api-ops.md");
        return;
    }
    assert!(
        current == expected,
        "docs/spec/generated/api-ops.md is stale; run \
         `UPDATE_API_DOCS=1 cargo test -p tenferro-linalg --features autodiff api_docs_are_current`"
    );
}

// ------------------------------------------- eager solve and `no_grad` (§5.7)

/// `sess.solve` honours the calling thread's `no_grad` guard, as the removed
/// `EagerTensor::solve` did (compared bit for bit before its removal).
#[cfg(feature = "autodiff")]
#[test]
fn eager_session_solve_honours_no_grad() -> TestResult<()> {
    use crate::EagerLinalgOps;
    use tenferro_ad::{AdContext, EagerRuntime};

    let ad = AdContext::builder()
        .with_semantic_extension_rules(crate::semantic_ad_rules()?)?
        .build()?;
    let ctx = EagerRuntime::with_ad_context(&ad)?;
    let a = ctx.variable_from(raw::<f64>(sq::<f64>())?)?;
    let b = ctx.variable_from(raw::<f64>(rhs::<f64>())?)?;
    for guarded in [false, true] {
        let x = ctx.with_eager_session(&mut CpuEngine::serial(), |s| {
            let _g = guarded.then(|| ctx.no_grad());
            s.solve(&a, &b)
        })?;
        assert_eq!(x.tracks_grad(), !guarded, "no_grad = {guarded}");
    }
    Ok(())
}

// ------------------------------------------------- typed LU parity dtypes (Q5)

/// `lu` returns `parity` in the input dtype on every route, so the typed
/// complex call equals the erased one bit for bit; typed complex
/// `full_piv_lu` is the declared omission of Q21.
#[test]
fn typed_complex_lu_matches_erased_and_full_piv_lu_is_a_declared_omission() -> TestResult<()> {
    let mut e = CpuEngine::serial();
    let typed_a = typed::<Complex64>(sq::<Complex64>())?;
    let erased = Tensor::from_typed::<Complex64>(typed_a.duplicate()?);

    let t = typed_a.lu(&mut e)?;
    let w = erased.lu(&mut e)?;
    let tb = [
        tensor_bits(&Tensor::from_typed(t.p))?,
        tensor_bits(&Tensor::from_typed(t.l))?,
        tensor_bits(&Tensor::from_typed(t.u))?,
        tensor_bits(&Tensor::from_typed(t.parity))?,
    ];
    let wb = [
        tensor_bits(&w.p)?,
        tensor_bits(&w.l)?,
        tensor_bits(&w.u)?,
        tensor_bits(&w.parity)?,
    ];
    assert_eq!(tb, wb, "lu");

    // full_piv_lu: the owned erased route returns a real parity; the borrowed
    // route (typed inputs) returns it complex, so the typed call is a declared
    // omission (Q21) and leaves nothing half-written.
    let w = erased.full_piv_lu(&mut e)?;
    assert_eq!(w.parity.dtype(), tenferro_tensor::DType::F64);
    let read = typed_a.as_view();
    let r = TensorRead::from_view(Complex64::tensor_view(read)).full_piv_lu(&mut e)?;
    assert_eq!(r.parity.dtype(), tenferro_tensor::DType::C64);
    let err = typed_a
        .full_piv_lu(&mut e)
        .expect_err("typed complex full_piv_lu");
    assert_eq!(err.kind(), ErrorKind::Unsupported, "{err}");
    let real = typed::<f64>(sq::<f64>())?.full_piv_lu(&mut e)?;
    assert_eq!(real.parity.host_data()?.len(), 1);
    Ok(())
}

// ------------------------------------------------ options only where they mean something

/// #2552 Q14: an operation has an options type and an `_with` form only when
/// it has semantic options. The set of options-bearing operations is fixed, every
/// options type has at least one field, and the generated method names of each
/// operation are exactly `op`, `op_with` (iff options) and `op_into` (iff `into`).
#[test]
fn options_exist_exactly_for_the_operations_that_declare_them() {
    use crate::api::registry::api_ops_markdown;

    let mut with_options: Vec<&str> = API_OPS
        .iter()
        .filter(|op| op.options != "none")
        .map(|op| op.name)
        .collect();
    with_options.sort_unstable();
    assert_eq!(
        with_options,
        [
            "cholesky",
            "eigh",
            "eigvalsh",
            "norm",
            "pinv",
            "qr",
            "rank_revealing_qr",
            "svd",
            "svdvals",
            "triangular_solve"
        ]
    );

    // Every declared options type is non-empty: a derived `Debug` prints
    // `Name { field: .. }` for a struct with fields and only `Name` for an
    // empty one.
    let defaults: &[(&str, String)] = &[
        (
            "CholeskyOptions",
            format!("{:?}", crate::CholeskyOptions::default()),
        ),
        (
            "EighOptions",
            format!("{:?}", crate::EighOptions::default()),
        ),
        (
            "EigvalshOptions",
            format!("{:?}", crate::EigvalshOptions::default()),
        ),
        (
            "NormOptions",
            format!("{:?}", crate::NormOptions::default()),
        ),
        (
            "PinvOptions",
            format!("{:?}", crate::PinvOptions::default()),
        ),
        ("QrOptions", format!("{:?}", crate::QrOptions::default())),
        (
            "RankRevealingQrOptions",
            format!("{:?}", crate::RankRevealingQrOptions::default()),
        ),
        ("SvdOptions", format!("{:?}", crate::SvdOptions::default())),
        (
            "SvdvalsOptions",
            format!("{:?}", crate::SvdvalsOptions::default()),
        ),
        (
            "TriangularSolveOptions",
            format!("{:?}", crate::TriangularSolveOptions::default()),
        ),
    ];
    for op in API_OPS.iter().filter(|op| op.options != "none") {
        let name = op.options.rsplit("::").next().unwrap_or(op.options);
        let (_, shown) = defaults
            .iter()
            .find(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("no default for options type {name} of {}", op.name));
        assert!(
            shown.contains(" { "),
            "{name} of {} has no field: {shown}",
            op.name
        );
    }

    let md = api_ops_markdown();
    for op in API_OPS {
        let mut want = vec![format!("`{}`", op.name)];
        if op.options != "none" {
            want.push(format!("`{}_with`", op.name));
        }
        if !op.into.is_empty() {
            want.push(format!("`{}_into` ({})", op.name, op.into.join(", ")));
        }
        let row = md
            .lines()
            .find(|line| line.starts_with(&format!("| `{}` |", op.name)))
            .unwrap_or_else(|| panic!("no table row for {}", op.name));
        let cell = row.trim_end_matches(" |").rsplit("| ").next().unwrap_or("");
        assert_eq!(cell, want.join(", "), "{}", op.name);
    }
}
