//! The guards of the generated core grammar (#2552 §4.4), signed axes
//! (§3.2), the E05 no-copy contract and the generated documentation (§4.5).

use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_tensor::{
    AsView, Conj, DType, Dest, DotGeneralConfig, DotGeneralOptions, ErrorKind, StridedSliceSpec,
    Tensor, TensorRead, TypedTensor,
};

use crate::api::registry::{api_docs_sync, API_EXCEPTIONS, API_OPS};
use crate::{AddOptions, CoreOps};

type TR<T> = Result<T, Box<dyn std::error::Error>>;

fn bits(t: &Tensor) -> TR<Vec<u64>> {
    let mut out: Vec<u64> = t.shape().iter().map(|&d| d as u64).collect();
    out.push(u64::MAX);
    match t.dtype() {
        DType::F64 => out.extend(t.as_slice::<f64>()?.iter().map(|x| x.to_bits())),
        DType::C64 => out.extend(
            t.as_slice::<Complex64>()?
                .iter()
                .flat_map(|x| [x.re.to_bits(), x.im.to_bits()]),
        ),
        other => return Err(format!("unexpected dtype {other:?}").into()),
    }
    Ok(out)
}

fn base() -> TR<TypedTensor<f64>> {
    Ok(TypedTensor::from_vec_col_major(
        vec![2, 3],
        vec![0.5, -1.25, 2.0, 0.75, 3.5, -0.25],
    )?)
}

// ------------------------------------------------------------ 1. forwarding

#[test]
fn ops_with_options_forward_to_op_with_default_options() -> TR<()> {
    let with_options: Vec<&str> = API_OPS
        .iter()
        .filter(|op| op.options != "none" && op.layers.contains(&"concrete"))
        .map(|op| op.name)
        .collect();
    assert_eq!(
        with_options,
        ["add", "dot_general"],
        "every op with options needs a case"
    );

    let mut e = CpuEngine::serial();
    let a = Tensor::from_typed(base()?);
    assert_eq!(
        bits(&a.add(&a, &mut e)?)?,
        bits(&a.add_with(&a, &AddOptions::default(), &mut e)?)?
    );
    let s = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    let cfg = DotGeneralConfig {
        lhs_contracting_dims: [1usize].into_iter().collect(),
        rhs_contracting_dims: [0usize].into_iter().collect(),
        lhs_batch_dims: Default::default(),
        rhs_batch_dims: Default::default(),
    };
    assert_eq!(
        bits(&s.dot_general(&a, cfg.clone(), &mut e)?)?,
        bits(&s.dot_general_with(&a, cfg, &DotGeneralOptions::default(), &mut e)?)?
    );
    Ok(())
}

#[test]
fn ops_without_options_have_no_with_form_in_the_registry() {
    for op in API_OPS.iter().filter(|op| op.options == "none") {
        assert!(
            op.into.is_empty(),
            "{} declares into without options",
            op.name
        );
    }
}

// ---------------------------------------------------------- 2. input shapes

/// The logical elements of `view` in a fresh compact tensor.
fn compact(view: &tenferro_tensor::TypedTensorView<'_, f64>) -> TR<Tensor> {
    let (m, n) = (view.shape()[0], view.shape()[1]);
    let mut data = Vec::with_capacity(m * n);
    for col in 0..n {
        for row in 0..m {
            data.push(*view.get(&[row, col]).ok_or("index")?);
        }
    }
    Ok(Tensor::from_typed(TypedTensor::from_vec_col_major(
        vec![m, n],
        data,
    )?))
}

#[test]
fn every_input_kind_returns_the_owned_result() -> TR<()> {
    let mut e = CpuEngine::serial();
    let typed = base()?;
    let owned = Tensor::from_typed(typed.duplicate()?);
    let check = |label: &str,
                 logical: &Tensor,
                 unary: Tensor,
                 binary: Tensor,
                 e: &mut CpuEngine|
     -> TR<()> {
        assert_eq!(
            bits(&unary)?,
            bits(&logical.exp(e)?)?,
            "exp of the {label} input"
        );
        assert_eq!(
            bits(&binary)?,
            bits(&logical.add(logical, e)?)?,
            "add of the {label} input"
        );
        Ok(())
    };
    check(
        "owned",
        &owned,
        owned.exp(&mut e)?,
        owned.add(&owned, &mut e)?,
        &mut e,
    )?;
    let r = owned.as_read();
    check(
        "compact view",
        &owned,
        r.exp(&mut e)?,
        r.add(&r, &mut e)?,
        &mut e,
    )?;
    for (label, view) in [
        ("transposed", typed.as_view().transpose_view([1, 0])?),
        (
            "negative-stride",
            typed
                .as_view()
                .slice_view(&[StridedSliceSpec::reverse(), StridedSliceSpec::reverse()])?,
        ),
    ] {
        let logical = compact(&view)?;
        let r = view.as_read();
        check(label, &logical, r.exp(&mut e)?, r.add(&r, &mut e)?, &mut e)?;
        let typed_unary = Tensor::from_typed(view.exp(&mut e)?);
        let typed_binary = Tensor::from_typed(view.add(&view, &mut e)?);
        check(
            &format!("typed {label}"),
            &logical,
            typed_unary,
            typed_binary,
            &mut e,
        )?;
    }
    let tu = Tensor::from_typed(typed.exp(&mut e)?);
    let tb = Tensor::from_typed(typed.add(&typed, &mut e)?);
    check("typed", &owned, tu, tb, &mut e)?;
    Ok(())
}

// ------------------------------------------------------------------ 3. _into

#[test]
fn only_add_declares_into_and_only_overwrite() {
    let into: Vec<(&str, &[&str])> = API_OPS
        .iter()
        .filter(|op| !op.into.is_empty())
        .map(|op| (op.name, op.into))
        .collect();
    assert_eq!(into, [("add", &["overwrite"][..])]);
}

#[test]
fn add_into_overwrite_never_reads_the_destination() -> TR<()> {
    let mut e = CpuEngine::serial();
    let a = base()?;
    let want = Tensor::from_typed(a.add(&a, &mut e)?);
    for poison in [f64::NAN, -0.0, f64::INFINITY] {
        let mut out = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![poison; 6])?;
        a.add_into(
            &a,
            &AddOptions::default(),
            &mut e,
            Dest::overwrite(&mut out),
        )?;
        assert_eq!(
            bits(&Tensor::from_typed(out))?,
            bits(&want)?,
            "poison {poison}"
        );
    }
    Ok(())
}

// ----------------------------------------- 4. undeclared modes are rejected

#[test]
fn undeclared_into_modes_are_unsupported_before_any_write() -> TR<()> {
    let mut e = CpuEngine::serial();
    let a = base()?;
    let sentinel = vec![7.5_f64, -3.25, 1.0, 2.0, 3.0, 4.0];
    type Mk = fn(&mut TypedTensor<f64>) -> Dest<tenferro_tensor::TypedTensorWrite<'_, f64>>;
    let modes: [(&str, Mk); 2] = [
        ("accumulate", |out| Dest::accumulate(out)),
        ("scaled", |out| Dest::scaled(out, 2.0, 0.0)),
    ];
    for (label, mk) in modes {
        let mut out = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], sentinel.clone())?;
        let err = a
            .add_into(&a, &AddOptions::default(), &mut e, mk(&mut out))
            .expect_err(label);
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
fn dot_general_with_conjugates_the_requested_operand() -> TR<()> {
    let mut e = CpuEngine::serial();
    let l = Tensor::from_vec_col_major(
        vec![1, 2],
        vec![Complex64::new(1.0, 2.0), Complex64::new(0.0, -1.0)],
    )?;
    let r = Tensor::from_vec_col_major(
        vec![2, 1],
        vec![Complex64::new(3.0, 0.5), Complex64::new(1.0, 1.0)],
    )?;
    let cfg = DotGeneralConfig {
        lhs_contracting_dims: [1usize].into_iter().collect(),
        rhs_contracting_dims: [0usize].into_iter().collect(),
        lhs_batch_dims: Default::default(),
        rhs_batch_dims: Default::default(),
    };
    let conj_l = l.conj(&mut e)?;
    let want = conj_l.dot_general(&r, cfg.clone(), &mut e)?;
    let got = l.dot_general_with(
        &r,
        cfg,
        &DotGeneralOptions::default().lhs(Conj::Yes),
        &mut e,
    )?;
    assert_eq!(bits(&got)?, bits(&want)?);
    Ok(())
}

// ---------------------------------------------------------- signed axes
// The traced layer is covered by `tests/integration/signed_axes.rs` (it needs
// the CPU engine registration, a dev-dependency that links its own runtime).

#[test]
fn negative_axes_equal_rank_minus_one_on_the_concrete_layer() -> TR<()> {
    let mut e = CpuEngine::serial();
    let a = Tensor::from_typed(base()?);
    assert_eq!(
        bits(&a.reduce_sum(Some(&[-1]), &mut e)?)?,
        bits(&a.reduce_sum(Some(&[1]), &mut e)?)?
    );
    assert_eq!(
        bits(&a.softmax(-1, &mut e)?)?,
        bits(&a.softmax(1, &mut e)?)?
    );
    assert_eq!(
        bits(&Tensor::concatenate(&[&a, &a], -1, &mut e)?)?,
        bits(&Tensor::concatenate(&[&a, &a], 1, &mut e)?)?
    );
    assert_eq!(
        bits(&a.transpose(&[-1, 0], &mut e)?)?,
        bits(&a.transpose(&[1, 0], &mut e)?)?
    );
    let err = a
        .reduce_sum(Some(&[-3]), &mut e)
        .expect_err("axis -3 of rank 2");
    assert!(matches!(err.kind(), ErrorKind::Validation(_)), "{err}");
    Ok(())
}

// ------------------------------------------------- E05: metadata, no copy

#[test]
fn metadata_queries_and_views_do_not_copy() -> TR<()> {
    let typed = base()?;
    let data = typed.host_data()?;
    let view = typed.as_view();
    // Shape and dtype are descriptor reads.
    assert_eq!(view.shape(), typed.shape());
    assert_eq!(typed.as_read().dtype(), DType::F64);
    // A transposed view addresses the same elements.
    let t = view.transpose_view([1, 0])?;
    assert!(std::ptr::eq(t.get(&[2, 1]).ok_or("index")?, &data[5]));
    // A reversed view addresses the same elements from the other end.
    let r = view.slice_view(&[StridedSliceSpec::reverse(), StridedSliceSpec::reverse()])?;
    assert!(std::ptr::eq(r.get(&[0, 0]).ok_or("index")?, &data[5]));
    // `as_read` of an owned tensor borrows it.
    let owned = Tensor::from_typed(typed.duplicate()?);
    let read: TensorRead<'_> = owned.as_read();
    assert!(read.as_tensor().is_some_and(|t| std::ptr::eq(t, &owned)));
    Ok(())
}

// --------------------------------------------------------- exception table

#[test]
fn exception_table_lists_concrete_symbols() {
    let ids: Vec<&str> = API_EXCEPTIONS.iter().map(|e| e.id).collect();
    assert_eq!(ids, ["E03", "E04", "E05", "E06", "E09", "E10"]);
    let allowlist = include_str!("../../../../../api/allowlist.txt");
    let e06 = API_EXCEPTIONS
        .iter()
        .find(|e| e.id == "E06")
        .map(|e| e.symbols);
    for symbol in e06.unwrap_or_default() {
        assert!(
            allowlist.lines().any(|line| line.trim() == *symbol),
            "E06 symbol {symbol} is missing from api/allowlist.txt"
        );
    }
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
         `UPDATE_API_DOCS=1 cargo test -p tenferro-runtime api_docs_are_current`"
    );
}
