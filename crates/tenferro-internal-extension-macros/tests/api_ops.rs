//! Integration tests for `api_ops!` using a self-contained dummy family.

mod dummy {
    pub trait Engine {
        fn id(&self) -> u32;
    }
    pub struct Cpu(pub u32);
    impl Engine for Cpu {
        fn id(&self) -> u32 {
            self.0
        }
    }
    pub type Result<T> = ::std::result::Result<T, String>;
    pub trait Marker {}
    pub struct Plain;
    impl Marker for Plain {}
    pub trait AsView {
        type Repr;
        fn values(&self) -> Vec<f64>;
    }
    #[derive(Debug, PartialEq)]
    pub struct Pair<A, B>(pub A, pub B);

    pub struct Vecf(pub Vec<f64>);
    impl AsView for Vecf {
        type Repr = Plain;
        fn values(&self) -> Vec<f64> {
            self.0.clone()
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct NormOptions {
        pub scale: f64,
    }
    impl Default for NormOptions {
        fn default() -> Self {
            Self { scale: 1.0 }
        }
    }
    #[derive(Debug, Default)]
    pub struct AddOptions {
        pub offset: f64,
    }
    #[derive(Debug, Default)]
    pub struct RollOptions;
    #[derive(Debug, Default)]
    pub struct SplitOptions;

    /// Session type: records the first operand it was handed.
    pub struct Session {
        pub calls: Vec<String>,
    }

    pub mod forward_concrete {
        use super::*;
        pub fn norm<X: AsView + ?Sized>(a: &X, o: &NormOptions, e: &mut dyn Engine) -> Result<f64> {
            Ok(o.scale * a.values().iter().map(|v| v.abs()).sum::<f64>() + f64::from(e.id()))
        }
        pub fn add<X: AsView + ?Sized>(
            a: &X,
            b: &impl AsView<Repr = X::Repr>,
            o: &AddOptions,
            _e: &mut dyn Engine,
        ) -> Result<Vec<f64>> {
            Ok(a.values()
                .iter()
                .zip(b.values())
                .map(|(x, y)| x + y + o.offset)
                .collect())
        }
        pub fn add_into<X: AsView + ?Sized>(
            a: &X,
            b: &impl AsView<Repr = X::Repr>,
            o: &AddOptions,
            e: &mut dyn Engine,
            dest: &mut Vec<f64>,
        ) -> Result<()> {
            *dest = add(a, b, o, e)?;
            Ok(())
        }
        pub fn roll<X: AsView + ?Sized>(
            a: &X,
            axes: Vec<isize>,
            _o: &RollOptions,
            _e: &mut dyn Engine,
        ) -> Result<Vec<f64>> {
            let mut v = a.values();
            let k = axes.first().copied().unwrap_or(0);
            let len = v.len() as isize;
            if len > 0 {
                v.rotate_left(k.rem_euclid(len) as usize);
            }
            Ok(v)
        }
        pub fn split<X: AsView + ?Sized>(
            a: &X,
            _o: &SplitOptions,
            _e: &mut dyn Engine,
        ) -> Result<Pair<Vec<f64>, f64>> {
            let v = a.values();
            let s = v.iter().sum();
            Ok(Pair(v, s))
        }
    }

    pub mod forward_session {
        use super::*;
        pub fn norm(s: &mut Session, a: &Vecf, o: &NormOptions) -> Result<f64> {
            s.calls.push("norm".into());
            Ok(o.scale * a.0.iter().sum::<f64>())
        }
        pub fn add(s: &mut Session, a: &Vecf, b: &Vecf, o: &AddOptions) -> Result<Vec<f64>> {
            s.calls.push(format!("add({:?})", a.0));
            Ok(a.0
                .iter()
                .zip(&b.0)
                .map(|(x, y)| x + y + o.offset)
                .collect())
        }
        pub fn roll(
            s: &mut Session,
            a: &Vecf,
            axes: Vec<isize>,
            _o: &RollOptions,
        ) -> Result<Vec<f64>> {
            s.calls.push(format!("roll{axes:?}"));
            Ok(a.0.clone())
        }
        pub fn split(s: &mut Session, a: &Vecf, _o: &SplitOptions) -> Result<Pair<Vec<f64>, f64>> {
            s.calls.push("split".into());
            Ok(Pair(a.0.clone(), a.0.iter().sum()))
        }
    }

    tenferro_extension_macros::api_ops! {
        family: dummy;

        layer concrete {
            trait_doc: "Dummy ops on anything that implements `AsView`.";
            trait_def: pub trait DummyOps: AsView;
            impl_def: impl<X: AsView + ?Sized> DummyOps for X;
            method_where: [Self::Repr: Marker];
            receiver: self;
            tensor_arg: &impl AsView<Repr = Self::Repr>;
            exec: &mut dyn Engine;
            dest: &mut Vec<f64>;
            result: Result;
            types { V = Vec<f64>, R = f64 }
            forward: self::forward_concrete;
        }
        layer session {
            trait_doc: "Dummy ops on a session.";
            trait_def: pub trait DummySession;
            impl_def: impl DummySession for Session;
            method_where: [];
            receiver: session;
            first_arg: &Vecf;
            tensor_arg: &Vecf;
            result: Result;
            types { V = Vec<f64>, R = f64 }
            forward: self::forward_session;
        }
        layer core_eager { describe_only; }

        op norm {
            summary: "Scaled one-norm.";
            operands: (a);
            options: NormOptions;
            output: R;
            dtypes: "real";
            layers: [concrete, session];
            into: [];
            layout: "Any strided layout.";
            errors: [Validation, NumericalFailure(NonConvergence)];
            omissions: [];
            example concrete: r#"
let _ = 1;
"#;
            example session: "let _ = 2;";
        }
        op add {
            summary: "Elementwise sum.";
            operands: (a, b);
            options: AddOptions;
            output: V;
            dtypes: "real";
            layers: [concrete, session];
            into: [overwrite];
            layout: "Compact.";
            errors: [Validation];
            omissions: ["layer session: no destination support"];
            example concrete: "let _ = 3;";
            example session: "let _ = 4;";
        }
        op roll {
            summary: "Rotate left.";
            operands: (a, axes: Vec<isize>);
            options: RollOptions;
            output: V;
            dtypes: "any";
            layers: [concrete, session];
            into: [];
            layout: "Compact.";
            errors: [Unsupported];
            omissions: [];
            example concrete: "let _ = 5;";
            example session: "let _ = 6;";
        }
        op split {
            summary: "Values and their sum.";
            operands: (a);
            options: SplitOptions;
            output: Pair<V, R>;
            dtypes: "real";
            layers: [concrete, session, core_eager];
            into: [];
            layout: "Compact.";
            errors: [Validation];
            omissions: [];
            example concrete: "let _ = 7;";
            example session: "let _ = 8;";
        }

        exceptions {
            E08 {
                symbols: ["dummy::svd_with_exec", "dummy::qr_with_exec"];
                scope: "free functions";
                category: "legacy | wrapper";
                owner_module: "dummy::with_exec";
                reason: "kept for migration";
                verification: "symbol grep";
                exit_condition: "v1.0";
                status: retired;
            }
        }
    }
}

use dummy::*;

fn v(x: &[f64]) -> Vecf {
    Vecf(x.to_vec())
}

#[test]
fn plain_and_with_agree_and_options_pass_through() {
    let a = v(&[1.0, -2.0, 3.0]);
    let mut e = Cpu(0);
    let plain = a.norm(&mut e).unwrap();
    let with_default = a.norm_with(&NormOptions::default(), &mut e).unwrap();
    assert_eq!(plain, with_default);
    assert_eq!(plain, 6.0);
    let scaled = a.norm_with(&NormOptions { scale: 3.0 }, &mut e).unwrap();
    assert_eq!(scaled, 18.0);
    // exec is forwarded too.
    assert_eq!(a.norm(&mut Cpu(5)).unwrap(), 11.0);

    let b = v(&[10.0, 20.0, 30.0]);
    assert_eq!(a.add(&b, &mut e).unwrap(), vec![11.0, 18.0, 33.0]);
    assert_eq!(
        a.add_with(&b, &AddOptions { offset: 1.0 }, &mut e).unwrap(),
        vec![12.0, 19.0, 34.0]
    );
    assert_eq!(a.split(&mut e).unwrap(), Pair(vec![1.0, -2.0, 3.0], 2.0));
}

#[test]
fn typed_operand_is_passed() {
    let a = v(&[1.0, 2.0, 3.0]);
    let mut e = Cpu(0);
    assert_eq!(a.roll(vec![1], &mut e).unwrap(), vec![2.0, 3.0, 1.0]);
    assert_eq!(
        a.roll_with(vec![2], &RollOptions, &mut e).unwrap(),
        vec![3.0, 1.0, 2.0]
    );
}

#[test]
fn into_forwards_and_returns_body_result() {
    let a = v(&[1.0, 2.0]);
    let b = v(&[3.0, 4.0]);
    let mut e = Cpu(0);
    let mut dest = vec![f64::NAN; 2];
    let r: Result<()> = a.add_into(&b, &AddOptions { offset: 0.5 }, &mut e, &mut dest);
    assert_eq!(r, Ok(()));
    assert_eq!(dest, vec![4.5, 6.5]);
}

#[test]
fn session_layer_passes_first_operand() {
    let mut s = Session { calls: vec![] };
    let a = v(&[1.0, 2.0]);
    let b = v(&[3.0, 4.0]);
    assert_eq!(s.add(&a, &b).unwrap(), vec![4.0, 6.0]);
    assert_eq!(
        s.add_with(&a, &b, &AddOptions { offset: 1.0 }).unwrap(),
        vec![5.0, 7.0]
    );
    assert_eq!(s.roll(&a, vec![3]).unwrap(), vec![1.0, 2.0]);
    assert_eq!(s.norm(&a).unwrap(), 3.0);
    assert_eq!(s.split(&b).unwrap(), Pair(vec![3.0, 4.0], 7.0));
    assert_eq!(
        s.calls,
        vec![
            "add([1.0, 2.0])",
            "add([1.0, 2.0])",
            "roll[3]",
            "norm",
            "split"
        ]
    );
}

#[test]
fn metadata_holds_declared_data() {
    assert_eq!(API_FAMILY, "dummy");
    assert_eq!(
        API_LAYERS,
        &[
            ("concrete", "DummyOps"),
            ("session", "DummySession"),
            ("core_eager", "")
        ]
    );
    let names: Vec<_> = API_OPS.iter().map(|o| o.name).collect();
    assert_eq!(names, ["norm", "add", "roll", "split"]);
    let add = &API_OPS[1];
    assert_eq!(add.family, "dummy");
    assert_eq!(add.operands, ["a", "b"]);
    assert_eq!(add.into, ["overwrite"]);
    assert_eq!(add.layers, ["concrete", "session"]);
    assert_eq!(add.omissions, ["layer session: no destination support"]);
    assert_eq!(add.options, "AddOptions");
    assert_eq!(API_OPS[2].operands, ["a", "axes: Vec<isize>"]);
    assert_eq!(API_OPS[3].output, "Pair<V, R>");
    assert_eq!(API_OPS[3].layers, ["concrete", "session", "core_eager"]);
    assert_eq!(
        API_OPS[0].errors,
        ["Validation", "NumericalFailure(NonConvergence)"]
    );
    assert_eq!(API_EXCEPTIONS.len(), 1);
    let e = &API_EXCEPTIONS[0];
    assert_eq!(e.id, "E08");
    assert_eq!(e.status, "retired");
    assert_eq!(e.symbols, ["dummy::svd_with_exec", "dummy::qr_with_exec"]);
    assert_eq!(e.owner_module, "dummy::with_exec");
}

#[test]
fn markdown_has_table_rows_and_exceptions() {
    let md = api_ops_markdown();
    assert!(md.starts_with("## dummy\n"));
    assert!(md.contains("| Op | concrete | session | core_eager | Suffixes |"));
    assert!(md.contains("| `add` | DummyOps | DummySession | — |"));
    assert!(md.contains("`add_into` (overwrite)"));
    assert!(md.contains("| `split` | DummyOps | DummySession | described |"));
    assert!(md.contains("| `norm` |"));
    assert!(md.contains("| `roll` |"));
    assert!(md.contains("### add"));
    assert!(md.contains("### Exceptions"));
    assert!(md.contains(
        "| ID | Status | Symbols | Scope | Category | Owner | Reason | Verification | Exit condition |"
    ));
    assert!(md.contains("| E08 | retired |"));
    assert!(md.contains("legacy \\| wrapper"));
}

#[test]
fn docs_sync_replaces_between_markers_only_and_is_idempotent() {
    let begin = "<!-- api-ops:dummy:begin -->";
    let end = "<!-- api-ops:dummy:end -->";
    let input = format!("# Title\n\n{begin}\nstale\n{end}\n\ntail\n");
    let once = api_docs_sync(&input);
    assert!(once.starts_with(&format!("# Title\n\n{begin}\n## dummy\n")));
    assert!(once.ends_with(&format!("{end}\n\ntail\n")));
    assert!(!once.contains("stale"));
    assert_eq!(api_docs_sync(&once), once);

    let appended = api_docs_sync("# Title\n");
    assert!(appended.starts_with("# Title\n\n<!-- api-ops:dummy:begin -->\n## dummy"));
    assert!(appended.ends_with("<!-- api-ops:dummy:end -->\n"));
    assert_eq!(api_docs_sync(&appended), appended);
}
