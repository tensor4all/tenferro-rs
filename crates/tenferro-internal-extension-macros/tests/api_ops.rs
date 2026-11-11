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
        pub fn negate<X: AsView + ?Sized>(a: &X, _e: &mut dyn Engine) -> Result<Vec<f64>> {
            Ok(a.values().iter().map(|x| -x).collect())
        }
        pub fn negate_into<X: AsView + ?Sized>(
            a: &X,
            e: &mut dyn Engine,
            dest: &mut Vec<f64>,
        ) -> Result<()> {
            *dest = negate(a, e)?;
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
        pub fn negate(s: &mut Session, a: &Vecf) -> Result<Vec<f64>> {
            s.calls.push("negate".into());
            Ok(a.0.iter().map(|x| -x).collect())
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
        op negate {
            summary: "Elementwise negation; it has no settings, so no options type and no `_with`.";
            operands: (a);
            options: none;
            output: V;
            dtypes: "real";
            layers: [concrete, session];
            into: [overwrite];
            layout: "Compact.";
            errors: [Validation];
            omissions: [];
            example concrete: "let _ = 9;";
            example session: "let _ = 10;";
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
fn an_op_without_options_has_no_with_form_and_its_into_takes_no_options() {
    let a = v(&[1.0, -2.0]);
    let mut e = Cpu(0);
    assert_eq!(a.negate(&mut e).unwrap(), vec![-1.0, 2.0]);
    let mut dest = vec![];
    let r: Result<()> = a.negate_into(&mut e, &mut dest);
    assert_eq!(r, Ok(()));
    assert_eq!(dest, vec![-1.0, 2.0]);
    let mut s = Session { calls: vec![] };
    assert_eq!(s.negate(&a).unwrap(), vec![-1.0, 2.0]);

    let negate = API_OPS.iter().find(|o| o.name == "negate").unwrap();
    assert_eq!(negate.options, "none");
    let md = api_ops_markdown();
    assert!(md.contains("`negate`, `negate_into` (overwrite)"));
    assert!(!md.contains("negate_with"));
}

/// The generated method names of every op are exactly `op`, `op_with` (iff the
/// op declares an options type) and `op_into` (iff it declares `into` modes).
#[test]
fn generated_method_names_follow_options_and_into() {
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
            .find(|l| l.starts_with(&format!("| `{}` |", op.name)))
            .unwrap_or_else(|| panic!("no table row for {}", op.name));
        let cell = row.trim_end_matches(" |").rsplit("| ").next().unwrap();
        assert_eq!(cell, want.join(", "), "{}", op.name);
    }
    // Options-bearing ops of this family, fixed.
    let with_options: Vec<_> = API_OPS
        .iter()
        .filter(|o| o.options != "none")
        .map(|o| o.name)
        .collect();
    assert_eq!(with_options, ["norm", "add", "roll", "split"]);
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
    assert_eq!(names, ["norm", "add", "negate", "roll", "split"]);
    let add = &API_OPS[1];
    assert_eq!(add.family, "dummy");
    assert_eq!(add.operands, ["a", "b"]);
    assert_eq!(add.into, ["overwrite"]);
    assert_eq!(add.layers, ["concrete", "session"]);
    assert_eq!(add.omissions, ["layer session: no destination support"]);
    assert_eq!(add.options, "AddOptions");
    assert_eq!(API_OPS[3].operands, ["a", "axes: Vec<isize>"]);
    assert_eq!(API_OPS[4].output, "Pair<V, R>");
    assert_eq!(API_OPS[4].layers, ["concrete", "session", "core_eager"]);
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

/// A layer with `cfg: all();` is generated; a layer with `cfg: any();` is not.
mod gated {
    pub struct On;
    #[allow(dead_code)]
    pub struct Off;
    pub type Result<T> = ::std::result::Result<T, String>;
    #[derive(Default)]
    pub struct PingOptions;

    pub mod forward_on {
        use super::*;
        pub fn ping(_s: &mut On, _a: (), _o: &PingOptions) -> Result<u32> {
            Ok(7)
        }
    }

    // `not(any())` is always true. If the macro generated `OffOps` despite
    // `cfg: any();`, this definition would be a duplicate and fail to compile.
    #[cfg(not(any()))]
    #[allow(dead_code)]
    pub trait OffOps {}

    tenferro_extension_macros::api_ops! {
        family: gated;

        layer on {
            trait_doc: "Enabled layer.";
            trait_def: pub trait OnOps;
            impl_def: impl OnOps for On;
            method_where: [];
            receiver: session;
            first_arg: ();
            tensor_arg: ();
            result: Result;
            types { V = u32 }
            forward: self::forward_on;
            cfg: all();
            extra_decls {
                /// A hand-written item (exception-table entry).
                fn pong(self) -> u32;
            }
            extra_impls {
                fn pong(self) -> u32 {
                    9
                }
            }
        }
        layer off {
            trait_doc: "Disabled layer; its forward module does not exist.";
            trait_def: pub trait OffOps;
            impl_def: impl OffOps for Off;
            method_where: [];
            receiver: session;
            first_arg: ();
            tensor_arg: ();
            result: Result;
            types { V = u32 }
            forward: self::forward_missing;
            cfg: any();
        }
        op ping {
            summary: "Ping.";
            operands: (a);
            options: PingOptions;
            output: V;
            dtypes: "any";
            layers: [on, off];
            into: [];
            layout: "None.";
            errors: [Validation];
            omissions: [];
            example on: "let _ = 1;";
            example off: "let _ = 2;";
        }
    }
}

#[test]
fn cfg_key_gates_generated_trait_and_impl() {
    use gated::OnOps;
    let mut s = gated::On;
    assert_eq!(s.ping(()).unwrap(), 7);
    assert_eq!(s.ping_with((), &gated::PingOptions).unwrap(), 7);
    // `extra_decls` / `extra_impls` are appended to the generated trait.
    assert_eq!(gated::On.pong(), 9);
    // Metadata is unaffected by `cfg`.
    assert_eq!(gated::API_LAYERS, &[("on", "OnOps"), ("off", "OffOps")]);
    assert_eq!(gated::API_OPS[0].layers, ["on", "off"]);
}

/// `receiver: context;` gives `&self` with the first operand as an argument,
/// and an operand typed `S` takes each layer's `S` mapping (#2552 S4).
mod context_layer {
    pub struct Trace;
    pub struct Value(pub u32);
    pub type Result<T> = ::std::result::Result<T, String>;
    #[derive(Default)]
    pub struct SumOptions {
        pub offset: u32,
    }

    pub mod forward_ctx {
        use super::*;
        pub fn sum(_t: &Trace, inputs: &[&Value], o: &SumOptions) -> Result<u32> {
            Ok(inputs.iter().map(|v| v.0).sum::<u32>() + o.offset)
        }
    }
    pub mod forward_slice {
        pub fn sum(inputs: &[u32], o: &super::SumOptions) -> super::Result<u32> {
            Ok(inputs.iter().sum::<u32>() + o.offset)
        }
    }

    tenferro_extension_macros::api_ops! {
        family: context;

        layer ctx {
            trait_doc: "Context receiver.";
            trait_def: pub trait CtxOps;
            impl_def: impl CtxOps for Trace;
            method_where: [];
            receiver: context;
            tensor_arg: &Value;
            result: Result;
            types { V = u32, S = &[&Value] }
            forward: self::forward_ctx;
        }
        layer slice {
            trait_doc: "Slice receiver.";
            trait_def: pub trait SliceOps;
            impl_def: impl SliceOps for [u32];
            method_where: [];
            receiver: self;
            tensor_arg: &u32;
            result: Result;
            types { V = u32 }
            forward: self::forward_slice;
        }
        op sum {
            summary: "Sum.";
            operands: (inputs: S);
            options: SumOptions;
            output: V;
            dtypes: "any";
            layers: [ctx, slice];
            into: [];
            layout: "None.";
            errors: [Validation];
            omissions: [];
            example ctx: "let _ = 1;";
            example slice: "let _ = 2;";
        }
    }
}

#[test]
fn context_receiver_and_layer_operand_types() {
    use context_layer::{CtxOps, SliceOps, SumOptions, Trace, Value};
    let (a, b) = (Value(2), Value(3));
    assert_eq!(Trace.sum(&[&a, &b]).unwrap(), 5);
    assert_eq!(
        Trace
            .sum_with(&[&a, &b], &SumOptions { offset: 1 })
            .unwrap(),
        6
    );
    // The receiver layer skips the typed first operand.
    assert_eq!([4u32, 5][..].sum().unwrap(), 9);
    assert_eq!(
        context_layer::API_LAYERS,
        &[("ctx", "CtxOps"), ("slice", "SliceOps")]
    );
}

/// `options: none` generates only the plain form; `generics` adds method
/// generics; `associated` generates a receiver-less `Self: Sized` method.
mod shapes {
    pub struct Num(pub u32);
    pub type Result<T> = ::std::result::Result<T, String>;

    pub mod forward {
        use super::*;
        pub fn double(a: &Num, exec: &mut u32) -> Result<u32> {
            *exec += 1;
            Ok(a.0 * 2)
        }
        pub fn pick<Y: Copy>(a: &Num, on_true: &Y, on_false: &Y, _exec: &mut u32) -> Result<Y> {
            Ok(if a.0 != 0 { *on_true } else { *on_false })
        }
        pub fn total(items: &[&Num], _exec: &mut u32) -> Result<u32> {
            Ok(items.iter().map(|n| n.0).sum())
        }
    }

    tenferro_extension_macros::api_ops! {
        family: shapes;

        layer concrete {
            trait_doc: "Shapes.";
            trait_def: pub trait ShapeOps;
            impl_def: impl ShapeOps for Num;
            method_where: [];
            receiver: self;
            tensor_arg: &Num;
            exec: &mut u32;
            result: Result;
            types { V = u32 }
            forward: self::forward;
        }
        op double {
            summary: "Double.";
            operands: (a);
            options: none;
            output: V;
            dtypes: "u32";
            layers: [concrete];
            into: [];
            layout: "None.";
            errors: [Validation];
            omissions: [];
            example concrete: "let _ = 1;";
        }
        op pick {
            summary: "Pick.";
            generics: <Y: Copy>;
            operands: (a, on_true: &Y, on_false: &Y);
            options: none;
            output: Y;
            dtypes: "any";
            layers: [concrete];
            into: [];
            layout: "None.";
            errors: [Validation];
            omissions: [];
            example concrete: "let _ = 2;";
        }
        op total {
            summary: "Total.";
            associated;
            operands: (items: &[&Self]);
            options: none;
            output: V;
            dtypes: "u32";
            layers: [concrete];
            into: [];
            layout: "None.";
            errors: [Validation];
            omissions: [];
            example concrete: "let _ = 3;";
        }
    }
}

#[test]
fn options_none_generics_and_associated_forms() {
    use shapes::{Num, ShapeOps};
    let mut exec = 0u32;
    assert_eq!(Num(3).double(&mut exec).unwrap(), 6);
    assert_eq!(exec, 1);
    assert_eq!(Num(1).pick(&'a', &'b', &mut exec).unwrap(), 'a');
    assert_eq!(Num(0).pick(&1.5_f64, &2.5, &mut exec).unwrap(), 2.5);
    assert_eq!(Num::total(&[&Num(1), &Num(2)], &mut exec).unwrap(), 3);
    assert_eq!(shapes::API_OPS[0].options, "none");
    let md = shapes::api_ops_markdown();
    assert!(md.contains("| `double` | ShapeOps | `double` |"), "{md}");
    assert!(md.contains("- Options: none"));
    assert!(!md.contains("double_with"));
}
