//! Implementation of the `api_ops!` operation-registry macro.
//!
//! The registry is parsed by [`parse`], validated here, and expanded by
//! [`expand`] (traits, impls, metadata) and [`markdown`] (generated spec).

mod expand;
mod markdown;
mod parse;

use proc_macro2::TokenStream;
use syn::{Ident, LitStr, Path, Type, WherePredicate};

pub(crate) use expand::expand;
pub(crate) use parse::parse_registry;

/// How the first operand reaches a generated method.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Receiver {
    /// `&self` receiver; the first operand is the receiver.
    SelfRef,
    /// `&mut self` receiver; the first operand is an ordinary argument.
    Session,
    /// `&self` receiver; the first operand is an ordinary argument (a context
    /// object such as a trace that records the operation).
    Context,
}

/// A layer that generates a trait and an impl.
pub(crate) struct GenLayer {
    pub trait_doc: LitStr,
    pub trait_def: TokenStream,
    pub trait_name: Ident,
    pub impl_def: TokenStream,
    pub method_where: Vec<WherePredicate>,
    pub receiver: Receiver,
    pub first_arg: Option<Type>,
    pub tensor_arg: Type,
    pub exec: Option<Type>,
    pub dest: Option<Type>,
    pub result: Path,
    pub types: Vec<(Ident, Type)>,
    pub forward: Path,
    /// Optional `cfg` predicate gating the generated trait and impl.
    pub cfg: Option<TokenStream>,
    /// Hand-written trait items appended to the generated trait (an
    /// exception-table entry, e.g. a consuming in-place method).
    pub extra_decls: TokenStream,
    /// Hand-written impl items matching `extra_decls`.
    pub extra_impls: TokenStream,
}

pub(crate) enum LayerKind {
    DescribeOnly,
    Generate(Box<GenLayer>),
}

pub(crate) struct Layer {
    pub name: Ident,
    pub kind: LayerKind,
}

pub(crate) struct Operand {
    pub name: Ident,
    pub ty: Option<Type>,
}

pub(crate) struct ErrorEntry {
    pub path: Path,
    pub detail: Option<TokenStream>,
}

pub(crate) struct Op {
    pub name: Ident,
    pub summary: LitStr,
    pub operands: Vec<Operand>,
    /// `None` for `options: none;`: the op has no options type and no `_with` form.
    pub options: Option<Path>,
    /// Method generics (`generics: <Y: Bound>;`), added to every generated method.
    pub generics: Option<syn::Generics>,
    /// `associated;`: no receiver; every operand is an argument and the method
    /// requires `Self: Sized` (for example `concatenate(&[&Self], axis, exec)`).
    pub associated: bool,
    pub output: Type,
    pub dtypes: LitStr,
    pub layers: Vec<Ident>,
    pub into: Vec<Ident>,
    pub layout: LitStr,
    pub errors: Vec<ErrorEntry>,
    pub omissions: Vec<LitStr>,
    pub examples: Vec<(Ident, LitStr)>,
}

pub(crate) struct Exception {
    pub id: Ident,
    pub symbols: Vec<LitStr>,
    pub scope: LitStr,
    pub category: LitStr,
    pub owner_module: LitStr,
    pub reason: LitStr,
    pub verification: LitStr,
    pub exit_condition: LitStr,
    pub status: Ident,
}

pub(crate) struct Registry {
    pub family: Ident,
    pub layers: Vec<Layer>,
    pub ops: Vec<Op>,
    pub exceptions: Vec<Exception>,
}

/// Token-stream text with whitespace collapsed and punctuation tidied.
pub(crate) fn tidy(ts: &TokenStream) -> String {
    let mut s = ts
        .to_string()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    for (from, to) in [
        (" :: ", "::"),
        (":: ", "::"),
        (" ,", ","),
        (" :", ":"),
        (" <", "<"),
        ("< ", "<"),
        (" >", ">"),
        ("& ", "&"),
        (" (", "("),
        ("( ", "("),
        (" )", ")"),
    ] {
        s = s.replace(from, to);
    }
    s
}

/// Rendered text of an error entry such as `NumericalFailure(NonConvergence)`.
pub(crate) fn error_text(entry: &ErrorEntry) -> String {
    let path = {
        let p = &entry.path;
        tidy(&quote::quote!(#p))
    };
    match &entry.detail {
        Some(d) => format!("{path}({})", tidy(d)),
        None => path,
    }
}

impl Registry {
    pub(crate) fn layer(&self, name: &str) -> Option<&Layer> {
        self.layers.iter().find(|l| l.name == name)
    }

    /// Checks cross-references and uniqueness rules.
    pub(crate) fn validate(&self) -> syn::Result<()> {
        for (i, layer) in self.layers.iter().enumerate() {
            if self.layers[..i].iter().any(|l| l.name == layer.name) {
                return Err(syn::Error::new_spanned(
                    &layer.name,
                    format!("duplicate layer `{}`", layer.name),
                ));
            }
        }
        for (i, op) in self.ops.iter().enumerate() {
            if self.ops[..i].iter().any(|o| o.name == op.name) {
                return Err(syn::Error::new_spanned(
                    &op.name,
                    format!("duplicate op `{}`", op.name),
                ));
            }
            for l in &op.layers {
                if self.layer(&l.to_string()).is_none() {
                    return Err(syn::Error::new_spanned(
                        l,
                        format!(
                            "op `{}` lists layer `{l}`, which is not declared by a `layer {l} {{ .. }}` block",
                            op.name
                        ),
                    ));
                }
            }
            for (l, _) in &op.examples {
                if self.layer(&l.to_string()).is_none() {
                    return Err(syn::Error::new_spanned(
                        l,
                        format!("op `{}` has an example for undeclared layer `{l}`", op.name),
                    ));
                }
            }
            for mode in &op.into {
                if !matches!(
                    mode.to_string().as_str(),
                    "overwrite" | "accumulate" | "scaled"
                ) {
                    return Err(syn::Error::new_spanned(
                        mode,
                        format!(
                            "op `{}`: unknown `into` mode `{mode}`; expected overwrite, accumulate or scaled",
                            op.name
                        ),
                    ));
                }
            }
            if op.operands.is_empty() {
                return Err(syn::Error::new_spanned(
                    &op.name,
                    format!("op `{}` needs at least one operand", op.name),
                ));
            }
        }
        for (i, ex) in self.exceptions.iter().enumerate() {
            if self.exceptions[..i].iter().any(|e| e.id == ex.id) {
                return Err(syn::Error::new_spanned(
                    &ex.id,
                    format!("duplicate exception id `{}`", ex.id),
                ));
            }
            if !matches!(ex.status.to_string().as_str(), "retained" | "retired") {
                return Err(syn::Error::new_spanned(
                    &ex.status,
                    format!(
                        "exception `{}`: `status` must be `retained` or `retired`",
                        ex.id
                    ),
                ));
            }
            if ex.symbols.is_empty() {
                return Err(syn::Error::new_spanned(
                    &ex.id,
                    format!("exception `{}`: `symbols` must not be empty", ex.id),
                ));
            }
        }
        Ok(())
    }
}
