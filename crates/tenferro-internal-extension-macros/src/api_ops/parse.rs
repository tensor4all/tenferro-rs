//! Parser for the `api_ops!` registry grammar.

use proc_macro2::{TokenStream, TokenTree};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{braced, bracketed, parenthesized, Ident, LitStr, Path, Token, Type, WherePredicate};

use super::{ErrorEntry, Exception, GenLayer, Layer, LayerKind, Op, Operand, Receiver, Registry};

/// Parses a whole `api_ops!` invocation.
pub(crate) fn parse_registry(input: ParseStream<'_>) -> syn::Result<Registry> {
    let mut family: Option<Ident> = None;
    let mut layers = Vec::new();
    let mut ops = Vec::new();
    let mut exceptions = Vec::new();
    let start = input.span();
    while !input.is_empty() {
        let key: Ident = input.parse()?;
        match key.to_string().as_str() {
            "family" => {
                input.parse::<Token![:]>()?;
                set(&mut family, &key, input.parse()?)?;
                end_key(input)?;
            }
            "layer" => layers.push(parse_layer(input)?),
            "op" => ops.push(parse_op(input)?),
            "exceptions" => {
                let content;
                braced!(content in input);
                while !content.is_empty() {
                    exceptions.push(parse_exception(&content)?);
                }
                let _ = input.parse::<Option<Token![;]>>()?;
            }
            other => {
                return Err(syn::Error::new_spanned(
                    &key,
                    format!(
                        "unknown item `{other}`; expected `family`, `layer`, `op` or `exceptions`"
                    ),
                ))
            }
        }
    }
    let family = family.ok_or_else(|| syn::Error::new(start, "missing required key `family`"))?;
    Ok(Registry {
        family,
        layers,
        ops,
        exceptions,
    })
}

fn set<T>(slot: &mut Option<T>, key: &Ident, value: T) -> syn::Result<()> {
    if slot.is_some() {
        return Err(syn::Error::new_spanned(
            key,
            format!("duplicate key `{key}`"),
        ));
    }
    *slot = Some(value);
    Ok(())
}

fn need<T>(slot: Option<T>, owner: &Ident, key: &str) -> syn::Result<T> {
    slot.ok_or_else(|| {
        syn::Error::new_spanned(owner, format!("`{owner}` is missing required key `{key}`"))
    })
}

fn end_key(input: ParseStream<'_>) -> syn::Result<()> {
    if input.is_empty() {
        return Ok(());
    }
    input.parse::<Token![;]>().map(|_| ())
}

fn list<T: Parse>(input: ParseStream<'_>) -> syn::Result<Vec<T>> {
    let content;
    bracketed!(content in input);
    Ok(Punctuated::<T, Token![,]>::parse_terminated(&content)?
        .into_iter()
        .collect())
}

fn tokens_until_semi(input: ParseStream<'_>, key: &Ident) -> syn::Result<TokenStream> {
    let mut ts = TokenStream::new();
    while !input.is_empty() && !input.peek(Token![;]) {
        let tt: TokenTree = input.parse()?;
        ts.extend([tt]);
    }
    if ts.is_empty() {
        return Err(syn::Error::new_spanned(
            key,
            format!("key `{key}` needs a value"),
        ));
    }
    Ok(ts)
}

fn trait_name(tokens: &TokenStream, key: &Ident) -> syn::Result<Ident> {
    let mut iter = tokens.clone().into_iter();
    while let Some(tt) = iter.next() {
        if matches!(&tt, TokenTree::Ident(i) if i == "trait") {
            if let Some(TokenTree::Ident(name)) = iter.next() {
                return Ok(name);
            }
        }
    }
    Err(syn::Error::new_spanned(
        key,
        "`trait_def` must contain `trait <Name>`",
    ))
}

fn parse_layer(input: ParseStream<'_>) -> syn::Result<Layer> {
    let name: Ident = input.parse()?;
    let content;
    braced!(content in input);
    let input = &content;

    let mut describe_only = false;
    let mut trait_doc = None;
    let mut trait_def = None;
    let mut impl_def = None;
    let mut method_where = None;
    let mut receiver = None;
    let mut first_arg = None;
    let mut tensor_arg = None;
    let mut exec = None;
    let mut dest = None;
    let mut result = None;
    let mut types = None;
    let mut forward = None;
    let mut cfg = None;
    let mut extra_decls = None;
    let mut extra_impls = None;
    let mut any_other = false;

    while !input.is_empty() {
        let key: Ident = input.parse()?;
        let k = key.to_string();
        if k == "describe_only" {
            describe_only = true;
            let _ = input.parse::<Option<Token![;]>>()?;
            continue;
        }
        any_other = true;
        if k == "extra_decls" || k == "extra_impls" {
            let body;
            braced!(body in input);
            let ts: TokenStream = body.parse()?;
            let slot = if k == "extra_decls" {
                &mut extra_decls
            } else {
                &mut extra_impls
            };
            set(slot, &key, ts)?;
            let _ = input.parse::<Option<Token![;]>>()?;
            continue;
        }
        if k == "types" {
            let body;
            braced!(body in input);
            let mut map = Vec::new();
            while !body.is_empty() {
                let id: Ident = body.parse()?;
                body.parse::<Token![=]>()?;
                let ty: Type = body.parse()?;
                map.push((id, ty));
                if !body.is_empty() {
                    body.parse::<Token![,]>()?;
                }
            }
            set(&mut types, &key, map)?;
            let _ = input.parse::<Option<Token![;]>>()?;
            continue;
        }
        input.parse::<Token![:]>()?;
        match k.as_str() {
            "trait_doc" => set(&mut trait_doc, &key, input.parse::<LitStr>()?)?,
            "trait_def" => set(&mut trait_def, &key, tokens_until_semi(input, &key)?)?,
            "impl_def" => set(&mut impl_def, &key, tokens_until_semi(input, &key)?)?,
            "method_where" => set(&mut method_where, &key, list::<WherePredicate>(input)?)?,
            "receiver" => {
                let r = if input.peek(Token![self]) {
                    input.parse::<Token![self]>()?;
                    Receiver::SelfRef
                } else {
                    let r: Ident = input.parse()?;
                    match r.to_string().as_str() {
                        "session" => Receiver::Session,
                        "context" => Receiver::Context,
                        _ => {
                            return Err(syn::Error::new_spanned(
                                &r,
                                "`receiver` must be `self`, `session` or `context`",
                            ))
                        }
                    }
                };
                set(&mut receiver, &key, r)?;
            }
            "first_arg" => set(&mut first_arg, &key, input.parse::<Type>()?)?,
            "tensor_arg" => set(&mut tensor_arg, &key, input.parse::<Type>()?)?,
            "exec" => set(&mut exec, &key, input.parse::<Type>()?)?,
            "dest" => set(&mut dest, &key, input.parse::<Type>()?)?,
            "result" => set(&mut result, &key, input.parse::<Path>()?)?,
            "forward" => set(&mut forward, &key, input.parse::<Path>()?)?,
            "cfg" => set(&mut cfg, &key, tokens_until_semi(input, &key)?)?,
            _ => {
                return Err(syn::Error::new_spanned(
                    &key,
                    format!("unknown key `{k}` in `layer {name}`"),
                ))
            }
        }
        end_key(input)?;
    }

    if describe_only {
        if any_other {
            return Err(syn::Error::new_spanned(
                &name,
                format!("`layer {name}` is `describe_only` and must not set other keys"),
            ));
        }
        return Ok(Layer {
            name,
            kind: LayerKind::DescribeOnly,
        });
    }

    let trait_def = need(trait_def, &name, "trait_def")?;
    let receiver = need(receiver, &name, "receiver")?;
    let first_arg = if receiver == Receiver::Session {
        Some(need(first_arg, &name, "first_arg")?)
    } else {
        first_arg
    };
    let trait_name = trait_name(&trait_def, &name)?;
    Ok(Layer {
        kind: LayerKind::Generate(Box::new(GenLayer {
            trait_doc: need(trait_doc, &name, "trait_doc")?,
            trait_def,
            trait_name,
            impl_def: need(impl_def, &name, "impl_def")?,
            method_where: method_where.unwrap_or_default(),
            receiver,
            first_arg,
            tensor_arg: need(tensor_arg, &name, "tensor_arg")?,
            exec,
            dest,
            result: need(result, &name, "result")?,
            types: types.unwrap_or_default(),
            forward: need(forward, &name, "forward")?,
            cfg,
            extra_decls: extra_decls.unwrap_or_default(),
            extra_impls: extra_impls.unwrap_or_default(),
        })),
        name,
    })
}

impl Parse for Operand {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let name: Ident = input.parse()?;
        let ty = if input.peek(Token![:]) {
            input.parse::<Token![:]>()?;
            Some(input.parse::<Type>()?)
        } else {
            None
        };
        Ok(Self { name, ty })
    }
}

impl Parse for ErrorEntry {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let path: Path = input.parse()?;
        let detail = if input.peek(syn::token::Paren) {
            let content;
            parenthesized!(content in input);
            Some(content.parse::<TokenStream>()?)
        } else {
            None
        };
        Ok(Self { path, detail })
    }
}

fn parse_op(input: ParseStream<'_>) -> syn::Result<Op> {
    let name: Ident = input.parse()?;
    let content;
    braced!(content in input);
    let input = &content;

    let mut summary = None;
    let mut operands = None;
    let mut options = None;
    let mut output = None;
    let mut dtypes = None;
    let mut layers = None;
    let mut into = None;
    let mut layout = None;
    let mut errors = None;
    let mut omissions = None;
    let mut generics = None;
    let mut associated = false;
    let mut examples: Vec<(Ident, LitStr)> = Vec::new();

    while !input.is_empty() {
        let key: Ident = input.parse()?;
        let k = key.to_string();
        if k == "associated" {
            if associated {
                return Err(syn::Error::new_spanned(
                    &key,
                    format!("duplicate `associated` in `op {name}`"),
                ));
            }
            associated = true;
            end_key(input)?;
            continue;
        }
        if k == "example" {
            let layer: Ident = input.parse()?;
            input.parse::<Token![:]>()?;
            let text: LitStr = input.parse()?;
            if examples.iter().any(|(l, _)| *l == layer) {
                return Err(syn::Error::new_spanned(
                    &layer,
                    format!("duplicate `example {layer}` in `op {name}`"),
                ));
            }
            examples.push((layer, text));
            end_key(input)?;
            continue;
        }
        input.parse::<Token![:]>()?;
        match k.as_str() {
            "summary" => set(&mut summary, &key, input.parse::<LitStr>()?)?,
            "operands" => {
                let body;
                parenthesized!(body in input);
                let v = Punctuated::<Operand, Token![,]>::parse_terminated(&body)?;
                set(&mut operands, &key, v.into_iter().collect::<Vec<_>>())?;
            }
            "options" => {
                let none = input.peek(Ident) && input.fork().parse::<Ident>()? == "none";
                let value = if none {
                    input.parse::<Ident>()?;
                    None
                } else {
                    Some(input.parse::<Path>()?)
                };
                set(&mut options, &key, value)?
            }
            "generics" => set(&mut generics, &key, input.parse::<syn::Generics>()?)?,
            "output" => set(&mut output, &key, input.parse::<Type>()?)?,
            "dtypes" => set(&mut dtypes, &key, input.parse::<LitStr>()?)?,
            "layers" => set(&mut layers, &key, list::<Ident>(input)?)?,
            "into" => set(&mut into, &key, list::<Ident>(input)?)?,
            "layout" => set(&mut layout, &key, input.parse::<LitStr>()?)?,
            "errors" => set(&mut errors, &key, list::<ErrorEntry>(input)?)?,
            "omissions" => set(&mut omissions, &key, list::<LitStr>(input)?)?,
            _ => {
                return Err(syn::Error::new_spanned(
                    &key,
                    format!("unknown key `{k}` in `op {name}`"),
                ))
            }
        }
        end_key(input)?;
    }

    Ok(Op {
        summary: need(summary, &name, "summary")?,
        operands: need(operands, &name, "operands")?,
        options: need(options, &name, "options")?,
        generics,
        associated,
        output: need(output, &name, "output")?,
        dtypes: need(dtypes, &name, "dtypes")?,
        layers: need(layers, &name, "layers")?,
        into: need(into, &name, "into")?,
        layout: need(layout, &name, "layout")?,
        errors: need(errors, &name, "errors")?,
        omissions: need(omissions, &name, "omissions")?,
        examples,
        name,
    })
}

fn parse_exception(input: ParseStream<'_>) -> syn::Result<Exception> {
    let id: Ident = input.parse()?;
    let content;
    braced!(content in input);
    let input = &content;
    let mut symbols = None;
    let mut scope = None;
    let mut category = None;
    let mut owner_module = None;
    let mut reason = None;
    let mut verification = None;
    let mut exit_condition = None;
    let mut status = None;
    while !input.is_empty() {
        let key: Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        match key.to_string().as_str() {
            "symbols" => set(&mut symbols, &key, list::<LitStr>(input)?)?,
            "scope" => set(&mut scope, &key, input.parse::<LitStr>()?)?,
            "category" => set(&mut category, &key, input.parse::<LitStr>()?)?,
            "owner_module" => set(&mut owner_module, &key, input.parse::<LitStr>()?)?,
            "reason" => set(&mut reason, &key, input.parse::<LitStr>()?)?,
            "verification" => set(&mut verification, &key, input.parse::<LitStr>()?)?,
            "exit_condition" => set(&mut exit_condition, &key, input.parse::<LitStr>()?)?,
            "status" => set(&mut status, &key, input.parse::<Ident>()?)?,
            other => {
                return Err(syn::Error::new_spanned(
                    &key,
                    format!("unknown key `{other}` in exception `{id}`"),
                ))
            }
        }
        end_key(input)?;
    }
    Ok(Exception {
        symbols: need(symbols, &id, "symbols")?,
        scope: need(scope, &id, "scope")?,
        category: need(category, &id, "category")?,
        owner_module: need(owner_module, &id, "owner_module")?,
        reason: need(reason, &id, "reason")?,
        verification: need(verification, &id, "verification")?,
        exit_condition: need(exit_condition, &id, "exit_condition")?,
        status: need(status, &id, "status")?,
        id,
    })
}
