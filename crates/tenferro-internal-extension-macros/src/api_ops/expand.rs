//! Code generation for `api_ops!`: traits, impls, metadata and renderers.

use proc_macro2::{Group, TokenStream, TokenTree};
use quote::{format_ident, quote, ToTokens};
use syn::{Ident, LitStr};

use super::{error_text, markdown, tidy, GenLayer, LayerKind, Op, Receiver, Registry};

/// Replaces the idents `V`/`R`/`C`/`I` in `ts` using the layer's `types`.
fn substitute(
    ts: TokenStream,
    types: &[(Ident, syn::Type)],
    op: &Op,
    layer: &Ident,
) -> syn::Result<TokenStream> {
    let mut out = TokenStream::new();
    for tt in ts {
        match tt {
            TokenTree::Group(g) => {
                let inner = substitute(g.stream(), types, op, layer)?;
                let mut ng = Group::new(g.delimiter(), inner);
                ng.set_span(g.span());
                out.extend([TokenTree::Group(ng)]);
            }
            TokenTree::Ident(id) if matches!(id.to_string().as_str(), "V" | "R" | "C" | "I") => {
                match types.iter().find(|(k, _)| *k == id) {
                    Some((_, ty)) => ty.to_tokens(&mut out),
                    None => {
                        return Err(syn::Error::new_spanned(
                            &id,
                            format!(
                                "op `{}` output uses `{id}`, but layer `{layer}` has no `types` mapping for it",
                                op.name
                            ),
                        ))
                    }
                }
            }
            other => out.extend([other]),
        }
    }
    Ok(out)
}

fn lit(s: &str) -> LitStr {
    LitStr::new(s, proc_macro2::Span::call_site())
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Form {
    Plain,
    With,
    Into,
}

fn method_doc(op: &Op, trait_name: &Ident, form: Form, example: &str) -> String {
    let n = op.name.to_string();
    let opts = {
        let p = &op.options;
        tidy(&quote!(#p))
    };
    let sentence = match form {
        Form::Plain => format!(
            "Uses the default [`{opts}`]; same as [`{n}_with`]({trait_name}::{n}_with) with `&{opts}::default()`."
        ),
        Form::With => format!("With explicit [`{opts}`]."),
        Form::Into => {
            let modes = op
                .into
                .iter()
                .map(|m| capitalize(&m.to_string()))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "Writes into `dest` instead of returning a new tensor. Accepted destination modes: {modes}; any other mode returns an `Unsupported` error before any data is read or written."
            )
        }
    };
    let mut d = format!(
        "{}\n\n{sentence}\n\n# Layout\n\n{}\n\n",
        op.summary.value(),
        op.layout.value()
    );
    d.push_str("# Errors\n\nReturns an error of kind:\n\n");
    for e in &op.errors {
        d.push_str(&format!("- `{}`\n", error_text(e)));
    }
    if !op.omissions.is_empty() {
        d.push_str("\n# Omissions\n\n");
        for o in &op.omissions {
            d.push_str(&format!("- {}\n", o.value()));
        }
    }
    d.push_str("\n# Examples\n\n```rust\n");
    d.push_str(example.trim_matches('\n'));
    d.push_str("\n```");
    d
}

/// Generates the trait and impl of one generating layer.
fn gen_layer(reg: &Registry, name: &Ident, g: &GenLayer) -> syn::Result<TokenStream> {
    let t = &g.trait_name;
    let result = &g.result;
    let preds = &g.method_where;
    let where_clause = if preds.is_empty() {
        quote!()
    } else {
        quote!(where #(#preds),*)
    };
    let forward = &g.forward;
    let family = reg.family.to_string();
    let trait_doc = g.trait_doc.value();
    let trait_def = &g.trait_def;
    let impl_def = &g.impl_def;
    let (receiver, first_skip) = match g.receiver {
        Receiver::SelfRef => (quote!(&self), 1usize),
        Receiver::Session => (quote!(&mut self), 0usize),
    };

    let mut decls = Vec::new();
    let mut impls = Vec::new();
    for op in reg
        .ops
        .iter()
        .filter(|o| o.layers.iter().any(|l| l == name))
    {
        let example = op
            .examples
            .iter()
            .find(|(l, _)| l == name)
            .map(|(_, e)| e.value())
            .ok_or_else(|| {
                syn::Error::new_spanned(
                    &op.name,
                    format!(
                        "op `{}` is missing `example {name}: r#\"...\"#;` required by generating layer `{name}`",
                        op.name
                    ),
                )
            })?;
        let out = substitute(op.output.to_token_stream(), &g.types, op, name)?;
        let n = &op.name;
        let n_with = format_ident!("{}_with", n);
        let n_into = format_ident!("{}_into", n);
        let opts = &op.options;

        let mut operand_params = Vec::new();
        let mut operand_names = Vec::new();
        for (i, operand) in op.operands.iter().enumerate().skip(first_skip) {
            let nm = &operand.name;
            let ty = if i == 0 {
                match &g.first_arg {
                    Some(t) => t.to_token_stream(),
                    None => g.tensor_arg.to_token_stream(),
                }
            } else {
                match &operand.ty {
                    Some(t) => t.to_token_stream(),
                    None => g.tensor_arg.to_token_stream(),
                }
            };
            operand_params.push(quote!(#nm: #ty));
            operand_names.push(quote!(#nm));
        }
        let exec_param = g.exec.as_ref().map(|e| quote!(exec: #e));
        let exec_name = g.exec.as_ref().map(|_| quote!(exec));
        let opt_param = quote!(options: &#opts);

        let mut build = |form: Form| {
            let mut params = operand_params.clone();
            let mut args = operand_names.clone();
            if form != Form::Plain {
                params.push(opt_param.clone());
                args.push(quote!(options));
            }
            if let Some(p) = &exec_param {
                params.push(p.clone());
            }
            if let Some(a) = &exec_name {
                args.push(a.clone());
            }
            let ret = if form == Form::Into {
                let dest = g.dest.as_ref().map(|d| quote!(dest: #d));
                if let Some(d) = dest {
                    params.push(d);
                    args.push(quote!(dest));
                }
                quote!(#result<()>)
            } else {
                quote!(#result<#out>)
            };
            let fname = match form {
                Form::Plain => n.clone(),
                Form::With => n_with.clone(),
                Form::Into => n_into.clone(),
            };
            let doc = lit(&method_doc(op, t, form, &example));
            let many = if params.len() + 1 > 7 {
                quote!(#[allow(clippy::too_many_arguments)])
            } else {
                quote!()
            };
            let sig = quote!(fn #fname(#receiver, #(#params),*) -> #ret #where_clause);
            let fwd_fn = if form == Form::Into {
                n_into.clone()
            } else {
                n.clone()
            };
            let body = match form {
                Form::Plain => {
                    // Default options are inserted before `exec`.
                    let mut a = operand_names.clone();
                    a.push(quote!(&<#opts as ::core::default::Default>::default()));
                    if let Some(e) = &exec_name {
                        a.push(e.clone());
                    }
                    quote!(<Self as #t>::#n_with(self, #(#a),*))
                }
                _ => quote!(#forward::#fwd_fn(self, #(#args),*)),
            };
            if form == Form::Into && g.dest.is_none() {
                return;
            }
            decls.push(quote! { #[doc = #doc] #many #sig; });
            impls.push(quote! { #many #sig { #body } });
        };
        build(Form::Plain);
        build(Form::With);
        if !op.into.is_empty() {
            build(Form::Into);
        }
    }

    let doc_a = lit(&trait_doc);
    let doc_b = lit(&format!(
        "\n\nGenerated by `api_ops!` from the {family} registry."
    ));
    Ok(quote! {
        #[doc = #doc_a]
        #[doc = #doc_b]
        #trait_def {
            #(#decls)*
        }
        #impl_def {
            #(#impls)*
        }
    })
}

/// Expands a validated registry into traits, impls and metadata.
pub(crate) fn expand(reg: &Registry) -> syn::Result<TokenStream> {
    let mut generated = Vec::new();
    for layer in &reg.layers {
        if let LayerKind::Generate(g) = &layer.kind {
            generated.push(gen_layer(reg, &layer.name, g)?);
        }
    }

    let family = reg.family.to_string();
    let layer_pairs = reg.layers.iter().map(|l| {
        let n = l.name.to_string();
        let t = match &l.kind {
            LayerKind::DescribeOnly => String::new(),
            LayerKind::Generate(g) => g.trait_name.to_string(),
        };
        quote!((#n, #t))
    });
    let ops = reg.ops.iter().map(|op| {
        let name = op.name.to_string();
        let operands = op.operands.iter().map(|o| {
            let n = &o.name;
            match &o.ty {
                Some(t) => tidy(&quote!(#n: #t)),
                None => n.to_string(),
            }
        });
        let options = {
            let p = &op.options;
            tidy(&quote!(#p))
        };
        let output = {
            let p = &op.output;
            tidy(&quote!(#p))
        };
        let dtypes = op.dtypes.value();
        let layers = op.layers.iter().map(ToString::to_string);
        let into = op.into.iter().map(ToString::to_string);
        let layout = op.layout.value();
        let errors = op.errors.iter().map(error_text);
        let omissions = op.omissions.iter().map(LitStr::value);
        quote! {
            ApiOpDescriptor {
                family: #family,
                name: #name,
                operands: &[#(#operands),*],
                options: #options,
                output: #output,
                dtypes: #dtypes,
                layers: &[#(#layers),*],
                into: &[#(#into),*],
                layout: #layout,
                errors: &[#(#errors),*],
                omissions: &[#(#omissions),*],
            }
        }
    });
    let exceptions = reg.exceptions.iter().map(|e| {
        let id = e.id.to_string();
        let symbols = e.symbols.iter().map(LitStr::value);
        let (scope, category, owner, reason, verification, exit, status) = (
            e.scope.value(),
            e.category.value(),
            e.owner_module.value(),
            e.reason.value(),
            e.verification.value(),
            e.exit_condition.value(),
            e.status.to_string(),
        );
        quote! {
            ApiExceptionDescriptor {
                id: #id,
                symbols: &[#(#symbols),*],
                scope: #scope,
                category: #category,
                owner_module: #owner,
                reason: #reason,
                verification: #verification,
                exit_condition: #exit,
                status: #status,
            }
        }
    });
    let md = markdown::render(reg);
    let begin = format!("<!-- api-ops:{family}:begin -->");
    let end = format!("<!-- api-ops:{family}:end -->");

    Ok(quote! {
        #(#generated)*

        /// Descriptor of one registry operation.
        #[allow(dead_code)]
        #[derive(Debug, Clone, Copy)]
        pub(crate) struct ApiOpDescriptor {
            pub family: &'static str,
            pub name: &'static str,
            pub operands: &'static [&'static str],
            pub options: &'static str,
            pub output: &'static str,
            pub dtypes: &'static str,
            pub layers: &'static [&'static str],
            pub into: &'static [&'static str],
            pub layout: &'static str,
            pub errors: &'static [&'static str],
            pub omissions: &'static [&'static str],
        }

        /// Descriptor of one registry exception.
        #[allow(dead_code)]
        #[derive(Debug, Clone, Copy)]
        pub(crate) struct ApiExceptionDescriptor {
            pub id: &'static str,
            pub symbols: &'static [&'static str],
            pub scope: &'static str,
            pub category: &'static str,
            pub owner_module: &'static str,
            pub reason: &'static str,
            pub verification: &'static str,
            pub exit_condition: &'static str,
            pub status: &'static str,
        }

        #[allow(dead_code)]
        pub(crate) const API_FAMILY: &str = #family;
        #[allow(dead_code)]
        pub(crate) const API_LAYERS: &[(&str, &str)] = &[#(#layer_pairs),*];
        #[allow(dead_code)]
        pub(crate) const API_OPS: &[ApiOpDescriptor] = &[#(#ops),*];
        #[allow(dead_code)]
        pub(crate) const API_EXCEPTIONS: &[ApiExceptionDescriptor] = &[#(#exceptions),*];

        /// Markdown section describing this registry.
        #[allow(dead_code)]
        pub(crate) fn api_ops_markdown() -> ::std::string::String {
            ::std::string::String::from(#md)
        }

        /// Returns `current_file` with this family's marker block replaced by
        /// [`api_ops_markdown`]; appends the block when the markers are absent.
        #[allow(dead_code)]
        pub(crate) fn api_docs_sync(current_file: &str) -> ::std::string::String {
            let markdown = api_ops_markdown();
            if let Some(start) = current_file.find(#begin) {
                let after_begin = start + #begin.len();
                if let Some(rel) = current_file[after_begin..].find(#end) {
                    let end_at = after_begin + rel;
                    return ::std::format!(
                        "{}\n{}{}",
                        &current_file[..after_begin],
                        markdown,
                        &current_file[end_at..]
                    );
                }
            }
            ::std::format!("{}\n{}\n{}{}\n", current_file, #begin, markdown, #end)
        }
    })
}
