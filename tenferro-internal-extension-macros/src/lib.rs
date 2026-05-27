//! Procedural macros for tenferro extension crates.
//!
//! # Examples
//!
//! ```
//! use tenferro_extension_macros::ExtensionFamilyId;
//!
//! #[derive(ExtensionFamilyId)]
//! #[tenferro_extension(namespace = "my-crate", name = "fft", version = 1)]
//! struct FftOp;
//!
//! assert_eq!(FftOp::FAMILY_ID, "my-crate.fft.v1");
//! ```

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, DeriveInput, Expr, ExprLit, Ident, Lit, Path, Token};

#[derive(Debug, Default)]
struct ExtensionArgs {
    namespace: Option<String>,
    name: Option<String>,
    version: Option<u64>,
}

struct RuntimeArgs {
    runtime: Ident,
    family_id: Path,
    op_type: Path,
    execute: Path,
    register_fn: Ident,
    backend_bound: Path,
}

impl Parse for ExtensionArgs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut args = Self::default();
        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            let value: Expr = input.parse()?;
            match key.to_string().as_str() {
                "namespace" => args.namespace = Some(expect_string(value, "namespace")?),
                "name" => args.name = Some(expect_string(value, "name")?),
                "version" => args.version = Some(expect_u64(value, "version")?),
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unsupported tenferro_extension argument {other:?}"),
                    ));
                }
            }
            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
        }
        Ok(args)
    }
}

impl Parse for RuntimeArgs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut runtime = None;
        let mut family_id = None;
        let mut op_type = None;
        let mut execute = None;
        let mut register_fn = None;
        let mut backend_bound = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "runtime" => runtime = Some(input.parse()?),
                "family_id" => family_id = Some(input.parse()?),
                "op_type" => op_type = Some(input.parse()?),
                "execute" => execute = Some(input.parse()?),
                "register_fn" => register_fn = Some(input.parse()?),
                "backend_bound" => backend_bound = Some(input.parse()?),
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unsupported define_extension_runtime argument {other:?}"),
                    ));
                }
            }
            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
        }

        Ok(Self {
            runtime: required(runtime, "runtime")?,
            family_id: required(family_id, "family_id")?,
            op_type: required(op_type, "op_type")?,
            execute: required(execute, "execute")?,
            register_fn: required(register_fn, "register_fn")?,
            backend_bound: backend_bound
                .unwrap_or_else(|| syn::parse_quote!(tenferro_tensor::TensorBackend)),
        })
    }
}

/// Derive an inherent `FAMILY_ID` constant for an extension payload type.
///
/// The required attribute is:
/// `#[tenferro_extension(namespace = "...", version = N)]`.
/// `name = "..."` is optional; when omitted, the Rust type name is converted
/// to snake_case.
#[proc_macro_derive(ExtensionFamilyId, attributes(tenferro_extension))]
pub fn derive_extension_family_id(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand_extension_family_id(input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Generate a standard extension runtime and registration function.
///
/// The `execute` function must have this signature:
/// `fn<B: BackendBound + 'static>(&OpType, &[&Tensor], &mut ExtensionExecutionContext<'_, B>)`.
#[proc_macro]
pub fn define_extension_runtime(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as RuntimeArgs);
    expand_extension_runtime(args).into()
}

fn expand_extension_family_id(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let mut parsed = None;
    for attr in &input.attrs {
        if attr.path().is_ident("tenferro_extension") {
            let args = attr.parse_args::<ExtensionArgs>()?;
            parsed = Some(args);
        }
    }
    let args = parsed.ok_or_else(|| {
        syn::Error::new_spanned(
            &input.ident,
            "missing #[tenferro_extension(namespace = \"...\", version = N)]",
        )
    })?;
    let namespace = args.namespace.ok_or_else(|| {
        syn::Error::new_spanned(&input.ident, "missing tenferro_extension namespace")
    })?;
    let version = args.version.ok_or_else(|| {
        syn::Error::new_spanned(&input.ident, "missing tenferro_extension version")
    })?;
    let name = args
        .name
        .unwrap_or_else(|| to_snake_case(&input.ident.to_string()));
    let family_id = format!("{namespace}.{name}.v{version}");
    let ident = input.ident;

    Ok(quote! {
        impl #ident {
            /// Stable extension family identifier generated by `ExtensionFamilyId`.
            pub const FAMILY_ID: &'static str = #family_id;
        }
    })
}

fn expand_extension_runtime(args: RuntimeArgs) -> proc_macro2::TokenStream {
    let RuntimeArgs {
        runtime,
        family_id,
        op_type,
        execute,
        register_fn,
        backend_bound,
    } = args;

    quote! {
        #[derive(Debug, Default)]
        pub(crate) struct #runtime;

        impl<B: #backend_bound + 'static> tenferro_runtime::extension::ExtensionRuntime<B>
            for #runtime
        {
            fn family_id(&self) -> &'static str {
                #family_id
            }

            fn execute(
                &self,
                op: &dyn tenferro_runtime::extension::ExtensionOpTrait,
                inputs: &[&tenferro_tensor::Tensor],
                ctx: &mut tenferro_runtime::extension::ExtensionExecutionContext<'_, B>,
            ) -> tenferro_tensor::Result<Vec<tenferro_tensor::Tensor>> {
                let op = op
                    .as_any()
                    .downcast_ref::<#op_type>()
                    .ok_or_else(|| tenferro_tensor::Error::InvalidConfig {
                        op: "extension_runtime",
                        message: format!("payload type mismatch for {}", #family_id),
                    })?;
                #execute(op, inputs, ctx)
            }
        }

        pub fn #register_fn<B: #backend_bound + 'static>(
            executor: &mut tenferro_runtime::extension::ExtensionExecutor<B>,
        ) -> std::result::Result<
            (),
            tenferro_runtime::extension::ExtensionRuntimeRegistryError,
        > {
            executor.registry_mut().register(std::sync::Arc::new(#runtime))
        }
    }
}

fn expect_string(value: Expr, field: &str) -> syn::Result<String> {
    match value {
        Expr::Lit(ExprLit {
            lit: Lit::Str(value),
            ..
        }) => Ok(value.value()),
        other => Err(syn::Error::new_spanned(
            other,
            format!("{field} must be a string literal"),
        )),
    }
}

fn expect_u64(value: Expr, field: &str) -> syn::Result<u64> {
    match value {
        Expr::Lit(ExprLit {
            lit: Lit::Int(value),
            ..
        }) => value.base10_parse(),
        other => Err(syn::Error::new_spanned(
            other,
            format!("{field} must be an integer literal"),
        )),
    }
}

fn required<T>(value: Option<T>, field: &str) -> syn::Result<T> {
    value.ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(), format!("missing {field}")))
}

fn to_snake_case(input: &str) -> String {
    let mut out = String::new();
    let mut prev_lower_or_digit = false;
    for ch in input.chars() {
        if ch.is_ascii_uppercase() {
            if prev_lower_or_digit {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
            prev_lower_or_digit = false;
        } else {
            prev_lower_or_digit = ch.is_ascii_lowercase() || ch.is_ascii_digit();
            out.push(ch);
        }
    }
    out
}

#[cfg(test)]
mod tests;
