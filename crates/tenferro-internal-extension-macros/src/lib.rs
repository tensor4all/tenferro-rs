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
use quote::{format_ident, quote};
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
    execute_reads: Path,
    execute_in_session: Option<Path>,
    session_supported: Option<Path>,
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
        let mut execute_reads = None;
        let mut execute_in_session = None;
        let mut session_supported = None;
        let mut backend_bound = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "runtime" => runtime = Some(input.parse()?),
                "family_id" => family_id = Some(input.parse()?),
                "op_type" => op_type = Some(input.parse()?),
                "execute" => execute = Some(input.parse()?),
                "execute_reads" => execute_reads = Some(input.parse()?),
                "execute_in_session" => execute_in_session = Some(input.parse()?),
                "session_supported" => session_supported = Some(input.parse()?),
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
            execute_reads: required(execute_reads, "execute_reads")?,
            execute_in_session,
            session_supported,
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

/// Generate a standard extension module, preparation engine, and prepared
/// operation.
///
/// The `execute` function must have this signature:
/// `fn<B: BackendBound + 'static>(&OpType, &[&Tensor], &mut ExtensionExecutionContext<'_, B>)`.
///
/// `execute_reads` is required. It must have this signature:
/// `fn<B: BackendBound + 'static>(&OpType, &[TensorRead<'_>], &mut ExtensionExecutionContext<'_, B>)`.
///
/// `session_supported` and `execute_in_session` are optional, but must be
/// supplied together. The former has signature
/// `fn<B: BackendBound + 'static>(&OpType) -> bool`; the latter has signature
/// `fn(&OpType, &mut dyn BackendSession, &mut ExtensionCacheStore, &[TensorRead<'_>])`.
#[proc_macro]
pub fn define_extension_runtime(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as RuntimeArgs);
    match expand_extension_runtime(args) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
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

fn expand_extension_runtime(args: RuntimeArgs) -> syn::Result<proc_macro2::TokenStream> {
    let RuntimeArgs {
        runtime,
        family_id,
        op_type,
        execute: _execute,
        execute_reads,
        execute_in_session,
        session_supported,
        backend_bound,
    } = args;
    let module = format_ident!("{}Module", runtime);
    let planning_config = format_ident!("{}PlanningConfig", runtime);
    let prepared_operation = format_ident!("{}PreparedOperation", runtime);
    let session_methods = match (execute_in_session, session_supported) {
        (Some(execute_in_session), Some(session_supported)) => quote! {
            fn supports_session(&self) -> bool {
                #session_supported::<B>(&self.op)
            }

            fn execute_in_session(
                &self,
                session: &mut dyn tenferro_tensor::BackendSession,
                extension_caches: &mut tenferro_runtime::ExtensionCacheStore,
                inputs: &[tenferro_tensor::TensorRead<'_>],
            ) -> tenferro_runtime::Result<Vec<tenferro_tensor::Tensor>> {
                Ok(#execute_in_session(&self.op, session, extension_caches, inputs)?)
            }
        },
        (None, None) => quote! {},
        (Some(path), None) | (None, Some(path)) => {
            return Err(syn::Error::new_spanned(
                path,
                "execute_in_session and session_supported must be supplied together",
            ))
        }
    };
    Ok(quote! {
        pub(crate) struct #runtime<B: #backend_bound + 'static> {
            engine_id: tenferro_runtime::EngineId,
            _backend: std::marker::PhantomData<fn() -> B>,
        }

        pub(crate) struct #module<B: #backend_bound + 'static> {
            module_id: tenferro_runtime::ExtensionModuleId,
            engine_id: tenferro_runtime::EngineId,
            _backend: std::marker::PhantomData<fn() -> B>,
        }

        #[derive(Debug, Default)]
        pub(crate) struct #planning_config;

        pub(crate) struct #prepared_operation<B: #backend_bound + 'static> {
            binding: tenferro_runtime::PreparedOperationBinding,
            specialization: tenferro_runtime::SpecializationProjection,
            op: #op_type,
            _backend: std::marker::PhantomData<fn() -> B>,
        }

        impl<B: #backend_bound + 'static> std::fmt::Debug for #runtime<B> {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter
                    .debug_struct(stringify!(#runtime))
                    .field("family_id", &#family_id)
                    .field("engine_id", &self.engine_id)
                    .field("backend_type", &std::any::type_name::<B>())
                    .finish()
            }
        }

        impl<B: #backend_bound + 'static> std::fmt::Debug for #module<B> {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter
                    .debug_struct(stringify!(#module))
                    .field("module_id", &self.module_id)
                    .field("engine_id", &self.engine_id)
                    .field("backend_type", &std::any::type_name::<B>())
                    .finish()
            }
        }

        impl<B: #backend_bound + 'static> std::fmt::Debug for #prepared_operation<B> {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter
                    .debug_struct(stringify!(#prepared_operation))
                    .field("family_id", &#family_id)
                    .field("binding", &self.binding)
                    .field("specialization", &self.specialization)
                    .field("backend_type", &std::any::type_name::<B>())
                    .finish_non_exhaustive()
            }
        }

        impl<B: #backend_bound + 'static> tenferro_runtime::ExtensionEngine for #runtime<B>
        where
            #op_type: Clone + Send + Sync + 'static,
        {
            fn family_id(&self) -> &'static str {
                #family_id
            }

            fn engine_id(&self) -> &tenferro_runtime::EngineId {
                &self.engine_id
            }

            fn context_identity(&self) -> tenferro_runtime::ExecutionContextIdentity {
                tenferro_runtime::ExecutionContextIdentity::of::<B>()
            }

            fn prepare(
                &self,
                request: tenferro_runtime::ExtensionPrepareRequest<'_>,
            ) -> std::result::Result<tenferro_runtime::PrepareCapability, tenferro_runtime::PrepareError> {
                let op = request
                    .operation()
                    .as_any()
                    .downcast_ref::<#op_type>()
                    .cloned()
                    .ok_or_else(|| tenferro_runtime::PrepareError::ProviderContract {
                        source: tenferro_runtime::ProviderContractError::WrongOperationFamily {
                            expected: tenferro_runtime::CoreCapabilityKind::Elementwise,
                            operation: #family_id,
                        },
                    })?;
                let prepared = std::sync::Arc::new(#prepared_operation::<B> {
                    binding: request.binding().clone(),
                    specialization: request.specialization().clone(),
                    op,
                    _backend: std::marker::PhantomData,
                });
                Ok(tenferro_runtime::PrepareCapability::Prepared(
                    tenferro_runtime::PreparedOperationPlan::executable(prepared.clone(), prepared)
                ))
            }
        }

        impl tenferro_runtime::ExtensionPlanningConfig for #planning_config {
            fn family_id(&self) -> &'static str {
                #family_id
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }

            fn payload_hash(&self, state: &mut dyn std::hash::Hasher) {
                state.write_u8(0);
            }

            fn payload_eq(&self, other: &dyn tenferro_runtime::ExtensionPlanningConfig) -> bool {
                other.as_any().downcast_ref::<Self>().is_some()
            }

            fn retained_bytes(&self) -> usize {
                0
            }
        }

        impl<B: #backend_bound + 'static> tenferro_runtime::PreparedOperation for #prepared_operation<B>
        where
            #op_type: Clone + Send + Sync + 'static,
        {
            fn binding(&self) -> &tenferro_runtime::PreparedOperationBinding {
                &self.binding
            }

            fn specialization(&self) -> &tenferro_runtime::SpecializationProjection {
                &self.specialization
            }

            fn retained_bytes(&self) -> usize {
                0
            }

        }

        impl<B: #backend_bound + 'static> tenferro_runtime::PreparedOperationExecutor for #prepared_operation<B>
        where
            #op_type: Clone + Send + Sync + 'static,
        {
            fn execute(
                &self,
                context: &mut tenferro_runtime::ErasedExecutionContext<'_>,
                extension_caches: &mut tenferro_runtime::ExtensionCacheStore,
                inputs: &[tenferro_tensor::TensorRead<'_>],
            ) -> tenferro_runtime::Result<Vec<tenferro_tensor::Tensor>> {
                let backend = context
                    .downcast_mut::<B>(self.binding.context_identity())
                    .map_err(|source| tenferro_runtime::Error::runtime_state_source(
                        "extension",
                        tenferro_runtime::ErrorPhase::Execution,
                        source,
                    ))?;
                let mut ctx = tenferro_runtime::ExtensionExecutionContext::new(
                    backend,
                    extension_caches,
                );
                Ok(#execute_reads(&self.op, inputs, &mut ctx)?)
            }

            #session_methods
        }

        impl<B: #backend_bound + 'static> tenferro_runtime::ExtensionModule for #module<B>
        where
            #op_type: Clone + Send + Sync + 'static,
        {
            fn module_id(&self) -> &tenferro_runtime::ExtensionModuleId {
                &self.module_id
            }

            fn configure(
                &self,
                registrar: &mut tenferro_runtime::ExtensionModuleRegistrar<'_>,
            ) -> std::result::Result<(), tenferro_runtime::ExtensionModuleError> {
                registrar.register_engine(std::sync::Arc::new(#runtime::<B> {
                    engine_id: self.engine_id.clone(),
                    _backend: std::marker::PhantomData,
                }))?;
                registrar.register_planning_config(
                    self.engine_id.clone(),
                    std::sync::Arc::new(#planning_config),
                )?;
                Ok(())
            }
        }

        #[doc = "Build this extension module for one runtime engine."]
        #[doc = "\n# Errors\n\nReturns `RuntimeConfigError::MalformedIdentity` when the generated module identifier is invalid."]
        pub fn extension_module<B: #backend_bound + 'static>(
            engine_id: tenferro_runtime::EngineId,
        ) -> std::result::Result<
            std::sync::Arc<dyn tenferro_runtime::ExtensionModule>,
            tenferro_runtime::RuntimeConfigError,
        >
        where
            #op_type: Clone + Send + Sync + 'static,
        {
            Ok(std::sync::Arc::new(#module::<B> {
                module_id: tenferro_runtime::ExtensionModuleId::new(format!("{}.module", #family_id))?,
                engine_id,
                _backend: std::marker::PhantomData,
            }))
        }

    })
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
