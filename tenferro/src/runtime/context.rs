#[cfg(test)]
pub(crate) use tenferro_internal_runtime::{
    set_default_runtime as set_runtime_context, with_default_runtime as with_runtime_context,
};

#[cfg(test)]
mod tests;
