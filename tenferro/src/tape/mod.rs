mod registry;
#[cfg(test)]
mod tensor_pullback;

pub(crate) use registry::register_closure_rule;
pub(crate) use registry::register_mixed_rule;
pub(crate) use registry::register_rule;
#[cfg(test)]
pub(crate) use tensor_pullback::pullback;

#[cfg(test)]
mod tests;
