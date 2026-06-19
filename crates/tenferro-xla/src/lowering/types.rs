use std::fmt;

use tenferro_tensor::DType;

use crate::{Error, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TensorType {
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: DType,
}

impl TensorType {
    pub(crate) fn new(shape: Vec<usize>, dtype: DType, context: &'static str) -> Result<Self> {
        validate_dtype(dtype, context)?;
        Ok(Self { shape, dtype })
    }

    pub(crate) fn scalar(dtype: DType, context: &'static str) -> Result<Self> {
        Self::new(Vec::new(), dtype, context)
    }

    pub(crate) fn element(&self) -> &'static str {
        // `TensorType::new` rejects unsupported dtypes before this formatting
        // helper can be called. Keep this infallible so `fmt::Display` cannot
        // panic on a public lowering path.
        match self.dtype {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::I32 | DType::I64 | DType::Bool | DType::C32 | DType::C64 => {
                debug_assert!(false, "TensorType validates dtype at construction");
                "!tenferro.unsupported_dtype"
            }
        }
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_tensor_type(self))
    }
}

pub(crate) fn format_tensor_type(ty: &TensorType) -> String {
    if ty.shape.is_empty() {
        format!("tensor<{}>", ty.element())
    } else {
        let dims = ty
            .shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("x");
        format!("tensor<{dims}x{}>", ty.element())
    }
}

pub(crate) fn validate_dtype(dtype: DType, context: &'static str) -> Result<()> {
    stablehlo_dtype(dtype)
        .map(|_| ())
        .ok_or(Error::UnsupportedDType { dtype, context })
}

fn stablehlo_dtype(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::F32 => Some("f32"),
        DType::F64 => Some("f64"),
        DType::I32 | DType::I64 | DType::Bool | DType::C32 | DType::C64 => None,
    }
}
