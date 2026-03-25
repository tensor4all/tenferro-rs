use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenferro_device::{Error, Result};

use super::Tensor;
use crate::layout::{compute_contiguous_strides, validate_layout_against_len};

mod basic;
mod complex;
