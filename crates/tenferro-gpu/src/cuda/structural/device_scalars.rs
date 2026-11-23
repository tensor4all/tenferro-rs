//! Host scalars kept resident on the device, for the scalar-operand engine hooks (#2701).
//!
//! A composite asks for the same few constants on every call: an activation coefficient, a
//! normalization epsilon, a mean count, a scale factor, a pseudoinverse threshold. The portable
//! [`tenferro_tensor::Engine::binary_scalar_read`] uploads each one per call, a host-to-device
//! transfer inside the operation. The CUDA engine uploads a scalar on its first use, keeps it, and
//! lends it to the same formula afterwards: a zero-stride view of the resident scalar and the
//! matching `*_read` entry, which reads that view in place. The result is the portable hook's, bit
//! for bit, without the transfer.

use tenferro_tensor::{
    default_binary_scalar_read, default_fill_scalar, default_select_scalar_read, ContractionScalar,
    DType, ScalarBinaryOp, ScalarOperandSide, Tensor, TensorRead,
};

use super::CudaEngine;

/// Most scalars one engine keeps. A composite uses a handful; the bound only stops a caller that
/// sweeps many distinct values from growing the set. Past it, a scalar is uploaded per call, as
/// by the portable hook.
const DEVICE_SCALAR_CAPACITY: usize = 64;

/// The resident scalars of one engine, keyed by dtype and bit pattern.
///
/// Owner: the [`CudaEngine`], for its whole lifetime. An entry is lent out (taken) for the
/// duration of one hook call and put back afterwards, so the engine stays free for the call.
#[derive(Default)]
pub(crate) struct DeviceScalars {
    entries: Vec<((DType, u128), Option<Tensor>)>,
}

/// The dtype and bit pattern of `scalar`.
fn scalar_key(scalar: ContractionScalar) -> (DType, u128) {
    let bits = match scalar {
        ContractionScalar::F32(value) => u128::from(value.to_bits()),
        ContractionScalar::F64(value) => u128::from(value.to_bits()),
        ContractionScalar::C32(value) => {
            (u128::from(value.re.to_bits()) << 64) | u128::from(value.im.to_bits())
        }
        ContractionScalar::C64(value) => {
            (u128::from(value.re.to_bits()) << 64) | u128::from(value.im.to_bits())
        }
    };
    (scalar.dtype(), bits)
}

/// A rank-0 host tensor holding `scalar`.
fn host_scalar(scalar: ContractionScalar) -> crate::Result<Tensor> {
    match scalar {
        ContractionScalar::F32(value) => Tensor::from_vec_col_major(Vec::new(), vec![value]),
        ContractionScalar::F64(value) => Tensor::from_vec_col_major(Vec::new(), vec![value]),
        ContractionScalar::C32(value) => Tensor::from_vec_col_major(Vec::new(), vec![value]),
        ContractionScalar::C64(value) => Tensor::from_vec_col_major(Vec::new(), vec![value]),
    }
}

impl CudaEngine {
    /// Run `f` with the resident device copy of `scalar`, uploading it on its first use.
    fn with_device_scalar<R>(
        &mut self,
        scalar: ContractionScalar,
        f: impl FnOnce(&mut Self, &Tensor) -> crate::Result<R>,
    ) -> crate::Result<R> {
        use tenferro_tensor::TensorDeviceTransfer as _;
        let key = scalar_key(scalar);
        let slot = self
            .device_scalars
            .entries
            .iter()
            .position(|(entry, _)| *entry == key);
        let lent = slot.and_then(|slot| self.device_scalars.entries[slot].1.take());
        let device = match lent {
            Some(device) => device,
            None => self.upload_host_tensor(TensorRead::from_tensor(&host_scalar(scalar)?))?,
        };
        let result = f(self, &device);
        match slot {
            Some(slot) => self.device_scalars.entries[slot].1 = Some(device),
            None if self.device_scalars.entries.len() < DEVICE_SCALAR_CAPACITY => {
                self.device_scalars.entries.push((key, Some(device)));
            }
            None => {}
        }
        result
    }

    /// [`tenferro_tensor::Engine::binary_scalar_read`] with a resident scalar.
    pub(super) fn binary_with_device_scalar(
        &mut self,
        op: ScalarBinaryOp,
        tensor: TensorRead<'_>,
        scalar: ContractionScalar,
        side: ScalarOperandSide,
    ) -> crate::Result<Tensor> {
        if scalar.dtype() != tensor.dtype() {
            // The portable hook types the refusal.
            return default_binary_scalar_read(self, op, tensor, scalar, side);
        }
        self.with_device_scalar(scalar, |engine, device| {
            use tenferro_tensor::TensorElementwise as _;
            let splat = tenferro_tensor::scalar_splat(device, tensor.shape())?;
            let (lhs, rhs) = match side {
                ScalarOperandSide::Left => (splat, tensor),
                ScalarOperandSide::Right => (tensor, splat),
            };
            match op {
                ScalarBinaryOp::Add => engine.add_read(lhs, rhs),
                ScalarBinaryOp::Subtract => engine.sub_read(lhs, rhs),
                ScalarBinaryOp::Multiply => engine.mul_read(lhs, rhs),
                ScalarBinaryOp::Divide => engine.div_read(lhs, rhs),
                ScalarBinaryOp::Maximum => engine.maximum_read(lhs, rhs),
                other => Err(crate::Error::unsupported(
                    other.label(),
                    "no CUDA scalar-operand form of this operation",
                )),
            }
        })
    }

    /// [`tenferro_tensor::Engine::select_scalar_read`] with a resident scalar.
    pub(super) fn select_with_device_scalar(
        &mut self,
        predicate: TensorRead<'_>,
        tensor: TensorRead<'_>,
        scalar: ContractionScalar,
        side: ScalarOperandSide,
    ) -> crate::Result<Tensor> {
        if scalar.dtype() != tensor.dtype() {
            return default_select_scalar_read(self, predicate, tensor, scalar, side);
        }
        self.with_device_scalar(scalar, |engine, device| {
            use tenferro_tensor::TensorElementwise as _;
            let splat = tenferro_tensor::scalar_splat(device, tensor.shape())?;
            let (on_true, on_false) = match side {
                ScalarOperandSide::Left => (splat, tensor),
                ScalarOperandSide::Right => (tensor, splat),
            };
            engine.select_read(predicate, on_true, on_false)
        })
    }

    /// [`tenferro_tensor::Engine::fill_scalar`] with a resident scalar.
    pub(super) fn fill_with_device_scalar(
        &mut self,
        shape: &[usize],
        scalar: ContractionScalar,
    ) -> crate::Result<Tensor> {
        if shape.is_empty() {
            // The result is a fresh rank-0 tensor the caller owns; the resident one is not lent.
            return default_fill_scalar(self, shape, scalar);
        }
        self.with_device_scalar(scalar, |engine, device| {
            use tenferro_tensor::TensorStructural as _;
            engine.broadcast_in_dim_read(TensorRead::from_tensor(device), shape, &[])
        })
    }
}
