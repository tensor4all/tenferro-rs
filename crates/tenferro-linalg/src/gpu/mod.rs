mod ffi;
mod kernels;
mod linalg;

use tenferro_gpu::cuda::CudaExecSession;
use tenferro_tensor::{Tensor, TensorRead, TensorView};

use crate::backend::{unsupported_dtype, LinalgBackend};
use crate::extension::{
    apply_eigh_gauge, apply_svd_gauge, validate_derivative_eps, EighDriver, EighOptions, SvdDriver,
    SvdOptions,
};
use crate::{QrOptions, RankRevealingQrOptions};

/// cuSOLVER needs compact column-major device storage, so a borrowed view is
/// made contiguous on the device and never crosses the host boundary.
fn svd_read_with_driver(
    session: &mut CudaExecSession<'_>,
    input: TensorRead<'_>,
    driver: SvdDriver,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let input = input.tensor_view();
    match input {
        TensorView::F32(view) => {
            let compact = session.to_contiguous(&view)?;
            let input = Tensor::from_typed::<f32>(compact);
            linalg::svd(session, &input, driver)
        }
        TensorView::F64(view) => {
            let compact = session.to_contiguous(&view)?;
            let input = Tensor::from_typed::<f64>(compact);
            linalg::svd(session, &input, driver)
        }
        TensorView::C32(view) => {
            let compact = session.to_contiguous(&view)?;
            let input = Tensor::from_typed::<num_complex::Complex32>(compact);
            linalg::svd(session, &input, driver)
        }
        TensorView::C64(view) => {
            let compact = session.to_contiguous(&view)?;
            let input = Tensor::from_typed::<num_complex::Complex64>(compact);
            linalg::svd(session, &input, driver)
        }
        TensorView::I32(_) | TensorView::I64(_) | TensorView::Bool(_) => {
            Err(unsupported_dtype("svd", input.dtype()))
        }
    }
}

fn svd_values_read_with_driver(
    session: &mut CudaExecSession<'_>,
    input: TensorRead<'_>,
    driver: SvdDriver,
) -> tenferro_tensor::Result<Tensor> {
    let input = input.tensor_view();
    match input {
        TensorView::F32(view) => session.to_contiguous(&view).and_then(|input| {
            linalg::svd_values(session, &Tensor::from_typed::<f32>(input), driver)
        }),
        TensorView::F64(view) => session.to_contiguous(&view).and_then(|input| {
            linalg::svd_values(session, &Tensor::from_typed::<f64>(input), driver)
        }),
        TensorView::C32(view) => session.to_contiguous(&view).and_then(|input| {
            linalg::svd_values(
                session,
                &Tensor::from_typed::<num_complex::Complex32>(input),
                driver,
            )
        }),
        TensorView::C64(view) => session.to_contiguous(&view).and_then(|input| {
            linalg::svd_values(
                session,
                &Tensor::from_typed::<num_complex::Complex64>(input),
                driver,
            )
        }),
        TensorView::I32(_) | TensorView::I64(_) | TensorView::Bool(_) => {
            Err(unsupported_dtype("svd_values", input.dtype()))
        }
    }
}

/// cuSOLVER needs compact column-major device storage, so a borrowed view is
/// made contiguous on the device and never crosses the host boundary.
fn eigh_read_with_driver(
    session: &mut CudaExecSession<'_>,
    input: TensorRead<'_>,
    driver: EighDriver,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let input = input.tensor_view();
    match input {
        TensorView::F32(view) => {
            let compact = session.to_contiguous(&view)?;
            let input = Tensor::from_typed::<f32>(compact);
            linalg::eigh(session, &input, driver)
        }
        TensorView::F64(view) => {
            let compact = session.to_contiguous(&view)?;
            let input = Tensor::from_typed::<f64>(compact);
            linalg::eigh(session, &input, driver)
        }
        TensorView::C32(view) => {
            let compact = session.to_contiguous(&view)?;
            let input = Tensor::from_typed::<num_complex::Complex32>(compact);
            linalg::eigh(session, &input, driver)
        }
        TensorView::C64(view) => {
            let compact = session.to_contiguous(&view)?;
            let input = Tensor::from_typed::<num_complex::Complex64>(compact);
            linalg::eigh(session, &input, driver)
        }
        TensorView::I32(_) | TensorView::I64(_) | TensorView::Bool(_) => {
            Err(unsupported_dtype("eigh", input.dtype()))
        }
    }
}

fn eigh_values_read_with_driver(
    session: &mut CudaExecSession<'_>,
    input: TensorRead<'_>,
    driver: EighDriver,
) -> tenferro_tensor::Result<Tensor> {
    let input = input.tensor_view();
    match input {
        TensorView::F32(view) => session.to_contiguous(&view).and_then(|input| {
            linalg::eigh_values(session, &Tensor::from_typed::<f32>(input), driver)
        }),
        TensorView::F64(view) => session.to_contiguous(&view).and_then(|input| {
            linalg::eigh_values(session, &Tensor::from_typed::<f64>(input), driver)
        }),
        TensorView::C32(view) => session.to_contiguous(&view).and_then(|input| {
            linalg::eigh_values(
                session,
                &Tensor::from_typed::<num_complex::Complex32>(input),
                driver,
            )
        }),
        TensorView::C64(view) => session.to_contiguous(&view).and_then(|input| {
            linalg::eigh_values(
                session,
                &Tensor::from_typed::<num_complex::Complex64>(input),
                driver,
            )
        }),
        TensorView::I32(_) | TensorView::I64(_) | TensorView::Bool(_) => {
            Err(unsupported_dtype("eigh_values", input.dtype()))
        }
    }
}

impl LinalgBackend for CudaExecSession<'_> {
    fn cholesky(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        linalg::cholesky(self, input)
    }

    fn triangular_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> tenferro_tensor::Result<Tensor> {
        linalg::triangular_solve(self, a, b, left_side, lower, transpose_a, unit_diagonal)
    }

    fn lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        linalg::lu(self, input)
    }

    fn lu_factor(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        linalg::lu_factor(self, input)
    }

    fn full_piv_lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        linalg::full_piv_lu(self, input)
    }

    fn full_piv_lu_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        transpose_a: bool,
    ) -> tenferro_tensor::Result<Tensor> {
        linalg::full_piv_lu_solve(self, a, b, transpose_a)
    }

    fn svd(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        linalg::svd(self, input, SvdDriver::Auto)
    }

    fn svd_with_options(
        &mut self,
        input: &Tensor,
        options: SvdOptions,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        validate_derivative_eps("svd_with_options", options.derivative_eps)?;
        let mut outputs = linalg::svd(self, input, options.driver)?;
        apply_svd_gauge(options.gauge, &mut outputs)?;
        Ok(outputs)
    }

    fn svd_full(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        linalg::svd_full(self, input)
    }

    fn svd_values(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        linalg::svd_values(self, input, SvdDriver::Auto)
    }

    fn svd_values_with_driver(
        &mut self,
        input: &Tensor,
        driver: SvdDriver,
    ) -> tenferro_tensor::Result<Tensor> {
        linalg::svd_values(self, input, driver)
    }

    fn svd_full_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        // Matches the existing CUDA `svd_read` contract: cuSOLVER needs compact
        // column-major device storage, so a view is canonicalized on the device
        // (never transferred across the host boundary) and then decomposed.
        let input = input.tensor_view();
        match input {
            TensorView::F32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<f32>(compact);
                self.svd_full(&input)
            }
            TensorView::F64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<f64>(compact);
                self.svd_full(&input)
            }
            TensorView::C32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<num_complex::Complex32>(compact);
                self.svd_full(&input)
            }
            TensorView::C64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<num_complex::Complex64>(compact);
                self.svd_full(&input)
            }
            TensorView::I32(_) | TensorView::I64(_) | TensorView::Bool(_) => {
                Err(unsupported_dtype("svd_full", input.dtype()))
            }
        }
    }

    fn svd_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        svd_read_with_driver(self, input, SvdDriver::Auto)
    }

    fn svd_with_options_read(
        &mut self,
        input: TensorRead<'_>,
        options: SvdOptions,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        validate_derivative_eps("svd_with_options_read", options.derivative_eps)?;
        let mut outputs = svd_read_with_driver(self, input, options.driver)?;
        apply_svd_gauge(options.gauge, &mut outputs)?;
        Ok(outputs)
    }

    fn svd_values_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        svd_values_read_with_driver(self, input, SvdDriver::Auto)
    }

    fn svd_values_with_driver_read(
        &mut self,
        input: TensorRead<'_>,
        driver: SvdDriver,
    ) -> tenferro_tensor::Result<Tensor> {
        svd_values_read_with_driver(self, input, driver)
    }

    fn qr(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        linalg::qr(self, input)
    }

    fn qr_with_options(
        &mut self,
        input: &Tensor,
        options: QrOptions,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        linalg::qr_with_options(self, input, options)
    }

    fn qr_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = input.tensor_view();
        match input {
            TensorView::F32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<f32>(compact);
                self.qr(&input)
            }
            TensorView::F64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<f64>(compact);
                self.qr(&input)
            }
            TensorView::C32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<num_complex::Complex32>(compact);
                self.qr(&input)
            }
            TensorView::C64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<num_complex::Complex64>(compact);
                self.qr(&input)
            }
            TensorView::I32(_) | TensorView::I64(_) | TensorView::Bool(_) => {
                Err(unsupported_dtype("qr", input.dtype()))
            }
        }
    }

    fn rank_revealing_qr(
        &mut self,
        input: &Tensor,
        options: RankRevealingQrOptions,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        linalg::rank_revealing_qr(self, input, options)
    }

    fn rank_revealing_qr_read(
        &mut self,
        input: TensorRead<'_>,
        options: RankRevealingQrOptions,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = input.tensor_view();
        match input {
            TensorView::F32(view) => self.to_contiguous(&view).and_then(|input| {
                self.rank_revealing_qr(&Tensor::from_typed::<f32>(input), options)
            }),
            TensorView::F64(view) => self.to_contiguous(&view).and_then(|input| {
                self.rank_revealing_qr(&Tensor::from_typed::<f64>(input), options)
            }),
            TensorView::C32(view) => self.to_contiguous(&view).and_then(|input| {
                self.rank_revealing_qr(
                    &Tensor::from_typed::<num_complex::Complex32>(input),
                    options,
                )
            }),
            TensorView::C64(view) => self.to_contiguous(&view).and_then(|input| {
                self.rank_revealing_qr(
                    &Tensor::from_typed::<num_complex::Complex64>(input),
                    options,
                )
            }),
            TensorView::I32(_) | TensorView::I64(_) | TensorView::Bool(_) => {
                Err(unsupported_dtype("rank_revealing_qr", input.dtype()))
            }
        }
    }

    fn householder_qr(
        &mut self,
        input: &Tensor,
    ) -> tenferro_tensor::Result<crate::backend::CompactQrResult> {
        linalg::householder_qr(self, input)
    }

    fn householder_qr_from_factors(
        &mut self,
        q: &Tensor,
        r: &Tensor,
    ) -> tenferro_tensor::Result<crate::backend::CompactQrResult> {
        linalg::householder_qr_from_factors(self, q, r)
    }

    fn householder_qr_append(
        &mut self,
        packed: &Tensor,
        coeff: &Tensor,
        block: &Tensor,
    ) -> tenferro_tensor::Result<crate::backend::CompactQrResult> {
        linalg::householder_qr_append(self, packed, coeff, block)
    }

    fn householder_qr_q_columns(
        &mut self,
        packed: &Tensor,
        coeff: &Tensor,
        range: std::ops::Range<usize>,
        options: QrOptions,
    ) -> tenferro_tensor::Result<Tensor> {
        linalg::householder_qr_q_columns(self, packed, coeff, range.start, range.end, options)
    }

    fn householder_qr_r(
        &mut self,
        packed: &Tensor,
        coeff: &Tensor,
        options: QrOptions,
    ) -> tenferro_tensor::Result<Tensor> {
        linalg::householder_qr_r(self, packed, coeff, options)
    }

    fn qr_with_options_read(
        &mut self,
        input: TensorRead<'_>,
        options: QrOptions,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = input.tensor_view();
        match input {
            TensorView::F32(view) => {
                let compact = self.to_contiguous(&view)?;
                linalg::qr_with_options(self, &Tensor::from_typed::<f32>(compact), options)
            }
            TensorView::F64(view) => {
                let compact = self.to_contiguous(&view)?;
                linalg::qr_with_options(self, &Tensor::from_typed::<f64>(compact), options)
            }
            TensorView::C32(view) => {
                let compact = self.to_contiguous(&view)?;
                linalg::qr_with_options(
                    self,
                    &Tensor::from_typed::<num_complex::Complex32>(compact),
                    options,
                )
            }
            TensorView::C64(view) => {
                let compact = self.to_contiguous(&view)?;
                linalg::qr_with_options(
                    self,
                    &Tensor::from_typed::<num_complex::Complex64>(compact),
                    options,
                )
            }
            TensorView::I32(_) | TensorView::I64(_) | TensorView::Bool(_) => {
                Err(unsupported_dtype("qr_with_options_read", input.dtype()))
            }
        }
    }

    fn eigh(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        linalg::eigh(self, input, EighDriver::Auto)
    }

    fn eigh_with_options(
        &mut self,
        input: &Tensor,
        options: EighOptions,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        validate_derivative_eps("eigh_with_options", options.derivative_eps)?;
        let mut outputs = linalg::eigh(self, input, options.driver)?;
        apply_eigh_gauge(options.gauge, &mut outputs)?;
        Ok(outputs)
    }

    fn eigh_with_options_read(
        &mut self,
        input: TensorRead<'_>,
        options: EighOptions,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        validate_derivative_eps("eigh_with_options_read", options.derivative_eps)?;
        let mut outputs = eigh_read_with_driver(self, input, options.driver)?;
        apply_eigh_gauge(options.gauge, &mut outputs)?;
        Ok(outputs)
    }

    fn eigh_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        eigh_read_with_driver(self, input, EighDriver::Auto)
    }

    fn cholesky_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        let input = input.tensor_view();
        match input {
            TensorView::F32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<f32>(compact);
                self.cholesky(&input)
            }
            TensorView::F64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<f64>(compact);
                self.cholesky(&input)
            }
            TensorView::C32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<num_complex::Complex32>(compact);
                self.cholesky(&input)
            }
            TensorView::C64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<num_complex::Complex64>(compact);
                self.cholesky(&input)
            }
            TensorView::I32(_) | TensorView::I64(_) | TensorView::Bool(_) => {
                Err(unsupported_dtype("cholesky", input.dtype()))
            }
        }
    }

    fn lu_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = input.tensor_view();
        match input {
            TensorView::F32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<f32>(compact);
                self.lu(&input)
            }
            TensorView::F64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<f64>(compact);
                self.lu(&input)
            }
            TensorView::C32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<num_complex::Complex32>(compact);
                self.lu(&input)
            }
            TensorView::C64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<num_complex::Complex64>(compact);
                self.lu(&input)
            }
            TensorView::I32(_) | TensorView::I64(_) | TensorView::Bool(_) => {
                Err(unsupported_dtype("lu", input.dtype()))
            }
        }
    }

    fn full_piv_lu_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = input.tensor_view();
        match input {
            TensorView::F32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<f32>(compact);
                self.full_piv_lu(&input)
            }
            TensorView::F64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<f64>(compact);
                self.full_piv_lu(&input)
            }
            TensorView::C32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<num_complex::Complex32>(compact);
                self.full_piv_lu(&input)
            }
            TensorView::C64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<num_complex::Complex64>(compact);
                self.full_piv_lu(&input)
            }
            TensorView::I32(_) | TensorView::I64(_) | TensorView::Bool(_) => {
                Err(unsupported_dtype("full_piv_lu", input.dtype()))
            }
        }
    }

    fn eig_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = input.tensor_view();
        match input {
            TensorView::F32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<f32>(compact);
                self.eig(&input)
            }
            TensorView::F64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<f64>(compact);
                self.eig(&input)
            }
            TensorView::C32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<num_complex::Complex32>(compact);
                self.eig(&input)
            }
            TensorView::C64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::from_typed::<num_complex::Complex64>(compact);
                self.eig(&input)
            }
            TensorView::I32(_) | TensorView::I64(_) | TensorView::Bool(_) => {
                Err(unsupported_dtype("eig", input.dtype()))
            }
        }
    }

    fn eigh_values(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        linalg::eigh_values(self, input, EighDriver::Auto)
    }

    fn eigh_values_with_driver(
        &mut self,
        input: &Tensor,
        driver: EighDriver,
    ) -> tenferro_tensor::Result<Tensor> {
        linalg::eigh_values(self, input, driver)
    }

    fn eigh_values_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        eigh_values_read_with_driver(self, input, EighDriver::Auto)
    }

    fn eigh_values_with_driver_read(
        &mut self,
        input: TensorRead<'_>,
        driver: EighDriver,
    ) -> tenferro_tensor::Result<Tensor> {
        eigh_values_read_with_driver(self, input, driver)
    }

    fn eig(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        linalg::eig(self, input)
    }

    fn solve(&mut self, a: &Tensor, b: &Tensor) -> tenferro_tensor::Result<Tensor> {
        linalg::solve(self, a, b)
    }

    fn lu_solve_prepared(
        &mut self,
        a: &Tensor,
        packed_lu: &Tensor,
        pivots: &Tensor,
        b: &Tensor,
        transpose_a: bool,
        conjugate_a: bool,
    ) -> tenferro_tensor::Result<Tensor> {
        linalg::lu_solve_prepared(self, a, packed_lu, pivots, b, transpose_a, conjugate_a)
    }
}
