//! One forwarding case per (operation, layer, input dtype): the plain call
//! and the call with default options must agree bit for bit.

use tenferro_cpu::CpuEngine;
use tenferro_runtime::TraceContext;

use super::support::*;
#[cfg(feature = "autodiff")]
use crate::EagerFftOps;
use crate::{FftOps, FftOptions, FftnOptions, TracedFftOps};

/// A forwarding case.
pub(super) struct Case {
    pub op: &'static str,
    pub layer: &'static str,
    pub input: &'static str,
    pub run: Box<dyn Fn() -> TestResult<Pair>>,
}

/// Every forwarding case.
pub(super) fn all() -> Vec<Case> {
    let mut cases = Vec::new();
    macro_rules! push {
        ($op:ident, $with:ident, $opts:ident, $input:ident) => {
            cases.push(Case {
                op: stringify!($op),
                layer: "concrete",
                input: stringify!($input),
                run: Box::new(|| {
                    let a = $input()?;
                    let mut e = CpuEngine::serial();
                    let p = a.$op(&mut e)?;
                    let w = a.$with(&$opts::default(), &mut e)?;
                    Ok((tensor_bits(&p)?, tensor_bits(&w)?))
                }),
            });
            cases.push(Case {
                op: stringify!($op),
                layer: "traced",
                input: stringify!($input),
                run: Box::new(|| {
                    let trace = TraceContext::new();
                    let a = trace.from_tensor_concrete_shape($input()?)?;
                    let p = a.$op()?;
                    let w = a.$with(&$opts::default())?;
                    traced_pair(&p, &w)
                }),
            });
            #[cfg(feature = "autodiff")]
            cases.push(Case {
                op: stringify!($op),
                layer: "eager",
                input: stringify!($input),
                run: Box::new(|| {
                    let runtime = tenferro_ad::EagerRuntime::new()?;
                    let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
                        let a = s.constant_from($input()?)?;
                        let p = s.$op(&a)?;
                        let w = s.$with(&a, &$opts::default())?;
                        Ok::<_, tenferro_ad::Error>((p, w))
                    })?;
                    let mut e = CpuEngine::serial();
                    Ok((
                        tensor_bits(
                            &p.runtime()
                                .with_eager_session(&mut e, |s| s.duplicate_value(&p))?,
                        )?,
                        tensor_bits(
                            &w.runtime()
                                .with_eager_session(&mut e, |s| s.duplicate_value(&w))?,
                        )?,
                    ))
                }),
            });
        };
    }
    push!(fft, fft_with, FftOptions, real);
    push!(fft, fft_with, FftOptions, complex);
    push!(ifft, ifft_with, FftOptions, complex);
    push!(rfft, rfft_with, FftOptions, real);
    push!(irfft, irfft_with, FftOptions, complex);
    push!(fftn, fftn_with, FftnOptions, real);
    push!(fftn, fftn_with, FftnOptions, complex);
    push!(ifftn, ifftn_with, FftnOptions, complex);
    push!(rfftn, rfftn_with, FftnOptions, real);
    push!(irfftn, irfftn_with, FftnOptions, complex);
    cases
}
