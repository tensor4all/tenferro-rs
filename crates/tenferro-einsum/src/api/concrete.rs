//! Concrete forward bodies of the registry (`EinsumOps`, `TensordotOps`) and
//! the grammar of [`ConcreteEinsumPlan`].
//!
//! Default options (automatic order without annealing, no conjugation) with an
//! overwriting or fresh result take the former entry for the input kind, so
//! results are bit-identical to the former API: inputs that are all owned
//! erased tensors take the owned route, every other input list (views, typed
//! tensors) the borrowed-read route. Every other choice runs through a
//! [`ConcreteEinsumPlan`] prepared with that order and conjugation.

use tenferro_tensor::{
    AsView, Conj, Dest, DestMode, Engine, Repr, Tensor, TensorRead, TensorWrite,
};

use super::options::{ConjMask, EinsumInputs, EinsumOptions, EinsumSpec, Spec, TensordotOptions};
use crate::concrete::{
    read_input_specs, resolve_read_notation, OwnedEinsum, OwnedEinsumInto, ReadEinsum,
    ReadEinsumInto,
};
use crate::{parse_einsum_notation, ConcreteEinsumPlan, Error, Subscripts, TensorDotAxes};

type Res<T> = tenferro_tensor::Result<T>;
type Val<I> = <<I as EinsumInputs>::Repr as Repr>::Value;
type Write<'a, I> = <<I as EinsumInputs>::Repr as Repr>::Write<'a>;

const OP: &str = "einsum";

pub(crate) fn tensor_err(error: Error) -> tenferro_tensor::Error {
    error.into_tensor_error(OP)
}

/// The inputs as owned erased tensors, when every one of them is exactly that.
fn all_owned<'a>(reads: &[TensorRead<'a>]) -> Option<Vec<&'a Tensor>> {
    reads.iter().map(TensorRead::as_tensor).collect()
}

/// Integer subscripts of `spec` for inputs of the given shapes.
pub(crate) fn resolve(spec: Spec<'_>, reads: &[TensorRead<'_>]) -> crate::Result<Subscripts> {
    match spec {
        Spec::Text(text) => resolve_read_notation(reads, &parse_einsum_notation(text)?),
        Spec::Notation(notation) => resolve_read_notation(reads, notation),
        Spec::Subscripts(subscripts) => Ok(Subscripts::from(subscripts)),
    }
}

/// Conjugated copies of the operands `mask` selects; `None` for the others.
fn conjugate(
    reads: &[TensorRead<'_>],
    mask: &ConjMask,
    exec: &mut dyn Engine,
) -> Res<Vec<Option<Tensor>>> {
    check_mask(mask, reads.len())?;
    reads
        .iter()
        .enumerate()
        .map(|(i, read)| {
            if mask.get(i).is_yes() {
                exec.conj_read(read.clone()).map(Some)
            } else {
                Ok(None)
            }
        })
        .collect()
}

fn check_mask(mask: &ConjMask, inputs: usize) -> Res<()> {
    if mask.len() > inputs {
        return Err(tensor_err(Error::invalid_argument(
            OP,
            "conj",
            format!(
                "the conjugation mask has {} entries for {inputs} inputs",
                mask.len()
            ),
        )));
    }
    Ok(())
}

/// `reads` with the conjugated copies substituted.
fn substituted<'a>(reads: &[TensorRead<'a>], conj: &'a [Option<Tensor>]) -> Vec<TensorRead<'a>> {
    reads
        .iter()
        .zip(conj)
        .map(|(read, c)| match c {
            Some(t) => TensorRead::from_tensor(t),
            None => read.clone(),
        })
        .collect()
}

// ---------------------------------------------------------------------- einsum

/// The former entries for default options.
fn einsum_plain(
    reads: &[TensorRead<'_>],
    spec: Spec<'_>,
    exec: &mut dyn Engine,
) -> crate::Result<Tensor> {
    match all_owned(reads) {
        Some(owned) => {
            let owned: &[&Tensor] = &owned;
            match spec {
                Spec::Text(t) => OwnedEinsum::einsum(owned, t, exec),
                Spec::Notation(n) => OwnedEinsum::einsum_notation(owned, n, exec),
                Spec::Subscripts(s) => OwnedEinsum::einsum_subscripts(owned, s, exec),
            }
        }
        None => match spec {
            Spec::Text(t) => ReadEinsum::einsum_read(reads, t, exec),
            Spec::Notation(n) => ReadEinsum::einsum_read_notation(reads, n, exec),
            Spec::Subscripts(s) => ReadEinsum::einsum_read_subscripts(reads, s, exec),
        },
    }
}

pub(crate) fn einsum_erased(
    reads: &[TensorRead<'_>],
    spec: Spec<'_>,
    options: &EinsumOptions,
    exec: &mut dyn Engine,
) -> Res<Tensor> {
    if options.is_plain() {
        return einsum_plain(reads, spec, exec).map_err(tensor_err);
    }
    let conj = conjugate(reads, &options.conj, exec)?;
    let reads = substituted(reads, &conj);
    let subscripts = resolve(spec, &reads).map_err(tensor_err)?;
    let plan = ConcreteEinsumPlan::prepare_planned(
        read_input_specs(&reads),
        &subscripts,
        &options.optimize,
        ConjMask::default(),
    )
    .map_err(tensor_err)?;
    plan.run_reads(&reads[..], exec).map_err(tensor_err)
}

pub(crate) fn einsum<I: EinsumInputs + ?Sized>(
    inputs: &I,
    spec: &(impl EinsumSpec + ?Sized),
    options: &EinsumOptions,
    exec: &mut dyn Engine,
) -> Res<Val<I>> {
    let reads = inputs.reads();
    let result = einsum_erased(&reads, spec.spec(), options, exec)?;
    <I::Repr as Repr>::wrap_value(result)
}

/// The former overwrite entries for default options.
fn einsum_into_plain(
    reads: &[TensorRead<'_>],
    spec: Spec<'_>,
    exec: &mut dyn Engine,
    out: TensorWrite<'_>,
) -> crate::Result<()> {
    match all_owned(reads) {
        Some(owned) => {
            let owned: &[&Tensor] = &owned;
            match spec {
                Spec::Text(t) => OwnedEinsumInto::einsum_into(owned, t, exec, out),
                Spec::Notation(n) => OwnedEinsumInto::einsum_into_notation(owned, n, exec, out),
                Spec::Subscripts(s) => OwnedEinsumInto::einsum_into_subscripts(owned, s, exec, out),
            }
        }
        None => match spec {
            Spec::Text(t) => ReadEinsumInto::einsum_read_into(reads, t, exec, out),
            Spec::Notation(n) => ReadEinsumInto::einsum_read_into_notation(reads, n, exec, out),
            Spec::Subscripts(s) => ReadEinsumInto::einsum_read_into_subscripts(reads, s, exec, out),
        },
    }
}

pub(crate) fn einsum_into<I: EinsumInputs + ?Sized>(
    inputs: &I,
    spec: &(impl EinsumSpec + ?Sized),
    options: &EinsumOptions,
    exec: &mut dyn Engine,
    dest: Dest<Write<'_, I>>,
) -> Res<()> {
    let reads = inputs.reads();
    let spec = spec.spec();
    if options.is_plain() && dest.mode() == DestMode::Overwrite {
        let out = <I::Repr as Repr>::erase_write(dest.into_target());
        return einsum_into_plain(&reads, spec, exec, out).map_err(tensor_err);
    }
    check_mask(&options.conj, reads.len())?;
    let subscripts = resolve(spec, &reads).map_err(tensor_err)?;
    let plan = ConcreteEinsumPlan::prepare_planned(
        read_input_specs(&reads),
        &subscripts,
        &options.optimize,
        options.conj.clone(),
    )
    .map_err(tensor_err)?;
    plan_into_reads::<I::Repr>(&plan, &reads, exec, dest)
}

// ------------------------------------------------------------------ the plan

/// Runs `plan` on borrowed reads, applying its conjugation.
pub(crate) fn plan_reads(
    plan: &ConcreteEinsumPlan,
    reads: &[TensorRead<'_>],
    exec: &mut dyn Engine,
) -> Res<Tensor> {
    if plan.conj_mask().is_none() {
        return match all_owned(reads) {
            Some(owned) => plan.run_owned(&owned[..], exec),
            None => plan.run_reads(reads, exec),
        }
        .map_err(tensor_err);
    }
    let conj = conjugate(reads, plan.conj_mask(), exec)?;
    let reads = substituted(reads, &conj);
    plan.run_reads(&reads[..], exec).map_err(tensor_err)
}

/// Writes `plan` on borrowed reads into a destination of representation `R`.
///
/// The selected operands are conjugated first, exactly as [`plan_reads`] and
/// the owned entrance do, so `einsum` and `einsum_into` with the same options
/// give equal values. Only the sign of an exact zero may differ between the
/// allocating and `_into` routes: on some hosts (notably x86_64) the fresh
/// route and the initialized beta-0 route round an exactly cancelled
/// imaginary part to `+0.0` and `-0.0` respectively. The conjugation never travels as contraction
/// operand flags: the flagged kernel evaluates `conj(a) conj(b)` as
/// `conj(a b)`, which turns an exactly cancelled zero into `-0.0` where the
/// owned entrance yields `+0.0` (#2580). Overwrite then writes the read
/// route; accumulation is one unconjugated accumulating contraction.
pub(crate) fn plan_into_reads<R: Repr>(
    plan: &ConcreteEinsumPlan,
    reads: &[TensorRead<'_>],
    exec: &mut dyn Engine,
    dest: Dest<R::Write<'_>>,
) -> Res<()> {
    let mask = plan.conj_mask();
    if mask.is_none() && dest.mode() == DestMode::Overwrite {
        let out = R::erase_write(dest.into_target());
        return match all_owned(reads) {
            Some(owned) => plan.run_owned_into(&owned[..], exec, out),
            None => plan.run_reads_into(reads, exec, out),
        }
        .map_err(tensor_err);
    }
    let conj = conjugate(reads, mask, exec)?;
    let reads = substituted(reads, &conj);
    if dest.mode() == DestMode::Overwrite {
        let out = R::erase_write(dest.into_target());
        return plan
            .run_reads_into(&reads[..], exec, out)
            .map_err(tensor_err);
    }
    let dtype = reads.first().map(TensorRead::dtype).ok_or_else(|| {
        tensor_err(Error::invalid_argument(
            OP,
            "inputs",
            "einsum requires at least one input tensor",
        ))
    })?;
    let (out, accumulation) = dest.into_accumulation(dtype, Conj::No, Conj::No)?;
    let out = R::erase_write(out);
    plan.run_reads_into_accum(&reads[..], exec, accumulation, out)
        .map_err(tensor_err)
}

// ------------------------------------------------------------------- tensordot

pub(crate) fn tensordot<X: AsView + ?Sized>(
    a: &X,
    b: &impl AsView<Repr = X::Repr>,
    axes: TensorDotAxes<'_>,
    _options: &TensordotOptions,
    exec: &mut dyn Engine,
) -> Res<<X::Repr as Repr>::Value> {
    let (a, b) = (a.as_read(), b.as_read());
    let result = match (a.as_tensor(), b.as_tensor()) {
        (Some(ta), Some(tb)) => crate::concrete::tensordot_owned(ta, tb, axes, exec),
        _ => crate::concrete::tensordot_reads(a, b, axes, exec),
    }
    .map_err(|e| e.into_tensor_error("tensordot"))?;
    <X::Repr as Repr>::wrap_value(result)
}
