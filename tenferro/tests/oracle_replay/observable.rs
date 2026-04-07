use std::collections::BTreeMap;

use tenferro::einsum::einsum;
use tenferro::engine::Engine;
use tenferro::{CpuBackend, TracedTensor};

use crate::dispatch::NamedTensor;

pub type ObservableResult = Result<Vec<NamedTensor>, String>;

pub fn apply_observable(
    kind: &str,
    outputs: Vec<NamedTensor>,
    engine: &mut Engine<CpuBackend>,
) -> ObservableResult {
    let outputs = output_map(outputs);
    match kind {
        "identity" => Ok(outputs
            .into_iter()
            .map(|(name, tensor)| named(&name, tensor))
            .collect()),
        "svd_s" => Ok(vec![named("s", required(&outputs, "s")?.clone())]),
        "svd_u_abs" => Ok(vec![named("u", required(&outputs, "u")?.abs())]),
        "svd_vh_abs" => Ok(vec![
            named("s", required(&outputs, "s")?.clone()),
            named("vh", required(&outputs, "vh")?.abs()),
        ]),
        "svd_uvh_product" => {
            let u = required(&outputs, "u")?;
            let s = required(&outputs, "s")?;
            let vh = required(&outputs, "vh")?;
            let subscripts = svd_uvh_subscripts(u.shape.len().saturating_sub(2))?;
            let product = einsum(engine, &[u, vh], &subscripts).map_err(|err| err.to_string())?;
            Ok(vec![named("s", s.clone()), named("uvh", product)])
        }
        "eigh_values_vectors_abs" => Ok(vec![
            named("values", required(&outputs, "values")?.clone()),
            named("vectors", required(&outputs, "vectors")?.abs()),
        ]),
        "eig_values_vectors_abs" => Ok(vec![
            named("values", required(&outputs, "values")?.clone()),
            named("vectors", required(&outputs, "vectors")?.abs()),
        ]),
        other => Err(format!("unsupported observable kind {other}")),
    }
}

fn output_map(outputs: Vec<NamedTensor>) -> BTreeMap<String, TracedTensor> {
    outputs
        .into_iter()
        .map(|output| (output.name, output.tensor))
        .collect()
}

fn required<'a>(
    outputs: &'a BTreeMap<String, TracedTensor>,
    name: &str,
) -> Result<&'a TracedTensor, String> {
    outputs
        .get(name)
        .ok_or_else(|| format!("observable is missing output {name}"))
}

fn named(name: &str, tensor: TracedTensor) -> NamedTensor {
    NamedTensor {
        name: name.to_string(),
        tensor,
    }
}

fn svd_uvh_subscripts(batch_rank: usize) -> Result<String, String> {
    const BATCH_AXES: &[u8] = b"abcdefghlmnopqrstuvwxyzABCDEFGHLMNOPQRSTUVWXYZ";
    if batch_rank > BATCH_AXES.len() {
        return Err(format!(
            "svd_uvh_product batch rank {batch_rank} exceeds supported axis budget"
        ));
    }

    let batch: String = BATCH_AXES
        .iter()
        .take(batch_rank)
        .map(|axis| char::from(*axis))
        .collect();
    Ok(format!("{batch}ij,{batch}jk->{batch}ik"))
}
