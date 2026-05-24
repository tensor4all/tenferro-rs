use std::collections::BTreeMap;

use tenferro::{DotGeneralConfig, GraphCompiler, TracedTensor};

use crate::dispatch::NamedTensor;

pub type ObservableResult = Result<Vec<NamedTensor>, String>;

pub fn apply_observable(
    kind: &str,
    outputs: Vec<NamedTensor>,
    _compiler: &mut GraphCompiler,
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
            let product = batched_matmul_preserve_batch_prefix(u, vh)?;
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

fn batched_matmul_preserve_batch_prefix(
    lhs: &TracedTensor,
    rhs: &TracedTensor,
) -> Result<TracedTensor, String> {
    if lhs.rank < 2 || rhs.rank < 2 {
        return Err("svd_uvh_product expects rank >= 2 inputs".to_string());
    }
    let batch_rank = lhs.rank - 2;
    if rhs.rank != lhs.rank {
        return Err(format!(
            "svd_uvh_product expects matching ranks, got lhs={} rhs={}",
            lhs.rank, rhs.rank
        ));
    }

    let product = lhs.dot_general(
        rhs,
        DotGeneralConfig {
            lhs_contracting_dims: vec![batch_rank + 1],
            rhs_contracting_dims: vec![batch_rank],
            lhs_batch_dims: (0..batch_rank).collect(),
            rhs_batch_dims: (0..batch_rank).collect(),
        },
    );

    if batch_rank == 0 {
        return Ok(product);
    }

    let mut perm: Vec<usize> = (2..2 + batch_rank).collect();
    perm.push(0);
    perm.push(1);
    Ok(product.transpose(&perm))
}
