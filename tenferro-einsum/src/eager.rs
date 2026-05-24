use std::borrow::Cow;
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::env;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use tenferro_tensor::{
    DotGeneralConfig, Error, Result, Tensor, TensorBackend, TensorExec, TensorRead, TensorView,
};

use crate::{ContractionTree, Subscripts};

const EAGER_EINSUM_OP: &str = "eager_einsum";

#[derive(Debug, Default, Clone)]
struct EagerEinsumProfileEntry {
    calls: usize,
    total_time: Duration,
}

thread_local! {
    static EAGER_EINSUM_PROFILE_STATE: RefCell<HashMap<&'static str, EagerEinsumProfileEntry>> =
        RefCell::new(HashMap::new());
}

fn eager_einsum_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env::var("TENFERRO_PROFILE_EAGER_EINSUM_AGG").is_ok())
}

fn eager_einsum_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env::var("TENFERRO_PROFILE_EAGER_EINSUM").is_ok())
}

fn record_eager_einsum_profile(section: &'static str, elapsed: Duration) {
    if !eager_einsum_profile_enabled() {
        return;
    }
    EAGER_EINSUM_PROFILE_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let entry = state.entry(section).or_default();
        entry.calls += 1;
        entry.total_time += elapsed;
    });
}

fn profile_eager_einsum_section<T>(section: &'static str, f: impl FnOnce() -> T) -> T {
    if !eager_einsum_profile_enabled() {
        return f();
    }
    let started = Instant::now();
    let result = f();
    record_eager_einsum_profile(section, started.elapsed());
    result
}

fn maybe_print_eager_einsum_profile() {
    if !eager_einsum_profile_enabled() {
        return;
    }
    let Ok(print_every) = env::var("TENFERRO_PROFILE_EAGER_EINSUM_PRINT_EVERY") else {
        return;
    };
    let Ok(print_every) = print_every.parse::<usize>() else {
        return;
    };
    if print_every == 0 {
        return;
    }

    let should_print = EAGER_EINSUM_PROFILE_STATE.with(|state| {
        state
            .borrow()
            .get("total")
            .is_some_and(|entry| entry.calls % print_every == 0)
    });
    if should_print {
        print_and_reset_eager_einsum_profile();
    }
}

fn print_and_reset_eager_einsum_profile() {
    EAGER_EINSUM_PROFILE_STATE.with(|state| {
        let mut entries: Vec<_> = state
            .borrow()
            .iter()
            .map(|(section, entry)| (*section, entry.clone()))
            .collect();
        state.borrow_mut().clear();
        entries.sort_by_key(|(_, entry)| Reverse(entry.total_time));

        eprintln!("=== tenferro eager einsum profile ===");
        for (section, entry) in entries {
            eprintln!(
                "{section}: calls={} total={:.6}ms per_call={:.3}us",
                entry.calls,
                entry.total_time.as_secs_f64() * 1.0e3,
                entry.total_time.as_secs_f64() * 1.0e6 / entry.calls as f64,
            );
        }
    });
}

enum TensorValue<'a> {
    Borrowed(&'a Tensor),
    View(TensorView<'a>),
    Owned(Tensor),
}

impl TensorValue<'_> {
    fn as_tensor(&self) -> Option<&Tensor> {
        match self {
            Self::Borrowed(tensor) => Some(tensor),
            Self::View(_) => None,
            Self::Owned(tensor) => Some(tensor),
        }
    }

    fn into_tensor(self) -> Tensor {
        match self {
            Self::Borrowed(tensor) => tensor.clone(),
            Self::View(view) => view.to_tensor(),
            Self::Owned(tensor) => tensor,
        }
    }

    fn tensor_read(&self) -> TensorRead<'_> {
        match self {
            Self::Borrowed(tensor) => TensorRead::from_tensor(tensor),
            Self::View(view) => TensorRead::from_view(*view),
            Self::Owned(tensor) => TensorRead::from_tensor(tensor),
        }
    }

    fn reclaim_if_owned(self, exec: &mut dyn TensorExec) {
        if let Self::Owned(tensor) = self {
            exec.reclaim_buffer(tensor);
        }
    }
}

struct LabeledTensor<'a> {
    tensor: TensorValue<'a>,
    labels: Vec<u32>,
}

impl LabeledTensor<'_> {
    fn tensor(&self) -> Option<&Tensor> {
        self.tensor.as_tensor()
    }

    fn tensor_read(&self) -> TensorRead<'_> {
        self.tensor.tensor_read()
    }

    fn tensor_cow(&self) -> Cow<'_, Tensor> {
        match self.tensor() {
            Some(tensor) => Cow::Borrowed(tensor),
            None => Cow::Owned(self.tensor_read().to_tensor()),
        }
    }

    fn shape(&self) -> &[usize] {
        match &self.tensor {
            TensorValue::Borrowed(tensor) => tensor.shape(),
            TensorValue::View(view) => view.shape(),
            TensorValue::Owned(tensor) => tensor.shape(),
        }
    }

    fn reclaim_if_owned(self, exec: &mut dyn TensorExec) {
        self.tensor.reclaim_if_owned(exec);
    }
}

#[derive(Debug)]
struct BinaryDotFastPlan {
    result_labels: Vec<u32>,
    target_labels: Vec<u32>,
    config: DotGeneralConfig,
}

fn small_contains(labels: &[u32], label: u32) -> bool {
    labels.iter().any(|candidate| *candidate == label)
}

fn labels_are_unique(labels: &[u32]) -> bool {
    let mut seen = Vec::with_capacity(labels.len());
    for &label in labels {
        if small_contains(&seen, label) {
            return false;
        }
        seen.push(label);
    }
    true
}

fn try_build_binary_dot_fast_plan(
    lhs_labels: &[u32],
    rhs_labels: &[u32],
    output_labels: &[u32],
) -> Option<BinaryDotFastPlan> {
    if !labels_are_unique(lhs_labels)
        || !labels_are_unique(rhs_labels)
        || !labels_are_unique(output_labels)
    {
        return None;
    }

    let mut lhs_contracting_dims = Vec::new();
    let mut rhs_contracting_dims = Vec::new();
    let mut lhs_batch_dims = Vec::new();
    let mut rhs_batch_dims = Vec::new();
    let mut lhs_free_labels = Vec::new();
    let mut rhs_free_labels = Vec::new();
    let mut batch_labels = Vec::new();

    for (lhs_axis, &label) in lhs_labels.iter().enumerate() {
        let rhs_axis = rhs_labels.iter().position(|candidate| *candidate == label);
        let in_output = small_contains(output_labels, label);
        match (rhs_axis, in_output) {
            (Some(rhs_axis), true) => {
                lhs_batch_dims.push(lhs_axis);
                rhs_batch_dims.push(rhs_axis);
                batch_labels.push(label);
            }
            (Some(rhs_axis), false) => {
                lhs_contracting_dims.push(lhs_axis);
                rhs_contracting_dims.push(rhs_axis);
            }
            (None, true) => lhs_free_labels.push(label),
            (None, false) => return None,
        }
    }

    for &label in rhs_labels {
        if !small_contains(lhs_labels, label) {
            if small_contains(output_labels, label) {
                rhs_free_labels.push(label);
            } else {
                return None;
            }
        }
    }

    if lhs_contracting_dims.is_empty() {
        return None;
    }

    for &label in output_labels {
        if !small_contains(lhs_labels, label) && !small_contains(rhs_labels, label) {
            return None;
        }
    }

    let mut result_labels =
        Vec::with_capacity(lhs_free_labels.len() + rhs_free_labels.len() + batch_labels.len());
    result_labels.extend(lhs_free_labels);
    result_labels.extend(rhs_free_labels);
    result_labels.extend(batch_labels);

    Some(BinaryDotFastPlan {
        result_labels,
        target_labels: output_labels.to_vec(),
        config: DotGeneralConfig {
            lhs_contracting_dims,
            rhs_contracting_dims,
            lhs_batch_dims,
            rhs_batch_dims,
        },
    })
}

fn execute_binary_dot_fast_plan<'a>(
    exec: &mut dyn TensorExec,
    lhs: LabeledTensor<'a>,
    rhs: LabeledTensor<'a>,
    plan: BinaryDotFastPlan,
    reorder_result: bool,
) -> Result<LabeledTensor<'a>> {
    let tensor = profile_eager_einsum_section("binary.fast_dot_general", || {
        exec.dot_general_read(lhs.tensor_read(), rhs.tensor_read(), &plan.config)
    })?;
    lhs.reclaim_if_owned(exec);
    rhs.reclaim_if_owned(exec);

    let result = LabeledTensor {
        tensor: TensorValue::Owned(tensor),
        labels: plan.result_labels,
    };

    if !reorder_result || result.labels == plan.target_labels {
        return Ok(result);
    }
    transpose_to_labels(exec, result, &plan.target_labels)
}

fn eager_invalid_config(message: impl Into<String>) -> Error {
    Error::InvalidConfig {
        op: EAGER_EINSUM_OP,
        message: message.into(),
    }
}

fn plan_subscripts(subs: &Subscripts, input_shapes: &[&[usize]]) -> Result<ContractionTree> {
    if input_shapes.is_empty() {
        return Err(eager_invalid_config(
            "eager einsum requires at least one input tensor",
        ));
    }

    if subs.inputs.len() != input_shapes.len() {
        return Err(eager_invalid_config(format!(
            "eager einsum subscripts expect {} inputs, got {}",
            subs.inputs.len(),
            input_shapes.len()
        )));
    }

    ContractionTree::optimize(subs, input_shapes)
        .map_err(|err| eager_invalid_config(format!("failed to optimize contraction tree: {err}")))
}

fn take_labeled<'a>(
    labeled: &mut [Option<LabeledTensor<'a>>],
    index: usize,
    role: &'static str,
) -> Result<LabeledTensor<'a>> {
    labeled
        .get_mut(index)
        .ok_or_else(|| eager_invalid_config(format!("missing {role} operand at index {index}")))?
        .take()
        .ok_or_else(|| eager_invalid_config(format!("missing {role} operand at index {index}")))
}

fn find_label_axis(labels: &[u32], label: u32) -> Result<usize> {
    labels
        .iter()
        .position(|candidate| *candidate == label)
        .ok_or_else(|| eager_invalid_config(format!("label {label} missing from tensor labels")))
}

fn label_size(label: u32, operands: &[&LabeledTensor<'_>]) -> Result<usize> {
    for operand in operands {
        if let Some(axis) = operand
            .labels
            .iter()
            .position(|candidate| *candidate == label)
        {
            return Ok(operand.shape()[axis]);
        }
    }
    Err(eager_invalid_config(format!(
        "label {label} missing from eager einsum operands"
    )))
}

fn reduce_tensor<'a>(
    exec: &mut dyn TensorExec,
    operand: LabeledTensor<'a>,
    reduce_labels: &HashSet<u32>,
) -> Result<LabeledTensor<'a>> {
    if reduce_labels.is_empty() {
        return Ok(operand);
    }

    let reduce_axes: Vec<usize> = operand
        .labels
        .iter()
        .enumerate()
        .filter(|(_, label)| reduce_labels.contains(label))
        .map(|(axis, _)| axis)
        .collect();
    if reduce_axes.is_empty() {
        return Ok(operand);
    }

    let reduce_set: HashSet<usize> = reduce_axes.iter().copied().collect();
    let labels: Vec<u32> = operand
        .labels
        .iter()
        .enumerate()
        .filter(|(axis, _)| !reduce_set.contains(axis))
        .map(|(_, label)| *label)
        .collect();
    let operand_tensor = operand.tensor_cow();
    let tensor = exec.reduce_sum(&operand_tensor, &reduce_axes)?;
    operand.reclaim_if_owned(exec);
    Ok(LabeledTensor {
        tensor: TensorValue::Owned(tensor),
        labels,
    })
}

fn diagonalize_repeated<'a>(
    exec: &mut dyn TensorExec,
    mut operand: LabeledTensor<'a>,
) -> Result<LabeledTensor<'a>> {
    loop {
        let mut seen = HashMap::new();
        let mut repeated_pair = None;
        for (axis, label) in operand.labels.iter().copied().enumerate() {
            if let Some(first_axis) = seen.insert(label, axis) {
                repeated_pair = Some((first_axis, axis));
                break;
            }
        }

        let Some((axis_a, axis_b)) = repeated_pair else {
            return Ok(operand);
        };

        let operand_tensor = operand.tensor_cow();
        let tensor = exec.extract_diagonal(&operand_tensor, axis_a, axis_b)?;
        let mut labels = operand.labels.clone();
        labels.remove(axis_b);
        operand.reclaim_if_owned(exec);
        operand = LabeledTensor {
            tensor: TensorValue::Owned(tensor),
            labels,
        };
    }
}

fn embed_repeated<'a>(
    exec: &mut dyn TensorExec,
    mut operand: LabeledTensor<'a>,
    output_labels: &[u32],
) -> Result<LabeledTensor<'a>> {
    loop {
        let mut embedded = false;
        for &label in output_labels {
            let current_count = operand
                .labels
                .iter()
                .filter(|candidate| **candidate == label)
                .count();
            let output_count = output_labels
                .iter()
                .filter(|candidate| **candidate == label)
                .count();
            if output_count > current_count {
                let axis_a = find_label_axis(&operand.labels, label)?;
                let axis_b = axis_a + 1;
                let operand_tensor = operand.tensor_cow();
                let tensor = exec.embed_diagonal(&operand_tensor, axis_a, axis_b)?;
                let mut labels = operand.labels.clone();
                labels.insert(axis_b, label);
                operand.reclaim_if_owned(exec);
                operand = LabeledTensor {
                    tensor: TensorValue::Owned(tensor),
                    labels,
                };
                embedded = true;
                break;
            }
        }

        if !embedded {
            return Ok(operand);
        }
    }
}

fn transpose_to_labels<'a>(
    exec: &mut dyn TensorExec,
    operand: LabeledTensor<'a>,
    target_labels: &[u32],
) -> Result<LabeledTensor<'a>> {
    if operand.labels == target_labels {
        return Ok(operand);
    }

    let perm: Vec<usize> = target_labels
        .iter()
        .map(|label| find_label_axis(&operand.labels, *label))
        .collect::<Result<_>>()?;
    if perm
        .iter()
        .enumerate()
        .all(|(axis, target)| axis == *target)
    {
        return Ok(operand);
    }

    let operand_tensor = operand.tensor_cow();
    let tensor = exec.transpose(&operand_tensor, &perm)?;
    operand.reclaim_if_owned(exec);
    Ok(LabeledTensor {
        tensor: TensorValue::Owned(tensor),
        labels: target_labels.to_vec(),
    })
}

fn outer_product<'a>(
    exec: &mut dyn TensorExec,
    lhs: LabeledTensor<'a>,
    rhs: LabeledTensor<'a>,
    batch_labels: &[u32],
    lhs_free_labels: &[u32],
    rhs_free_labels: &[u32],
) -> Result<LabeledTensor<'a>> {
    if lhs.labels == rhs.labels {
        let lhs_tensor = lhs.tensor_cow();
        let rhs_tensor = rhs.tensor_cow();
        let tensor = exec.mul(&lhs_tensor, &rhs_tensor)?;
        let labels = lhs.labels.clone();
        lhs.reclaim_if_owned(exec);
        rhs.reclaim_if_owned(exec);
        return Ok(LabeledTensor {
            tensor: TensorValue::Owned(tensor),
            labels,
        });
    }

    let combined_labels: Vec<u32> = lhs_free_labels
        .iter()
        .chain(rhs_free_labels.iter())
        .chain(batch_labels.iter())
        .copied()
        .collect();
    let combined_shape: Vec<usize> = combined_labels
        .iter()
        .map(|label| label_size(*label, &[&lhs, &rhs]))
        .collect::<Result<_>>()?;
    let lhs_dims: Vec<usize> = lhs
        .labels
        .iter()
        .map(|label| find_label_axis(&combined_labels, *label))
        .collect::<Result<_>>()?;
    let rhs_dims: Vec<usize> = rhs
        .labels
        .iter()
        .map(|label| find_label_axis(&combined_labels, *label))
        .collect::<Result<_>>()?;

    let lhs_input = lhs.tensor_cow();
    let rhs_input = rhs.tensor_cow();
    let lhs_tensor = exec.broadcast_in_dim(&lhs_input, &combined_shape, &lhs_dims)?;
    let rhs_tensor = exec.broadcast_in_dim(&rhs_input, &combined_shape, &rhs_dims)?;
    let tensor = exec.mul(&lhs_tensor, &rhs_tensor)?;
    lhs.reclaim_if_owned(exec);
    rhs.reclaim_if_owned(exec);
    exec.reclaim_buffer(lhs_tensor);
    exec.reclaim_buffer(rhs_tensor);
    Ok(LabeledTensor {
        tensor: TensorValue::Owned(tensor),
        labels: combined_labels,
    })
}

fn binary_contract<'a>(
    exec: &mut dyn TensorExec,
    lhs: LabeledTensor<'a>,
    rhs: LabeledTensor<'a>,
    survive_labels: &[u32],
    reorder_result: bool,
) -> Result<LabeledTensor<'a>> {
    if let Some(plan) = try_build_binary_dot_fast_plan(&lhs.labels, &rhs.labels, survive_labels) {
        return profile_eager_einsum_section("binary.fast_path", || {
            execute_binary_dot_fast_plan(exec, lhs, rhs, plan, reorder_result)
        });
    }

    let (survive_set, lhs_reduce, rhs_reduce) =
        profile_eager_einsum_section("binary.pre_reduce_label_analysis", || {
            let survive_set: HashSet<u32> = survive_labels.iter().copied().collect();
            let rhs_label_set: HashSet<u32> = rhs.labels.iter().copied().collect();
            let lhs_label_set: HashSet<u32> = lhs.labels.iter().copied().collect();

            let lhs_reduce: HashSet<u32> = lhs
                .labels
                .iter()
                .filter(|label| !rhs_label_set.contains(label) && !survive_set.contains(label))
                .copied()
                .collect();
            let rhs_reduce: HashSet<u32> = rhs
                .labels
                .iter()
                .filter(|label| !lhs_label_set.contains(label) && !survive_set.contains(label))
                .copied()
                .collect();
            (survive_set, lhs_reduce, rhs_reduce)
        });

    let lhs = profile_eager_einsum_section("binary.reduce_lhs", || {
        reduce_tensor(exec, lhs, &lhs_reduce)
    })?;
    let rhs = profile_eager_einsum_section("binary.reduce_rhs", || {
        reduce_tensor(exec, rhs, &rhs_reduce)
    })?;

    let (batch_labels, contracting_labels, lhs_free_labels, rhs_free_labels) =
        profile_eager_einsum_section("binary.classify_labels", || {
            let lhs_label_set: HashSet<u32> = lhs.labels.iter().copied().collect();
            let rhs_label_set: HashSet<u32> = rhs.labels.iter().copied().collect();

            let mut batch_labels = Vec::new();
            let mut contracting_labels = Vec::new();
            let mut lhs_free_labels = Vec::new();
            let mut rhs_free_labels = Vec::new();

            for &label in &lhs.labels {
                if rhs_label_set.contains(&label) {
                    if survive_set.contains(&label) {
                        if !batch_labels.contains(&label) {
                            batch_labels.push(label);
                        }
                    } else if !contracting_labels.contains(&label) {
                        contracting_labels.push(label);
                    }
                } else if !lhs_free_labels.contains(&label) {
                    lhs_free_labels.push(label);
                }
            }

            for &label in &rhs.labels {
                if !lhs_label_set.contains(&label) && !rhs_free_labels.contains(&label) {
                    rhs_free_labels.push(label);
                }
            }
            (
                batch_labels,
                contracting_labels,
                lhs_free_labels,
                rhs_free_labels,
            )
        });

    let result = if contracting_labels.is_empty() {
        outer_product(
            exec,
            lhs,
            rhs,
            &batch_labels,
            &lhs_free_labels,
            &rhs_free_labels,
        )?
    } else {
        let (labels, config) = profile_eager_einsum_section("binary.build_dot_config", || {
            let lhs_contracting_dims: Vec<usize> = contracting_labels
                .iter()
                .map(|label| find_label_axis(&lhs.labels, *label))
                .collect::<Result<_>>()?;
            let rhs_contracting_dims: Vec<usize> = contracting_labels
                .iter()
                .map(|label| find_label_axis(&rhs.labels, *label))
                .collect::<Result<_>>()?;
            let lhs_batch_dims: Vec<usize> = batch_labels
                .iter()
                .map(|label| find_label_axis(&lhs.labels, *label))
                .collect::<Result<_>>()?;
            let rhs_batch_dims: Vec<usize> = batch_labels
                .iter()
                .map(|label| find_label_axis(&rhs.labels, *label))
                .collect::<Result<_>>()?;
            let labels: Vec<u32> = lhs_free_labels
                .iter()
                .chain(rhs_free_labels.iter())
                .chain(batch_labels.iter())
                .copied()
                .collect();
            let config = DotGeneralConfig {
                lhs_contracting_dims,
                rhs_contracting_dims,
                lhs_batch_dims,
                rhs_batch_dims,
            };
            Ok::<_, Error>((labels, config))
        })?;
        let tensor = profile_eager_einsum_section("binary.dot_general", || {
            exec.dot_general_read(lhs.tensor_read(), rhs.tensor_read(), &config)
        })?;
        lhs.reclaim_if_owned(exec);
        rhs.reclaim_if_owned(exec);
        LabeledTensor {
            tensor: TensorValue::Owned(tensor),
            labels,
        }
    };

    if !reorder_result {
        return Ok(result);
    }

    let result_label_set: HashSet<u32> = result.labels.iter().copied().collect();
    let target_labels: Vec<u32> = survive_labels
        .iter()
        .filter(|label| result_label_set.contains(label))
        .copied()
        .collect();
    profile_eager_einsum_section("binary.final_transpose", || {
        transpose_to_labels(exec, result, &target_labels)
    })
}

fn eager_einsum_exec_values<'a>(
    exec: &mut dyn TensorExec,
    inputs: Vec<TensorValue<'a>>,
    tree: &ContractionTree,
) -> Result<Tensor> {
    record_eager_einsum_profile("exec_values.enter", Duration::ZERO);
    let subscripts = &tree.subscripts;
    let n_inputs = subscripts.inputs.len();
    let output_labels = &subscripts.output;

    let mut labeled: Vec<Option<LabeledTensor<'a>>> =
        profile_eager_einsum_section("exec.init_labeled", || {
            inputs
                .into_iter()
                .zip(subscripts.inputs.iter())
                .map(|(tensor, labels)| {
                    Some(LabeledTensor {
                        tensor,
                        labels: labels.clone(),
                    })
                })
                .collect()
        });

    profile_eager_einsum_section("exec.diagonalize_inputs", || -> Result<()> {
        for index in 0..labeled.len() {
            let operand = take_labeled(&mut labeled, index, "input")?;
            labeled[index] = Some(diagonalize_repeated(exec, operand)?);
        }
        Ok(())
    })?;

    if n_inputs == 1 || tree.step_count() == 0 {
        let operand = take_labeled(&mut labeled, 0, "input")?;
        let output_set: HashSet<u32> = output_labels.iter().copied().collect();
        let reduce_labels: HashSet<u32> = operand
            .labels
            .iter()
            .filter(|label| !output_set.contains(label))
            .copied()
            .collect();
        let reduced = reduce_tensor(exec, operand, &reduce_labels)?;
        let embedded = embed_repeated(exec, reduced, output_labels)?;
        let reordered = transpose_to_labels(exec, embedded, output_labels)?;
        return Ok(reordered.tensor.into_tensor());
    }

    for step_idx in 0..tree.step_count() {
        let (left, right) = tree.step_pair(step_idx).ok_or_else(|| {
            eager_invalid_config(format!("missing contraction pair for step {step_idx}"))
        })?;
        let (_, _, step_output_labels) = tree.step_subscripts(step_idx).ok_or_else(|| {
            eager_invalid_config(format!(
                "missing contraction subscripts for step {step_idx}"
            ))
        })?;
        let lhs = take_labeled(&mut labeled, left, "lhs")?;
        let rhs = take_labeled(&mut labeled, right, "rhs")?;
        let result = profile_eager_einsum_section("exec.binary_contract", || {
            binary_contract(
                exec,
                lhs,
                rhs,
                step_output_labels,
                step_idx + 1 == tree.step_count(),
            )
        })?;
        labeled.push(Some(result));
    }

    let final_index = n_inputs + tree.step_count() - 1;
    let result = take_labeled(&mut labeled, final_index, "result")?;
    let extra_labels: HashSet<u32> =
        profile_eager_einsum_section("exec.final_label_analysis", || {
            let output_set: HashSet<u32> = output_labels.iter().copied().collect();
            result
                .labels
                .iter()
                .filter(|label| !output_set.contains(label))
                .copied()
                .collect()
        });
    let reduced = profile_eager_einsum_section("exec.final_reduce", || {
        reduce_tensor(exec, result, &extra_labels)
    })?;
    let reordered = profile_eager_einsum_section("exec.final_transpose", || {
        transpose_to_labels(exec, reduced, output_labels)
    })?;
    Ok(reordered.tensor.into_tensor())
}

fn eager_einsum_exec(
    exec: &mut dyn TensorExec,
    inputs: &[&Tensor],
    tree: &ContractionTree,
) -> Result<Tensor> {
    record_eager_einsum_profile("exec.enter", Duration::ZERO);
    let values = inputs
        .iter()
        .map(|tensor| TensorValue::Borrowed(*tensor))
        .collect();
    eager_einsum_exec_values(exec, values, tree)
}

#[cfg(test)]
fn eager_einsum_exec_read(
    exec: &mut dyn TensorExec,
    inputs: &[TensorRead<'_>],
    tree: &ContractionTree,
) -> Result<Tensor> {
    record_eager_einsum_profile("exec_read.enter", Duration::ZERO);
    let values = inputs
        .iter()
        .map(|input| match input {
            TensorRead::Tensor(tensor) => TensorValue::Borrowed(*tensor),
            TensorRead::View(view) => TensorValue::View(*view),
        })
        .collect();
    eager_einsum_exec_values(exec, values, tree)
}

fn tensor_value_from_read(input: TensorRead<'_>) -> TensorValue<'_> {
    match input {
        TensorRead::Tensor(tensor) => TensorValue::Borrowed(tensor),
        TensorRead::View(view) => TensorValue::View(view),
    }
}

fn eager_einsum_exec_binary_read_fast(
    exec: &mut dyn TensorExec,
    inputs: &[TensorRead<'_>],
    subscripts: &Subscripts,
    plan: BinaryDotFastPlan,
) -> Result<Tensor> {
    let lhs = LabeledTensor {
        tensor: tensor_value_from_read(inputs[0]),
        labels: subscripts.inputs[0].clone(),
    };
    let rhs = LabeledTensor {
        tensor: tensor_value_from_read(inputs[1]),
        labels: subscripts.inputs[1].clone(),
    };
    execute_binary_dot_fast_plan(exec, lhs, rhs, plan, true)
        .map(|result| result.tensor.into_tensor())
}

fn try_eager_einsum_binary_read_fast(
    ctx: &mut impl TensorBackend,
    inputs: &[TensorRead<'_>],
    subscripts: &Subscripts,
) -> Option<Result<Tensor>> {
    let total_started = eager_einsum_profile_enabled().then(Instant::now);
    if inputs.len() != 2 || subscripts.inputs.len() != 2 {
        return None;
    }
    if inputs[0].shape().len() != subscripts.inputs[0].len()
        || inputs[1].shape().len() != subscripts.inputs[1].len()
    {
        return None;
    }
    let plan = profile_eager_einsum_section("fast.build_plan", || {
        try_build_binary_dot_fast_plan(
            &subscripts.inputs[0],
            &subscripts.inputs[1],
            &subscripts.output,
        )
    })?;
    let result = profile_eager_einsum_section("fast.with_exec_session", || {
        ctx.with_exec_session(|exec| {
            eager_einsum_exec_binary_read_fast(exec, inputs, subscripts, plan)
        })
    });
    if let Some(started) = total_started {
        record_eager_einsum_profile("total", started.elapsed());
        maybe_print_eager_einsum_profile();
    }
    Some(result)
}

/// Eager N-ary einsum on concrete [`Tensor`] values.
///
/// This applies the same contraction-tree optimization strategy used by the
/// traced einsum path, but executes each contraction immediately against the
/// provided backend context.
///
#[cfg(test)]
pub(crate) fn eager_einsum(
    ctx: &mut impl TensorBackend,
    inputs: &[&Tensor],
    subscripts: &str,
) -> Result<Tensor> {
    let subscripts = Subscripts::parse(subscripts)
        .map_err(|err| eager_invalid_config(format!("invalid subscripts: {err}")))?;
    eager_einsum_subscripts(ctx, inputs, &subscripts)
}

/// Eager N-ary einsum on concrete [`Tensor`] values using integer labels.
pub(crate) fn eager_einsum_subscripts(
    ctx: &mut impl TensorBackend,
    inputs: &[&Tensor],
    subscripts: &Subscripts,
) -> Result<Tensor> {
    if inputs.len() == 2 {
        let read_inputs = [
            TensorRead::from_tensor(inputs[0]),
            TensorRead::from_tensor(inputs[1]),
        ];
        if let Some(result) = try_eager_einsum_binary_read_fast(ctx, &read_inputs, subscripts) {
            return result;
        }
    }

    if eager_einsum_profile_enabled() {
        let total_started = Instant::now();
        let shapes = profile_eager_einsum_section("shape_collect", || {
            inputs
                .iter()
                .map(|tensor| tensor.shape())
                .collect::<Vec<_>>()
        });
        let tree = profile_eager_einsum_section("plan_subscripts", || {
            plan_subscripts(subscripts, &shapes)
        })?;
        let result = profile_eager_einsum_section("with_exec_session", || {
            ctx.with_exec_session(|exec| eager_einsum_exec(exec, inputs, &tree))
        });
        record_eager_einsum_profile("total", total_started.elapsed());
        maybe_print_eager_einsum_profile();
        return result;
    }

    if eager_einsum_trace_enabled() {
        let total_started = Instant::now();
        let started = Instant::now();
        let shapes: Vec<&[usize]> = inputs.iter().map(|tensor| tensor.shape()).collect();
        let shape_us = started.elapsed().as_secs_f64() * 1.0e6;

        let started = Instant::now();
        let tree = plan_subscripts(subscripts, &shapes)?;
        let plan_us = started.elapsed().as_secs_f64() * 1.0e6;

        let started = Instant::now();
        let result = ctx.with_exec_session(|exec| eager_einsum_exec(exec, inputs, &tree));
        let exec_us = started.elapsed().as_secs_f64() * 1.0e6;
        let total_us = total_started.elapsed().as_secs_f64() * 1.0e6;

        eprintln!(
            "tenferro_eager_einsum_profile,n_inputs={},shapes={:?},steps={},shape_us={shape_us:.3},plan_us={plan_us:.3},exec_us={exec_us:.3},total_us={total_us:.3}",
            inputs.len(),
            shapes,
            tree.step_count(),
        );
        return result;
    }

    let shapes: Vec<&[usize]> = inputs.iter().map(|tensor| tensor.shape()).collect();
    let tree = plan_subscripts(subscripts, &shapes)?;
    ctx.with_exec_session(|exec| eager_einsum_exec(exec, inputs, &tree))
}

/// Eager N-ary einsum on read-only tensor inputs.
///
/// Owned tensors and borrowed host views share this entry point. Backends may
/// consume views directly when their execution model supports it, or
/// materialize/upload them inside the execution session.
#[cfg(test)]
pub(crate) fn eager_einsum_read_subscripts(
    ctx: &mut impl TensorBackend,
    inputs: &[TensorRead<'_>],
    subscripts: &Subscripts,
) -> Result<Tensor> {
    if let Some(result) = try_eager_einsum_binary_read_fast(ctx, inputs, subscripts) {
        return result;
    }

    let shapes: Vec<&[usize]> = inputs.iter().map(TensorRead::shape).collect();
    let tree = plan_subscripts(subscripts, &shapes)?;
    ctx.with_exec_session(|exec| eager_einsum_exec_read(exec, inputs, &tree))
}

/// Eager N-ary einsum that consumes concrete [`Tensor`] inputs.
///
/// This has the same parsing, contraction planning, execution semantics, and
/// output ordering as [`eager_einsum`]. Unlike the borrowed API, inputs are
/// moved into the executor, so eligible buffers may be reclaimed by the backend
/// after their last use.
///
/// Downstream crates should use this N-ary API even for two-input contractions
/// instead of depending on binary lowering details.
///
#[cfg(test)]
pub(crate) fn eager_einsum_owned(
    ctx: &mut impl TensorBackend,
    inputs: Vec<Tensor>,
    subscripts: &str,
) -> Result<Tensor> {
    let subscripts = Subscripts::parse(subscripts)
        .map_err(|err| eager_invalid_config(format!("invalid subscripts: {err}")))?;
    eager_einsum_owned_subscripts(ctx, inputs, &subscripts)
}

/// Eager N-ary einsum that consumes concrete [`Tensor`] inputs using integer labels.
#[cfg(test)]
pub(crate) fn eager_einsum_owned_subscripts(
    ctx: &mut impl TensorBackend,
    inputs: Vec<Tensor>,
    subscripts: &Subscripts,
) -> Result<Tensor> {
    let shapes: Vec<&[usize]> = inputs.iter().map(|tensor| tensor.shape()).collect();
    let tree = plan_subscripts(subscripts, &shapes)?;
    let values = inputs.into_iter().map(TensorValue::Owned).collect();
    ctx.with_exec_session(|exec| eager_einsum_exec_values(exec, values, &tree))
}

#[cfg(test)]
mod tests;
