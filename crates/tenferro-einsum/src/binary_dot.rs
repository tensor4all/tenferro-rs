use tenferro_tensor::DotGeneralConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BinaryDotOperandOrder {
    Original,
    Swapped,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BinaryDotPlan {
    pub(crate) operand_order: BinaryDotOperandOrder,
    pub(crate) result_labels: Vec<u32>,
    pub(crate) target_labels: Vec<u32>,
    pub(crate) config: DotGeneralConfig,
}

fn small_contains<L: PartialEq>(labels: &[L], label: L) -> bool {
    labels.contains(&label)
}

fn labels_are_unique<L: PartialEq>(labels: &[L]) -> bool {
    // INVARIANT: label lists are bounded by tensor rank; the existing planner already
    // uses quadratic membership scans, so checking prior labels avoids scratch allocation.
    labels
        .iter()
        .enumerate()
        .all(|(i, label)| !labels[..i].contains(label))
}

pub(crate) fn try_build_exact_output_binary_dot_config<L: Copy + PartialEq>(
    lhs_labels: &[L],
    rhs_labels: &[L],
    output_labels: &[L],
) -> Option<(BinaryDotOperandOrder, DotGeneralConfig)> {
    if !labels_are_unique(lhs_labels)
        || !labels_are_unique(rhs_labels)
        || !labels_are_unique(output_labels)
    {
        return None;
    }
    try_build_exact_output_binary_dot_config_with_order(
        lhs_labels,
        rhs_labels,
        output_labels,
        BinaryDotOperandOrder::Original,
    )
    .or_else(|| {
        try_build_exact_output_binary_dot_config_with_order(
            rhs_labels,
            lhs_labels,
            output_labels,
            BinaryDotOperandOrder::Swapped,
        )
    })
}

fn try_build_exact_output_binary_dot_config_with_order<L: Copy + PartialEq>(
    lhs_labels: &[L],
    rhs_labels: &[L],
    output_labels: &[L],
    operand_order: BinaryDotOperandOrder,
) -> Option<(BinaryDotOperandOrder, DotGeneralConfig)> {
    let mut result_pos = 0;
    for &label in lhs_labels {
        if !small_contains(rhs_labels, label) {
            if !small_contains(output_labels, label)
                || output_labels.get(result_pos) != Some(&label)
            {
                return None;
            }
            result_pos += 1;
        } else if small_contains(output_labels, label) {
            // Shared output labels follow both operands' free labels.
            continue;
        }
    }
    for &label in rhs_labels {
        if !small_contains(lhs_labels, label) {
            if !small_contains(output_labels, label)
                || output_labels.get(result_pos) != Some(&label)
            {
                return None;
            }
            result_pos += 1;
        }
    }
    for &label in lhs_labels {
        if small_contains(rhs_labels, label) && small_contains(output_labels, label) {
            if output_labels.get(result_pos) != Some(&label) {
                return None;
            }
            result_pos += 1;
        }
    }
    if result_pos != output_labels.len()
        || !lhs_labels.iter().any(|label| {
            small_contains(rhs_labels, *label) && !small_contains(output_labels, *label)
        })
    {
        return None;
    }

    let mut config = DotGeneralConfig {
        lhs_contracting_dims: Default::default(),
        rhs_contracting_dims: Default::default(),
        lhs_batch_dims: Default::default(),
        rhs_batch_dims: Default::default(),
    };
    for (lhs_axis, &label) in lhs_labels.iter().enumerate() {
        if let Some(rhs_axis) = rhs_labels.iter().position(|candidate| *candidate == label) {
            if small_contains(output_labels, label) {
                config.lhs_batch_dims.push(lhs_axis);
                config.rhs_batch_dims.push(rhs_axis);
            } else {
                config.lhs_contracting_dims.push(lhs_axis);
                config.rhs_contracting_dims.push(rhs_axis);
            }
        }
    }
    Some((operand_order, config))
}

pub(crate) fn try_build_binary_dot_plan(
    lhs_labels: &[u32],
    rhs_labels: &[u32],
    output_labels: &[u32],
) -> Option<BinaryDotPlan> {
    try_build_binary_dot_plan_with_order(
        lhs_labels,
        rhs_labels,
        output_labels,
        BinaryDotOperandOrder::Original,
    )
}

pub(crate) fn try_build_exact_output_binary_dot_plan(
    lhs_labels: &[u32],
    rhs_labels: &[u32],
    output_labels: &[u32],
) -> Option<BinaryDotPlan> {
    if let Some(plan) = try_build_binary_dot_plan(lhs_labels, rhs_labels, output_labels) {
        if plan.result_labels == plan.target_labels {
            return Some(plan);
        }
    }

    let plan = try_build_binary_dot_plan_with_order(
        rhs_labels,
        lhs_labels,
        output_labels,
        BinaryDotOperandOrder::Swapped,
    )?;
    if plan.result_labels == plan.target_labels {
        Some(plan)
    } else {
        None
    }
}

fn try_build_binary_dot_plan_with_order(
    lhs_labels: &[u32],
    rhs_labels: &[u32],
    output_labels: &[u32],
    operand_order: BinaryDotOperandOrder,
) -> Option<BinaryDotPlan> {
    if !labels_are_unique(lhs_labels)
        || !labels_are_unique(rhs_labels)
        || !labels_are_unique(output_labels)
    {
        return None;
    }

    let mut lhs_contracting_dims = smallvec::SmallVec::<[usize; 4]>::new();
    let mut rhs_contracting_dims = smallvec::SmallVec::<[usize; 4]>::new();
    let mut lhs_batch_dims = smallvec::SmallVec::<[usize; 4]>::new();
    let mut rhs_batch_dims = smallvec::SmallVec::<[usize; 4]>::new();
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

    Some(BinaryDotPlan {
        operand_order,
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

#[cfg(test)]
mod tests;
