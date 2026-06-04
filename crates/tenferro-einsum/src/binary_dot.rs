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

fn small_contains(labels: &[u32], label: u32) -> bool {
    labels.contains(&label)
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
mod tests {
    use tenferro_tensor::DotGeneralConfig;

    use super::{try_build_exact_output_binary_dot_plan, BinaryDotOperandOrder};

    #[test]
    fn exact_binary_dot_plan_accepts_original_output_order() {
        let plan = try_build_exact_output_binary_dot_plan(
            &[b'i' as u32, b'j' as u32],
            &[b'j' as u32, b'k' as u32],
            &[b'i' as u32, b'k' as u32],
        )
        .expect("matmul should lower exactly");

        assert_eq!(plan.operand_order, BinaryDotOperandOrder::Original);
        assert_eq!(
            plan.config,
            DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            }
        );
    }

    #[test]
    fn exact_binary_dot_plan_accepts_swapped_col_major_matmul() {
        let plan = try_build_exact_output_binary_dot_plan(
            &[b'j' as u32, b'i' as u32],
            &[b'k' as u32, b'j' as u32],
            &[b'k' as u32, b'i' as u32],
        )
        .expect("col-major matmul should lower exactly after swapping operands");

        assert_eq!(plan.operand_order, BinaryDotOperandOrder::Swapped);
        assert_eq!(
            plan.config,
            DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            }
        );
    }

    #[test]
    fn exact_binary_dot_plan_accepts_swapped_col_major_batched_matmul() {
        let plan = try_build_exact_output_binary_dot_plan(
            &[b'j' as u32, b'i' as u32, b'b' as u32],
            &[b'k' as u32, b'j' as u32, b'b' as u32],
            &[b'k' as u32, b'i' as u32, b'b' as u32],
        )
        .expect("col-major batched matmul should lower exactly after swapping operands");

        assert_eq!(plan.operand_order, BinaryDotOperandOrder::Swapped);
        assert_eq!(
            plan.config,
            DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![2],
                rhs_batch_dims: vec![2],
            }
        );
    }

    #[test]
    fn exact_binary_dot_plan_rejects_repeated_labels() {
        assert!(try_build_exact_output_binary_dot_plan(
            &[b'i' as u32, b'i' as u32],
            &[b'j' as u32],
            &[b'j' as u32],
        )
        .is_none());
    }
}
