use std::collections::{BTreeMap, HashMap, HashSet};

use tenferro_einsum::Subscripts;

use super::types::{
    AxisClassMergePlan, AxisClassPlanError, OperandAxisClassPlan, OperandAxisClasses,
};

/// Build a metadata-only merge plan for PartialDiagonal contraction with
/// integer-label einsum subscripts.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{plan_axis_classes_for_subscripts, OperandAxisClasses};
/// use tenferro_einsum::Subscripts;
///
/// let operands = vec![
///     OperandAxisClasses::new(vec![3, 3], vec![0, 0]).unwrap(),
///     OperandAxisClasses::new(vec![3, 3], vec![0, 0]).unwrap(),
/// ];
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
/// let plan = plan_axis_classes_for_subscripts(&operands, &subs).unwrap();
/// assert_eq!(plan.output_axis_classes, vec![0, 0]);
/// ```
pub fn plan_axis_classes_for_subscripts(
    operands: &[OperandAxisClasses],
    subscripts: &Subscripts,
) -> Result<AxisClassMergePlan, AxisClassPlanError> {
    if operands.len() != subscripts.inputs.len() {
        return Err(AxisClassPlanError::InvalidOperandCount {
            expected: subscripts.inputs.len(),
            found: operands.len(),
        });
    }
    validate_operands(operands)?;
    validate_subscripts_ranks(operands, subscripts)?;

    let node_offsets = build_node_offsets(operands);
    let total_axes = node_offsets.last().copied().unwrap_or(0);
    let mut uf = NodeUnionFind::new(total_axes);

    for (operand_idx, operand) in operands.iter().enumerate() {
        let mut first_node_of_class: HashMap<usize, usize> = HashMap::new();
        for (axis, &class_id) in operand.axis_classes.iter().enumerate() {
            let node = node_offsets[operand_idx] + axis;
            if let Some(&first) = first_node_of_class.get(&class_id) {
                uf.union(first, node);
            } else {
                first_node_of_class.insert(class_id, node);
            }
        }
    }

    let mut label_nodes: HashMap<u32, Vec<usize>> = HashMap::new();
    let mut label_dims: HashMap<u32, usize> = HashMap::new();
    for (operand_idx, labels) in subscripts.inputs.iter().enumerate() {
        for (axis, &label) in labels.iter().enumerate() {
            let dim = operands[operand_idx].dims[axis];
            if let Some(&expected) = label_dims.get(&label) {
                if expected != dim {
                    return Err(AxisClassPlanError::LabelDimensionMismatch {
                        label,
                        expected,
                        actual: dim,
                    });
                }
            } else {
                label_dims.insert(label, dim);
            }
            let node = node_offsets[operand_idx] + axis;
            label_nodes.entry(label).or_default().push(node);
        }
    }
    for nodes in label_nodes.values() {
        if nodes.len() >= 2 {
            let first = nodes[0];
            for &node in nodes.iter().skip(1) {
                uf.union(first, node);
            }
        }
    }

    let node_root_ids = canonicalize_roots(&mut uf, total_axes);
    validate_merged_dims(operands, &node_offsets, &node_root_ids)?;

    let mut operand_plans = Vec::with_capacity(operands.len());
    let mut operand_axis_roots = Vec::with_capacity(operands.len());
    for (operand_idx, operand) in operands.iter().enumerate() {
        let axis_roots: Vec<usize> = (0..operand.axis_classes.len())
            .map(|axis| {
                let node = node_offsets[operand_idx] + axis;
                node_root_ids[node]
            })
            .collect();
        operand_axis_roots.push(axis_roots);

        let local_class_axes = local_class_axes(&operand.axis_classes);
        let class_roots: Vec<usize> = local_class_axes
            .iter()
            .map(|axes| {
                let node = node_offsets[operand_idx] + axes[0];
                node_root_ids[node]
            })
            .collect();

        let mut grouped: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (class_pos, &root) in class_roots.iter().enumerate() {
            grouped.entry(root).or_default().push(class_pos);
        }
        let duplicate_class_groups: Vec<Vec<usize>> =
            grouped.values().filter(|g| g.len() >= 2).cloned().collect();

        let normalized_class_roots = unique_in_first_appearance_order(&class_roots);
        operand_plans.push(OperandAxisClassPlan {
            class_roots,
            duplicate_class_groups,
            normalized_class_roots,
        });
    }

    let mut output_root_seq = Vec::with_capacity(subscripts.output.len());
    let mut output_dims = Vec::with_capacity(subscripts.output.len());
    for &label in &subscripts.output {
        let nodes = label_nodes
            .get(&label)
            .ok_or(AxisClassPlanError::MissingOutputLabel { label })?;
        let root = node_root_ids[nodes[0]];
        let dim = *label_dims
            .get(&label)
            .ok_or(AxisClassPlanError::MissingOutputLabel { label })?;
        output_root_seq.push(root);
        output_dims.push(dim);
    }
    let output_axis_classes = canonicalize_sequence(&output_root_seq);
    let output_compressed_roots = unique_in_first_appearance_order(&output_root_seq);

    Ok(AxisClassMergePlan {
        operand_plans,
        operand_axis_roots,
        output_class_roots: output_root_seq,
        output_axis_classes,
        output_dims,
        output_compressed_roots,
    })
}

fn validate_operands(operands: &[OperandAxisClasses]) -> Result<(), AxisClassPlanError> {
    for (operand_idx, operand) in operands.iter().enumerate() {
        if operand.dims.len() != operand.axis_classes.len() {
            return Err(AxisClassPlanError::InvalidOperand {
                operand: Some(operand_idx),
                message: format!(
                    "dims length ({}) must match axis_classes length ({})",
                    operand.dims.len(),
                    operand.axis_classes.len()
                ),
            });
        }
        let mut class_dims: HashMap<usize, usize> = HashMap::new();
        for (&dim, &class_id) in operand.dims.iter().zip(operand.axis_classes.iter()) {
            if let Some(&expected) = class_dims.get(&class_id) {
                if expected != dim {
                    return Err(AxisClassPlanError::InvalidOperand {
                        operand: Some(operand_idx),
                        message: format!(
                            "axis class {class_id} has inconsistent dims: {expected} vs {dim}"
                        ),
                    });
                }
            } else {
                class_dims.insert(class_id, dim);
            }
        }
    }
    Ok(())
}

fn validate_subscripts_ranks(
    operands: &[OperandAxisClasses],
    subscripts: &Subscripts,
) -> Result<(), AxisClassPlanError> {
    for (operand_idx, operand) in operands.iter().enumerate() {
        let label_rank = subscripts.inputs[operand_idx].len();
        if label_rank != operand.dims.len() {
            return Err(AxisClassPlanError::InvalidSubscripts {
                operand: operand_idx,
                message: format!(
                    "subscript rank {} does not match operand rank {}",
                    label_rank,
                    operand.dims.len()
                ),
            });
        }
    }
    Ok(())
}

fn build_node_offsets(operands: &[OperandAxisClasses]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(operands.len() + 1);
    let mut next_offset = 0usize;
    offsets.push(next_offset);
    for operand in operands {
        next_offset += operand.axis_classes.len();
        offsets.push(next_offset);
    }
    offsets
}

fn local_class_axes(axis_classes: &[usize]) -> Vec<Vec<usize>> {
    let mut class_pos_map: HashMap<usize, usize> = HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (axis, &class_id) in axis_classes.iter().enumerate() {
        if let Some(&pos) = class_pos_map.get(&class_id) {
            groups[pos].push(axis);
        } else {
            class_pos_map.insert(class_id, groups.len());
            groups.push(vec![axis]);
        }
    }
    groups
}

fn canonicalize_roots(uf: &mut NodeUnionFind, total_axes: usize) -> Vec<usize> {
    let mut root_to_canonical: HashMap<usize, usize> = HashMap::new();
    let mut next = 0usize;
    let mut node_root_ids = vec![0usize; total_axes];
    for (node, slot) in node_root_ids.iter_mut().enumerate() {
        let root = uf.find(node);
        let cid = if let Some(&cid) = root_to_canonical.get(&root) {
            cid
        } else {
            let cid = next;
            next += 1;
            root_to_canonical.insert(root, cid);
            cid
        };
        *slot = cid;
    }
    node_root_ids
}

fn validate_merged_dims(
    operands: &[OperandAxisClasses],
    node_offsets: &[usize],
    node_root_ids: &[usize],
) -> Result<(), AxisClassPlanError> {
    let mut expected_dim_by_root: HashMap<usize, usize> = HashMap::new();
    for (operand_idx, operand) in operands.iter().enumerate() {
        for (axis, &dim) in operand.dims.iter().enumerate() {
            let node = node_offsets[operand_idx] + axis;
            let root = node_root_ids[node];
            if let Some(&expected) = expected_dim_by_root.get(&root) {
                if expected != dim {
                    return Err(AxisClassPlanError::MergedClassDimensionMismatch {
                        root,
                        expected,
                        actual: dim,
                    });
                }
            } else {
                expected_dim_by_root.insert(root, dim);
            }
        }
    }
    Ok(())
}

fn unique_in_first_appearance_order(values: &[usize]) -> Vec<usize> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for &v in values {
        if seen.insert(v) {
            out.push(v);
        }
    }
    out
}

fn canonicalize_sequence(values: &[usize]) -> Vec<usize> {
    let mut id_map: HashMap<usize, usize> = HashMap::new();
    let mut next = 0usize;
    values
        .iter()
        .map(|v| {
            if let Some(&id) = id_map.get(v) {
                id
            } else {
                let id = next;
                next += 1;
                id_map.insert(*v, id);
                id
            }
        })
        .collect()
}

#[derive(Debug, Clone)]
struct NodeUnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl NodeUnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            let root = self.find(self.parent[x]);
            self.parent[x] = root;
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        let rank_a = self.rank[ra];
        let rank_b = self.rank[rb];
        if rank_a < rank_b {
            self.parent[ra] = rb;
        } else if rank_a > rank_b {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] = self.rank[ra].saturating_add(1);
        }
    }
}
