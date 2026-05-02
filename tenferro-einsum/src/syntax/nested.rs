use std::collections::HashSet;

use tenferro_device::{Error, Result};

use crate::syntax::notation::{char_to_label, split_and_validate_notation};
use crate::syntax::subscripts::Subscripts;

/// Recursive einsum tree that preserves parenthesized grouping.
///
/// `NestedEinsum` mirrors OMEinsum.jl's `NestedEinsum`: each internal node
/// holds [`Subscripts`] describing how its children are contracted, and leaf
/// nodes reference an original input tensor by index.
///
/// # Construction
///
/// Use [`NestedEinsum::parse`] to build a tree from parenthesized string
/// notation such as `"(ij,jk),kl->il"`.  Without parentheses the result is
/// a flat root node whose children are all leaves.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::NestedEinsum;
///
/// // Flat (no grouping): root with two leaves
/// let flat = NestedEinsum::parse("ij,jk->ik").unwrap();
/// assert!(matches!(flat, NestedEinsum::Node { .. }));
///
/// // Grouped: contract first two operands, then with third
/// let grouped = NestedEinsum::parse("(ij,jk),kl->il").unwrap();
/// assert!(matches!(grouped, NestedEinsum::Node { .. }));
/// ```
#[derive(Debug, Clone)]
pub enum NestedEinsum {
    /// A leaf referencing one of the original input tensors by index.
    Leaf(usize),
    /// An internal node that contracts its children according to `subscripts`.
    Node {
        /// The subscripts for this contraction: one input per child, plus output.
        subscripts: Subscripts,
        /// Child sub-expressions (leaves or further nodes).
        children: Vec<NestedEinsum>,
    },
}

impl NestedEinsum {
    /// Count the total number of leaf operands in the tree.
    pub fn count_leaves(&self) -> usize {
        match self {
            Self::Leaf(_) => 1,
            Self::Node { children, .. } => children.iter().map(|c| c.count_leaves()).sum(),
        }
    }

    /// Parse parenthesized einsum notation into a recursive tree.
    ///
    /// Notation follows the standard `"inputs->output"` format with optional
    /// parentheses to specify contraction order. Each parenthesized group
    /// becomes an internal [`NestedEinsum::Node`]; bare operands become
    /// [`NestedEinsum::Leaf`] nodes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::NestedEinsum;
    ///
    /// let nested = NestedEinsum::parse("(ij,jk),kl->il").unwrap();
    /// // Root has two children: a group node and a leaf
    /// match &nested {
    ///     NestedEinsum::Node { children, .. } => assert_eq!(children.len(), 2),
    ///     _ => panic!("expected Node"),
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if parentheses are mismatched or the notation is
    /// otherwise malformed.
    pub fn parse(notation: &str) -> Result<Self> {
        let (lhs, output_str) = split_and_validate_notation(notation)?;

        let output: Vec<u32> = output_str
            .chars()
            .map(char_to_label)
            .collect::<Result<_>>()?;

        let mut leaf_counter: usize = 0;
        let outer_needed: HashSet<u32> = output.iter().copied().collect();
        Self::parse_group(lhs, &outer_needed, &output, &mut leaf_counter)
    }

    /// Recursively parse a group (possibly containing sub-groups) into a Node.
    ///
    /// `group_str` is a comma-separated list of items (at the top level),
    /// where each item is either a bare operand (e.g. `"ij"`) or a
    /// parenthesized sub-group (e.g. `"(ij,jk)"`).
    ///
    /// `outer_needed` contains labels that the parent or siblings need from
    /// this group.  `final_output` is the overall output of the entire
    /// expression.
    fn parse_group(
        group_str: &str,
        outer_needed: &HashSet<u32>,
        final_output: &[u32],
        leaf_counter: &mut usize,
    ) -> Result<Self> {
        let items = Self::split_top_level(group_str)?;

        let mut children = Vec::with_capacity(items.len());
        let mut child_subscript_inputs: Vec<Vec<u32>> = Vec::with_capacity(items.len());

        for (idx, item) in items.iter().enumerate() {
            if item.starts_with('(') && item.ends_with(')') {
                // Sub-group: strip outer parens and recurse
                let inner = &item[1..item.len() - 1];

                // Compute what this sub-group needs to output:
                // labels in this group that appear in outer_needed or in sibling items
                let group_labels = Self::collect_labels_in_order(inner)?;
                let sibling_labels = Self::collect_sibling_labels(&items, idx)?;
                let mut needed: HashSet<u32> = HashSet::new();
                let mut sub_output = Vec::new();
                for label in group_labels {
                    if outer_needed.contains(&label) || sibling_labels.contains(&label) {
                        needed.insert(label);
                        sub_output.push(label);
                    }
                }

                let child = Self::parse_group(inner, &needed, &sub_output, leaf_counter)?;
                child_subscript_inputs.push(sub_output);
                children.push(child);
            } else {
                // Bare operand -> Leaf
                let labels: Vec<u32> = item.chars().map(char_to_label).collect::<Result<_>>()?;
                child_subscript_inputs.push(labels);
                children.push(NestedEinsum::Leaf(*leaf_counter));
                *leaf_counter += 1;
            }
        }

        // Build subscripts for this node
        let node_output: Vec<u32> = final_output.to_vec();
        let subscripts = Subscripts {
            inputs: child_subscript_inputs,
            output: node_output,
        };

        Ok(NestedEinsum::Node {
            subscripts,
            children,
        })
    }

    /// Split a string on commas at the top level (depth 0), respecting parentheses.
    fn split_top_level(s: &str) -> Result<Vec<&str>> {
        let mut items = Vec::new();
        let mut depth: usize = 0;
        let mut start = 0;

        for (pos, c) in s.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    if depth == 0 {
                        return Err(Error::InvalidArgument(format!(
                            "unmatched ')' in einsum group: {s}"
                        )));
                    }
                    depth -= 1;
                }
                ',' if depth == 0 => {
                    items.push(&s[start..pos]);
                    start = pos + 1; // skip the comma
                }
                _ => {}
            }
        }
        // Push the last item
        items.push(&s[start..]);
        Ok(items)
    }

    /// Collect all unique labels from a (possibly nested) string, ignoring
    /// parentheses and commas.
    fn collect_labels(s: &str) -> Result<HashSet<u32>> {
        let mut labels = HashSet::new();
        for c in s.chars() {
            match c {
                '(' | ')' | ',' => continue,
                _ => {
                    labels.insert(char_to_label(c)?);
                }
            }
        }
        Ok(labels)
    }

    /// Collect unique labels in first-appearance order, ignoring
    /// parentheses and commas.
    fn collect_labels_in_order(s: &str) -> Result<Vec<u32>> {
        let mut seen = HashSet::new();
        let mut labels = Vec::new();
        for c in s.chars() {
            match c {
                '(' | ')' | ',' => continue,
                _ => {
                    let label = char_to_label(c)?;
                    if seen.insert(label) {
                        labels.push(label);
                    }
                }
            }
        }
        Ok(labels)
    }

    /// Collect all labels from sibling items (all items except the one at `current_idx`).
    fn collect_sibling_labels(items: &[&str], current_idx: usize) -> Result<HashSet<u32>> {
        let mut labels = HashSet::new();
        for (idx, item) in items.iter().enumerate() {
            if idx == current_idx {
                continue;
            }
            for label in Self::collect_labels(item)? {
                labels.insert(label);
            }
        }
        Ok(labels)
    }
}

#[cfg(test)]
mod tests;
