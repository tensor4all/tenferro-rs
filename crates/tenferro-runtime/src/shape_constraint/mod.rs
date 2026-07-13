use std::collections::HashSet;
use std::fmt;
use std::sync::{Arc, OnceLock};

use computegraph::types::ValueKey;
use tenferro_ops::{dim_expr::DimExpr, std_tensor_op::StdTensorOp, ShapeRelation};

mod solver;

// INVARIANT: Task 3 defines the solver boundary before the compiler and
// executor integrations in later tasks consume these crate-private items.
#[allow(unused_imports)]
pub(crate) use solver::discharge;
pub use solver::ShapeGuard;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ConstraintSource {
    pub(crate) family_id: &'static str,
    pub(crate) instruction_index: Option<usize>,
}

impl ConstraintSource {
    // INVARIANT: graph recording assigns instruction provenance after local
    // inference; the transition remains unused until that pipeline stage.
    #[allow(dead_code)]
    pub(crate) fn with_instruction(mut self, instruction_index: usize) -> Self {
        self.instruction_index = Some(instruction_index);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct LocalShapeConstraint {
    pub(crate) source: ConstraintSource,
    pub(crate) relation: ShapeRelation,
    pub(crate) lhs: DimExpr,
    pub(crate) rhs: DimExpr,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ScopedShapeConstraint {
    pub(crate) origins: Vec<ValueKey<StdTensorOp>>,
    pub(crate) inputs: Vec<ValueKey<StdTensorOp>>,
    pub(crate) local: LocalShapeConstraint,
}

#[derive(Clone, Debug)]
pub(crate) struct SlotScopedShapeConstraint {
    pub(crate) origin_slots: Vec<usize>,
    pub(crate) input_slots: Vec<usize>,
    pub(crate) local: LocalShapeConstraint,
}

/// Shape constraints recorded while analyzing one graph scope.
///
/// The constraint representation is intentionally private. This type crosses
/// only the documented [`crate::ad_support`] boundary so AD graph transforms
/// can preserve graph analysis without depending on its internal encoding.
///
/// # Examples
///
/// ```rust
/// use tenferro_runtime::ad_support::register_scoped_graph_analysis;
/// use tenferro_runtime::{DType, TracedTensor};
///
/// let input = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
/// let analysis = register_scoped_graph_analysis(input.graph(), []).unwrap();
/// assert!(analysis.constraints.is_empty());
/// ```
#[derive(Clone, Default)]
pub struct ShapeConstraintScope {
    constraints: Vec<ScopedShapeConstraint>,
}

impl ShapeConstraintScope {
    pub(crate) fn new(constraints: Vec<ScopedShapeConstraint>) -> Self {
        Self { constraints }
    }

    /// Return whether this scope contains no shape constraints.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::ad_support::register_scoped_graph_analysis;
    /// use tenferro_runtime::{DType, TracedTensor};
    ///
    /// let input = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    /// let analysis = register_scoped_graph_analysis(input.graph(), []).unwrap();
    /// assert!(analysis.constraints.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    pub(crate) fn constraints(&self) -> &[ScopedShapeConstraint] {
        &self.constraints
    }
}

impl fmt::Debug for ShapeConstraintScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShapeConstraintScope")
            .field("constraint_count", &self.constraints.len())
            .finish()
    }
}

#[derive(Clone)]
pub(crate) struct ConstraintScopeChain {
    node: Arc<ConstraintScopeChainNode>,
}

struct ConstraintScopeChainNode {
    scope: Option<Arc<ShapeConstraintScope>>,
    parents: Vec<ConstraintScopeChain>,
    materialized: OnceLock<Vec<Arc<ShapeConstraintScope>>>,
}

impl ConstraintScopeChain {
    pub(crate) fn empty() -> Self {
        Self {
            node: Arc::new(ConstraintScopeChainNode {
                scope: None,
                parents: Vec::new(),
                materialized: OnceLock::new(),
            }),
        }
    }

    pub(crate) fn with_scope<'a>(
        scope: Arc<ShapeConstraintScope>,
        inherited: impl IntoIterator<Item = &'a ConstraintScopeChain>,
    ) -> Self {
        Self {
            node: Arc::new(ConstraintScopeChainNode {
                scope: Some(scope),
                parents: inherited.into_iter().cloned().collect(),
                materialized: OnceLock::new(),
            }),
        }
    }

    pub(crate) fn merge<'a>(inherited: impl IntoIterator<Item = &'a ConstraintScopeChain>) -> Self {
        let parents: Vec<_> = inherited.into_iter().cloned().collect();
        match parents.as_slice() {
            [] => Self::empty(),
            [only] => only.clone(),
            _ => Self {
                node: Arc::new(ConstraintScopeChainNode {
                    scope: None,
                    parents,
                    materialized: OnceLock::new(),
                }),
            },
        }
    }

    pub(crate) fn from_materialized(scopes: Vec<Arc<ShapeConstraintScope>>) -> Self {
        let mut chain = Self::empty();
        for scope in scopes.into_iter().rev() {
            chain = Self::with_scope(scope, [&chain]);
        }
        chain
    }

    pub(crate) fn materialize(&self) -> Vec<Arc<ShapeConstraintScope>> {
        self.as_slice().to_vec()
    }

    pub(crate) fn as_slice(&self) -> &[Arc<ShapeConstraintScope>] {
        self.node
            .materialized
            .get_or_init(|| {
                let mut scopes = Vec::new();
                let mut seen_scopes = HashSet::new();
                let mut seen_nodes = HashSet::new();
                let mut visited_nodes = 0;
                self.extend_materialized(
                    &mut scopes,
                    &mut seen_scopes,
                    &mut seen_nodes,
                    &mut visited_nodes,
                );
                scopes
            })
            .as_slice()
    }

    #[cfg(test)]
    pub(crate) fn materialize_with_visit_count(&self) -> (Vec<Arc<ShapeConstraintScope>>, usize) {
        let mut scopes = Vec::new();
        let mut seen_scopes = HashSet::new();
        let mut seen_nodes = HashSet::new();
        let mut visited_nodes = 0;
        self.extend_materialized(
            &mut scopes,
            &mut seen_scopes,
            &mut seen_nodes,
            &mut visited_nodes,
        );
        (scopes, visited_nodes)
    }

    fn extend_materialized(
        &self,
        scopes: &mut Vec<Arc<ShapeConstraintScope>>,
        seen_scopes: &mut HashSet<*const ShapeConstraintScope>,
        seen_nodes: &mut HashSet<*const ConstraintScopeChainNode>,
        visited_nodes: &mut usize,
    ) {
        if !seen_nodes.insert(Arc::as_ptr(&self.node)) {
            return;
        }
        *visited_nodes += 1;
        if let Some(scope) = &self.node.scope {
            if seen_scopes.insert(Arc::as_ptr(scope)) {
                scopes.push(Arc::clone(scope));
            }
        }
        for parent in &self.node.parents {
            parent.extend_materialized(scopes, seen_scopes, seen_nodes, visited_nodes);
        }
    }
}

#[cfg(test)]
mod tests;
