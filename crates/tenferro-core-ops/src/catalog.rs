/// High-level category for a core primitive operation.
///
/// # Examples
///
/// ```rust
/// use tenferro_core_ops::{descriptor, OpCategory, PrimitiveOpKind};
///
/// assert_eq!(
///     descriptor(PrimitiveOpKind::ShapeOf).category,
///     OpCategory::Host
/// );
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpCategory {
    Elementwise,
    Analytic,
    Structural,
    Reduction,
    Contraction,
    Indexing,
    Dynamic,
    Host,
}

/// Dtype compatibility policy for a core primitive operation.
///
/// # Examples
///
/// ```rust
/// use tenferro_core_ops::{descriptor, DTypePolicy, PrimitiveOpKind};
///
/// assert_eq!(
///     descriptor(PrimitiveOpKind::Compare).dtype_policy,
///     DTypePolicy::CompareToBool
/// );
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DTypePolicy {
    SameAny,
    SameNumeric,
    SameFloat,
    SameFloatOrComplex,
    CompareToBool,
    BoolSelect,
    Convert,
    Shape,
    Constant,
}

/// Static metadata for one core primitive operation.
///
/// # Examples
///
/// ```rust
/// use tenferro_core_ops::{descriptor, PrimitiveOpKind};
///
/// let add = descriptor(PrimitiveOpKind::Add);
/// assert_eq!(add.min_inputs, 2);
/// assert_eq!(add.max_inputs, 2);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PrimitiveOpDescriptor {
    /// Catalog key for this operation.
    pub kind: PrimitiveOpKind,
    /// Stable snake-case operation name for diagnostics and descriptors.
    pub name: &'static str,
    /// Broad execution category.
    pub category: OpCategory,
    /// Dtype compatibility policy.
    pub dtype_policy: DTypePolicy,
    /// Minimum number of inputs accepted by the op.
    pub min_inputs: u8,
    /// Maximum number of inputs accepted by the op.
    pub max_inputs: u8,
    /// Whether this op is executed by host/runtime logic rather than a tensor backend.
    pub host_only: bool,
}

macro_rules! primitive_ops {
    ($macro:ident) => {
        $macro! {
            Add, "add", Elementwise, SameNumeric, 2, 2, false;
            Mul, "mul", Elementwise, SameNumeric, 2, 2, false;
            Neg, "neg", Elementwise, SameNumeric, 1, 1, false;
            Conj, "conj", Elementwise, SameFloatOrComplex, 1, 1, false;
            Div, "div", Elementwise, SameFloatOrComplex, 2, 2, false;
            Abs, "abs", Elementwise, SameFloat, 1, 1, false;
            Sign, "sign", Elementwise, SameFloat, 1, 1, false;
            Maximum, "maximum", Elementwise, SameFloat, 2, 2, false;
            Minimum, "minimum", Elementwise, SameFloat, 2, 2, false;
            Compare, "compare", Elementwise, CompareToBool, 2, 2, false;
            Select, "select", Elementwise, BoolSelect, 3, 3, false;
            Clamp, "clamp", Elementwise, SameFloat, 3, 3, false;
            Exp, "exp", Analytic, SameFloatOrComplex, 1, 1, false;
            Log, "log", Analytic, SameFloatOrComplex, 1, 1, false;
            Sin, "sin", Analytic, SameFloatOrComplex, 1, 1, false;
            Cos, "cos", Analytic, SameFloatOrComplex, 1, 1, false;
            Tanh, "tanh", Analytic, SameFloatOrComplex, 1, 1, false;
            Sqrt, "sqrt", Analytic, SameFloatOrComplex, 1, 1, false;
            Rsqrt, "rsqrt", Analytic, SameFloatOrComplex, 1, 1, false;
            Pow, "pow", Analytic, SameFloatOrComplex, 2, 2, false;
            Expm1, "expm1", Analytic, SameFloatOrComplex, 1, 1, false;
            Log1p, "log1p", Analytic, SameFloatOrComplex, 1, 1, false;
            DotGeneral, "dot_general", Contraction, SameFloatOrComplex, 2, 2, false;
            ReduceSum, "reduce_sum", Reduction, SameNumeric, 1, 1, false;
            ReduceProd, "reduce_prod", Reduction, SameNumeric, 1, 1, false;
            ReduceMax, "reduce_max", Reduction, SameFloat, 1, 1, false;
            ReduceMin, "reduce_min", Reduction, SameFloat, 1, 1, false;
            Transpose, "transpose", Structural, SameAny, 1, 1, false;
            Reshape, "reshape", Structural, SameAny, 1, 1, false;
            BroadcastInDim, "broadcast_in_dim", Structural, SameAny, 1, 1, false;
            Convert, "convert", Structural, Convert, 1, 1, false;
            ExtractDiag, "extract_diag", Structural, SameAny, 1, 1, false;
            EmbedDiag, "embed_diag", Structural, SameAny, 1, 1, false;
            Tril, "tril", Structural, SameAny, 1, 1, false;
            Triu, "triu", Structural, SameAny, 1, 1, false;
            Gather, "gather", Indexing, SameAny, 2, 2, false;
            GatherDynamicSliceSizes, "gather_dynamic_slice_sizes", Indexing, SameAny, 2, 2, false;
            Scatter, "scatter", Indexing, SameAny, 3, 3, false;
            Slice, "slice", Indexing, SameAny, 1, 1, false;
            DynamicSlice, "dynamic_slice", Indexing, SameAny, 2, 2, false;
            DynamicUpdateSlice, "dynamic_update_slice", Indexing, SameAny, 3, 3, false;
            Pad, "pad", Indexing, SameAny, 1, 1, false;
            Concatenate, "concatenate", Indexing, SameAny, 1, u8::MAX, false;
            Reverse, "reverse", Indexing, SameAny, 1, 1, false;
            ShapeOf, "shape_of", Host, Shape, 1, 1, true;
            DynamicTruncate, "dynamic_truncate", Dynamic, SameAny, 2, 2, true;
            PadToMatch, "pad_to_match", Dynamic, SameAny, 2, 2, true;
            Constant, "constant", Host, Constant, 0, 0, true;
        }
    };
}

macro_rules! define_kind {
    ($( $variant:ident, $name:literal, $category:ident, $policy:ident, $min:expr, $max:expr, $host:expr; )*) => {
        /// Catalog key for a core primitive operation.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use tenferro_core_ops::{descriptor, PrimitiveOpKind};
        ///
        /// assert_eq!(descriptor(PrimitiveOpKind::Add).name, "add");
        /// ```
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum PrimitiveOpKind {
            $( $variant, )*
        }

        impl PrimitiveOpKind {
            /// Number of primitive operation kinds in the catalog.
            ///
            /// # Examples
            ///
            /// ```rust
            /// use tenferro_core_ops::PrimitiveOpKind;
            ///
            /// assert!(PrimitiveOpKind::COUNT > 0);
            /// ```
            pub const COUNT: usize = [$(PrimitiveOpKind::$variant),*].len();

            /// Return this kind's dense catalog index.
            ///
            /// # Examples
            ///
            /// ```rust
            /// use tenferro_core_ops::PrimitiveOpKind;
            ///
            /// assert_eq!(PrimitiveOpKind::Add.as_index(), 0);
            /// ```
            pub const fn as_index(self) -> usize {
                self as usize
            }
        }
    };
}

primitive_ops!(define_kind);

macro_rules! define_descriptors {
    ($( $variant:ident, $name:literal, $category:ident, $policy:ident, $min:expr, $max:expr, $host:expr; )*) => {
        const DESCRIPTORS: &[PrimitiveOpDescriptor] = &[
            $(
                PrimitiveOpDescriptor {
                    kind: PrimitiveOpKind::$variant,
                    name: $name,
                    category: OpCategory::$category,
                    dtype_policy: DTypePolicy::$policy,
                    min_inputs: $min,
                    max_inputs: $max,
                    host_only: $host,
                },
            )*
        ];

        /// Return the descriptor for a primitive operation kind.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use tenferro_core_ops::{descriptor, PrimitiveOpKind};
        ///
        /// assert_eq!(descriptor(PrimitiveOpKind::Add).name, "add");
        /// ```
        pub fn descriptor(kind: PrimitiveOpKind) -> &'static PrimitiveOpDescriptor {
            match kind {
                $(
                    PrimitiveOpKind::$variant => &DESCRIPTORS[PrimitiveOpKind::$variant as usize],
                )*
            }
        }
    };
}

primitive_ops!(define_descriptors);

/// Return all core primitive operation descriptors in catalog order.
///
/// # Examples
///
/// ```rust
/// use tenferro_core_ops::all_primitive_descriptors;
///
/// assert!(all_primitive_descriptors()
///     .iter()
///     .any(|descriptor| descriptor.name == "add"));
/// ```
pub fn all_primitive_descriptors() -> &'static [PrimitiveOpDescriptor] {
    DESCRIPTORS
}

#[doc(hidden)]
#[macro_export]
macro_rules! define_std_tensor_op {
    () => {
        #[derive(Clone, Debug)]
        pub enum StdTensorOp {
            // Semiring arithmetic core
            Add,
            Mul,
            Neg,
            Conj,
            DotGeneral {
                config: DotGeneralConfig,
            },
            Transpose {
                perm: Vec<usize>,
            },
            Reshape {
                to_shape: Vec<DimExpr>,
            },
            BroadcastInDim {
                shape: Vec<DimExpr>,
                dims: Vec<usize>,
            },
            Convert {
                from: DType,
                to: DType,
            },
            Constant {
                dtype: DType,
                bytes: Vec<u8>,
            },
            ReduceSum {
                axes: Vec<usize>,
            },

            // Elementwise (non-semiring)
            Div,
            Abs,
            Sign,
            Maximum,
            Minimum,
            Compare(CompareDir),
            Select,
            Clamp,

            // Analytic
            Exp,
            Log,
            Sin,
            Cos,
            Tanh,
            Sqrt,
            Rsqrt,
            Pow,
            Expm1,
            Log1p,

            // Diagonal extraction / embedding (AD-closed pair)
            ExtractDiag {
                axis_a: usize,
                axis_b: usize,
            },
            EmbedDiag {
                axis_a: usize,
                axis_b: usize,
            },
            Tril {
                k: i64,
            },
            Triu {
                k: i64,
            },

            // Indexing
            Gather(GatherConfig),
            GatherDynamicSliceSizes {
                offset_dims: Vec<usize>,
                collapsed_slice_dims: Vec<usize>,
                start_index_map: Vec<usize>,
                index_vector_dim: usize,
                slice_sizes: Vec<DimExpr>,
            },
            Scatter(ScatterConfig),
            Slice(SliceConfig),
            DynamicSlice {
                slice_sizes: Vec<usize>,
            },
            DynamicUpdateSlice,
            Pad(PadConfig),
            Concatenate {
                axis: usize,
                input_count: usize,
            },
            Reverse {
                axes: Vec<usize>,
            },
            ShapeOf {
                axis: usize,
            },
            DynamicTruncate {
                axis: usize,
            },
            PadToMatch {
                axis: usize,
            },

            // Reductions
            ReduceProd {
                axes: Vec<usize>,
            },
            ReduceMax {
                axes: Vec<usize>,
            },
            ReduceMin {
                axes: Vec<usize>,
            },

            /// Out-of-tree extension carrier.
            ///
            /// See [`crate::ext_op`] and `docs/spec/extension-op.md`. Identity,
            /// hashing, equality, arity, shape inference, and AD rules are delegated
            /// to the inner [`ExtensionOp`] trait object.
            Extension(Arc<dyn ExtensionOp>),
        }

        impl StdTensorOp {
            /// Return the core primitive catalog kind for this graph operation.
            ///
            /// Extension operations do not claim a core primitive kind; they are
            /// dispatched through their extension family id instead.
            ///
            /// # Examples
            ///
            /// ```rust
            /// use tenferro_core_ops::PrimitiveOpKind;
            /// use tenferro_ops::std_tensor_op::StdTensorOp;
            ///
            /// assert_eq!(StdTensorOp::Add.primitive_kind(), Some(PrimitiveOpKind::Add));
            /// ```
            pub fn primitive_kind(&self) -> Option<$crate::PrimitiveOpKind> {
                let kind = match self {
                    Self::Add => $crate::PrimitiveOpKind::Add,
                    Self::Mul => $crate::PrimitiveOpKind::Mul,
                    Self::Neg => $crate::PrimitiveOpKind::Neg,
                    Self::Conj => $crate::PrimitiveOpKind::Conj,
                    Self::DotGeneral { .. } => $crate::PrimitiveOpKind::DotGeneral,
                    Self::Transpose { .. } => $crate::PrimitiveOpKind::Transpose,
                    Self::Reshape { .. } => $crate::PrimitiveOpKind::Reshape,
                    Self::BroadcastInDim { .. } => $crate::PrimitiveOpKind::BroadcastInDim,
                    Self::Convert { .. } => $crate::PrimitiveOpKind::Convert,
                    Self::Constant { .. } => $crate::PrimitiveOpKind::Constant,
                    Self::ReduceSum { .. } => $crate::PrimitiveOpKind::ReduceSum,
                    Self::Div => $crate::PrimitiveOpKind::Div,
                    Self::Abs => $crate::PrimitiveOpKind::Abs,
                    Self::Sign => $crate::PrimitiveOpKind::Sign,
                    Self::Maximum => $crate::PrimitiveOpKind::Maximum,
                    Self::Minimum => $crate::PrimitiveOpKind::Minimum,
                    Self::Compare(_) => $crate::PrimitiveOpKind::Compare,
                    Self::Select => $crate::PrimitiveOpKind::Select,
                    Self::Clamp => $crate::PrimitiveOpKind::Clamp,
                    Self::Exp => $crate::PrimitiveOpKind::Exp,
                    Self::Log => $crate::PrimitiveOpKind::Log,
                    Self::Sin => $crate::PrimitiveOpKind::Sin,
                    Self::Cos => $crate::PrimitiveOpKind::Cos,
                    Self::Tanh => $crate::PrimitiveOpKind::Tanh,
                    Self::Sqrt => $crate::PrimitiveOpKind::Sqrt,
                    Self::Rsqrt => $crate::PrimitiveOpKind::Rsqrt,
                    Self::Pow => $crate::PrimitiveOpKind::Pow,
                    Self::Expm1 => $crate::PrimitiveOpKind::Expm1,
                    Self::Log1p => $crate::PrimitiveOpKind::Log1p,
                    Self::ExtractDiag { .. } => $crate::PrimitiveOpKind::ExtractDiag,
                    Self::EmbedDiag { .. } => $crate::PrimitiveOpKind::EmbedDiag,
                    Self::Tril { .. } => $crate::PrimitiveOpKind::Tril,
                    Self::Triu { .. } => $crate::PrimitiveOpKind::Triu,
                    Self::Gather(_) => $crate::PrimitiveOpKind::Gather,
                    Self::GatherDynamicSliceSizes { .. } => {
                        $crate::PrimitiveOpKind::GatherDynamicSliceSizes
                    }
                    Self::Scatter(_) => $crate::PrimitiveOpKind::Scatter,
                    Self::Slice(_) => $crate::PrimitiveOpKind::Slice,
                    Self::DynamicSlice { .. } => $crate::PrimitiveOpKind::DynamicSlice,
                    Self::DynamicUpdateSlice => $crate::PrimitiveOpKind::DynamicUpdateSlice,
                    Self::Pad(_) => $crate::PrimitiveOpKind::Pad,
                    Self::Concatenate { .. } => $crate::PrimitiveOpKind::Concatenate,
                    Self::Reverse { .. } => $crate::PrimitiveOpKind::Reverse,
                    Self::ShapeOf { .. } => $crate::PrimitiveOpKind::ShapeOf,
                    Self::DynamicTruncate { .. } => $crate::PrimitiveOpKind::DynamicTruncate,
                    Self::PadToMatch { .. } => $crate::PrimitiveOpKind::PadToMatch,
                    Self::ReduceProd { .. } => $crate::PrimitiveOpKind::ReduceProd,
                    Self::ReduceMax { .. } => $crate::PrimitiveOpKind::ReduceMax,
                    Self::ReduceMin { .. } => $crate::PrimitiveOpKind::ReduceMin,
                    Self::Extension(_) => return None,
                };
                Some(kind)
            }

            #[cfg(test)]
            pub(crate) fn sample_from_kind(kind: $crate::PrimitiveOpKind) -> Self {
                match kind {
                    $crate::PrimitiveOpKind::Add => Self::Add,
                    $crate::PrimitiveOpKind::Mul => Self::Mul,
                    $crate::PrimitiveOpKind::Neg => Self::Neg,
                    $crate::PrimitiveOpKind::Conj => Self::Conj,
                    $crate::PrimitiveOpKind::DotGeneral => Self::DotGeneral {
                        config: DotGeneralConfig {
                            lhs_contracting_dims: vec![0],
                            rhs_contracting_dims: vec![0],
                            lhs_batch_dims: vec![],
                            rhs_batch_dims: vec![],
                        },
                    },
                    $crate::PrimitiveOpKind::Transpose => Self::Transpose { perm: vec![0] },
                    $crate::PrimitiveOpKind::Reshape => Self::Reshape {
                        to_shape: vec![DimExpr::Const(1)],
                    },
                    $crate::PrimitiveOpKind::BroadcastInDim => Self::BroadcastInDim {
                        shape: vec![DimExpr::Const(1)],
                        dims: vec![0],
                    },
                    $crate::PrimitiveOpKind::Convert => Self::Convert {
                        from: DType::F32,
                        to: DType::F64,
                    },
                    $crate::PrimitiveOpKind::Constant => Self::Constant {
                        dtype: DType::F64,
                        bytes: 0.0_f64.to_le_bytes().to_vec(),
                    },
                    $crate::PrimitiveOpKind::ReduceSum => Self::ReduceSum { axes: vec![0] },
                    $crate::PrimitiveOpKind::Div => Self::Div,
                    $crate::PrimitiveOpKind::Abs => Self::Abs,
                    $crate::PrimitiveOpKind::Sign => Self::Sign,
                    $crate::PrimitiveOpKind::Maximum => Self::Maximum,
                    $crate::PrimitiveOpKind::Minimum => Self::Minimum,
                    $crate::PrimitiveOpKind::Compare => Self::Compare(CompareDir::Eq),
                    $crate::PrimitiveOpKind::Select => Self::Select,
                    $crate::PrimitiveOpKind::Clamp => Self::Clamp,
                    $crate::PrimitiveOpKind::Exp => Self::Exp,
                    $crate::PrimitiveOpKind::Log => Self::Log,
                    $crate::PrimitiveOpKind::Sin => Self::Sin,
                    $crate::PrimitiveOpKind::Cos => Self::Cos,
                    $crate::PrimitiveOpKind::Tanh => Self::Tanh,
                    $crate::PrimitiveOpKind::Sqrt => Self::Sqrt,
                    $crate::PrimitiveOpKind::Rsqrt => Self::Rsqrt,
                    $crate::PrimitiveOpKind::Pow => Self::Pow,
                    $crate::PrimitiveOpKind::Expm1 => Self::Expm1,
                    $crate::PrimitiveOpKind::Log1p => Self::Log1p,
                    $crate::PrimitiveOpKind::ExtractDiag => Self::ExtractDiag {
                        axis_a: 0,
                        axis_b: 1,
                    },
                    $crate::PrimitiveOpKind::EmbedDiag => Self::EmbedDiag {
                        axis_a: 0,
                        axis_b: 1,
                    },
                    $crate::PrimitiveOpKind::Tril => Self::Tril { k: 0 },
                    $crate::PrimitiveOpKind::Triu => Self::Triu { k: 0 },
                    $crate::PrimitiveOpKind::Gather => Self::Gather(GatherConfig {
                        offset_dims: vec![],
                        collapsed_slice_dims: vec![0],
                        start_index_map: vec![0],
                        index_vector_dim: 1,
                        slice_sizes: vec![1],
                    }),
                    $crate::PrimitiveOpKind::GatherDynamicSliceSizes => {
                        Self::GatherDynamicSliceSizes {
                            offset_dims: vec![],
                            collapsed_slice_dims: vec![0],
                            start_index_map: vec![0],
                            index_vector_dim: 1,
                            slice_sizes: vec![DimExpr::Const(1)],
                        }
                    }
                    $crate::PrimitiveOpKind::Scatter => Self::Scatter(ScatterConfig {
                        update_window_dims: vec![],
                        inserted_window_dims: vec![0],
                        scatter_dims_to_operand_dims: vec![0],
                        index_vector_dim: 1,
                    }),
                    $crate::PrimitiveOpKind::Slice => Self::Slice(SliceConfig {
                        starts: vec![0],
                        limits: vec![1],
                        strides: vec![1],
                    }),
                    $crate::PrimitiveOpKind::DynamicSlice => Self::DynamicSlice {
                        slice_sizes: vec![1],
                    },
                    $crate::PrimitiveOpKind::DynamicUpdateSlice => Self::DynamicUpdateSlice,
                    $crate::PrimitiveOpKind::Pad => Self::Pad(PadConfig {
                        edge_padding_low: vec![0],
                        edge_padding_high: vec![0],
                        interior_padding: vec![0],
                    }),
                    $crate::PrimitiveOpKind::Concatenate => Self::Concatenate {
                        axis: 0,
                        input_count: 1,
                    },
                    $crate::PrimitiveOpKind::Reverse => Self::Reverse { axes: vec![0] },
                    $crate::PrimitiveOpKind::ShapeOf => Self::ShapeOf { axis: 0 },
                    $crate::PrimitiveOpKind::DynamicTruncate => Self::DynamicTruncate { axis: 0 },
                    $crate::PrimitiveOpKind::PadToMatch => Self::PadToMatch { axis: 0 },
                    $crate::PrimitiveOpKind::ReduceProd => Self::ReduceProd { axes: vec![0] },
                    $crate::PrimitiveOpKind::ReduceMax => Self::ReduceMax { axes: vec![0] },
                    $crate::PrimitiveOpKind::ReduceMin => Self::ReduceMin { axes: vec![0] },
                }
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! define_elementwise_fusion_op {
    () => {
        /// Elementwise op kinds supported by backend fusion implementations.
        #[doc(hidden)]
        #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
        pub enum ElementwiseFusionOp {
            Add,
            Multiply,
            Negate,
            Conj,
            Divide,
            Abs,
            Maximum,
            Minimum,
            Clamp,
            Exp,
            Log,
            Sin,
            Cos,
            Tanh,
            Sqrt,
            Rsqrt,
            Pow,
            Expm1,
            Log1p,
        }

        #[cfg(test)]
        impl ElementwiseFusionOp {
            pub(crate) fn iter() -> impl Iterator<Item = Self> {
                [
                    Self::Add,
                    Self::Multiply,
                    Self::Negate,
                    Self::Conj,
                    Self::Divide,
                    Self::Abs,
                    Self::Maximum,
                    Self::Minimum,
                    Self::Clamp,
                    Self::Exp,
                    Self::Log,
                    Self::Sin,
                    Self::Cos,
                    Self::Tanh,
                    Self::Sqrt,
                    Self::Rsqrt,
                    Self::Pow,
                    Self::Expm1,
                    Self::Log1p,
                ]
                .into_iter()
            }

            pub(crate) fn from_primitive_kind(kind: $crate::PrimitiveOpKind) -> Option<Self> {
                match kind {
                    $crate::PrimitiveOpKind::Add => Some(Self::Add),
                    $crate::PrimitiveOpKind::Mul => Some(Self::Multiply),
                    $crate::PrimitiveOpKind::Neg => Some(Self::Negate),
                    $crate::PrimitiveOpKind::Conj => Some(Self::Conj),
                    $crate::PrimitiveOpKind::Div => Some(Self::Divide),
                    $crate::PrimitiveOpKind::Abs => Some(Self::Abs),
                    $crate::PrimitiveOpKind::Maximum => Some(Self::Maximum),
                    $crate::PrimitiveOpKind::Minimum => Some(Self::Minimum),
                    $crate::PrimitiveOpKind::Clamp => Some(Self::Clamp),
                    $crate::PrimitiveOpKind::Exp => Some(Self::Exp),
                    $crate::PrimitiveOpKind::Log => Some(Self::Log),
                    $crate::PrimitiveOpKind::Sin => Some(Self::Sin),
                    $crate::PrimitiveOpKind::Cos => Some(Self::Cos),
                    $crate::PrimitiveOpKind::Tanh => Some(Self::Tanh),
                    $crate::PrimitiveOpKind::Sqrt => Some(Self::Sqrt),
                    $crate::PrimitiveOpKind::Rsqrt => Some(Self::Rsqrt),
                    $crate::PrimitiveOpKind::Pow => Some(Self::Pow),
                    $crate::PrimitiveOpKind::Expm1 => Some(Self::Expm1),
                    $crate::PrimitiveOpKind::Log1p => Some(Self::Log1p),
                    _ => None,
                }
            }

            pub(crate) fn primitive_kind(self) -> $crate::PrimitiveOpKind {
                match self {
                    Self::Add => $crate::PrimitiveOpKind::Add,
                    Self::Multiply => $crate::PrimitiveOpKind::Mul,
                    Self::Negate => $crate::PrimitiveOpKind::Neg,
                    Self::Conj => $crate::PrimitiveOpKind::Conj,
                    Self::Divide => $crate::PrimitiveOpKind::Div,
                    Self::Abs => $crate::PrimitiveOpKind::Abs,
                    Self::Maximum => $crate::PrimitiveOpKind::Maximum,
                    Self::Minimum => $crate::PrimitiveOpKind::Minimum,
                    Self::Clamp => $crate::PrimitiveOpKind::Clamp,
                    Self::Exp => $crate::PrimitiveOpKind::Exp,
                    Self::Log => $crate::PrimitiveOpKind::Log,
                    Self::Sin => $crate::PrimitiveOpKind::Sin,
                    Self::Cos => $crate::PrimitiveOpKind::Cos,
                    Self::Tanh => $crate::PrimitiveOpKind::Tanh,
                    Self::Sqrt => $crate::PrimitiveOpKind::Sqrt,
                    Self::Rsqrt => $crate::PrimitiveOpKind::Rsqrt,
                    Self::Pow => $crate::PrimitiveOpKind::Pow,
                    Self::Expm1 => $crate::PrimitiveOpKind::Expm1,
                    Self::Log1p => $crate::PrimitiveOpKind::Log1p,
                }
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! define_exec_op {
    () => {
        #[derive(Clone, Debug)]
        pub enum ExecOp {
            Transpose {
                perm: Vec<usize>,
            },
            Reshape {
                shape: Vec<DimExpr>,
            },
            BroadcastInDim {
                shape: Vec<DimExpr>,
                dims: Vec<usize>,
            },
            Convert {
                to: DType,
            },
            Constant {
                dtype: DType,
                bytes: Vec<u8>,
            },
            DotGeneral(DotGeneralConfig),
            DotGeneralWithConj {
                config: DotGeneralConfig,
                lhs_conj: bool,
                rhs_conj: bool,
            },
            ReduceSum {
                axes: Vec<usize>,
            },
            ExtractDiag {
                axis_a: usize,
                axis_b: usize,
            },
            EmbedDiag {
                axis_a: usize,
                axis_b: usize,
            },
            Tril {
                k: i64,
            },
            Triu {
                k: i64,
            },
            Add,
            Multiply,
            Negate,
            Conj,
            Divide,
            Abs,
            Sign,
            Maximum,
            Minimum,
            Compare(CompareDir),
            Select,
            Clamp,
            Exp,
            Log,
            Sin,
            Cos,
            Tanh,
            Sqrt,
            Rsqrt,
            Pow,
            Expm1,
            Log1p,
            Gather(GatherConfig),
            GatherDynamicSliceSizes {
                offset_dims: Vec<usize>,
                collapsed_slice_dims: Vec<usize>,
                start_index_map: Vec<usize>,
                index_vector_dim: usize,
                slice_sizes: Vec<DimExpr>,
            },
            Scatter(ScatterConfig),
            Slice(SliceConfig),
            DynamicSlice {
                slice_sizes: Vec<usize>,
            },
            DynamicUpdateSlice,
            Pad(PadConfig),
            Concatenate {
                axis: usize,
            },
            Reverse {
                axes: Vec<usize>,
            },
            ShapeOf {
                axis: usize,
            },
            DynamicTruncate {
                axis: usize,
            },
            PadToMatch {
                axis: usize,
            },
            ReduceProd {
                axes: Vec<usize>,
            },
            ReduceMax {
                axes: Vec<usize>,
            },
            ReduceMin {
                axes: Vec<usize>,
            },
            /// Out-of-tree extension carrier in the execution IR.
            ///
            /// Payload and dispatch are defined by the inner [`ExtensionOp`]. The
            /// execution pipeline treats extensions as single-instruction FFI
            /// boundaries (spec Section 8): no elementwise fusion, and dispatch is
            /// routed through the executor's registered extension runtime.
            Extension(Arc<dyn ExtensionOp>),
        }

        impl ExecOp {
            pub(crate) fn primitive_kind(&self) -> Option<$crate::PrimitiveOpKind> {
                let kind = match self {
                    Self::Transpose { .. } => $crate::PrimitiveOpKind::Transpose,
                    Self::Reshape { .. } => $crate::PrimitiveOpKind::Reshape,
                    Self::BroadcastInDim { .. } => $crate::PrimitiveOpKind::BroadcastInDim,
                    Self::Convert { .. } => $crate::PrimitiveOpKind::Convert,
                    Self::Constant { .. } => $crate::PrimitiveOpKind::Constant,
                    Self::DotGeneral(_) | Self::DotGeneralWithConj { .. } => {
                        $crate::PrimitiveOpKind::DotGeneral
                    }
                    Self::ReduceSum { .. } => $crate::PrimitiveOpKind::ReduceSum,
                    Self::ExtractDiag { .. } => $crate::PrimitiveOpKind::ExtractDiag,
                    Self::EmbedDiag { .. } => $crate::PrimitiveOpKind::EmbedDiag,
                    Self::Tril { .. } => $crate::PrimitiveOpKind::Tril,
                    Self::Triu { .. } => $crate::PrimitiveOpKind::Triu,
                    Self::Add => $crate::PrimitiveOpKind::Add,
                    Self::Multiply => $crate::PrimitiveOpKind::Mul,
                    Self::Negate => $crate::PrimitiveOpKind::Neg,
                    Self::Conj => $crate::PrimitiveOpKind::Conj,
                    Self::Divide => $crate::PrimitiveOpKind::Div,
                    Self::Abs => $crate::PrimitiveOpKind::Abs,
                    Self::Sign => $crate::PrimitiveOpKind::Sign,
                    Self::Maximum => $crate::PrimitiveOpKind::Maximum,
                    Self::Minimum => $crate::PrimitiveOpKind::Minimum,
                    Self::Compare(_) => $crate::PrimitiveOpKind::Compare,
                    Self::Select => $crate::PrimitiveOpKind::Select,
                    Self::Clamp => $crate::PrimitiveOpKind::Clamp,
                    Self::Exp => $crate::PrimitiveOpKind::Exp,
                    Self::Log => $crate::PrimitiveOpKind::Log,
                    Self::Sin => $crate::PrimitiveOpKind::Sin,
                    Self::Cos => $crate::PrimitiveOpKind::Cos,
                    Self::Tanh => $crate::PrimitiveOpKind::Tanh,
                    Self::Sqrt => $crate::PrimitiveOpKind::Sqrt,
                    Self::Rsqrt => $crate::PrimitiveOpKind::Rsqrt,
                    Self::Pow => $crate::PrimitiveOpKind::Pow,
                    Self::Expm1 => $crate::PrimitiveOpKind::Expm1,
                    Self::Log1p => $crate::PrimitiveOpKind::Log1p,
                    Self::Gather(_) => $crate::PrimitiveOpKind::Gather,
                    Self::GatherDynamicSliceSizes { .. } => {
                        $crate::PrimitiveOpKind::GatherDynamicSliceSizes
                    }
                    Self::Scatter(_) => $crate::PrimitiveOpKind::Scatter,
                    Self::Slice(_) => $crate::PrimitiveOpKind::Slice,
                    Self::DynamicSlice { .. } => $crate::PrimitiveOpKind::DynamicSlice,
                    Self::DynamicUpdateSlice => $crate::PrimitiveOpKind::DynamicUpdateSlice,
                    Self::Pad(_) => $crate::PrimitiveOpKind::Pad,
                    Self::Concatenate { .. } => $crate::PrimitiveOpKind::Concatenate,
                    Self::Reverse { .. } => $crate::PrimitiveOpKind::Reverse,
                    Self::ShapeOf { .. } => $crate::PrimitiveOpKind::ShapeOf,
                    Self::DynamicTruncate { .. } => $crate::PrimitiveOpKind::DynamicTruncate,
                    Self::PadToMatch { .. } => $crate::PrimitiveOpKind::PadToMatch,
                    Self::ReduceProd { .. } => $crate::PrimitiveOpKind::ReduceProd,
                    Self::ReduceMax { .. } => $crate::PrimitiveOpKind::ReduceMax,
                    Self::ReduceMin { .. } => $crate::PrimitiveOpKind::ReduceMin,
                    Self::Extension(_) => return None,
                };
                Some(kind)
            }

            pub(crate) fn from_std_tensor_op(
                op: &tenferro_ops::std_tensor_op::StdTensorOp,
            ) -> Self {
                match op {
                    tenferro_ops::std_tensor_op::StdTensorOp::Add => Self::Add,
                    tenferro_ops::std_tensor_op::StdTensorOp::Mul => Self::Multiply,
                    tenferro_ops::std_tensor_op::StdTensorOp::Neg => Self::Negate,
                    tenferro_ops::std_tensor_op::StdTensorOp::Conj => Self::Conj,
                    tenferro_ops::std_tensor_op::StdTensorOp::Div => Self::Divide,
                    tenferro_ops::std_tensor_op::StdTensorOp::Abs => Self::Abs,
                    tenferro_ops::std_tensor_op::StdTensorOp::Sign => Self::Sign,
                    tenferro_ops::std_tensor_op::StdTensorOp::Maximum => Self::Maximum,
                    tenferro_ops::std_tensor_op::StdTensorOp::Minimum => Self::Minimum,
                    tenferro_ops::std_tensor_op::StdTensorOp::Compare(dir) => {
                        Self::Compare(dir.clone())
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::Select => Self::Select,
                    tenferro_ops::std_tensor_op::StdTensorOp::Clamp => Self::Clamp,
                    tenferro_ops::std_tensor_op::StdTensorOp::Exp => Self::Exp,
                    tenferro_ops::std_tensor_op::StdTensorOp::Log => Self::Log,
                    tenferro_ops::std_tensor_op::StdTensorOp::Sin => Self::Sin,
                    tenferro_ops::std_tensor_op::StdTensorOp::Cos => Self::Cos,
                    tenferro_ops::std_tensor_op::StdTensorOp::Tanh => Self::Tanh,
                    tenferro_ops::std_tensor_op::StdTensorOp::Sqrt => Self::Sqrt,
                    tenferro_ops::std_tensor_op::StdTensorOp::Rsqrt => Self::Rsqrt,
                    tenferro_ops::std_tensor_op::StdTensorOp::Pow => Self::Pow,
                    tenferro_ops::std_tensor_op::StdTensorOp::Expm1 => Self::Expm1,
                    tenferro_ops::std_tensor_op::StdTensorOp::Log1p => Self::Log1p,
                    tenferro_ops::std_tensor_op::StdTensorOp::Transpose { perm } => {
                        Self::Transpose { perm: perm.clone() }
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::Reshape { to_shape } => {
                        Self::Reshape {
                            shape: to_shape.clone(),
                        }
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::BroadcastInDim { shape, dims } => {
                        Self::BroadcastInDim {
                            shape: shape.clone(),
                            dims: dims.clone(),
                        }
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::Convert { to, .. } => {
                        Self::Convert { to: *to }
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::Constant { dtype, bytes } => {
                        Self::Constant {
                            dtype: *dtype,
                            bytes: bytes.clone(),
                        }
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::DotGeneral { config } => {
                        Self::DotGeneral(config.clone())
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::ReduceSum { axes } => {
                        Self::ReduceSum { axes: axes.clone() }
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::ReduceProd { axes } => {
                        Self::ReduceProd { axes: axes.clone() }
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::ReduceMax { axes } => {
                        Self::ReduceMax { axes: axes.clone() }
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::ReduceMin { axes } => {
                        Self::ReduceMin { axes: axes.clone() }
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::ExtractDiag { axis_a, axis_b } => {
                        Self::ExtractDiag {
                            axis_a: *axis_a,
                            axis_b: *axis_b,
                        }
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::EmbedDiag { axis_a, axis_b } => {
                        Self::EmbedDiag {
                            axis_a: *axis_a,
                            axis_b: *axis_b,
                        }
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::Tril { k } => Self::Tril { k: *k },
                    tenferro_ops::std_tensor_op::StdTensorOp::Triu { k } => Self::Triu { k: *k },
                    tenferro_ops::std_tensor_op::StdTensorOp::Gather(config) => {
                        Self::Gather(config.clone())
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::GatherDynamicSliceSizes {
                        offset_dims,
                        collapsed_slice_dims,
                        start_index_map,
                        index_vector_dim,
                        slice_sizes,
                    } => Self::GatherDynamicSliceSizes {
                        offset_dims: offset_dims.clone(),
                        collapsed_slice_dims: collapsed_slice_dims.clone(),
                        start_index_map: start_index_map.clone(),
                        index_vector_dim: *index_vector_dim,
                        slice_sizes: slice_sizes.clone(),
                    },
                    tenferro_ops::std_tensor_op::StdTensorOp::Scatter(config) => {
                        Self::Scatter(config.clone())
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::Slice(config) => {
                        Self::Slice(config.clone())
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::DynamicSlice { slice_sizes } => {
                        Self::DynamicSlice {
                            slice_sizes: slice_sizes.clone(),
                        }
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::DynamicUpdateSlice => {
                        Self::DynamicUpdateSlice
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::Pad(config) => {
                        Self::Pad(config.clone())
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::Concatenate { axis, .. } => {
                        Self::Concatenate { axis: *axis }
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::Reverse { axes } => {
                        Self::Reverse { axes: axes.clone() }
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::ShapeOf { axis } => {
                        Self::ShapeOf { axis: *axis }
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::DynamicTruncate { axis } => {
                        Self::DynamicTruncate { axis: *axis }
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::PadToMatch { axis } => {
                        Self::PadToMatch { axis: *axis }
                    }
                    tenferro_ops::std_tensor_op::StdTensorOp::Extension(op) => {
                        Self::Extension(op.clone())
                    }
                }
            }

            pub(crate) fn elementwise_fusion_op(&self) -> Option<ElementwiseFusionOp> {
                match self {
                    Self::Add => Some(ElementwiseFusionOp::Add),
                    Self::Multiply => Some(ElementwiseFusionOp::Multiply),
                    Self::Negate => Some(ElementwiseFusionOp::Negate),
                    Self::Conj => Some(ElementwiseFusionOp::Conj),
                    Self::Divide => Some(ElementwiseFusionOp::Divide),
                    Self::Abs => Some(ElementwiseFusionOp::Abs),
                    Self::Maximum => Some(ElementwiseFusionOp::Maximum),
                    Self::Minimum => Some(ElementwiseFusionOp::Minimum),
                    Self::Clamp => Some(ElementwiseFusionOp::Clamp),
                    Self::Exp => Some(ElementwiseFusionOp::Exp),
                    Self::Log => Some(ElementwiseFusionOp::Log),
                    Self::Sin => Some(ElementwiseFusionOp::Sin),
                    Self::Cos => Some(ElementwiseFusionOp::Cos),
                    Self::Tanh => Some(ElementwiseFusionOp::Tanh),
                    Self::Sqrt => Some(ElementwiseFusionOp::Sqrt),
                    Self::Rsqrt => Some(ElementwiseFusionOp::Rsqrt),
                    Self::Pow => Some(ElementwiseFusionOp::Pow),
                    Self::Expm1 => Some(ElementwiseFusionOp::Expm1),
                    Self::Log1p => Some(ElementwiseFusionOp::Log1p),
                    _ => None,
                }
            }

            #[cfg(test)]
            pub(crate) fn input_arity_bounds(&self) -> Option<(u8, u8)> {
                self.primitive_kind().map(|kind| {
                    let descriptor = $crate::descriptor(kind);
                    (descriptor.min_inputs, descriptor.max_inputs)
                })
            }

            #[cfg(test)]
            pub(crate) fn sample_from_kind(kind: $crate::PrimitiveOpKind) -> Self {
                match kind {
                    $crate::PrimitiveOpKind::Transpose => Self::Transpose { perm: vec![0] },
                    $crate::PrimitiveOpKind::Reshape => Self::Reshape {
                        shape: vec![DimExpr::Const(1)],
                    },
                    $crate::PrimitiveOpKind::BroadcastInDim => Self::BroadcastInDim {
                        shape: vec![DimExpr::Const(1)],
                        dims: vec![0],
                    },
                    $crate::PrimitiveOpKind::Convert => Self::Convert { to: DType::F64 },
                    $crate::PrimitiveOpKind::Constant => Self::Constant {
                        dtype: DType::F64,
                        bytes: 0.0_f64.to_le_bytes().to_vec(),
                    },
                    $crate::PrimitiveOpKind::DotGeneral => Self::DotGeneral(DotGeneralConfig {
                        lhs_contracting_dims: vec![0],
                        rhs_contracting_dims: vec![0],
                        lhs_batch_dims: vec![],
                        rhs_batch_dims: vec![],
                    }),
                    $crate::PrimitiveOpKind::ReduceSum => Self::ReduceSum { axes: vec![0] },
                    $crate::PrimitiveOpKind::ExtractDiag => Self::ExtractDiag {
                        axis_a: 0,
                        axis_b: 1,
                    },
                    $crate::PrimitiveOpKind::EmbedDiag => Self::EmbedDiag {
                        axis_a: 0,
                        axis_b: 1,
                    },
                    $crate::PrimitiveOpKind::Tril => Self::Tril { k: 0 },
                    $crate::PrimitiveOpKind::Triu => Self::Triu { k: 0 },
                    $crate::PrimitiveOpKind::Add => Self::Add,
                    $crate::PrimitiveOpKind::Mul => Self::Multiply,
                    $crate::PrimitiveOpKind::Neg => Self::Negate,
                    $crate::PrimitiveOpKind::Conj => Self::Conj,
                    $crate::PrimitiveOpKind::Div => Self::Divide,
                    $crate::PrimitiveOpKind::Abs => Self::Abs,
                    $crate::PrimitiveOpKind::Sign => Self::Sign,
                    $crate::PrimitiveOpKind::Maximum => Self::Maximum,
                    $crate::PrimitiveOpKind::Minimum => Self::Minimum,
                    $crate::PrimitiveOpKind::Compare => Self::Compare(CompareDir::Eq),
                    $crate::PrimitiveOpKind::Select => Self::Select,
                    $crate::PrimitiveOpKind::Clamp => Self::Clamp,
                    $crate::PrimitiveOpKind::Exp => Self::Exp,
                    $crate::PrimitiveOpKind::Log => Self::Log,
                    $crate::PrimitiveOpKind::Sin => Self::Sin,
                    $crate::PrimitiveOpKind::Cos => Self::Cos,
                    $crate::PrimitiveOpKind::Tanh => Self::Tanh,
                    $crate::PrimitiveOpKind::Sqrt => Self::Sqrt,
                    $crate::PrimitiveOpKind::Rsqrt => Self::Rsqrt,
                    $crate::PrimitiveOpKind::Pow => Self::Pow,
                    $crate::PrimitiveOpKind::Expm1 => Self::Expm1,
                    $crate::PrimitiveOpKind::Log1p => Self::Log1p,
                    $crate::PrimitiveOpKind::Gather => Self::Gather(GatherConfig {
                        offset_dims: vec![],
                        collapsed_slice_dims: vec![0],
                        start_index_map: vec![0],
                        index_vector_dim: 1,
                        slice_sizes: vec![1],
                    }),
                    $crate::PrimitiveOpKind::GatherDynamicSliceSizes => {
                        Self::GatherDynamicSliceSizes {
                            offset_dims: vec![],
                            collapsed_slice_dims: vec![0],
                            start_index_map: vec![0],
                            index_vector_dim: 1,
                            slice_sizes: vec![DimExpr::Const(1)],
                        }
                    }
                    $crate::PrimitiveOpKind::Scatter => Self::Scatter(ScatterConfig {
                        update_window_dims: vec![],
                        inserted_window_dims: vec![0],
                        scatter_dims_to_operand_dims: vec![0],
                        index_vector_dim: 1,
                    }),
                    $crate::PrimitiveOpKind::Slice => Self::Slice(SliceConfig {
                        starts: vec![0],
                        limits: vec![1],
                        strides: vec![1],
                    }),
                    $crate::PrimitiveOpKind::DynamicSlice => Self::DynamicSlice {
                        slice_sizes: vec![1],
                    },
                    $crate::PrimitiveOpKind::DynamicUpdateSlice => Self::DynamicUpdateSlice,
                    $crate::PrimitiveOpKind::Pad => Self::Pad(PadConfig {
                        edge_padding_low: vec![0],
                        edge_padding_high: vec![0],
                        interior_padding: vec![0],
                    }),
                    $crate::PrimitiveOpKind::Concatenate => Self::Concatenate { axis: 0 },
                    $crate::PrimitiveOpKind::Reverse => Self::Reverse { axes: vec![0] },
                    $crate::PrimitiveOpKind::ShapeOf => Self::ShapeOf { axis: 0 },
                    $crate::PrimitiveOpKind::DynamicTruncate => Self::DynamicTruncate { axis: 0 },
                    $crate::PrimitiveOpKind::PadToMatch => Self::PadToMatch { axis: 0 },
                    $crate::PrimitiveOpKind::ReduceProd => Self::ReduceProd { axes: vec![0] },
                    $crate::PrimitiveOpKind::ReduceMax => Self::ReduceMax { axes: vec![0] },
                    $crate::PrimitiveOpKind::ReduceMin => Self::ReduceMin { axes: vec![0] },
                }
            }
        }
    };
}
