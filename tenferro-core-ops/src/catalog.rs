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
