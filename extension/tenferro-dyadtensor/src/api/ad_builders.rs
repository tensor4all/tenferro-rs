use super::runtime::*;
use super::*;

macro_rules! dispatch_linalg_ad_runtime {
    ($ty:ty, $capability:expr, $op:literal, |$ctx:ident, $backend:ident| $body:expr) => {{
        with_linalg_runtime::<$ty, _>(
            $op,
            $capability,
            |$ctx| {
                type $backend = CpuBackend;
                $body
            },
            |$ctx| {
                type $backend = tenferro_prims::CudaBackend;
                $body
            },
            |$ctx| {
                type $backend = tenferro_prims::RocmBackend;
                $body
            },
        )
    }};
}

macro_rules! run_unary_tensor_ad {
    (
        ty = $ty:ty,
        capability = $capability:expr,
        op = $op_name:literal,
        pullback = $pullback_op_name:literal,
        input = $input:expr,
        primal = |$primal_ctx:ident, $primal_backend:ident, $primal_tensor:ident| $primal_body:expr,
        frule = |$frule_ctx:ident, $frule_backend:ident, $frule_tensor:ident, $frule_tangent:ident| $frule_body:expr,
        rrule = |$rrule_ctx:ident, $rrule_backend:ident, $rrule_tensor:ident, $rrule_cotangent:ident| $rrule_body:expr $(,)?
    ) => {{
        let operands = [$input];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
        let (input_primal, input_tangent) =
            dispatch_linalg_ad_runtime!($ty, $capability, $op_name, |ctx, Backend| {
                dense_input_snapshot_in_backend::<Backend, _, $ty>(ctx, $input, needs_tangent)
            })?;

        let (primal, tangent) = if needs_tangent {
            let input_tangent = input_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: format!("{} missing materialized tangent", $op_name),
            })?;
            let (p, d) = dispatch_linalg_ad_runtime!(
                $ty,
                $capability,
                $op_name,
                |$frule_ctx, $frule_backend| {
                    let $frule_tensor = &input_primal;
                    let $frule_tangent = &input_tangent;
                    $frule_body
                }
            )?;
            (p, Some(d))
        } else {
            let primal = dispatch_linalg_ad_runtime!(
                $ty,
                $capability,
                $op_name,
                |$primal_ctx, $primal_backend| {
                    let $primal_tensor = &input_primal;
                    $primal_body
                }
            )?;
            (primal, None)
        };

        let out = wrap_dense_ad_output($op_name, &operands, primal, tangent, 0)?;

        if let AdValue::Reverse { node, tape, .. } = out.as_value() {
            let input_spec = collect_reverse_input_specs(&operands)
                .into_iter()
                .next()
                .flatten();
            let output_node = *node;
            let tape_id = *tape;

            reverse_tape::register_rule::<$ty>(
                tape_id,
                output_node,
                Box::new(move |cotangent| {
                    let grad = dispatch_linalg_ad_runtime!(
                        $ty,
                        $capability,
                        $pullback_op_name,
                        |$rrule_ctx, $rrule_backend| {
                            let $rrule_tensor = &input_primal;
                            let $rrule_cotangent = cotangent;
                            $rrule_body
                        }
                    )?;

                    let Some(spec) = &input_spec else {
                        return Ok(Vec::new());
                    };
                    let grad = compress_pullback_like($op_name, grad, &spec.layout)?;
                    Ok(vec![(spec.node, grad)])
                }),
            )?;
        }

        Ok(out)
    }};
}

macro_rules! run_binary_tensor_ad {
    (
        ty = $ty:ty,
        capability = $capability:expr,
        op = $op_name:literal,
        pullback = $pullback_op_name:literal,
        lhs = $lhs:expr,
        rhs = $rhs:expr,
        primal = |$primal_ctx:ident, $primal_backend:ident, $lhs_primal:ident, $rhs_primal:ident| $primal_body:expr,
        frule = |$frule_ctx:ident, $frule_backend:ident, $frule_lhs:ident, $frule_rhs:ident, $lhs_tangent:ident, $rhs_tangent:ident| $frule_body:expr,
        rrule = |$rrule_ctx:ident, $rrule_backend:ident, $rrule_lhs:ident, $rrule_rhs:ident, $rrule_cotangent:ident| $rrule_body:expr $(,)?
    ) => {{
        let operands = [$lhs, $rhs];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
        let ((lhs_primal, lhs_tangent), (rhs_primal, rhs_tangent)) =
            dispatch_linalg_ad_runtime!($ty, $capability, $op_name, |ctx, Backend| {
                Ok((
                    dense_input_snapshot_in_backend::<Backend, _, $ty>(ctx, $lhs, needs_tangent)?,
                    dense_input_snapshot_in_backend::<Backend, _, $ty>(ctx, $rhs, needs_tangent)?,
                ))
            })?;

        let (primal, tangent) = if needs_tangent {
            let lhs_tangent = lhs_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: format!("{} missing materialized lhs tangent", $op_name),
            })?;
            let rhs_tangent = rhs_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: format!("{} missing materialized rhs tangent", $op_name),
            })?;
            let (p, d) = dispatch_linalg_ad_runtime!(
                $ty,
                $capability,
                $op_name,
                |$frule_ctx, $frule_backend| {
                    let $frule_lhs = &lhs_primal;
                    let $frule_rhs = &rhs_primal;
                    let $lhs_tangent = &lhs_tangent;
                    let $rhs_tangent = &rhs_tangent;
                    $frule_body
                }
            )?;
            (p, Some(d))
        } else {
            let primal = dispatch_linalg_ad_runtime!(
                $ty,
                $capability,
                $op_name,
                |$primal_ctx, $primal_backend| {
                    let $lhs_primal = &lhs_primal;
                    let $rhs_primal = &rhs_primal;
                    $primal_body
                }
            )?;
            (primal, None)
        };

        let out = wrap_dense_ad_output($op_name, &operands, primal, tangent, 0)?;

        if let AdValue::Reverse { node, tape, .. } = out.as_value() {
            let reverse_specs = collect_reverse_input_specs(&operands);
            let output_node = *node;
            let tape_id = *tape;

            reverse_tape::register_rule::<$ty>(
                tape_id,
                output_node,
                Box::new(move |cotangent| {
                    let (grad_lhs, grad_rhs) = dispatch_linalg_ad_runtime!(
                        $ty,
                        $capability,
                        $pullback_op_name,
                        |$rrule_ctx, $rrule_backend| {
                            let $rrule_lhs = &lhs_primal;
                            let $rrule_rhs = &rhs_primal;
                            let $rrule_cotangent = cotangent;
                            $rrule_body
                        }
                    )?;

                    let mut input_grads = Vec::new();
                    if let Some(spec) = &reverse_specs[0] {
                        let grad_lhs = compress_pullback_like($op_name, grad_lhs, &spec.layout)?;
                        input_grads.push((spec.node, grad_lhs));
                    }
                    if let Some(spec) = &reverse_specs[1] {
                        let grad_rhs = compress_pullback_like($op_name, grad_rhs, &spec.layout)?;
                        input_grads.push((spec.node, grad_rhs));
                    }
                    Ok(input_grads)
                }),
            )?;
        }

        Ok(out)
    }};
}

/// Builder for AD einsum.
/// # Examples
///
/// ```text
/// // Construct `EinsumAdBuilder` via its corresponding operation constructor.
/// ```
pub struct EinsumAdBuilder<'a, T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
    tenferro_prims::CudaBackend: TensorPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    tenferro_prims::RocmBackend: TensorPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
{
    subscripts: &'a str,
    operands: &'a [&'a AdTensor<T>],
    size_dict: Option<&'a HashMap<u32, usize>>,
}

fn structured_einsum_pullback_in_backend<B, C, T>(
    ctx: &mut C,
    subscripts: &Subscripts,
    reverse_nodes: &[Option<NodeId>],
    primals: &[StructuredTensor<T>],
    cotangent: &StructuredTensor<T>,
) -> Result<Vec<(NodeId, Tensor<T>)>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    B: TensorPrims<Standard<T>, Context = C>,
{
    let mut input_grads = Vec::new();

    for (k, maybe_node) in reverse_nodes.iter().enumerate() {
        let Some(node) = maybe_node else {
            continue;
        };
        let rev_subs = reverse_subscripts(subscripts, k);
        let mut rev_operands: Vec<&StructuredTensor<T>> = Vec::with_capacity(primals.len());
        rev_operands.push(cotangent);
        for (idx, operand) in primals.iter().enumerate() {
            if idx != k {
                rev_operands.push(operand);
            }
        }
        let grad = einsum_with_subscripts_in_ctx::<B, _, T>(ctx, &rev_subs, &rev_operands)?;
        input_grads.push((*node, grad.into_payload()));
    }

    Ok(input_grads)
}

fn dense_einsum_pullback_in_backend<B, C, T>(
    ctx: &mut C,
    subscripts: &str,
    reverse_specs: &[Option<ReverseInputSpec<T>>],
    primals: &[Tensor<T>],
    cotangent: &Tensor<T>,
) -> Result<Vec<(NodeId, Tensor<T>)>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    B: TensorPrims<Standard<T>, Context = C>,
{
    let primal_refs: Vec<&Tensor<T>> = primals.iter().collect();
    let gradients =
        tf_einsum::einsum_rrule::<Standard<T>, B>(ctx, subscripts, &primal_refs, cotangent)
            .map_err(Error::from)?;
    if gradients.len() != reverse_specs.len() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "einsum_ad pullback arity mismatch: expected {}, got {}",
                reverse_specs.len(),
                gradients.len()
            ),
        });
    }

    let mut input_grads = Vec::new();
    for (k, grad) in gradients.into_iter().enumerate() {
        let Some(spec) = &reverse_specs[k] else {
            continue;
        };
        let grad =
            compress_pullback_like_in_backend::<B, _, T>(ctx, "einsum_ad", grad, &spec.layout)?;
        input_grads.push((spec.node, grad));
    }
    Ok(input_grads)
}

fn run_einsum_ad_in_backend<B, C, T>(
    ctx: &mut C,
    subscripts: &str,
    operands: &[&AdTensor<T>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<AdTensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>> + 'static,
    B: TensorPrims<Standard<T>, Context = C>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
    tenferro_prims::CudaBackend: TensorPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    tenferro_prims::RocmBackend: TensorPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
{
    if size_dict.is_none() && !subscripts.contains('(') {
        let subs = Subscripts::parse(subscripts).map_err(Error::from)?;
        let primals: Vec<&StructuredTensor<T>> =
            operands.iter().map(|op| op.structured_primal()).collect();
        let primal_out = einsum_with_subscripts_in_ctx::<B, _, T>(ctx, &subs, &primals)?;

        let tangents = collect_structured_ad_tangents(operands);
        let tangent_out = if has_forward(operands) || has_any_tangent(operands) {
            sum_structured_einsum_tangent_terms::<B, _, T>(ctx, &subs, &primals, &tangents)?
        } else {
            None
        };

        let out = wrap_structured_ad_output("einsum_ad", operands, primal_out, tangent_out, 0)?;

        if let AdValue::Reverse { node, tape, .. } = out.as_value() {
            let subscripts = subs.clone();
            let reverse_nodes = collect_reverse_input_nodes(operands);
            let primal_owned: Vec<StructuredTensor<T>> =
                primals.iter().map(|tensor| (*tensor).clone()).collect();
            let output_layout = out.structured_primal().clone();
            let output_node = *node;
            let tape_id = *tape;

            reverse_tape::register_rule::<T>(
                tape_id,
                output_node,
                Box::new(move |cotangent| {
                    let cotangent = output_layout.with_payload_like(cotangent.clone())?;
                    with_einsum_runtime::<T, _>(
                        "einsum_ad_pullback_structured",
                        |ctx| {
                            structured_einsum_pullback_in_backend::<CpuBackend, _, T>(
                                ctx,
                                &subscripts,
                                &reverse_nodes,
                                &primal_owned,
                                &cotangent,
                            )
                        },
                        |ctx| {
                            structured_einsum_pullback_in_backend::<tenferro_prims::CudaBackend, _, T>(
                                ctx,
                                &subscripts,
                                &reverse_nodes,
                                &primal_owned,
                                &cotangent,
                            )
                        },
                        |ctx| {
                            structured_einsum_pullback_in_backend::<tenferro_prims::RocmBackend, _, T>(
                                ctx,
                                &subscripts,
                                &reverse_nodes,
                                &primal_owned,
                                &cotangent,
                            )
                        },
                    )
                }),
            )?;
        }

        return Ok(out);
    }

    let needs_tangent = has_forward(operands) || has_any_tangent(operands);
    let dense_inputs: Vec<(Tensor<T>, Option<Tensor<T>>)> = operands
        .iter()
        .map(|op| dense_input_snapshot_in_backend::<B, _, T>(ctx, op, needs_tangent))
        .collect::<Result<_>>()?;
    let primal_owned: Vec<Tensor<T>> = dense_inputs
        .iter()
        .map(|(primal, _)| primal.clone())
        .collect();
    let tangent_owned: Vec<Option<Tensor<T>>> = dense_inputs
        .iter()
        .map(|(_, tangent)| tangent.clone())
        .collect();
    let primals: Vec<&Tensor<T>> = primal_owned.iter().collect();
    let primal_out = tf_einsum::einsum::<Standard<T>, B>(ctx, subscripts, &primals, size_dict)
        .map_err(Error::from)?;

    let tangents: Vec<Option<&Tensor<T>>> = tangent_owned
        .iter()
        .map(|tangent| tangent.as_ref())
        .collect();
    let tangent_out = if needs_tangent {
        sum_einsum_tangent_terms::<B, _, T>(ctx, subscripts, &primals, &tangents, size_dict)?
    } else {
        None
    };

    let out = wrap_dense_ad_output("einsum_ad", operands, primal_out, tangent_out, 0)?;

    if let AdValue::Reverse { node, tape, .. } = out.as_value() {
        let subscripts = subscripts.to_string();
        let reverse_specs = collect_reverse_input_specs(operands);
        let output_node = *node;
        let tape_id = *tape;

        reverse_tape::register_rule::<T>(
            tape_id,
            output_node,
            Box::new(move |cotangent| {
                with_einsum_runtime::<T, _>(
                    "einsum_ad_pullback",
                    |ctx| {
                        dense_einsum_pullback_in_backend::<CpuBackend, _, T>(
                            ctx,
                            &subscripts,
                            &reverse_specs,
                            &primal_owned,
                            cotangent,
                        )
                    },
                    |ctx| {
                        dense_einsum_pullback_in_backend::<tenferro_prims::CudaBackend, _, T>(
                            ctx,
                            &subscripts,
                            &reverse_specs,
                            &primal_owned,
                            cotangent,
                        )
                    },
                    |ctx| {
                        dense_einsum_pullback_in_backend::<tenferro_prims::RocmBackend, _, T>(
                            ctx,
                            &subscripts,
                            &reverse_specs,
                            &primal_owned,
                            cotangent,
                        )
                    },
                )
            }),
        )?;
    }

    Ok(out)
}

impl<'a, T> EinsumAdBuilder<'a, T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
    tenferro_prims::CudaBackend: TensorPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    tenferro_prims::RocmBackend: TensorPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
{
    /// Sets optional size dictionary for output-only labels.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.size_dict(&size_dict);
    /// ```
    pub fn size_dict(mut self, size_dict: &'a HashMap<u32, usize>) -> Self {
        self.size_dict = Some(size_dict);
        self
    }

    /// Executes AD einsum with mode propagation.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>>
    where
        T: 'static,
    {
        let subscripts = self.subscripts;
        let operands = self.operands;
        let size_dict = self.size_dict;
        with_einsum_runtime::<T, _>(
            "einsum_ad",
            |ctx| {
                run_einsum_ad_in_backend::<CpuBackend, _, T>(ctx, subscripts, operands, size_dict)
            },
            |ctx| {
                run_einsum_ad_in_backend::<tenferro_prims::CudaBackend, _, T>(
                    ctx, subscripts, operands, size_dict,
                )
            },
            |ctx| {
                run_einsum_ad_in_backend::<tenferro_prims::RocmBackend, _, T>(
                    ctx, subscripts, operands, size_dict,
                )
            },
        )
    }
}

/// Creates a builder for AD einsum.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{einsum_ad, set_default_runtime, AdTensor, RuntimeContext};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
/// let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
/// let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
/// let ad_a = AdTensor::new_primal(a);
/// let ad_b = AdTensor::new_primal(b);
/// let out = einsum_ad("ij,jk->ik", &[&ad_a, &ad_b]).run().unwrap();
/// assert_eq!(out.dims(), &[2, 2]);
/// ```
pub fn einsum_ad<'a, T>(
    subscripts: &'a str,
    operands: &'a [&'a AdTensor<T>],
) -> EinsumAdBuilder<'a, T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
    tenferro_prims::CudaBackend: TensorPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    tenferro_prims::RocmBackend: TensorPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
{
    EinsumAdBuilder {
        subscripts,
        operands,
        size_dict: None,
    }
}

/// Builder for AD full reduction / sum.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::sum_ad(&x).run()?;
/// ```
pub struct SumAdBuilder<'a, T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    tensor: &'a AdTensor<T>,
}

impl<'a, T> SumAdBuilder<'a, T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    /// Executes AD full reduction / sum with mode propagation.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = builder.run()?;
    /// ```
    pub fn run(self) -> Result<AdTensor<T>>
    where
        T: Copy + 'static,
        CpuBackend: tenferro_prims::TensorScalarPrims<Standard<T>, Context = CpuContext>,
        tenferro_prims::CudaBackend:
            tenferro_prims::TensorScalarPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
        tenferro_prims::RocmBackend:
            tenferro_prims::TensorScalarPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
    {
        let operands = [self.tensor];
        let primal =
            StructuredTensor::from_dense(super::scalar_runtime::scalar_full_reduction_primal(
                "sum_ad",
                tenferro_prims::ScalarReductionOp::Sum,
                self.tensor.primal(),
            )?);
        let tangent = if has_forward(&operands) || has_any_tangent(&operands) {
            let tangent_input = if let Some(tangent) = self.tensor.structured_tangent() {
                tangent.payload()
            } else {
                &zero_like(self.tensor.primal())
            };
            Some(StructuredTensor::from_dense(
                super::scalar_runtime::scalar_full_reduction_primal(
                    "sum_ad",
                    tenferro_prims::ScalarReductionOp::Sum,
                    tangent_input,
                )?,
            ))
        } else {
            None
        };

        let out = wrap_structured_ad_output("sum_ad", &operands, primal, tangent, 0)?;

        if let AdValue::Reverse { node, tape, .. } = out.as_value() {
            let input_node = collect_reverse_input_nodes(&operands)
                .into_iter()
                .next()
                .flatten();
            let input_layout = self.tensor.structured_primal().clone();
            let output_node = *node;
            let tape_id = *tape;

            reverse_tape::register_rule::<T>(
                tape_id,
                output_node,
                Box::new(move |cotangent| {
                    let Some(input_node) = input_node else {
                        return Ok(Vec::new());
                    };
                    with_runtime(
                        |_ctx| {
                            let scalar = scalar_from_rank0_tensor(cotangent, "sum_ad")?;
                            let payload = broadcast_scalar_like(scalar, input_layout.payload())?;
                            let grad = input_layout.with_payload_like(payload)?;
                            Ok(vec![(input_node, grad.into_payload())])
                        },
                        |_ctx| Err(unsupported_runtime_capability("sum_ad_pullback", "cuda")),
                        |_ctx| Err(unsupported_runtime_capability("sum_ad_pullback", "rocm")),
                    )
                }),
            )?;
        }

        Ok(out)
    }
}

/// Creates a builder for AD full reduction / sum.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::sum_ad(&x).run()?;
/// ```
pub fn sum_ad<'a, T>(tensor: &'a AdTensor<T>) -> SumAdBuilder<'a, T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    SumAdBuilder { tensor }
}

/// Builder for AD SVD.
/// # Examples
///
/// ```text
/// // Construct `SvdAdBuilder` via its corresponding operation constructor.
/// ```
pub struct SvdAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
    options: Option<&'a SvdOptions>,
}

impl<'a, T> SvdAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    /// Sets optional SVD options.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.options(&options);
    /// ```
    pub fn options(mut self, options: &'a SvdOptions) -> Self {
        self.options = Some(options);
        self
    }

    /// Executes AD SVD.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdSvdResult<T>> {
        let operands = [self.tensor];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
        let (input_primal, input_tangent) = dispatch_linalg_ad_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::ThinSvd,
            "svd_ad",
            |ctx, Backend| {
                dense_input_snapshot_in_backend::<Backend, _, T>(ctx, self.tensor, needs_tangent)
            }
        )?;

        let (primal, tangent) = if needs_tangent {
            let dt = input_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: "svd_ad missing materialized tangent".to_string(),
            })?;
            let (p, d) = dispatch_linalg_ad_runtime!(
                T,
                tenferro_linalg::backend::LinalgCapabilityOp::ThinSvd,
                "svd_ad",
                |ctx, Backend| {
                    let _ = std::marker::PhantomData::<Backend>;
                    tenferro_linalg::svd_frule::<T, _>(ctx, &input_primal, &dt, self.options)
                        .map_err(Error::from)
                }
            )?;
            (p, Some(d))
        } else {
            (
                dispatch_linalg_ad_runtime!(
                    T,
                    tenferro_linalg::backend::LinalgCapabilityOp::ThinSvd,
                    "svd_ad",
                    |ctx, Backend| {
                        let _ = std::marker::PhantomData::<Backend>;
                        tenferro_linalg::svd::<T, _>(ctx, &input_primal, self.options)
                            .map_err(Error::from)
                    }
                )?,
                None,
            )
        };

        let (du, ds, dvt) = if let Some(d) = tangent {
            (Some(d.u), Some(d.s), Some(d.vt))
        } else {
            (None, None, None)
        };

        let out_u = wrap_dense_ad_output("svd_ad", &operands, primal.u, du, 1)?;
        let out_s = wrap_dense_ad_output("svd_ad", &operands, primal.s, ds, 2)?;
        let out_vt = wrap_dense_ad_output("svd_ad", &operands, primal.vt, dvt, 3)?;

        let input_spec = collect_reverse_input_specs(&operands)
            .into_iter()
            .next()
            .flatten();
        if let Some(spec) = input_spec {
            let a_primal = input_primal.clone();
            let options = self.options.cloned();

            if let AdValue::Reverse { node, tape, .. } = out_u.as_value() {
                let output_node = *node;
                let tape_id = *tape;
                let spec = spec.clone();
                let options = options.clone();
                reverse_tape::register_rule::<T>(
                    tape_id,
                    output_node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::ThinSvd,
                            "svd_ad_pullback_u",
                            |ctx, Backend| {
                                let _ = std::marker::PhantomData::<Backend>;
                                tenferro_linalg::svd_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::SvdCotangent {
                                        u: Some(cotangent.clone()),
                                        s: None,
                                        vt: None,
                                    },
                                    options.as_ref(),
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = compress_pullback_like("svd_ad", grad, &spec.layout)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                )?;
            }

            if let AdValue::Reverse { node, tape, .. } = out_s.as_value() {
                let output_node = *node;
                let tape_id = *tape;
                let spec = spec.clone();
                let options = options.clone();
                let a_primal = input_primal.clone();
                reverse_tape::register_rule::<T>(
                    tape_id,
                    output_node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::ThinSvd,
                            "svd_ad_pullback_s",
                            |ctx, Backend| {
                                let _ = std::marker::PhantomData::<Backend>;
                                tenferro_linalg::svd_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::SvdCotangent {
                                        u: None,
                                        s: Some(cotangent.clone()),
                                        vt: None,
                                    },
                                    options.as_ref(),
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = compress_pullback_like("svd_ad", grad, &spec.layout)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                )?;
            }

            if let AdValue::Reverse { node, tape, .. } = out_vt.as_value() {
                let output_node = *node;
                let tape_id = *tape;
                let spec = spec.clone();
                let options = options.clone();
                let a_primal = input_primal.clone();
                reverse_tape::register_rule::<T>(
                    tape_id,
                    output_node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::ThinSvd,
                            "svd_ad_pullback_vt",
                            |ctx, Backend| {
                                let _ = std::marker::PhantomData::<Backend>;
                                tenferro_linalg::svd_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::SvdCotangent {
                                        u: None,
                                        s: None,
                                        vt: Some(cotangent.clone()),
                                    },
                                    options.as_ref(),
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = compress_pullback_like("svd_ad", grad, &spec.layout)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                )?;
            }
        }

        Ok(AdSvdResult {
            u: out_u,
            s: out_s,
            vt: out_vt,
        })
    }
}

/// Creates an AD SVD builder.
/// # Examples
///
/// ```ignore
/// let _ = svd_ad(/* ... */);
/// ```
pub fn svd_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> SvdAdBuilder<'a, T> {
    SvdAdBuilder {
        tensor,
        options: None,
    }
}

/// Builder for AD QR.
/// # Examples
///
/// ```text
/// // Construct `QrAdBuilder` via its corresponding operation constructor.
/// ```
pub struct QrAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> QrAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    /// Executes AD QR.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdQrResult<T>> {
        let operands = [self.tensor];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
        let (input_primal, input_tangent) = dispatch_linalg_ad_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::Qr,
            "qr_ad",
            |ctx, Backend| {
                dense_input_snapshot_in_backend::<Backend, _, T>(ctx, self.tensor, needs_tangent)
            }
        )?;

        let (primal, tangent) = if needs_tangent {
            let dt = input_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: "qr_ad missing materialized tangent".to_string(),
            })?;
            let (p, d) = dispatch_linalg_ad_runtime!(
                T,
                tenferro_linalg::backend::LinalgCapabilityOp::Qr,
                "qr_ad",
                |ctx, Backend| {
                    let _ = std::marker::PhantomData::<Backend>;
                    tenferro_linalg::qr_frule::<T, _>(ctx, &input_primal, &dt).map_err(Error::from)
                }
            )?;
            (p, Some(d))
        } else {
            (
                dispatch_linalg_ad_runtime!(
                    T,
                    tenferro_linalg::backend::LinalgCapabilityOp::Qr,
                    "qr_ad",
                    |ctx, Backend| {
                        let _ = std::marker::PhantomData::<Backend>;
                        tenferro_linalg::qr::<T, _>(ctx, &input_primal).map_err(Error::from)
                    }
                )?,
                None,
            )
        };

        let (dq, dr) = if let Some(d) = tangent {
            (Some(d.q), Some(d.r))
        } else {
            (None, None)
        };

        let out_q = wrap_dense_ad_output("qr_ad", &operands, primal.q, dq, 1)?;
        let out_r = wrap_dense_ad_output("qr_ad", &operands, primal.r, dr, 2)?;

        let input_spec = collect_reverse_input_specs(&operands)
            .into_iter()
            .next()
            .flatten();
        if let Some(spec) = input_spec {
            if let AdValue::Reverse { node, tape, .. } = out_q.as_value() {
                let output_node = *node;
                let tape_id = *tape;
                let spec = spec.clone();
                let a_primal = input_primal.clone();
                reverse_tape::register_rule::<T>(
                    tape_id,
                    output_node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::Qr,
                            "qr_ad_pullback_q",
                            |ctx, Backend| {
                                let _ = std::marker::PhantomData::<Backend>;
                                tenferro_linalg::qr_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::QrCotangent {
                                        q: Some(cotangent.clone()),
                                        r: None,
                                    },
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = compress_pullback_like("qr_ad", grad, &spec.layout)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                )?;
            }

            if let AdValue::Reverse { node, tape, .. } = out_r.as_value() {
                let output_node = *node;
                let tape_id = *tape;
                let spec = spec.clone();
                let a_primal = input_primal.clone();
                reverse_tape::register_rule::<T>(
                    tape_id,
                    output_node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::Qr,
                            "qr_ad_pullback_r",
                            |ctx, Backend| {
                                let _ = std::marker::PhantomData::<Backend>;
                                tenferro_linalg::qr_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::QrCotangent {
                                        q: None,
                                        r: Some(cotangent.clone()),
                                    },
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = compress_pullback_like("qr_ad", grad, &spec.layout)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                )?;
            }
        }

        Ok(AdQrResult { q: out_q, r: out_r })
    }
}

/// Creates an AD QR builder.
/// # Examples
///
/// ```ignore
/// let _ = qr_ad(/* ... */);
/// ```
pub fn qr_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> QrAdBuilder<'a, T> {
    QrAdBuilder { tensor }
}

/// Builder for AD LU.
/// # Examples
///
/// ```text
/// // Construct `LuAdBuilder` via its corresponding operation constructor.
/// ```
pub struct LuAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
    pivot: LuPivot,
}

impl<'a, T> LuAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    /// Sets LU pivot policy.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.pivot(pivot);
    /// ```
    pub fn pivot(mut self, pivot: LuPivot) -> Self {
        self.pivot = pivot;
        self
    }

    /// Executes AD LU.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdLuResult<T>> {
        let operands = [self.tensor];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
        let (input_primal, input_tangent) = dispatch_linalg_ad_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::LuFactor,
            "lu_ad",
            |ctx, Backend| {
                dense_input_snapshot_in_backend::<Backend, _, T>(ctx, self.tensor, needs_tangent)
            }
        )?;

        let (primal, tangent) = if needs_tangent {
            let dt = input_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: "lu_ad missing materialized tangent".to_string(),
            })?;
            let (p, d) = dispatch_linalg_ad_runtime!(
                T,
                tenferro_linalg::backend::LinalgCapabilityOp::LuFactor,
                "lu_ad",
                |ctx, Backend| {
                    let _ = std::marker::PhantomData::<Backend>;
                    tenferro_linalg::lu_frule::<T, _>(ctx, &input_primal, &dt, self.pivot)
                        .map_err(Error::from)
                }
            )?;
            (p, Some(d))
        } else {
            (
                dispatch_linalg_ad_runtime!(
                    T,
                    tenferro_linalg::backend::LinalgCapabilityOp::LuFactor,
                    "lu_ad",
                    |ctx, Backend| {
                        let _ = std::marker::PhantomData::<Backend>;
                        tenferro_linalg::lu::<T, _>(ctx, &input_primal, self.pivot)
                            .map_err(Error::from)
                    }
                )?,
                None,
            )
        };

        let (dl, du) = if let Some(d) = tangent {
            (Some(d.l), Some(d.u))
        } else {
            (None, None)
        };

        let out_l = wrap_dense_ad_output("lu_ad", &operands, primal.l, dl, 1)?;
        let out_u = wrap_dense_ad_output("lu_ad", &operands, primal.u, du, 2)?;

        let input_spec = collect_reverse_input_specs(&operands)
            .into_iter()
            .next()
            .flatten();
        if let Some(spec) = input_spec {
            if let AdValue::Reverse { node, tape, .. } = out_l.as_value() {
                let output_node = *node;
                let tape_id = *tape;
                let spec = spec.clone();
                let a_primal = input_primal.clone();
                let pivot = self.pivot;
                reverse_tape::register_rule::<T>(
                    tape_id,
                    output_node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::LuFactor,
                            "lu_ad_pullback_l",
                            |ctx, Backend| {
                                let _ = std::marker::PhantomData::<Backend>;
                                tenferro_linalg::lu_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::LuCotangent {
                                        l: Some(cotangent.clone()),
                                        u: None,
                                    },
                                    pivot,
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = compress_pullback_like("lu_ad", grad, &spec.layout)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                )?;
            }

            if let AdValue::Reverse { node, tape, .. } = out_u.as_value() {
                let output_node = *node;
                let tape_id = *tape;
                let spec = spec.clone();
                let a_primal = input_primal.clone();
                let pivot = self.pivot;
                reverse_tape::register_rule::<T>(
                    tape_id,
                    output_node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::LuFactor,
                            "lu_ad_pullback_u",
                            |ctx, Backend| {
                                let _ = std::marker::PhantomData::<Backend>;
                                tenferro_linalg::lu_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::LuCotangent {
                                        l: None,
                                        u: Some(cotangent.clone()),
                                    },
                                    pivot,
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = compress_pullback_like("lu_ad", grad, &spec.layout)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                )?;
            }
        }

        Ok(AdLuResult {
            p: primal.p,
            l: out_l,
            u: out_u,
        })
    }
}

/// Creates an AD LU builder.
/// # Examples
///
/// ```ignore
/// let _ = lu_ad(/* ... */);
/// ```
pub fn lu_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> LuAdBuilder<'a, T> {
    LuAdBuilder {
        tensor,
        pivot: LuPivot::Partial,
    }
}

/// Builder for AD eigen decomposition.
/// # Examples
///
/// ```text
/// // Construct `EigenAdBuilder` via its corresponding operation constructor.
/// ```
pub struct EigenAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> EigenAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    /// Executes AD eigen decomposition.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdEigenResult<T>> {
        let operands = [self.tensor];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
        let (input_primal, input_tangent) = dispatch_linalg_ad_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::EigenSym,
            "eigen_ad",
            |ctx, Backend| {
                dense_input_snapshot_in_backend::<Backend, _, T>(ctx, self.tensor, needs_tangent)
            }
        )?;

        let (primal, tangent) = if needs_tangent {
            let dt = input_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: "eigen_ad missing materialized tangent".to_string(),
            })?;
            let (p, d) = dispatch_linalg_ad_runtime!(
                T,
                tenferro_linalg::backend::LinalgCapabilityOp::EigenSym,
                "eigen_ad",
                |ctx, Backend| {
                    let _ = std::marker::PhantomData::<Backend>;
                    tenferro_linalg::eigen_frule::<T, _>(ctx, &input_primal, &dt)
                        .map_err(Error::from)
                }
            )?;
            (p, Some(d))
        } else {
            (
                dispatch_linalg_ad_runtime!(
                    T,
                    tenferro_linalg::backend::LinalgCapabilityOp::EigenSym,
                    "eigen_ad",
                    |ctx, Backend| {
                        let _ = std::marker::PhantomData::<Backend>;
                        tenferro_linalg::eigen::<T, _>(ctx, &input_primal).map_err(Error::from)
                    }
                )?,
                None,
            )
        };

        let (dvalues, dvectors) = if let Some(d) = tangent {
            (Some(d.values), Some(d.vectors))
        } else {
            (None, None)
        };

        let out_values = wrap_dense_ad_output("eigen_ad", &operands, primal.values, dvalues, 1)?;
        let out_vectors = wrap_dense_ad_output("eigen_ad", &operands, primal.vectors, dvectors, 2)?;

        let input_spec = collect_reverse_input_specs(&operands)
            .into_iter()
            .next()
            .flatten();
        if let Some(spec) = input_spec {
            if let AdValue::Reverse { node, tape, .. } = out_values.as_value() {
                let output_node = *node;
                let tape_id = *tape;
                let spec = spec.clone();
                let a_primal = input_primal.clone();
                reverse_tape::register_rule::<T>(
                    tape_id,
                    output_node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::EigenSym,
                            "eigen_ad_pullback_values",
                            |ctx, Backend| {
                                let _ = std::marker::PhantomData::<Backend>;
                                tenferro_linalg::eigen_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::EigenCotangent {
                                        values: Some(cotangent.clone()),
                                        vectors: None,
                                    },
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = compress_pullback_like("eigen_ad", grad, &spec.layout)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                )?;
            }

            if let AdValue::Reverse { node, tape, .. } = out_vectors.as_value() {
                let output_node = *node;
                let tape_id = *tape;
                let spec = spec.clone();
                let a_primal = input_primal.clone();
                reverse_tape::register_rule::<T>(
                    tape_id,
                    output_node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::EigenSym,
                            "eigen_ad_pullback_vectors",
                            |ctx, Backend| {
                                let _ = std::marker::PhantomData::<Backend>;
                                tenferro_linalg::eigen_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::EigenCotangent {
                                        values: None,
                                        vectors: Some(cotangent.clone()),
                                    },
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = compress_pullback_like("eigen_ad", grad, &spec.layout)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                )?;
            }
        }

        Ok(AdEigenResult {
            values: out_values,
            vectors: out_vectors,
        })
    }
}

/// Creates an AD eigen builder.
/// # Examples
///
/// ```ignore
/// let _ = eigen_ad(/* ... */);
/// ```
pub fn eigen_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> EigenAdBuilder<'a, T> {
    EigenAdBuilder { tensor }
}

/// Builder for AD least squares.
/// # Examples
///
/// ```text
/// // Construct `LstsqAdBuilder` via its corresponding operation constructor.
/// ```
pub struct LstsqAdBuilder<'a, T: Scalar> {
    a: &'a AdTensor<T>,
    b: &'a AdTensor<T>,
}

impl<'a, T> LstsqAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    /// Executes AD least squares.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdLstsqResult<T>> {
        let operands = [self.a, self.b];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
        let ((a_primal, a_tangent), (b_primal, b_tangent)) = dispatch_linalg_ad_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::Lstsq,
            "lstsq_ad",
            |ctx, Backend| {
                Ok((
                    dense_input_snapshot_in_backend::<Backend, _, T>(ctx, self.a, needs_tangent)?,
                    dense_input_snapshot_in_backend::<Backend, _, T>(ctx, self.b, needs_tangent)?,
                ))
            }
        )?;

        let (primal, tangent) = if needs_tangent {
            let da = a_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: "lstsq_ad missing materialized lhs tangent".to_string(),
            })?;
            let db = b_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: "lstsq_ad missing materialized rhs tangent".to_string(),
            })?;
            let (p, d) = dispatch_linalg_ad_runtime!(
                T,
                tenferro_linalg::backend::LinalgCapabilityOp::Lstsq,
                "lstsq_ad",
                |ctx, Backend| {
                    let _ = std::marker::PhantomData::<Backend>;
                    tenferro_linalg::lstsq_frule::<T, _>(ctx, &a_primal, &b_primal, &da, &db)
                        .map_err(Error::from)
                }
            )?;
            (p, Some(d))
        } else {
            (
                dispatch_linalg_ad_runtime!(
                    T,
                    tenferro_linalg::backend::LinalgCapabilityOp::Lstsq,
                    "lstsq_ad",
                    |ctx, Backend| {
                        let _ = std::marker::PhantomData::<Backend>;
                        tenferro_linalg::lstsq::<T, _>(ctx, &a_primal, &b_primal)
                            .map_err(Error::from)
                    }
                )?,
                None,
            )
        };

        let (dx, dresidual) = if let Some(d) = tangent {
            (Some(d.x), Some(d.residual))
        } else {
            (None, None)
        };

        let out_x = wrap_dense_ad_output("lstsq_ad", &operands, primal.x, dx, 1)?;
        let out_residual =
            wrap_dense_ad_output("lstsq_ad", &operands, primal.residual, dresidual, 2)?;

        let reverse_specs = collect_reverse_input_specs(&operands);
        if has_reverse(&operands) {
            if let AdValue::Reverse { node, tape, .. } = out_x.as_value() {
                let output_node = *node;
                let tape_id = *tape;
                let reverse_specs = reverse_specs.clone();
                let a_primal = a_primal.clone();
                let b_primal = b_primal.clone();
                reverse_tape::register_rule::<T>(
                    tape_id,
                    output_node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::Lstsq,
                            "lstsq_ad_pullback_x",
                            |ctx, Backend| {
                                let _ = std::marker::PhantomData::<Backend>;
                                tenferro_linalg::lstsq_rrule::<T, _>(
                                    ctx, &a_primal, &b_primal, cotangent,
                                )
                                .map_err(Error::from)
                            }
                        )?;

                        let mut input_grads = Vec::new();
                        if let Some(spec) = &reverse_specs[0] {
                            let grad_a = compress_pullback_like("lstsq_ad", grad.a, &spec.layout)?;
                            input_grads.push((spec.node, grad_a));
                        }
                        if let Some(spec) = &reverse_specs[1] {
                            let grad_b = compress_pullback_like("lstsq_ad", grad.b, &spec.layout)?;
                            input_grads.push((spec.node, grad_b));
                        }
                        Ok(input_grads)
                    }),
                )?;
            }

            if let AdValue::Reverse { node, tape, .. } = out_residual.as_value() {
                let output_node = *node;
                let tape_id = *tape;
                let reverse_specs = reverse_specs.clone();
                let zero_a = zero_like(self.a.structured_primal().payload());
                let zero_b = zero_like(self.b.structured_primal().payload());
                reverse_tape::register_rule::<T>(
                    tape_id,
                    output_node,
                    Box::new(move |_cotangent| {
                        let mut input_grads = Vec::new();
                        if let Some(spec) = &reverse_specs[0] {
                            input_grads.push((spec.node, zero_a.clone()));
                        }
                        if let Some(spec) = &reverse_specs[1] {
                            input_grads.push((spec.node, zero_b.clone()));
                        }
                        Ok(input_grads)
                    }),
                )?;
            }
        }

        Ok(AdLstsqResult {
            x: out_x,
            residual: out_residual,
        })
    }
}

/// Creates an AD lstsq builder.
/// # Examples
///
/// ```ignore
/// let _ = lstsq_ad(/* ... */);
/// ```
pub fn lstsq_ad<'a, T: Scalar>(a: &'a AdTensor<T>, b: &'a AdTensor<T>) -> LstsqAdBuilder<'a, T> {
    LstsqAdBuilder { a, b }
}

/// Builder for AD Cholesky.
/// # Examples
///
/// ```text
/// // Construct `CholeskyAdBuilder` via its corresponding operation constructor.
/// ```
pub struct CholeskyAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> CholeskyAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    /// Executes AD Cholesky.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad!(
            ty = T,
            capability = tenferro_linalg::backend::LinalgCapabilityOp::Cholesky,
            op = "cholesky_ad",
            pullback = "cholesky_ad_pullback",
            input = self.tensor,
            primal = |ctx, Backend, tensor| {
                let _ = std::marker::PhantomData::<Backend>;
                tenferro_linalg::cholesky::<T, _>(ctx, tensor).map_err(Error::from)
            },
            frule = |ctx, Backend, tensor, dt| {
                let _ = std::marker::PhantomData::<Backend>;
                tenferro_linalg::cholesky_frule::<T, _>(ctx, tensor, dt).map_err(Error::from)
            },
            rrule = |ctx, Backend, tensor, cotangent| {
                let _ = std::marker::PhantomData::<Backend>;
                tenferro_linalg::cholesky_rrule::<T, _>(ctx, tensor, cotangent).map_err(Error::from)
            },
        )
    }
}

/// Creates an AD cholesky builder.
/// # Examples
///
/// ```ignore
/// let _ = cholesky_ad(/* ... */);
/// ```
pub fn cholesky_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> CholeskyAdBuilder<'a, T> {
    CholeskyAdBuilder { tensor }
}

/// Builder for AD solve.
/// # Examples
///
/// ```text
/// // Construct `SolveAdBuilder` via its corresponding operation constructor.
/// ```
pub struct SolveAdBuilder<'a, T: Scalar> {
    a: &'a AdTensor<T>,
    b: &'a AdTensor<T>,
}

impl<'a, T> SolveAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    /// Executes AD solve.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_binary_tensor_ad!(
            ty = T,
            capability = tenferro_linalg::backend::LinalgCapabilityOp::Solve,
            op = "solve_ad",
            pullback = "solve_ad_pullback",
            lhs = self.a,
            rhs = self.b,
            primal = |ctx, Backend, a, b| {
                let _ = std::marker::PhantomData::<Backend>;
                tenferro_linalg::solve::<T, _>(ctx, a, b).map_err(Error::from)
            },
            frule = |ctx, Backend, a, b, da, db| {
                let _ = std::marker::PhantomData::<Backend>;
                tenferro_linalg::solve_frule::<T, _>(ctx, a, b, da, db).map_err(Error::from)
            },
            rrule = |ctx, Backend, a, b, cotangent| {
                let _ = std::marker::PhantomData::<Backend>;
                let grad = tenferro_linalg::solve_rrule::<T, _>(ctx, a, b, cotangent)
                    .map_err(Error::from)?;
                Ok((grad.a, grad.b))
            },
        )
    }
}

/// Creates an AD solve builder.
/// # Examples
///
/// ```ignore
/// let _ = solve_ad(/* ... */);
/// ```
pub fn solve_ad<'a, T: Scalar>(a: &'a AdTensor<T>, b: &'a AdTensor<T>) -> SolveAdBuilder<'a, T> {
    SolveAdBuilder { a, b }
}

/// Builder for AD inverse.
/// # Examples
///
/// ```text
/// // Construct `InvAdBuilder` via its corresponding operation constructor.
/// ```
pub struct InvAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> InvAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    /// Executes AD inverse.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad!(
            ty = T,
            capability = tenferro_linalg::backend::LinalgCapabilityOp::Inv,
            op = "inv_ad",
            pullback = "inv_ad_pullback",
            input = self.tensor,
            primal = |ctx, Backend, tensor| {
                let _ = std::marker::PhantomData::<Backend>;
                tenferro_linalg::inv::<T, _>(ctx, tensor).map_err(Error::from)
            },
            frule = |ctx, Backend, tensor, dt| {
                let _ = std::marker::PhantomData::<Backend>;
                tenferro_linalg::inv_frule::<T, _>(ctx, tensor, dt).map_err(Error::from)
            },
            rrule = |ctx, Backend, tensor, cotangent| {
                let _ = std::marker::PhantomData::<Backend>;
                tenferro_linalg::inv_rrule::<T, _>(ctx, tensor, cotangent).map_err(Error::from)
            },
        )
    }
}

/// Creates an AD inv builder.
/// # Examples
///
/// ```ignore
/// let _ = inv_ad(/* ... */);
/// ```
pub fn inv_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> InvAdBuilder<'a, T> {
    InvAdBuilder { tensor }
}

/// Builder for AD det.
/// # Examples
///
/// ```text
/// // Construct `DetAdBuilder` via its corresponding operation constructor.
/// ```
pub struct DetAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> DetAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    /// Executes AD determinant.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad!(
            ty = T,
            capability = tenferro_linalg::backend::LinalgCapabilityOp::Det,
            op = "det_ad",
            pullback = "det_ad_pullback",
            input = self.tensor,
            primal = |ctx, Backend, tensor| {
                let _ = std::marker::PhantomData::<Backend>;
                tenferro_linalg::det::<T, _>(ctx, tensor).map_err(Error::from)
            },
            frule = |ctx, Backend, tensor, dt| {
                let _ = std::marker::PhantomData::<Backend>;
                tenferro_linalg::det_frule::<T, _>(ctx, tensor, dt).map_err(Error::from)
            },
            rrule = |ctx, Backend, tensor, cotangent| {
                let _ = std::marker::PhantomData::<Backend>;
                tenferro_linalg::det_rrule::<T, _>(ctx, tensor, cotangent).map_err(Error::from)
            },
        )
    }
}

/// Creates an AD det builder.
/// # Examples
///
/// ```ignore
/// let _ = det_ad(/* ... */);
/// ```
pub fn det_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> DetAdBuilder<'a, T> {
    DetAdBuilder { tensor }
}

/// Builder for AD slogdet.
/// # Examples
///
/// ```text
/// // Construct `SlogdetAdBuilder` via its corresponding operation constructor.
/// ```
pub struct SlogdetAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> SlogdetAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    /// Executes AD slogdet.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdSlogdetResult<T>> {
        let operands = [self.tensor];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
        let (input_primal, input_tangent) = dispatch_linalg_ad_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::Slogdet,
            "slogdet_ad",
            |ctx, Backend| {
                dense_input_snapshot_in_backend::<Backend, _, T>(ctx, self.tensor, needs_tangent)
            }
        )?;

        let (primal, tangent) = if needs_tangent {
            let dt = input_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: "slogdet_ad missing materialized tangent".to_string(),
            })?;
            let (p, d) = dispatch_linalg_ad_runtime!(
                T,
                tenferro_linalg::backend::LinalgCapabilityOp::Slogdet,
                "slogdet_ad",
                |ctx, Backend| {
                    let _ = std::marker::PhantomData::<Backend>;
                    tenferro_linalg::slogdet_frule::<T, _>(ctx, &input_primal, &dt)
                        .map_err(Error::from)
                }
            )?;
            (p, Some(d))
        } else {
            (
                dispatch_linalg_ad_runtime!(
                    T,
                    tenferro_linalg::backend::LinalgCapabilityOp::Slogdet,
                    "slogdet_ad",
                    |ctx, Backend| {
                        let _ = std::marker::PhantomData::<Backend>;
                        tenferro_linalg::slogdet::<T, _>(ctx, &input_primal).map_err(Error::from)
                    }
                )?,
                None,
            )
        };

        let (dsign, dlogabsdet) = if let Some(d) = tangent {
            (Some(d.sign), Some(d.logabsdet))
        } else {
            (None, None)
        };

        let out_sign = wrap_dense_ad_output("slogdet_ad", &operands, primal.sign, dsign, 1)?;
        let out_logabsdet =
            wrap_dense_ad_output("slogdet_ad", &operands, primal.logabsdet, dlogabsdet, 2)?;

        let input_spec = collect_reverse_input_specs(&operands)
            .into_iter()
            .next()
            .flatten();
        if let Some(spec) = input_spec {
            if let AdValue::Reverse { node, tape, .. } = out_sign.as_value() {
                let output_node = *node;
                let tape_id = *tape;
                let spec = spec.clone();
                let zero = zero_like(spec.layout.payload());
                reverse_tape::register_rule::<T>(
                    tape_id,
                    output_node,
                    Box::new(move |_cotangent| Ok(vec![(spec.node, zero.clone())])),
                )?;
            }

            if let AdValue::Reverse { node, tape, .. } = out_logabsdet.as_value() {
                let output_node = *node;
                let tape_id = *tape;
                let spec = spec.clone();
                let a_primal = input_primal.clone();
                reverse_tape::register_rule::<T>(
                    tape_id,
                    output_node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::Slogdet,
                            "slogdet_ad_pullback_logabsdet",
                            |ctx, Backend| {
                                let _ = std::marker::PhantomData::<Backend>;
                                tenferro_linalg::slogdet_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::SlogdetCotangent {
                                        logabsdet: Some(cotangent.clone()),
                                    },
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = compress_pullback_like("slogdet_ad", grad, &spec.layout)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                )?;
            }
        }

        Ok(AdSlogdetResult {
            sign: out_sign,
            logabsdet: out_logabsdet,
        })
    }
}

/// Creates an AD slogdet builder.
/// # Examples
///
/// ```ignore
/// let _ = slogdet_ad(/* ... */);
/// ```
pub fn slogdet_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> SlogdetAdBuilder<'a, T> {
    SlogdetAdBuilder { tensor }
}

/// Builder for AD eig.
/// # Examples
///
/// ```text
/// // Construct `EigAdBuilder` via its corresponding operation constructor.
/// ```
pub struct EigAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> EigAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T, Complex = Complex<T>>
        + Float
        + CpuLinalgScalar
        + HasAlgebra<Algebra = Standard<T>>,
    Complex<T>: Scalar,
{
    /// Executes AD eig.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdEigResult<T>> {
        let operands = [self.tensor];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
        let (input_primal, input_tangent) = dispatch_linalg_ad_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::Eig,
            "eig_ad",
            |ctx, Backend| {
                dense_input_snapshot_in_backend::<Backend, _, T>(ctx, self.tensor, needs_tangent)
            }
        )?;

        let (primal, tangent) = if needs_tangent {
            let dt = input_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: "eig_ad missing materialized tangent".to_string(),
            })?;
            let (p, d) = dispatch_linalg_ad_runtime!(
                T,
                tenferro_linalg::backend::LinalgCapabilityOp::Eig,
                "eig_ad",
                |ctx, Backend| {
                    let _ = std::marker::PhantomData::<Backend>;
                    tenferro_linalg::eig_frule::<T, _>(ctx, &input_primal, &dt).map_err(Error::from)
                }
            )?;
            (p, Some(d))
        } else {
            (
                dispatch_linalg_ad_runtime!(
                    T,
                    tenferro_linalg::backend::LinalgCapabilityOp::Eig,
                    "eig_ad",
                    |ctx, Backend| {
                        let _ = std::marker::PhantomData::<Backend>;
                        tenferro_linalg::eig::<T, _>(ctx, &input_primal).map_err(Error::from)
                    }
                )?,
                None,
            )
        };

        let (dvalues, dvectors) = if let Some(d) = tangent {
            (Some(d.values), Some(d.vectors))
        } else {
            (None, None)
        };

        let out_values = wrap_dense_ad_output("eig_ad", &operands, primal.values, dvalues, 1)?;
        let out_vectors = wrap_dense_ad_output("eig_ad", &operands, primal.vectors, dvectors, 2)?;

        let input_spec = collect_reverse_input_specs(&operands)
            .into_iter()
            .next()
            .flatten();
        if let Some(spec) = input_spec {
            if let AdValue::Reverse { node, tape, .. } = out_values.as_value() {
                let output_node = *node;
                let tape_id = *tape;
                let spec = spec.clone();
                let a_primal = input_primal.clone();
                reverse_tape::register_bridge_rule::<Complex<T>, T>(
                    tape_id,
                    output_node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::Eig,
                            "eig_ad_pullback_values",
                            |ctx, Backend| {
                                let _ = std::marker::PhantomData::<Backend>;
                                tenferro_linalg::eig_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::EigCotangent {
                                        values: Some(cotangent.clone()),
                                        vectors: None,
                                    },
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = compress_pullback_like("eig_ad", grad, &spec.layout)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                )?;
            }

            if let AdValue::Reverse { node, tape, .. } = out_vectors.as_value() {
                let output_node = *node;
                let tape_id = *tape;
                let spec = spec.clone();
                let a_primal = input_primal.clone();
                reverse_tape::register_bridge_rule::<Complex<T>, T>(
                    tape_id,
                    output_node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::Eig,
                            "eig_ad_pullback_vectors",
                            |ctx, Backend| {
                                let _ = std::marker::PhantomData::<Backend>;
                                tenferro_linalg::eig_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::EigCotangent {
                                        values: None,
                                        vectors: Some(cotangent.clone()),
                                    },
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = compress_pullback_like("eig_ad", grad, &spec.layout)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                )?;
            }
        }

        Ok(AdEigResult {
            values: out_values,
            vectors: out_vectors,
        })
    }
}

/// Creates an AD eig builder.
/// # Examples
///
/// ```ignore
/// let _ = eig_ad(/* ... */);
/// ```
pub fn eig_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> EigAdBuilder<'a, T> {
    EigAdBuilder { tensor }
}

/// Builder for AD pinv.
/// # Examples
///
/// ```text
/// // Construct `PinvAdBuilder` via its corresponding operation constructor.
/// ```
pub struct PinvAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
    rcond: Option<f64>,
}

impl<'a, T> PinvAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    /// Sets rcond.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.rcond(1e-12);
    /// ```
    pub fn rcond(mut self, rcond: f64) -> Self {
        self.rcond = Some(rcond);
        self
    }

    /// Executes AD pinv.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad!(
            ty = T,
            capability = tenferro_linalg::backend::LinalgCapabilityOp::Pinv,
            op = "pinv_ad",
            pullback = "pinv_ad_pullback",
            input = self.tensor,
            primal = |ctx, Backend, tensor| {
                let _ = std::marker::PhantomData::<Backend>;
                tenferro_linalg::pinv::<T, _>(ctx, tensor, self.rcond).map_err(Error::from)
            },
            frule = |ctx, Backend, tensor, dt| {
                let _ = std::marker::PhantomData::<Backend>;
                tenferro_linalg::pinv_frule::<T, _>(ctx, tensor, dt, self.rcond)
                    .map_err(Error::from)
            },
            rrule = |ctx, Backend, tensor, cotangent| {
                let _ = std::marker::PhantomData::<Backend>;
                tenferro_linalg::pinv_rrule::<T, _>(ctx, tensor, cotangent, self.rcond)
                    .map_err(Error::from)
            },
        )
    }
}

/// Creates an AD pinv builder.
/// # Examples
///
/// ```ignore
/// let _ = pinv_ad(/* ... */);
/// ```
pub fn pinv_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> PinvAdBuilder<'a, T> {
    PinvAdBuilder {
        tensor,
        rcond: None,
    }
}

/// Builder for AD matrix exponential.
/// # Examples
///
/// ```text
/// // Construct `MatrixExpAdBuilder` via its corresponding operation constructor.
/// ```
pub struct MatrixExpAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> MatrixExpAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    /// Executes AD matrix exponential.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad!(
            ty = T,
            capability = tenferro_linalg::backend::LinalgCapabilityOp::MatrixExp,
            op = "matrix_exp_ad",
            pullback = "matrix_exp_ad_pullback",
            input = self.tensor,
            primal = |ctx, Backend, tensor| {
                let _ = std::marker::PhantomData::<Backend>;
                tenferro_linalg::matrix_exp::<T, _>(ctx, tensor).map_err(Error::from)
            },
            frule = |ctx, Backend, tensor, dt| {
                let _ = std::marker::PhantomData::<Backend>;
                tenferro_linalg::matrix_exp_frule::<T, _>(ctx, tensor, dt).map_err(Error::from)
            },
            rrule = |ctx, Backend, tensor, cotangent| {
                let _ = std::marker::PhantomData::<Backend>;
                tenferro_linalg::matrix_exp_rrule::<T, _>(ctx, tensor, cotangent)
                    .map_err(Error::from)
            },
        )
    }
}

/// Creates an AD matrix_exp builder.
/// # Examples
///
/// ```ignore
/// let _ = matrix_exp_ad(/* ... */);
/// ```
pub fn matrix_exp_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> MatrixExpAdBuilder<'a, T> {
    MatrixExpAdBuilder { tensor }
}

/// Builder for AD solve_triangular.
/// # Examples
///
/// ```text
/// // Construct `SolveTriangularAdBuilder` via its corresponding operation constructor.
/// ```
pub struct SolveTriangularAdBuilder<'a, T: Scalar> {
    a: &'a AdTensor<T>,
    b: &'a AdTensor<T>,
    upper: bool,
}

impl<'a, T> SolveTriangularAdBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    /// Sets whether the matrix is upper triangular.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.upper(true);
    /// ```
    pub fn upper(mut self, upper: bool) -> Self {
        self.upper = upper;
        self
    }

    /// Executes AD triangular solve.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>>
    where
        T: 'static,
    {
        run_binary_tensor_ad!(
            ty = T,
            capability = tenferro_linalg::backend::LinalgCapabilityOp::SolveTriangular,
            op = "solve_triangular_ad",
            pullback = "solve_triangular_ad_pullback",
            lhs = self.a,
            rhs = self.b,
            primal = |ctx, Backend, a, b| {
                let _ = std::marker::PhantomData::<Backend>;
                tenferro_linalg::solve_triangular::<T, _>(ctx, a, b, self.upper)
                    .map_err(Error::from)
            },
            frule = |ctx, Backend, a, b, da, db| {
                let _ = std::marker::PhantomData::<Backend>;
                tenferro_linalg::solve_triangular_frule::<T, _>(ctx, a, b, da, db, self.upper)
                    .map_err(Error::from)
            },
            rrule = |ctx, Backend, a, b, cotangent| {
                let _ = std::marker::PhantomData::<Backend>;
                let grad = tenferro_linalg::solve_triangular_rrule::<T, _>(
                    ctx, a, b, cotangent, self.upper,
                )
                .map_err(Error::from)?;
                Ok((grad.a, grad.b))
            },
        )
    }
}

/// Creates an AD solve_triangular builder.
/// # Examples
///
/// ```ignore
/// let _ = solve_triangular_ad(/* ... */);
/// ```
pub fn solve_triangular_ad<'a, T: Scalar>(
    a: &'a AdTensor<T>,
    b: &'a AdTensor<T>,
) -> SolveTriangularAdBuilder<'a, T> {
    SolveTriangularAdBuilder { a, b, upper: true }
}

/// Builder for AD norm.
/// # Examples
///
/// ```text
/// // Construct `NormAdBuilder` via its corresponding operation constructor.
/// ```
pub struct NormAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
    kind: NormKind,
}

impl<'a, T> NormAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    /// Sets norm kind.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.kind(kind);
    /// ```
    pub fn kind(mut self, kind: NormKind) -> Self {
        self.kind = kind;
        self
    }

    /// Executes AD norm.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad!(
            ty = T,
            capability = tenferro_linalg::backend::LinalgCapabilityOp::Norm,
            op = "norm_ad",
            pullback = "norm_ad_pullback",
            input = self.tensor,
            primal = |ctx, Backend, tensor| {
                let _ = std::marker::PhantomData::<Backend>;
                tenferro_linalg::norm::<T, _>(ctx, tensor, self.kind).map_err(Error::from)
            },
            frule = |ctx, Backend, tensor, dt| {
                let _ = std::marker::PhantomData::<Backend>;
                tenferro_linalg::norm_frule::<T, _>(ctx, tensor, dt, self.kind).map_err(Error::from)
            },
            rrule = |ctx, Backend, tensor, cotangent| {
                let _ = std::marker::PhantomData::<Backend>;
                tenferro_linalg::norm_rrule::<T, _>(ctx, tensor, cotangent, self.kind)
                    .map_err(Error::from)
            },
        )
    }
}

/// Creates an AD norm builder.
/// # Examples
///
/// ```ignore
/// let _ = norm_ad(/* ... */);
/// ```
pub fn norm_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> NormAdBuilder<'a, T> {
    NormAdBuilder {
        tensor,
        kind: NormKind::Fro,
    }
}
