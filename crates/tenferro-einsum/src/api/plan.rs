//! The #2552 grammar of [`ConcreteEinsumPlan`]: one `prepare`, `execute` and
//! `execute_into`.

use tenferro_tensor::{Dest, Engine, Repr};

use super::concrete::{plan_into_reads, plan_reads, resolve, tensor_err};
use super::options::{EinsumInputs, EinsumOptions, EinsumSpec};
use crate::concrete::read_input_specs;
use crate::ConcreteEinsumPlan;

type Res<T> = tenferro_tensor::Result<T>;

impl ConcreteEinsumPlan {
    /// Plan a concrete einsum once from the inputs' dtypes and shapes; the
    /// element data is not read. The plan honours `options.optimize` (an
    /// explicit path, nested grouping or tree is used as given) and records
    /// `options.conj` for every later execution.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_einsum::{ConcreteEinsumPlan, EinsumOptions};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    /// let plan = ConcreteEinsumPlan::prepare(&[&a, &a], "ij,jk->ik", &EinsumOptions::default())?;
    /// let out = plan.execute(&[&a, &a], &mut CpuEngine::serial())?;
    /// assert_eq!(out.as_slice::<f64>()?, &[7.0, 10.0, 15.0, 22.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// `Validation` for a malformed spec (with a `tenferro_einsum::Error`
    /// source), mismatched ranks or extents, or an invalid order.
    pub fn prepare<I: EinsumInputs + ?Sized>(
        inputs: &I,
        spec: &(impl EinsumSpec + ?Sized),
        options: &EinsumOptions,
    ) -> Res<Self> {
        let reads = inputs.reads();
        let subscripts = resolve(spec.spec(), &reads).map_err(tensor_err)?;
        Self::prepare_planned(
            read_input_specs(&reads),
            &subscripts,
            &options.optimize,
            options.conj.clone(),
        )
        .map_err(tensor_err)
    }

    /// Run the plan on inputs with the prepared dtypes and shapes. The result
    /// takes the inputs' representation (typed inputs give a typed result).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_einsum::{ConcreteEinsumPlan, EinsumOptions};
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0])?;
    /// let plan = ConcreteEinsumPlan::prepare(&[&a, &a], "i,i->", &EinsumOptions::default())?;
    /// let dot: TypedTensor<f64> = plan.execute(&[&a, &a], &mut CpuEngine::serial())?;
    /// assert_eq!(dot.host_data()?, &[5.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// `Validation` when an input's dtype or shape differs from the prepared
    /// one; backend failures otherwise.
    pub fn execute<I: EinsumInputs + ?Sized>(
        &self,
        inputs: &I,
        exec: &mut dyn Engine,
    ) -> Res<<I::Repr as Repr>::Value> {
        let reads = inputs.reads();
        <I::Repr as Repr>::wrap_value(plan_reads(self, &reads, exec)?)
    }

    /// Run the plan into `dest`: `Overwrite`, `Accumulate` (`out += r`) or
    /// `Scaled` (`out = alpha r + beta out`; `beta = 0` never reads `out`).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_einsum::{ConcreteEinsumPlan, EinsumOptions};
    /// use tenferro_tensor::{Dest, Tensor};
    ///
    /// let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// let plan = ConcreteEinsumPlan::prepare(&[&a, &a], "i,i->", &EinsumOptions::default())?;
    /// let mut out = Tensor::from_vec_col_major(Vec::<usize>::new(), vec![10.0_f64])?;
    /// plan.execute_into(&[&a, &a], &mut CpuEngine::serial(), Dest::scaled(&mut out, 2.0, 1.0))?;
    /// assert_eq!(out.as_slice::<f64>()?, &[20.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// `Validation` when an input or the destination has another dtype or
    /// shape than the plan, or `alpha`/`beta` have another dtype; checked
    /// before any write.
    pub fn execute_into<I: EinsumInputs + ?Sized>(
        &self,
        inputs: &I,
        exec: &mut dyn Engine,
        dest: Dest<<I::Repr as Repr>::Write<'_>>,
    ) -> Res<()> {
        let reads = inputs.reads();
        plan_into_reads::<I::Repr>(self, &reads, exec, dest)
    }
}
