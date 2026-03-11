use chainrules_scalarops::ScalarAd;
use num_traits::Float;
use tenferro_algebra::{HasAlgebra, Scalar, Standard};
use tenferro_prims::{
    CpuBackend, CudaBackend, RocmBackend, TensorAnalyticPrims, TensorPrims, TensorScalarPrims,
};

pub(crate) trait StandardScalarValue:
    Scalar + HasAlgebra<Algebra = Standard<Self>> + Copy + 'static
{
}

impl<T> StandardScalarValue for T where
    T: Scalar + HasAlgebra<Algebra = Standard<T>> + Copy + 'static
{
}

pub(crate) trait GenericAdScalar: StandardScalarValue + ScalarAd {}

impl<T> GenericAdScalar for T where T: StandardScalarValue + ScalarAd {}

pub(crate) trait RealAdScalar: GenericAdScalar + ScalarAd<Real = Self> + Float {}

impl<T> RealAdScalar for T where T: GenericAdScalar + ScalarAd<Real = T> + Float {}

pub(crate) trait CpuScalarBackend<T: StandardScalarValue>:
    TensorPrims<Standard<T>, Context = tenferro_prims::CpuContext>
    + TensorScalarPrims<Standard<T>, Context = tenferro_prims::CpuContext>
{
}

impl<T, B> CpuScalarBackend<T> for B
where
    T: StandardScalarValue,
    B: TensorPrims<Standard<T>, Context = tenferro_prims::CpuContext>
        + TensorScalarPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
{
}

pub(crate) trait CudaScalarBackend<T: StandardScalarValue>:
    TensorScalarPrims<Standard<T>, Context = tenferro_prims::CudaContext>
{
}

impl<T, B> CudaScalarBackend<T> for B
where
    T: StandardScalarValue,
    B: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
{
}

pub(crate) trait RocmScalarBackend<T: StandardScalarValue>:
    TensorScalarPrims<Standard<T>, Context = tenferro_prims::RocmContext>
{
}

impl<T, B> RocmScalarBackend<T> for B
where
    T: StandardScalarValue,
    B: TensorScalarPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
{
}

pub(crate) trait CpuAnalyticBackend<T: StandardScalarValue>:
    TensorPrims<Standard<T>, Context = tenferro_prims::CpuContext>
    + TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CpuContext>
{
}

impl<T, B> CpuAnalyticBackend<T> for B
where
    T: StandardScalarValue,
    B: TensorPrims<Standard<T>, Context = tenferro_prims::CpuContext>
        + TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
{
}

pub(crate) trait CudaAnalyticBackend<T: StandardScalarValue>:
    TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CudaContext>
{
}

impl<T, B> CudaAnalyticBackend<T> for B
where
    T: StandardScalarValue,
    B: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
{
}

pub(crate) trait RocmAnalyticBackend<T: StandardScalarValue>:
    TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::RocmContext>
{
}

impl<T, B> RocmAnalyticBackend<T> for B
where
    T: StandardScalarValue,
    B: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
{
}

pub(crate) trait CpuScalarAnalyticBackend<T: StandardScalarValue>:
    CpuScalarBackend<T> + CpuAnalyticBackend<T>
{
}

impl<T, B> CpuScalarAnalyticBackend<T> for B
where
    T: StandardScalarValue,
    B: CpuScalarBackend<T> + CpuAnalyticBackend<T>,
{
}

pub(crate) trait CudaScalarAnalyticBackend<T: StandardScalarValue>:
    CudaScalarBackend<T> + CudaAnalyticBackend<T>
{
}

impl<T, B> CudaScalarAnalyticBackend<T> for B
where
    T: StandardScalarValue,
    B: CudaScalarBackend<T> + CudaAnalyticBackend<T>,
{
}

pub(crate) trait RocmScalarAnalyticBackend<T: StandardScalarValue>:
    RocmScalarBackend<T> + RocmAnalyticBackend<T>
{
}

impl<T, B> RocmScalarAnalyticBackend<T> for B
where
    T: StandardScalarValue,
    B: RocmScalarBackend<T> + RocmAnalyticBackend<T>,
{
}

#[allow(dead_code)]
const _: fn() = || {
    fn assert_default_backends<T>()
    where
        T: StandardScalarValue,
        CpuBackend: CpuScalarAnalyticBackend<T>,
        CudaBackend: CudaScalarAnalyticBackend<T>,
        RocmBackend: RocmScalarAnalyticBackend<T>,
    {
    }
};
