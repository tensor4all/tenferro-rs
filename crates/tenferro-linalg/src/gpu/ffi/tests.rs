use super::cusolver::CudaDataType;

#[test]
fn cusolver_cuda_data_type_has_c_abi_integer_layout() {
    assert_eq!(
        std::mem::size_of::<CudaDataType>(),
        std::mem::size_of::<i32>()
    );
    assert_eq!(
        std::mem::align_of::<CudaDataType>(),
        std::mem::align_of::<i32>()
    );
}
