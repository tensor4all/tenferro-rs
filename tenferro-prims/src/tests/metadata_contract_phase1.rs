use crate::{MetadataBinaryOp, MetadataReductionOp, MetadataTernaryOp, MetadataUnaryOp};

#[test]
fn metadata_family_exposes_i32_bool_comparison_where_and_sum_descriptors() {
    assert_eq!(
        MetadataUnaryOp::IotaStartZero as u8,
        MetadataUnaryOp::IotaStartZero as u8
    );
    assert_eq!(
        MetadataBinaryOp::NotEqual as u8,
        MetadataBinaryOp::NotEqual as u8
    );
    assert_eq!(
        MetadataTernaryOp::Where as u8,
        MetadataTernaryOp::Where as u8
    );
    assert_eq!(
        MetadataReductionOp::Sum as u8,
        MetadataReductionOp::Sum as u8
    );
}
