use std::sync::Arc;

use computegraph::fragment::Fragment;
use tenferro_ops::std_tensor_op::StdTensorOp;

pub struct Engine {
    _fragments: Vec<Arc<Fragment<StdTensorOp>>>,
}

impl Engine {
    pub fn new() -> Self {
        todo!()
    }
}
