#[derive(Clone, Debug)]
pub struct Subscripts {
    pub input_labels: Vec<Vec<u32>>,
    pub output_labels: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct ContractionStep {
    pub left: usize,
    pub right: usize,
}

#[derive(Clone, Debug)]
pub struct ContractionPath {
    pub subscripts: Subscripts,
    pub steps: Vec<ContractionStep>,
}
