use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct Subscripts {
    pub input_labels: Vec<Vec<u32>>,
    pub output_labels: Vec<u32>,
}

impl Subscripts {
    pub fn new(input_labels: Vec<Vec<u32>>, output_labels: Vec<u32>) -> Self {
        Self {
            input_labels,
            output_labels,
        }
    }

    pub fn parse(s: &str) -> Self {
        let parts: Vec<&str> = s.split("->").collect();
        assert_eq!(
            parts.len(),
            2,
            "einsum string must contain exactly one '->'"
        );
        let inputs_str = parts[0];
        let output_str = parts[1];

        let input_labels: Vec<Vec<u32>> = inputs_str
            .split(',')
            .map(|sub| sub.chars().map(|c| c as u32).collect())
            .collect();

        let output_labels: Vec<u32> = output_str.chars().map(|c| c as u32).collect();

        Self {
            input_labels,
            output_labels,
        }
    }

    pub fn n_inputs(&self) -> usize {
        self.input_labels.len()
    }

    pub fn all_labels(&self) -> HashSet<u32> {
        let mut set = HashSet::new();
        for labels in &self.input_labels {
            set.extend(labels);
        }
        set.extend(&self.output_labels);
        set
    }
}
