#[derive(Debug)]
pub struct ReplaySummary {
    pub validated_records: usize,
    pub expected_error_case_ids: Vec<String>,
    pub failures: Vec<String>,
}

pub fn run_database_replay() -> ReplaySummary {
    todo!("implement oracle replay")
}
