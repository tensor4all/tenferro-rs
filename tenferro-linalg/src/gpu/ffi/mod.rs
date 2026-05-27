pub(super) mod cusolver;

#[cfg(test)]
mod tests;

fn library_search_paths(env_var: &str, default_paths: &[&str]) -> Vec<String> {
    if let Ok(val) = std::env::var(env_var) {
        val.split(':')
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    } else {
        default_paths.iter().map(|s| s.to_string()).collect()
    }
}
