#[derive(Default)]
pub(crate) struct Emitter {
    next_value: usize,
    lines: Vec<String>,
}

impl Emitter {
    pub(crate) fn value(&mut self) -> String {
        let value = format!("%{}", self.next_value);
        self.next_value += 1;
        value
    }

    pub(crate) fn line(&mut self, text: impl Into<String>) {
        self.lines.push(text.into());
    }

    pub(crate) fn finish(self) -> Vec<String> {
        self.lines
    }
}

pub(crate) fn format_usize_list(values: &[usize]) -> String {
    let inner = values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{inner}]")
}
