//! Markdown rendering of one registry family section.

use super::{error_text, tidy, LayerKind, Op, Registry};
use quote::quote;

fn esc(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}

/// Generated method names of an op, e.g. `svd`, `svd_with`, `svd_into (overwrite)`.
fn suffixes(op: &Op) -> String {
    let n = op.name.to_string();
    let mut parts = vec![format!("`{n}`")];
    if op.options.is_some() {
        parts.push(format!("`{n}_with`"));
    }
    if !op.into.is_empty() {
        let modes = op
            .into
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        parts.push(format!("`{n}_into` ({modes})"));
    }
    parts.join(", ")
}

/// Renders the deterministic Markdown section for the registry.
pub(crate) fn render(reg: &Registry) -> String {
    let mut out = format!("## {}\n\n", reg.family);
    out.push_str("| Op |");
    for l in &reg.layers {
        out.push_str(&format!(" {} |", l.name));
    }
    out.push_str(" Suffixes |\n|---|");
    for _ in &reg.layers {
        out.push_str("---|");
    }
    out.push_str("---|\n");
    for op in &reg.ops {
        out.push_str(&format!("| `{}` |", op.name));
        for l in &reg.layers {
            let cell = if op.layers.contains(&l.name) {
                match &l.kind {
                    LayerKind::DescribeOnly => "described".to_owned(),
                    LayerKind::Generate(g) => g.trait_name.to_string(),
                }
            } else {
                "—".to_owned()
            };
            out.push_str(&format!(" {cell} |"));
        }
        out.push_str(&format!(" {} |\n", esc(&suffixes(op))));
    }
    out.push('\n');
    for op in &reg.ops {
        let output = &op.output;
        out.push_str(&format!("### {}\n\n", op.name));
        out.push_str(&format!("{}\n\n", op.summary.value()));
        match &op.options {
            Some(opts) => out.push_str(&format!("- Options: `{}`\n", tidy(&quote!(#opts)))),
            None => out.push_str("- Options: none\n"),
        }
        out.push_str(&format!("- Output: `{}`\n", tidy(&quote!(#output))));
        out.push_str(&format!("- Dtypes: {}\n", op.dtypes.value()));
        out.push_str(&format!("- Layout: {}\n", op.layout.value()));
        let errs = op
            .errors
            .iter()
            .map(|e| format!("`{}`", error_text(e)))
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!("- Errors: {errs}\n"));
        if op.omissions.is_empty() {
            out.push_str("- Omissions: none\n");
        } else {
            out.push_str("- Omissions:\n");
            for o in &op.omissions {
                out.push_str(&format!("  - {}\n", o.value()));
            }
        }
        out.push('\n');
    }
    if !reg.exceptions.is_empty() {
        out.push_str("### Exceptions\n\n");
        out.push_str(
            "| ID | Status | Symbols | Scope | Category | Owner | Reason | Verification | Exit condition |\n",
        );
        out.push_str("|---|---|---|---|---|---|---|---|---|\n");
        for e in &reg.exceptions {
            let symbols = e
                .symbols
                .iter()
                .map(|s| format!("`{}`", s.value()))
                .collect::<Vec<_>>()
                .join("<br>");
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} | `{}` | {} | {} | {} |\n",
                e.id,
                e.status,
                esc(&symbols),
                esc(&e.scope.value()),
                esc(&e.category.value()),
                esc(&e.owner_module.value()),
                esc(&e.reason.value()),
                esc(&e.verification.value()),
                esc(&e.exit_condition.value()),
            ));
        }
        out.push('\n');
    }
    out
}
