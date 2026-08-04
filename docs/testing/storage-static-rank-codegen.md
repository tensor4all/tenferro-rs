# Static-rank storage codegen

```json
{
  "schema": "tenferro.storage-static-rank-codegen.v1",
  "candidate_commit": "e8c7d3019df316424de12dc6d69b3f67bce87e14",
  "command": "cargo rustc -p tenferro-tensor --bench element_access --release -- --emit=asm",
  "rustc": "rustc 1.97.1 (8bab26f4f 2026-07-14)\nbinary: rustc\ncommit-hash: 8bab26f4f68e0e26f0bb7960be334d5b520ea452\ncommit-date: 2026-07-14\nhost: x86_64-unknown-linux-gnu\nrelease: 1.97.1\nLLVM version: 22.1.6",
  "target": "x86_64-unknown-linux-gnu",
  "probes": [
    "tensor_static_rank_read_probe",
    "tensor_static_rank_write_probe"
  ],
  "status": "pass",
  "observations": [
    "both fixed-rank probes are present and their backward loops contain no prohibited setup calls"
  ],
  "assembly": "target/release/deps/element_access-b8d5525be22d5bd4.s"
}
```
