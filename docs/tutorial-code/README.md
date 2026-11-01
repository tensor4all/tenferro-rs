# tenferro Tutorial Code

Runnable source code for the tenferro tutorial pages.

The online tutorials quote these binaries as their source of truth. Run them
with:

```bash
cargo test -p tenferro-tutorial-code --release
```

In CI, this package is executed by the existing workspace test command. Do not
add a separate tutorial workflow that recompiles tenferro after unit tests.
