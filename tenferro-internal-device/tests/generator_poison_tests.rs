use std::panic::{catch_unwind, AssertUnwindSafe};

use tenferro_device::{with_default_generator, Error, LogicalMemorySpace};

#[test]
fn with_default_generator_reports_poisoned_cpu_mutex() {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let _ = with_default_generator(LogicalMemorySpace::MainMemory, |_| -> Result<(), Error> {
            panic!("poison the default generator mutex");
        });
    }));

    let err = with_default_generator(LogicalMemorySpace::MainMemory, |_| Ok::<(), Error>(()))
        .unwrap_err();
    assert!(err
        .to_string()
        .contains("default CPU generator mutex poisoned"));
}
