//! Cross-tool interoperability with h5py, gated on `TENFERRO_HDF5_INTEROP=1`.
//!
//! Rust writes a file, `tests/interop/h5py_check.py` asserts the documented
//! schema from the Python side, then Python writes a file with the same schema
//! and Rust reads it back.
use std::path::Path;
use std::process::Command;

use num_complex::Complex64;
use tenferro_hdf5::hdf5;
use tenferro_hdf5::{read_info, read_typed, write_typed, Error, SchemaViolation};
use tenferro_tensor::{DynRank, Host, TypedTensor};

/// Both tests spawn an HDF5 process while this process holds HDF5 handles; two
/// concurrent opens raced on the file lock on this host, so they take turns.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serialize() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn enabled() -> bool {
    std::env::var("TENFERRO_HDF5_INTEROP").as_deref() == Ok("1")
}

fn run_h5py_check(path: &Path) {
    let python = std::env::var("TENFERRO_HDF5_PYTHON").unwrap_or_else(|_| "python3".to_owned());
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/interop/h5py_check.py");
    let output = Command::new(&python)
        .arg(script)
        .arg(path)
        .output()
        .unwrap_or_else(|error| panic!("cannot run {python}: {error}"));
    assert!(
        output.status.success(),
        "{python} {script} failed: status {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn write<T>(group: &hdf5::Group, name: &str, shape: Vec<usize>, values: Vec<T>)
where
    T: tenferro_tensor::TensorScalar + hdf5::H5Type,
{
    let tensor = TypedTensor::<T, DynRank, Host>::from_host_vec_col_major(shape, values).unwrap();
    write_typed(group, name, tensor.as_view()).unwrap();
}

#[test]
fn h5py_reads_the_documented_schema() {
    let _serial = serialize();
    if !enabled() {
        eprintln!("skipped: set TENFERRO_HDF5_INTEROP=1 to run the h5py interop test");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rust.h5");
    {
        let file = hdf5::File::create(&path).unwrap();
        let group = file.create_group("tensors").unwrap();

        write(&group, "rank0_f64", vec![], vec![2.5_f64]);
        write(&group, "empty_f64", vec![2, 0, 3], Vec::<f64>::new());
        write(
            &group,
            "f32_3x4",
            vec![3, 4],
            (0..12).map(|i| i as f32 * 0.5 - 2.0).collect(),
        );
        write(
            &group,
            "f64_3x4",
            vec![3, 4],
            (0..12).map(|i| i as f64 * 0.5 - 2.0).collect(),
        );
        write(
            &group,
            "i32_2x2",
            vec![2, 2],
            (0..4).map(|i: i32| i * 7 - 11).collect(),
        );
        write(
            &group,
            "i64_2x2",
            vec![2, 2],
            (0..4).map(|i| i as i64 * 7 - 11).collect(),
        );
        write(
            &group,
            "c64_2x2",
            vec![2, 2],
            (0..4)
                .map(|i| Complex64::new(i as f64 * 0.25, -(i as f64)))
                .collect(),
        );
        write(
            &group,
            "bool_5",
            vec![5],
            (0..5).map(|i| i % 3 == 1).collect(),
        );
    }
    run_h5py_check(&path);
}

#[test]
fn rust_reads_what_h5py_writes() {
    let _serial = serialize();
    if !enabled() {
        eprintln!("skipped: set TENFERRO_HDF5_INTEROP=1 to run the h5py interop test");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("python.h5");
    {
        let file = hdf5::File::create(&path).unwrap();
        file.create_group("tensors").unwrap();
    }
    run_h5py_check(&path);

    let file = hdf5::File::open(&path).unwrap();
    let group = file.group("tensors").unwrap();

    // Big-endian on disk: the descriptor check ignores byte order and HDF5
    // converts, so the values still come back unchanged.
    let big_endian: TypedTensor<f64, DynRank, Host> = read_typed(&group, "py_be_f64").unwrap();
    assert_eq!(big_endian.shape(), &[2, 3]);
    let expected: Vec<f64> = (0..6).map(|i| i as f64 * 0.5 - 2.0).collect();
    assert_eq!(big_endian.as_slice(), expected.as_slice());

    let complex: TypedTensor<Complex64, DynRank, Host> = read_typed(&group, "py_c64").unwrap();
    let expected: Vec<Complex64> = (0..4)
        .map(|i| Complex64::new(i as f64 * 0.25, -(i as f64)))
        .collect();
    assert_eq!(complex.as_slice(), expected.as_slice());

    let booleans: TypedTensor<bool, DynRank, Host> = read_typed(&group, "py_bool").unwrap();
    let expected: Vec<bool> = (0..5).map(|i| i % 3 == 1).collect();
    assert_eq!(booleans.as_slice(), expected.as_slice());

    // A low-level writer can store a byte that is no `bool` member, so the
    // guard in front of the `Vec<i8>` reinterpretation needs its own fixture.
    let error = read_typed::<bool, DynRank>(&group, "py_bad_bool").unwrap_err();
    assert!(
        matches!(
            error,
            Error::InvalidBool {
                flat_index: 0,
                byte: 2
            }
        ),
        "{error:?}"
    );

    let error = read_info(&group, "py_bad_utf8").unwrap_err();
    assert!(
        matches!(
            error,
            Error::Schema {
                violation: SchemaViolation::AttributeNotUtf8("type"),
                ..
            }
        ),
        "{error:?}"
    );
}
