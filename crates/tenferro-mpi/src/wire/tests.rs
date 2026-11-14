//! Wire codec tests: no MPI, no launcher.

use tenferro_tensor::{Complex32, Complex64, DType};

use super::{
    checked_elements, checked_payload_bytes, dtype_code, dtype_from_code, element_bytes, Ack,
    DecodeError, Header, WireScalar, ACK_LEN, HEADER_PROLOGUE, HOST_BYTE_ORDER, KIND_ACK,
    KIND_HEADER, MAGIC, MAX_HEADER_LEN, MAX_WIRE_ELEMENTS, MAX_WIRE_RANK, PROTOCOL_VERSION,
    SHAPE_STRIDE,
};

fn sample_header() -> Header {
    Header::new(5, HOST_BYTE_ORDER, 6, &[2, 3])
}

#[test]
fn header_round_trips_through_its_golden_bytes() {
    let header = sample_header();
    let mut buffer = [0u8; MAX_HEADER_LEN];
    let len = header.encode(&mut buffer);
    assert_eq!(len, HEADER_PROLOGUE + 2 * SHAPE_STRIDE);

    let mut expected = Vec::new();
    expected.extend_from_slice(&MAGIC);
    expected.extend_from_slice(&PROTOCOL_VERSION.to_le_bytes());
    expected.push(KIND_HEADER);
    expected.push(5);
    expected.push(0);
    expected.push(HOST_BYTE_ORDER);
    expected.extend_from_slice(&[0, 0]);
    expected.extend_from_slice(&2u32.to_le_bytes());
    expected.extend_from_slice(&6u64.to_le_bytes());
    expected.extend_from_slice(&2u64.to_le_bytes());
    expected.extend_from_slice(&3u64.to_le_bytes());
    assert_eq!(&buffer[..len], expected.as_slice());

    let decoded = Header::decode(&buffer[..len]).expect("golden bytes decode");
    assert_eq!(decoded.version, PROTOCOL_VERSION);
    assert_eq!(decoded.dtype_code, 5);
    assert_eq!(decoded.status, 0);
    assert_eq!(decoded.byte_order, HOST_BYTE_ORDER);
    assert_eq!(decoded.elements, 6);
    assert_eq!(decoded.shape(), &[2, 3]);
}

#[test]
fn abort_header_carries_no_dtype_rank_or_payload() {
    let header = Header::abort(2);
    let mut buffer = [0u8; MAX_HEADER_LEN];
    let len = header.encode(&mut buffer);
    assert_eq!(len, HEADER_PROLOGUE);
    let decoded = Header::decode(&buffer[..len]).expect("abort header decodes");
    assert_eq!(decoded.status, 2);
    assert_eq!(decoded.dtype_code, 0);
    assert_eq!(decoded.rank(), 0);
    assert_eq!(decoded.elements, 0);
}

#[test]
fn header_rank_zero_and_largest_rank_round_trip() {
    let rank_zero = Header::new(4, HOST_BYTE_ORDER, 1, &[]);
    let mut buffer = [0u8; MAX_HEADER_LEN];
    let len = rank_zero.encode(&mut buffer);
    assert_eq!(len, HEADER_PROLOGUE);
    assert_eq!(
        Header::decode(&buffer[..len]).unwrap().shape(),
        &[] as &[usize]
    );

    let shape: Vec<usize> = vec![1; MAX_WIRE_RANK];
    let largest = Header::new(
        5,
        HOST_BYTE_ORDER,
        checked_elements(&shape).unwrap(),
        &shape,
    );
    let len = largest.encode(&mut buffer);
    assert_eq!(len, MAX_HEADER_LEN);
    assert_eq!(
        Header::decode(&buffer[..len]).unwrap().shape(),
        shape.as_slice()
    );
}

#[test]
fn header_decode_rejects_structural_damage() {
    let header = sample_header();
    let mut buffer = [0u8; MAX_HEADER_LEN];
    let len = header.encode(&mut buffer);

    assert_eq!(
        Header::decode(&buffer[..len - 1]).unwrap_err(),
        DecodeError::Length("length does not match the declared rank")
    );
    assert_eq!(
        Header::decode(&buffer[..8]).unwrap_err(),
        DecodeError::Length("shorter than the fixed header")
    );

    let mut bad_magic = buffer;
    bad_magic[0] = b'X';
    assert_eq!(
        Header::decode(&bad_magic[..len]).unwrap_err(),
        DecodeError::Magic
    );

    let mut bad_kind = buffer;
    bad_kind[6] = KIND_ACK;
    assert_eq!(
        Header::decode(&bad_kind[..len]).unwrap_err(),
        DecodeError::Kind(KIND_ACK)
    );

    let mut bad_rank = buffer;
    bad_rank[12..16].copy_from_slice(&((MAX_WIRE_RANK + 1) as u32).to_le_bytes());
    assert_eq!(
        Header::decode(&bad_rank[..len]).unwrap_err(),
        DecodeError::Length("declared rank exceeds MAX_WIRE_RANK")
    );
}

#[test]
fn ack_round_trips_through_its_golden_bytes() {
    let ack = Ack::new(3);
    let bytes = ack.encode();
    assert_eq!(bytes.len(), ACK_LEN);
    assert_eq!(&bytes[..4], &MAGIC);
    assert_eq!(&bytes[4..6], &PROTOCOL_VERSION.to_le_bytes());
    assert_eq!(bytes[6], KIND_ACK);
    assert_eq!(bytes[7], 3);
    let decoded = Ack::decode(&bytes).expect("ack decodes");
    assert_eq!(decoded.version, PROTOCOL_VERSION);
    assert_eq!(decoded.verdict, 3);

    assert_eq!(
        Ack::decode(&bytes[..4]).unwrap_err(),
        DecodeError::Length("ack frame is shorter than 8 bytes")
    );
}

#[test]
fn dtype_codes_are_pinned() {
    let table = [
        (DType::Bool, 1u8, 1usize),
        (DType::I32, 2, 4),
        (DType::I64, 3, 8),
        (DType::F32, 4, 4),
        (DType::F64, 5, 8),
        (DType::C32, 6, 8),
        (DType::C64, 7, 16),
    ];
    for (dtype, code, bytes) in table {
        assert_eq!(dtype_code(dtype), Some(code));
        assert_eq!(dtype_from_code(code), Some(dtype));
        assert_eq!(element_bytes(code), Some(bytes), "{dtype:?} element size");
    }
    assert_eq!(dtype_from_code(0), None);
    assert_eq!(dtype_from_code(8), None);
    assert_eq!(dtype_from_code(200), None);
    assert_eq!(element_bytes(0), None);
}

#[test]
fn element_counts_are_checked_and_bounded() {
    assert_eq!(checked_elements(&[]), Some(1));
    assert_eq!(checked_elements(&[2, 3, 4]), Some(24));
    assert_eq!(checked_elements(&[3, 0, 2]), Some(0));
    assert_eq!(checked_elements(&[usize::MAX, usize::MAX, 2]), None);

    assert_eq!(checked_payload_bytes(6, 8), Some(48));
    assert_eq!(checked_payload_bytes(0, 16), Some(0));
    assert_eq!(checked_payload_bytes(u64::MAX, 16), None);
    assert_eq!(MAX_WIRE_ELEMENTS, i32::MAX as u64);
}

#[test]
fn bool_payloads_are_validated() {
    let values = [true, false, true];
    assert_eq!(<bool as WireScalar>::bytes(&values), &[1u8, 0, 1]);
    let mut out = [false; 4];
    assert!(<bool as WireScalar>::assign_from_bytes(
        &mut out,
        &[0, 1, 1, 0]
    ));
    assert_eq!(out, [false, true, true, false]);
    assert!(!<bool as WireScalar>::assign_from_bytes(
        &mut out,
        &[0, 2, 1, 0]
    ));
    assert_eq!(
        out,
        [false, true, true, false],
        "a rejected payload changes nothing"
    );
    assert!(!<bool as WireScalar>::assign_from_bytes(&mut out, &[0xFF]));
    assert!(
        !<bool as WireScalar>::assign_from_bytes(&mut out, &[0, 1]),
        "length mismatch"
    );

    let mut mutable = [false, true];
    <bool as WireScalar>::bytes_mut(&mut mutable).copy_from_slice(&[1, 0]);
    assert_eq!(mutable, [true, false]);
}

#[test]
fn plain_scalars_need_no_payload_validation() {
    let floats = [1.0f64, -0.0, f64::NAN];
    assert_eq!(<f64 as WireScalar>::bytes(&floats).len(), 24);
    const { assert!(!<f64 as WireScalar>::VALIDATES_PAYLOAD) };

    let complex = [Complex64::new(1.0, 2.0)];
    assert_eq!(<Complex64 as WireScalar>::bytes(&complex).len(), 16);
    let complex32 = [Complex32::new(1.0, 2.0)];
    assert_eq!(<Complex32 as WireScalar>::bytes(&complex32).len(), 8);
    const { assert!(!<Complex64 as WireScalar>::VALIDATES_PAYLOAD) };
}
