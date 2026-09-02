//! Tensor and QuantMatrix tests.

use super::*;
use crate::gguf::{GgmlType, Gguf};
use crate::quant::{self, QuantError};

#[test]
fn shape_strides_are_row_major() {
    let s = Shape::new(&[2, 3, 4]);
    assert_eq!(s.rank(), 3);
    assert_eq!(s.dims(), &[2, 3, 4]);
    assert_eq!(s.n_elements(), 24);
    // Innermost axis is contiguous; each step out multiplies by the axis below.
    assert_eq!(&s.strides()[..3], &[12, 4, 1]);
    assert_eq!(s.offset(&[0, 0, 0]), 0);
    assert_eq!(s.offset(&[1, 0, 0]), 12);
    assert_eq!(s.offset(&[0, 1, 0]), 4);
    assert_eq!(s.offset(&[1, 2, 3]), 12 + 8 + 3);
}

#[test]
fn shape_of_rank_one_is_contiguous() {
    let s = Shape::new(&[7]);
    assert_eq!(&s.strides()[..1], &[1]);
    assert_eq!(s.n_elements(), 7);
}

#[test]
#[should_panic(expected = "at least one dimension")]
fn shape_rejects_rank_zero() {
    Shape::new(&[]);
}

#[test]
#[should_panic(expected = "exceeds")]
fn shape_rejects_rank_above_four() {
    Shape::new(&[1, 2, 3, 4, 5]);
}

#[test]
fn tensor_view_rows() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let v = TensorView::new(&data, Shape::new(&[3, 4]));
    assert_eq!(v.row(0).as_slice(), &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(v.row(2).as_slice(), &[8.0, 9.0, 10.0, 11.0]);
    assert_eq!(v.row(1).shape().dims(), &[4]);
}

#[test]
#[should_panic(expected = "does not match shape")]
fn tensor_view_rejects_a_length_mismatch() {
    let data = [0.0f32; 5];
    TensorView::new(&data, Shape::new(&[2, 3]));
}

#[test]
fn tensor_view_mut_writes_through() {
    let mut data = vec![0.0f32; 6];
    let mut v = TensorViewMut::new(&mut data, Shape::new(&[2, 3]));
    v.row_mut(1).copy_from_slice(&[7.0, 8.0, 9.0]);
    assert_eq!(v.as_slice(), &[0.0, 0.0, 0.0, 7.0, 8.0, 9.0]);
    assert_eq!(v.as_view().row(1).as_slice(), &[7.0, 8.0, 9.0]);
}

// ---------------------------------------------------------- QuantMatrix ----

/// One Q8_0 block whose dequantised values are `scale * (i - 64)`.
fn q8_block(scale: f32) -> [u8; 34] {
    let mut b = [0u8; 34];
    b[0..2].copy_from_slice(&quant::f32_to_f16(scale).to_le_bytes());
    for i in 0..32 {
        b[2 + i] = (i as i32 - 64) as i8 as u8;
    }
    b
}

#[test]
fn quant_matrix_rows_and_geometry() {
    // 3 rows of 64 columns = 2 Q8_0 blocks per row.
    let mut data = Vec::new();
    for _ in 0..6 {
        data.extend_from_slice(&q8_block(0.5));
    }
    let m = QuantMatrix::new(GgmlType::Q8_0, &data, 3, 64).unwrap();

    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 64);
    assert_eq!(m.ggml_type(), GgmlType::Q8_0);
    assert_eq!(m.bytes_per_row(), 68);
    assert_eq!(m.byte_len(), 204);
    assert_eq!(m.row_bytes(1).len(), 68);

    let mut out = vec![0.0f32; 64];
    m.dequant_row(2, &mut out).unwrap();
    for (i, v) in out.iter().enumerate() {
        assert_eq!(*v, 0.5 * ((i % 32) as f32 - 64.0), "element {i}");
    }
}

#[test]
fn quant_matrix_rejects_bad_geometry() {
    let data = vec![0u8; 34];
    // 40 columns is not a whole number of 32-element Q8_0 blocks.
    assert_eq!(
        QuantMatrix::new(GgmlType::Q8_0, &data, 1, 40).unwrap_err(),
        QuantError::NotBlockAligned {
            ty: GgmlType::Q8_0,
            len: 40,
            block: 32
        }
    );
    // Right shape, wrong number of bytes.
    assert_eq!(
        QuantMatrix::new(GgmlType::Q8_0, &data, 2, 32).unwrap_err(),
        QuantError::BadSourceLength {
            ty: GgmlType::Q8_0,
            got: 34,
            want: 68
        }
    );
    // A format we can size but not decode.
    let data = vec![0u8; 176];
    assert_eq!(
        QuantMatrix::new(GgmlType::Q5_K, &data, 1, 256).unwrap_err(),
        QuantError::UnsupportedType(GgmlType::Q5_K)
    );
}

#[test]
fn quant_matrix_from_a_real_gguf_tensor_keeps_ggml_axis_order() {
    // GGUF dims [cols, rows]: 64 contiguous columns, 3 rows. The matrix must
    // read that as 3 rows of 64, not 64 rows of 3 -- getting it backwards would
    // transpose every weight in the model.
    use crate::gguf::tests_support::Builder;

    let mut data = Vec::new();
    for r in 0..3 {
        for _ in 0..2 {
            data.extend_from_slice(&q8_block(0.25 * (r + 1) as f32));
        }
    }
    let bytes = Builder::new()
        .tensor("w", &[64, 3], GgmlType::Q8_0, &data)
        .build();
    let g = Gguf::parse(&bytes).unwrap();
    let info = g.find_tensor("w").unwrap();

    let m = QuantMatrix::from_tensor(info, g.tensor_data(info)).unwrap();
    assert_eq!(m.cols(), 64, "dims[0] is the contiguous axis");
    assert_eq!(m.rows(), 3);

    // Each row was built with its own scale, so a transpose would be obvious.
    for r in 0..3 {
        let mut out = vec![0.0f32; 64];
        m.dequant_row(r, &mut out).unwrap();
        let scale = 0.25 * (r + 1) as f32;
        assert_eq!(out[0], scale * -64.0, "row {r}");
    }
}
