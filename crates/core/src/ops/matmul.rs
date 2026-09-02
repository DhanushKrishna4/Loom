//! Matrix multiplication, naive.
//!
//! Both entry points are "normal times transposed": the right-hand operand is
//! indexed by row, not column, because that is how GGUF stores weights and it
//! means every dot product reads two contiguous runs of memory.

use super::elementwise::dot;
use crate::gguf::GgmlType;
use crate::quant::{self, ActivationQ8, QuantError, UnpackedRow};
use crate::tensor::QuantMatrix;

/// `out[i, j] = dot(lhs[i, ..], rhs[j, ..])`
///
/// * `lhs` is `[m, k]` row-major -- one row per token during prefill
/// * `rhs` is `[n, k]` row-major -- one row per output neuron, as stored in GGUF
/// * `out` is `[m, n]` row-major
///
/// The `n`/`k` naming follows the usual GEMM convention, but note `rhs` is
/// *already* in the orientation we want; there is no implicit transpose and no
/// transpose is ever materialised.
///
/// This is the PREFILL path: the whole prompt at once. Decode uses [`matvec`].
pub fn matmul_nt(lhs: &[f32], rhs: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(lhs.len(), m * k, "lhs must be m*k");
    assert_eq!(rhs.len(), n * k, "rhs must be n*k");
    assert_eq!(out.len(), m * n, "out must be m*n");

    // Naive triple loop, deliberately. Blocking for L1 and unrolling the inner
    // loop is step 9, and it gets benchmarked against this.
    for i in 0..m {
        let lhs_row = &lhs[i * k..(i + 1) * k];
        for j in 0..n {
            out[i * n + j] = dot(lhs_row, &rhs[j * k..(j + 1) * k]);
        }
    }
}

/// `out[r] = dot(w[r, ..], x)` -- one matrix, one vector.
///
/// This is the DECODE path, and it is where essentially all generation time
/// goes. It is not a special case of matmul in performance terms: with a single
/// activation vector there is no data reuse across output rows, so it is
/// memory-bandwidth bound rather than compute bound, and it wants a completely
/// different optimisation strategy (streaming the weights, not blocking them).
/// Hence a separate entry point from the start, even though today the body is
/// the same loop.
pub fn matvec(w: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    assert_eq!(w.len(), rows * cols, "weight must be rows*cols");
    assert_eq!(x.len(), cols, "input must be cols long");
    assert_eq!(out.len(), rows, "output must be rows long");

    for (r, o) in out.iter_mut().enumerate() {
        *o = dot(&w[r * cols..(r + 1) * cols], x);
    }
}

/// `out[r] = dot(dequant(w[r]), x)` -- matvec straight out of quantised weights.
///
/// One row is unpacked into `scratch` at a time, used, and thrown away. Nothing
/// ever holds a dequantised copy of a whole tensor: at Q5_0, Qwen2.5-0.5B's
/// weights are 463 MB on disk and would be 2 GB as f32, which does not fit in
/// wasm's practical budget and would be bandwidth suicide even if it did.
///
/// This is still the **unfused** form: it writes 896 floats to memory and reads
/// them straight back. Step 6 replaces it with a kernel that quantises the
/// activation vector once and keeps the unpacked weights in registers, and this
/// function stays as the oracle that kernel is tested against.
pub fn matvec_dequant(
    w: &QuantMatrix<'_>,
    x: &[f32],
    out: &mut [f32],
    scratch: &mut [f32],
) -> Result<(), QuantError> {
    assert_eq!(x.len(), w.cols(), "input must match the weight row length");
    assert_eq!(
        out.len(),
        w.rows(),
        "output must match the weight row count"
    );
    assert!(scratch.len() >= w.cols(), "scratch must hold one row");

    let row = &mut scratch[..w.cols()];
    for (r, o) in out.iter_mut().enumerate() {
        w.dequant_row(r as u32, row)?;
        *o = dot(row, x);
    }
    Ok(())
}

/// Fused quantised matvec: the decode hot path.
///
/// `a` must already hold the activation vector, quantised once for the whole
/// matmul. Nothing dequantised ever reaches memory -- see [`crate::quant`]'s
/// `vecdot` module for why each format needs the kernel it does.
///
/// F32 weight *matrices* are rejected rather than silently handled: there are
/// none in a real GGUF (F32 tensors are norms and biases, which are not
/// matmuls), and quietly falling back would hide a model that is not what the
/// caller thinks it is. [`matvec_dequant`] covers that case explicitly.
pub fn matvec_fused(
    w: &QuantMatrix<'_>,
    a: &ActivationQ8,
    out: &mut [f32],
) -> Result<(), QuantError> {
    assert_eq!(
        a.len(),
        w.cols(),
        "activation must match the weight row length"
    );
    assert_eq!(
        out.len(),
        w.rows(),
        "output must match the weight row count"
    );

    // One dispatch per matmul, hoisted above every loop underneath it.
    let kernel: fn(&[u8], &ActivationQ8) -> f32 = match w.ggml_type() {
        GgmlType::Q8_0 => quant::row_dot_q8_0,
        GgmlType::Q5_0 => quant::row_dot_q5_0,
        GgmlType::Q4_0 => quant::row_dot_q4_0,
        GgmlType::Q4_K => quant::row_dot_q4_k,
        GgmlType::Q6_K => quant::row_dot_q6_k,
        other => return Err(QuantError::UnsupportedType(other)),
    };
    for (r, o) in out.iter_mut().enumerate() {
        *o = kernel(w.row_bytes(r), a);
    }
    Ok(())
}

/// Batched fused matmul: one weight matrix against `acts.len()` activation
/// vectors, writing `[t][row]` into `out`.
///
/// # Why this exists
///
/// [`matvec_fused`] re-reads and re-unpacks the entire weight matrix for every
/// token. At batch size 1 there is nothing to be done about that — the model is
/// ~463 MB and every token genuinely needs all of it. During prefill there are
/// many tokens available at once, and unpacking a row once to serve all of them
/// turns 463 MB of memory traffic *per token* into 463 MB *per chunk*.
///
/// The unpacked row also stays in cache across the inner loop, which is what
/// moves prefill from memory-bound toward compute-bound.
///
/// Results are bit-identical to calling [`matvec_fused`] once per activation —
/// see [`crate::quant::UnpackedRow`] for why that took deliberate care.
pub fn matmul_fused(
    w: &QuantMatrix<'_>,
    acts: &[ActivationQ8],
    out: &mut [f32],
    scratch: &mut UnpackedRow,
) -> Result<(), QuantError> {
    let t = acts.len();
    let rows = w.rows();
    assert_eq!(out.len(), t * rows, "out must be t*rows");
    for a in acts {
        assert_eq!(
            a.len(),
            w.cols(),
            "every activation must match the row length"
        );
    }
    if !quant::is_supported(w.ggml_type()) || w.ggml_type() == GgmlType::F32 {
        return Err(QuantError::UnsupportedType(w.ggml_type()));
    }

    for r in 0..rows {
        // Once per row, not once per row per token: this is the whole point.
        quant::unpack_row(w.ggml_type(), w.row_bytes(r), scratch);
        for (ti, a) in acts.iter().enumerate() {
            out[ti * rows + r] = quant::row_dot_unpacked(scratch, a);
        }
    }
    Ok(())
}
