//! Locating and validating every tensor the forward pass needs.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::gguf::{GgmlType, Gguf, ModelConfig, TensorInfo};
use crate::quant::{self, QuantError};
use crate::tensor::QuantMatrix;

#[derive(Debug, Clone, PartialEq)]
pub enum WeightError {
    Missing(String),
    Quant {
        name: String,
        err: QuantError,
    },
    /// A tensor is the wrong shape for the config, which almost always means
    /// the GGUF axis order was misread somewhere.
    BadShape {
        name: String,
        got: (usize, usize),
        want: (usize, usize),
    },
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeightError::Missing(n) => write!(f, "model is missing tensor {n:?}"),
            WeightError::Quant { name, err } => write!(f, "tensor {name:?}: {err}"),
            WeightError::BadShape { name, got, want } => write!(
                f,
                "tensor {name:?} is {}x{} (rows x cols), expected {}x{}",
                got.0, got.1, want.0, want.1
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for WeightError {}

/// One transformer block's parameters.
#[derive(Debug)]
pub struct LayerWeights<'a> {
    pub attn_norm: Vec<f32>,
    pub attn_q: QuantMatrix<'a>,
    pub attn_k: QuantMatrix<'a>,
    pub attn_v: QuantMatrix<'a>,
    pub attn_output: QuantMatrix<'a>,
    /// Qwen2 has biases on Q, K and V. Llama does not, and `attn_output` here
    /// does not either -- so these are `Option`, present for Qwen2 and absent
    /// for a Llama-family model going through the same loader.
    pub attn_q_bias: Option<Vec<f32>>,
    pub attn_k_bias: Option<Vec<f32>>,
    pub attn_v_bias: Option<Vec<f32>>,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: QuantMatrix<'a>,
    pub ffn_up: QuantMatrix<'a>,
    pub ffn_down: QuantMatrix<'a>,
}

/// Every parameter, borrowed from the model buffer.
#[derive(Debug)]
pub struct ModelWeights<'a> {
    pub config: ModelConfig,
    pub token_embd: QuantMatrix<'a>,
    pub layers: Vec<LayerWeights<'a>>,
    pub output_norm: Vec<f32>,
    /// The unembedding. Qwen2.5-0.5B ties this to `token_embd` in HF, but the
    /// GGUF writes it as its own tensor -- and quantises it differently (Q8_0
    /// here against Q5_0 for the embedding), so they are not interchangeable.
    pub output: QuantMatrix<'a>,
}

fn find<'a, 'g>(g: &'g Gguf<'a>, name: &str) -> Result<&'g TensorInfo<'a>, WeightError> {
    g.find_tensor(name)
        .ok_or_else(|| WeightError::Missing(name.to_string()))
}

/// Load a small F32 tensor into an owned vector.
///
/// F32 tensor data cannot be borrowed as `&[f32]`: the file buffer has no
/// alignment guarantee, and a misaligned `&[f32]` is undefined behaviour. These
/// are the norm weights and biases -- 280 KiB in total against a 463 MB model --
/// so copying them is free.
fn load_f32(g: &Gguf<'_>, name: &str) -> Result<Vec<f32>, WeightError> {
    let info = find(g, name)?;
    let n = info.n_elements() as usize;
    let mut out = vec![0.0f32; n];
    quant::dequantize_row(info.ggml_type, g.tensor_data(info), &mut out).map_err(|err| {
        WeightError::Quant {
            name: name.to_string(),
            err,
        }
    })?;
    Ok(out)
}

fn load_f32_opt(g: &Gguf<'_>, name: &str) -> Result<Option<Vec<f32>>, WeightError> {
    if g.find_tensor(name).is_none() {
        return Ok(None);
    }
    load_f32(g, name).map(Some)
}

fn load_matrix<'a>(
    g: &Gguf<'a>,
    name: &str,
    want: (usize, usize),
) -> Result<QuantMatrix<'a>, WeightError> {
    let info = find(g, name)?;
    let m =
        QuantMatrix::from_tensor(info, g.tensor_data(info)).map_err(|err| WeightError::Quant {
            name: name.to_string(),
            err,
        })?;
    // Catching a transposed matrix here rather than at the first NaN is worth
    // the four lines: GGUF's axis order is reversed from torch's, and a
    // transposed weight produces plausible garbage rather than an error.
    if (m.rows(), m.cols()) != want {
        return Err(WeightError::BadShape {
            name: name.to_string(),
            got: (m.rows(), m.cols()),
            want,
        });
    }
    Ok(m)
}

impl<'a> ModelWeights<'a> {
    pub fn from_gguf(g: &Gguf<'a>) -> Result<Self, WeightError> {
        let config = g
            .config()
            .map_err(|e| WeightError::Missing(e.to_string()))?;
        let d = config.embedding_length;
        let ff = config.feed_forward_length;
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();
        let vocab = config.vocab_size;

        let mut layers = Vec::with_capacity(config.block_count);
        for i in 0..config.block_count {
            let p = |suffix: &str| format!("blk.{i}.{suffix}");
            layers.push(LayerWeights {
                attn_norm: load_f32(g, &p("attn_norm.weight"))?,
                // Projections are [out, in]: `out` rows of `in` contiguous
                // weights, which is exactly GGUF's layout with dims reversed.
                attn_q: load_matrix(g, &p("attn_q.weight"), (q_dim, d))?,
                attn_k: load_matrix(g, &p("attn_k.weight"), (kv_dim, d))?,
                attn_v: load_matrix(g, &p("attn_v.weight"), (kv_dim, d))?,
                attn_output: load_matrix(g, &p("attn_output.weight"), (d, q_dim))?,
                attn_q_bias: load_f32_opt(g, &p("attn_q.bias"))?,
                attn_k_bias: load_f32_opt(g, &p("attn_k.bias"))?,
                attn_v_bias: load_f32_opt(g, &p("attn_v.bias"))?,
                ffn_norm: load_f32(g, &p("ffn_norm.weight"))?,
                ffn_gate: load_matrix(g, &p("ffn_gate.weight"), (ff, d))?,
                ffn_up: load_matrix(g, &p("ffn_up.weight"), (ff, d))?,
                // Note the flip: down projects ff -> d.
                ffn_down: load_matrix(g, &p("ffn_down.weight"), (d, ff))?,
            });
        }

        Ok(ModelWeights {
            token_embd: load_matrix(g, "token_embd.weight", (vocab, d))?,
            layers,
            output_norm: load_f32(g, "output_norm.weight")?,
            output: load_matrix(g, "output.weight", (vocab, d))?,
            config,
        })
    }

    /// True if the model carries Qwen2-style attention biases.
    pub fn has_attn_bias(&self) -> bool {
        self.layers.first().is_some_and(|l| l.attn_q_bias.is_some())
    }

    /// Bytes of weight data referenced (not copied) plus the owned norms.
    pub fn byte_len(&self) -> usize {
        let owned: usize = self
            .layers
            .iter()
            .map(|l| {
                (l.attn_norm.len() + l.ffn_norm.len()) * 4
                    + [&l.attn_q_bias, &l.attn_k_bias, &l.attn_v_bias]
                        .iter()
                        .filter_map(|b| b.as_ref().map(|v| v.len() * 4))
                        .sum::<usize>()
            })
            .sum::<usize>()
            + self.output_norm.len() * 4;
        let borrowed: usize = self
            .layers
            .iter()
            .map(|l| {
                l.attn_q.byte_len()
                    + l.attn_k.byte_len()
                    + l.attn_v.byte_len()
                    + l.attn_output.byte_len()
                    + l.ffn_gate.byte_len()
                    + l.ffn_up.byte_len()
                    + l.ffn_down.byte_len()
            })
            .sum::<usize>()
            + self.token_embd.byte_len()
            + self.output.byte_len();
        owned + borrowed
    }

    /// The distinct quantisation formats in use, with tensor counts.
    pub fn formats(&self) -> Vec<(GgmlType, usize)> {
        let mut out: Vec<(GgmlType, usize)> = Vec::new();
        let mut bump = |t: GgmlType| match out.iter_mut().find(|(x, _)| *x == t) {
            Some(e) => e.1 += 1,
            None => out.push((t, 1)),
        };
        bump(self.token_embd.ggml_type());
        bump(self.output.ggml_type());
        for l in &self.layers {
            for m in [
                &l.attn_q,
                &l.attn_k,
                &l.attn_v,
                &l.attn_output,
                &l.ffn_gate,
                &l.ffn_up,
                &l.ffn_down,
            ] {
                bump(m.ggml_type());
            }
        }
        out.sort_by_key(|(t, _)| *t);
        out
    }
}
