//! Turning logits into a token.
//!
//! The filters run in this order, which is the one HuggingFace and llama.cpp
//! both use:
//!
//! 1. repetition penalty  (on raw logits)
//! 2. temperature
//! 3. top-k
//! 4. top-p (nucleus)
//! 5. softmax over the survivors
//! 6. draw
//!
//! Order is not cosmetic. Temperature before top-p changes which tokens fall
//! inside the nucleus, because it changes the probabilities the cumulative sum
//! is computed over. Applying the repetition penalty *after* temperature would
//! make the penalty's strength depend on the temperature.

#![deny(unsafe_code)]

mod rng;

#[cfg(all(test, feature = "std"))]
mod tests;

pub use rng::Pcg32;

use alloc::vec;
use alloc::vec::Vec;

use crate::math::exp;

/// Everything that shapes the draw.
///
/// The defaults are "greedy": deterministic, and the baseline every other
/// setting is a deviation from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerConfig {
    /// `0.0` means greedy. Higher flattens the distribution.
    pub temperature: f32,
    /// Keep only the `k` highest-scoring tokens. `0` disables.
    pub top_k: usize,
    /// Keep the smallest set of tokens whose probabilities sum to at least this.
    /// `1.0` disables.
    pub top_p: f32,
    /// Divides the logits of recently-seen tokens. `1.0` disables.
    pub repetition_penalty: f32,
    /// How many recent tokens the penalty considers.
    pub repetition_window: usize,
    /// Fixed seed, so a generation can be reproduced from a URL.
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            repetition_window: 64,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    id: u32,
    /// Logit while filtering, probability after the softmax step.
    score: f32,
}

/// Stateful sampler: owns the RNG, the recent-token history, and its scratch.
#[derive(Debug)]
pub struct Sampler {
    cfg: SamplerConfig,
    rng: Pcg32,
    /// Recent tokens, most recent last. Bounded by `repetition_window`.
    history: Vec<u32>,
    /// Preallocated candidate buffer, sized to the vocabulary. Rebuilt in place
    /// each step so sampling never allocates.
    cand: Vec<Candidate>,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig, vocab_size: usize) -> Self {
        Sampler {
            rng: Pcg32::new(cfg.seed),
            cfg,
            history: Vec::with_capacity(256),
            cand: vec![Candidate { id: 0, score: 0.0 }; vocab_size],
        }
    }

    pub fn config(&self) -> &SamplerConfig {
        &self.cfg
    }

    /// Change settings without disturbing the RNG stream or the history.
    pub fn set_config(&mut self, cfg: SamplerConfig) {
        self.cfg = cfg;
    }

    /// Restart: same seed, same sequence, empty history.
    pub fn reset(&mut self) {
        self.rng = Pcg32::new(self.cfg.seed);
        self.history.clear();
    }

    /// Record a token so the repetition penalty can see it.
    pub fn accept(&mut self, token: u32) {
        self.history.push(token);
        // Keep only what the window can reach. `repetition_window` is small, so
        // the shift costs less than a ring buffer's index arithmetic would.
        let w = self.cfg.repetition_window.max(1);
        if self.history.len() > w {
            let drop = self.history.len() - w;
            self.history.drain(..drop);
        }
    }

    pub fn history(&self) -> &[u32] {
        &self.history
    }

    /// Pick the next token.
    ///
    /// `logits` is read, not written: the caller may want the originals for
    /// telemetry, and copying 151936 floats per step to protect them would be a
    /// real cost.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        assert_eq!(
            logits.len(),
            self.cand.len(),
            "logits do not match the vocabulary"
        );

        // Greedy is not "temperature = 0 with a divide guard" -- it is a separate
        // path, because it skips building and sorting the candidate list at all.
        if self.cfg.temperature <= 0.0 && self.cfg.repetition_penalty == 1.0 {
            return argmax(logits);
        }

        for (i, (c, &l)) in self.cand.iter_mut().zip(logits).enumerate() {
            c.id = i as u32;
            c.score = l;
        }
        let mut n = self.cand.len();

        // 1. Repetition penalty (CTRL's formulation, which HuggingFace adopted):
        //    divide positive logits, multiply negative ones. Both moves push the
        //    token *down*; a plain divide would promote negative logits instead.
        let p = self.cfg.repetition_penalty;
        if p != 1.0 {
            for &t in &self.history {
                let c = &mut self.cand[t as usize];
                c.score = if c.score > 0.0 {
                    c.score / p
                } else {
                    c.score * p
                };
            }
        }

        if self.cfg.temperature <= 0.0 {
            // Greedy, but only after the penalty has had its say.
            let best =
                self.cand[..n]
                    .iter()
                    .fold(self.cand[0], |a, &b| if b.score > a.score { b } else { a });
            return best.id;
        }

        // 2. Temperature.
        let inv_t = 1.0 / self.cfg.temperature;
        for c in &mut self.cand[..n] {
            c.score *= inv_t;
        }

        // 3. top-k, by partial selection rather than a full sort.
        if self.cfg.top_k > 0 && self.cfg.top_k < n {
            let k = self.cfg.top_k;
            self.cand[..n].select_nth_unstable_by(k - 1, |a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(core::cmp::Ordering::Equal)
            });
            n = k;
        }

        // 4. top-p. Needs the survivors in descending order.
        if self.cfg.top_p < 1.0 {
            self.cand[..n].sort_unstable_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(core::cmp::Ordering::Equal)
            });
            // Softmax over the survivors so the cumulative sum is in probability
            // space; doing it on raw logits would cut at a meaningless place.
            softmax_scores(&mut self.cand[..n]);
            let mut cum = 0.0f32;
            let mut keep = n;
            for (i, c) in self.cand[..n].iter().enumerate() {
                cum += c.score;
                if cum >= self.cfg.top_p {
                    keep = i + 1;
                    break;
                }
            }
            // Always keep at least one: a p below the top token's probability
            // must still leave something to draw from.
            n = keep.max(1);
            // Renormalise over what survived.
            let total: f32 = self.cand[..n].iter().map(|c| c.score).sum();
            if total > 0.0 {
                for c in &mut self.cand[..n] {
                    c.score /= total;
                }
            }
        } else {
            softmax_scores(&mut self.cand[..n]);
        }

        // 5. Draw.
        let r = self.rng.next_f32();
        let mut cum = 0.0f32;
        for c in &self.cand[..n] {
            cum += c.score;
            if r < cum {
                return c.id;
            }
        }
        // Only reachable through accumulated rounding; the last candidate is the
        // right answer in that case.
        self.cand[n - 1].id
    }
}

/// Index of the largest value. Ties go to the lowest index, matching
/// `torch.argmax`, so a golden-token test is stable.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Max-subtracted softmax over a candidate slice.
fn softmax_scores(cand: &mut [Candidate]) {
    if cand.is_empty() {
        return;
    }
    let max = cand.iter().fold(f32::NEG_INFINITY, |a, c| a.max(c.score));
    if !max.is_finite() {
        for c in cand.iter_mut() {
            c.score = 0.0;
        }
        return;
    }
    let mut sum = 0.0f32;
    for c in cand.iter_mut() {
        c.score = exp(c.score - max);
        sum += c.score;
    }
    let inv = 1.0 / sum;
    for c in cand.iter_mut() {
        c.score *= inv;
    }
}
