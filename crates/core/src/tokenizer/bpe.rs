//! Byte-level BPE encode and decode.

use alloc::string::String;
use alloc::vec::Vec;

use super::byte_level;
use super::pretokenize::pretokenize;
use super::vocab::{TokenType, Vocab, VocabError};
use crate::gguf::Gguf;

#[derive(Debug, Clone, Copy, Default)]
pub struct EncodeOptions {
    /// Recognise special-token strings (`<|im_start|>`, `<|endoftext|>`) in the
    /// input rather than letting BPE shred them into pieces.
    ///
    /// This is a real security boundary once user text is involved: with it on,
    /// a user who types `<|im_end|>` into a chat message can close the turn and
    /// impersonate the system. Chat *scaffolding* should be encoded with it on;
    /// untrusted message bodies should not.
    pub parse_special: bool,
    /// Prepend BOS, if the model asks for one.
    pub add_bos: bool,
}

#[derive(Debug)]
pub struct Tokenizer<'a> {
    vocab: Vocab<'a>,
}

impl<'a> Tokenizer<'a> {
    pub fn from_gguf(g: &Gguf<'a>) -> Result<Self, VocabError> {
        Ok(Tokenizer {
            vocab: Vocab::from_gguf(g)?,
        })
    }

    pub fn vocab(&self) -> &Vocab<'a> {
        &self.vocab
    }

    // ------------------------------------------------------------- encode ----

    pub fn encode(&self, text: &str, opts: EncodeOptions) -> Vec<u32> {
        let mut out = Vec::new();
        if opts.add_bos && self.vocab.add_bos {
            if let Some(b) = self.vocab.bos {
                out.push(b);
            }
        }

        let mut pos = 0usize;
        while pos < text.len() {
            let hit = if opts.parse_special {
                self.next_special(text, pos)
            } else {
                None
            };
            match hit {
                Some((at, id)) => {
                    if at > pos {
                        self.encode_plain(&text[pos..at], &mut out);
                    }
                    out.push(id);
                    pos = at + self.vocab.token(id).map_or(0, str::len);
                }
                None => {
                    self.encode_plain(&text[pos..], &mut out);
                    break;
                }
            }
        }
        out
    }

    /// Earliest special-token occurrence at or after `from`; ties go to the
    /// longest match, so `<|im_start|>` is never read as a shorter prefix.
    fn next_special(&self, text: &str, from: usize) -> Option<(usize, u32)> {
        let mut best: Option<(usize, u32)> = None;
        // `specials` is sorted longest-first, so at equal offsets the first hit
        // found is the longest one and later equal-offset hits are rejected.
        for &sid in self.vocab.specials() {
            let s = self.vocab.token(sid)?;
            if s.is_empty() {
                continue;
            }
            if let Some(off) = text[from..].find(s) {
                let at = from + off;
                if best.is_none_or(|(b, _)| at < b) {
                    best = Some((at, sid));
                }
            }
        }
        best
    }

    /// Encode a stretch of text known to contain no special tokens.
    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        let mut mapped = String::new();
        let mut symbols: Vec<u32> = Vec::new();

        for range in pretokenize(text) {
            let chunk = &text[range];

            // Into the byte-mapped alphabet the vocab is written in.
            mapped.clear();
            byte_level::encode_into(chunk.as_bytes(), &mut mapped);

            // Seed with one symbol per mapped character. A GPT-2 vocab contains
            // all 256 single-character tokens, so this always resolves; the
            // fallback exists only so a malformed vocab degrades rather than
            // panics.
            symbols.clear();
            let mut buf = [0u8; 4];
            for c in mapped.chars() {
                let s = c.encode_utf8(&mut buf);
                match self.vocab.id_of(s) {
                    Some(id) => symbols.push(id),
                    None => {
                        debug_assert!(false, "vocab is missing byte-level token {s:?}");
                        if let Some(unk) = self.vocab.unk {
                            symbols.push(unk);
                        }
                    }
                }
            }

            self.apply_merges(&mut symbols);
            out.extend_from_slice(&symbols);
        }
    }

    /// The merge loop: repeatedly take the lowest-ranked adjacent pair and merge
    /// every non-overlapping occurrence of it, left to right.
    ///
    /// Merging all occurrences of the winning pair in one pass (rather than one
    /// occurrence at a time) is what the reference implementation does, and it
    /// matters: they share a rank, so doing them together cannot change which
    /// pair wins next, but doing them one at a time and re-scanning can pick a
    /// different pair in between.
    fn apply_merges(&self, symbols: &mut Vec<u32>) {
        let mut scratch: Vec<u32> = Vec::new();
        loop {
            let mut best: Option<(u32, u32, u32, u32)> = None; // (rank, left, right, result)
            for w in symbols.windows(2) {
                if let Some(m) = self.vocab.merge(w[0], w[1]) {
                    if best.is_none_or(|(r, ..)| m.rank < r) {
                        best = Some((m.rank, m.left, m.right, m.result));
                    }
                }
            }
            let Some((_, left, right, result)) = best else {
                return;
            };

            scratch.clear();
            let mut i = 0usize;
            while i < symbols.len() {
                if i + 1 < symbols.len() && symbols[i] == left && symbols[i + 1] == right {
                    scratch.push(result);
                    i += 2;
                } else {
                    scratch.push(symbols[i]);
                    i += 1;
                }
            }
            core::mem::swap(symbols, &mut scratch);
        }
    }

    // ------------------------------------------------------------- decode ----

    /// Decode to raw bytes.
    ///
    /// Bytes rather than a `String` because a token boundary is not a character
    /// boundary: one token can be half a UTF-8 sequence, so a streaming caller
    /// has to buffer until the bytes form valid text. Step 10's `decode_step`
    /// depends on this.
    pub fn decode_bytes(&self, ids: &[u32], skip_special: bool) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            if skip_special && self.vocab.token_type(id).is_some_and(TokenType::is_special) {
                continue;
            }
            if let Some(s) = self.vocab.token(id) {
                byte_level::decode_into(s, &mut out);
            }
        }
        out
    }

    /// Decode to text, replacing anything that is not valid UTF-8.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids, skip_special)).into_owned()
    }
}
