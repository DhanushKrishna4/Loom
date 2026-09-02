//! BPE tokenizer. **In progress** -- character classification only so far.
//!
//! Qwen2 uses GPT-2 style byte-level BPE (`tokenizer.ggml.model == "gpt2"`),
//! which has four parts: a byte-to-unicode remapping, a pre-tokenizer that
//! splits text into chunks merges may not cross, the merge loop itself, and
//! special-token handling. This module currently has the classification
//! primitives the pre-tokenizer is built on.
//!
//! # Why the tables and not a regex
//!
//! Qwen2's pre-tokenizer pattern is
//!
//! ```text
//! (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//! ```
//!
//! `\p{L}`, `\p{N}` and `\s` are Unicode properties, and approximating them with
//! ASCII would put token boundaries in different places for any non-English
//! input -- which changes the token IDs, which changes the output, silently. The
//! `regex` crate is not in the allowed dependency list and would be the largest
//! thing in the binary, so the classification comes from generated range tables
//! (`tools/gen_unicode_tables.py`, ~6.6 KB) searched at runtime.

#![deny(unsafe_code)]

mod bpe;
mod byte_level;
mod chat;
mod pretokenize;
mod unicode_tables;
mod vocab;

pub use bpe::{EncodeOptions, Tokenizer};
pub use byte_level::{byte_to_char, char_to_byte};
pub use chat::{apply_chat_template, ChatMessage, DEFAULT_SYSTEM_PROMPT};
pub use pretokenize::pretokenize;
pub use vocab::{Merge, TokenType, Vocab, VocabError};

#[cfg(all(test, feature = "std"))]
mod tests;

/// Binary search a sorted, non-overlapping, inclusive range table.
#[inline]
fn in_ranges(table: &[(u32, u32)], cp: u32) -> bool {
    table
        .binary_search_by(|&(lo, hi)| {
            if cp < lo {
                core::cmp::Ordering::Greater
            } else if cp > hi {
                core::cmp::Ordering::Less
            } else {
                core::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// `\p{L}` -- Unicode categories Lu, Ll, Lt, Lm, Lo.
#[inline]
pub fn is_letter(c: char) -> bool {
    in_ranges(unicode_tables::LETTER, c as u32)
}

/// `\p{N}` -- Unicode categories Nd, Nl, No.
#[inline]
pub fn is_number(c: char) -> bool {
    in_ranges(unicode_tables::NUMBER, c as u32)
}

/// `\s` -- the Unicode White_Space property.
#[inline]
pub fn is_whitespace(c: char) -> bool {
    in_ranges(unicode_tables::WHITESPACE, c as u32)
}
