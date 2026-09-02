//! The vocabulary and merge table, loaded from GGUF metadata.
//!
//! Everything here borrows from the model buffer: 151936 token strings would be
//! 151936 allocations and ~2 MB of copies otherwise, for no benefit.
//!
//! There is no `HashMap`. `core` has none, `hashbrown` is not in the allowed
//! dependency list, and the two lookups we need (string -> id, pair -> merge)
//! are both fine as sorted arrays with a binary search: ~17 comparisons against
//! a hash plus a probe, on tables built once at load.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::gguf::Gguf;

/// `tokenizer.ggml.token_type` values, from `llama_token_type` in llama.cpp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TokenType {
    Undefined = 0,
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
}

impl TokenType {
    fn from_i32(v: i32) -> Self {
        match v {
            1 => TokenType::Normal,
            2 => TokenType::Unknown,
            3 => TokenType::Control,
            4 => TokenType::UserDefined,
            5 => TokenType::Unused,
            6 => TokenType::Byte,
            _ => TokenType::Undefined,
        }
    }

    /// Tokens that must be matched literally in the input rather than reached
    /// through BPE: `<|im_start|>`, `<|endoftext|>` and friends.
    pub fn is_special(self) -> bool {
        matches!(self, TokenType::Control | TokenType::UserDefined)
    }
}

/// One entry of the merge table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Merge {
    pub left: u32,
    pub right: u32,
    /// Position in the merge list. Lower wins.
    pub rank: u32,
    /// Id of the concatenated result, resolved at load time so the merge loop
    /// never has to build a string and look it up.
    pub result: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VocabError {
    MissingKey(&'static str),
    WrongType(&'static str),
    /// A merge rule referenced a token that is not in the vocab.
    NoMergesUsable {
        total: usize,
    },
    EmptyVocab,
}

impl fmt::Display for VocabError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VocabError::MissingKey(k) => write!(f, "tokenizer metadata key {k:?} is missing"),
            VocabError::WrongType(k) => {
                write!(f, "tokenizer metadata key {k:?} has the wrong type")
            }
            VocabError::NoMergesUsable { total } => write!(
                f,
                "none of the {total} merge rules resolved against the vocab; \
                 the merges and tokens arrays disagree"
            ),
            VocabError::EmptyVocab => write!(f, "vocabulary is empty"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for VocabError {}

/// Token strings, their types, and the merge table.
#[derive(Debug)]
pub struct Vocab<'a> {
    tokens: Vec<&'a str>,
    types: Vec<TokenType>,
    /// Token ids ordered by their string, for binary search.
    by_str: Vec<u32>,
    /// Merges ordered by `(left, right)`, for binary search.
    merges: Vec<Merge>,
    /// Special-token ids, longest string first so that scanning finds the
    /// longest match at a position rather than a prefix of it.
    specials: Vec<u32>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub pad: Option<u32>,
    pub unk: Option<u32>,
    pub add_bos: bool,
    /// Merge rules that referenced an unknown token and were dropped.
    pub skipped_merges: usize,
}

impl<'a> Vocab<'a> {
    pub fn from_gguf(g: &Gguf<'a>) -> Result<Self, VocabError> {
        let tokens: Vec<&'a str> = g
            .get_array("tokenizer.ggml.tokens")
            .ok_or(VocabError::MissingKey("tokenizer.ggml.tokens"))?
            .as_str_slice()
            .ok_or(VocabError::WrongType("tokenizer.ggml.tokens"))?
            .to_vec();
        if tokens.is_empty() {
            return Err(VocabError::EmptyVocab);
        }

        // token_type is optional; absent means "everything is a normal token".
        let types: Vec<TokenType> = match g.get_array("tokenizer.ggml.token_type") {
            Some(a) => a
                .as_i32_slice()
                .ok_or(VocabError::WrongType("tokenizer.ggml.token_type"))?
                .iter()
                .map(|v| TokenType::from_i32(*v))
                .collect(),
            None => alloc::vec![TokenType::Normal; tokens.len()],
        };

        // Sorted index for string -> id.
        let mut by_str: Vec<u32> = (0..tokens.len() as u32).collect();
        by_str.sort_unstable_by_key(|&i| tokens[i as usize]);

        let lookup = |s: &str| -> Option<u32> {
            by_str
                .binary_search_by(|&i| tokens[i as usize].cmp(s))
                .ok()
                .map(|k| by_str[k])
        };

        // Merges: "A B" -> (id(A), id(B), rank, id(AB)).
        let raw_merges = g
            .get_array("tokenizer.ggml.merges")
            .ok_or(VocabError::MissingKey("tokenizer.ggml.merges"))?
            .as_str_slice()
            .ok_or(VocabError::WrongType("tokenizer.ggml.merges"))?;

        let mut merges = Vec::with_capacity(raw_merges.len());
        let mut skipped_merges = 0usize;
        let mut joined = String::new();
        for (rank, rule) in raw_merges.iter().enumerate() {
            // The two halves are separated by a literal space. A space inside a
            // token is impossible here: everything is in the byte-mapped
            // alphabet, where 0x20 is written as U+0120.
            let Some(sp) = rule.find(' ') else {
                skipped_merges += 1;
                continue;
            };
            let (a, b) = (&rule[..sp], &rule[sp + 1..]);
            joined.clear();
            joined.push_str(a);
            joined.push_str(b);
            match (lookup(a), lookup(b), lookup(&joined)) {
                (Some(left), Some(right), Some(result)) => merges.push(Merge {
                    left,
                    right,
                    rank: rank as u32,
                    result,
                }),
                _ => skipped_merges += 1,
            }
        }
        if merges.is_empty() {
            return Err(VocabError::NoMergesUsable {
                total: raw_merges.len(),
            });
        }
        merges.sort_unstable_by_key(|m| (m.left, m.right));

        // Specials, longest first for longest-match scanning.
        let mut specials: Vec<u32> = (0..tokens.len() as u32)
            .filter(|&i| types[i as usize].is_special())
            .collect();
        specials.sort_unstable_by(|&a, &b| {
            tokens[b as usize]
                .len()
                .cmp(&tokens[a as usize].len())
                .then(a.cmp(&b))
        });

        Ok(Vocab {
            tokens,
            types,
            by_str,
            merges,
            specials,
            bos: g.get_u32("tokenizer.ggml.bos_token_id"),
            eos: g.get_u32("tokenizer.ggml.eos_token_id"),
            pad: g.get_u32("tokenizer.ggml.padding_token_id"),
            unk: g.get_u32("tokenizer.ggml.unknown_token_id"),
            add_bos: g.get_bool("tokenizer.ggml.add_bos_token").unwrap_or(false),
            skipped_merges,
        })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn n_merges(&self) -> usize {
        self.merges.len()
    }

    /// The token's string, in the byte-mapped alphabet.
    pub fn token(&self, id: u32) -> Option<&'a str> {
        self.tokens.get(id as usize).copied()
    }

    pub fn token_type(&self, id: u32) -> Option<TokenType> {
        self.types.get(id as usize).copied()
    }

    pub fn id_of(&self, s: &str) -> Option<u32> {
        self.by_str
            .binary_search_by(|&i| self.tokens[i as usize].cmp(s))
            .ok()
            .map(|k| self.by_str[k])
    }

    /// Special-token ids, longest string first.
    pub fn specials(&self) -> &[u32] {
        &self.specials
    }

    /// The merge rule for an adjacent pair, if one exists.
    #[inline]
    pub fn merge(&self, left: u32, right: u32) -> Option<&Merge> {
        self.merges
            .binary_search_by_key(&(left, right), |m| (m.left, m.right))
            .ok()
            .map(|i| &self.merges[i])
    }
}
