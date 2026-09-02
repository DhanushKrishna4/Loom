//! Qwen2's pre-tokenizer, hand-written.
//!
//! BPE merges may never cross a pre-token boundary, so this split decides the
//! final token IDs just as much as the merge table does. Qwen2's pattern (from
//! `tokenizer.json`, and matching llama.cpp's `LLAMA_VOCAB_PRE_TYPE_QWEN2`) is:
//!
//! ```text
//! (?i:'s|'t|'re|'ve|'m|'ll|'d)
//! | [^\r\n\p{L}\p{N}]?\p{L}+
//! | \p{N}
//! | ?[^\s\p{L}\p{N}]+[\r\n]*
//! | \s*[\r\n]+
//! | \s+(?!\S)
//! | \s+
//! ```
//!
//! Two details that are easy to get wrong:
//!
//! * **`\p{N}` matches a single digit**, not `\p{N}{1,3}`. GPT-4 and Llama 3 use
//!   the `{1,3}` form; Qwen2 does not, so "2024" is four tokens' worth of
//!   pre-tokens, not two.
//! * **`\s+(?!\S)` is what attaches a space to the following word.** Greedy `\s+`
//!   takes the whole whitespace run, the lookahead then fails if a non-space
//!   follows, and backtracking gives back exactly one character — which the next
//!   iteration picks up as the optional prefix of `[^\r\n\p{L}\p{N}]?\p{L}+`.
//!   That is why `" hello"` is one pre-token but `"  hello"` is `" "` + `" hello"`.
//!
//! Alternation is leftmost-first: at each position the first alternative that
//! matches wins, and each is greedy within itself.

use alloc::vec::Vec;
use core::ops::Range;

use super::{is_letter, is_number, is_whitespace};

#[inline]
fn is_crlf(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// True for `[^\s\p{L}\p{N}]` -- the "punctuation and symbols" class.
#[inline]
fn is_other(c: char) -> bool {
    !is_whitespace(c) && !is_letter(c) && !is_number(c)
}

/// Split `text` into pre-token byte ranges.
pub fn pretokenize(text: &str) -> Vec<Range<usize>> {
    // Char-indexed scan with byte offsets kept alongside, because the pattern
    // needs lookahead and backtracking that `char_indices` alone cannot express.
    let cs: Vec<(usize, char)> = text.char_indices().collect();
    let n = cs.len();
    let mut out = Vec::new();
    let mut i = 0usize;

    while i < n {
        let e = next_end(&cs, i);
        debug_assert!(e > i, "pre-tokenizer must always advance");
        let start = cs[i].0;
        let end = if e < n { cs[e].0 } else { text.len() };
        out.push(start..end);
        i = e;
    }
    out
}

/// Char index one past the end of the pre-token starting at `i`.
fn next_end(cs: &[(usize, char)], i: usize) -> usize {
    let n = cs.len();
    let ch = |k: usize| cs[k].1;

    // 1. Contractions. The (?i:) applies only to the ASCII letters.
    if ch(i) == '\'' && i + 1 < n {
        let c1 = ch(i + 1).to_ascii_lowercase();
        match c1 {
            's' | 't' | 'm' | 'd' => return i + 2,
            'r' | 'v' if i + 2 < n && ch(i + 2).eq_ignore_ascii_case(&'e') => return i + 3,
            'l' if i + 2 < n && ch(i + 2).eq_ignore_ascii_case(&'l') => return i + 3,
            _ => {}
        }
    }

    // 2. [^\r\n\p{L}\p{N}]? \p{L}+   -- optional prefix, then letters.
    //    The optional group is tried present-first, matching regex backtracking.
    if !is_crlf(ch(i))
        && !is_letter(ch(i))
        && !is_number(ch(i))
        && i + 1 < n
        && is_letter(ch(i + 1))
    {
        let mut j = i + 1;
        while j < n && is_letter(ch(j)) {
            j += 1;
        }
        return j;
    }
    if is_letter(ch(i)) {
        let mut j = i;
        while j < n && is_letter(ch(j)) {
            j += 1;
        }
        return j;
    }

    // 3. \p{N}  -- exactly one digit.
    if is_number(ch(i)) {
        return i + 1;
    }

    // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
    //    If the leading space is present, the body must start after it; the body
    //    class excludes whitespace, so there is no second branch to try.
    {
        let body = if ch(i) == ' ' { i + 1 } else { i };
        if body < n && is_other(ch(body)) {
            let mut j = body;
            while j < n && is_other(ch(j)) {
                j += 1;
            }
            while j < n && is_crlf(ch(j)) {
                j += 1;
            }
            return j;
        }
    }

    // Everything left starts with whitespace. Measure the run once.
    let mut ws_end = i;
    while ws_end < n && is_whitespace(ch(ws_end)) {
        ws_end += 1;
    }

    if ws_end > i {
        // 5. \s* [\r\n]+  -- ends at the last newline in the run, if any.
        let mut last_nl = None;
        for k in i..ws_end {
            if is_crlf(ch(k)) {
                last_nl = Some(k);
            }
        }
        if let Some(k) = last_nl {
            return k + 1;
        }

        // 6. \s+(?!\S)  -- the whole run at end of input, otherwise all but the
        //    last character, which the next iteration hands to a word.
        if ws_end == n {
            return ws_end;
        }
        if ws_end - 1 > i {
            return ws_end - 1;
        }

        // 7. \s+
        return ws_end;
    }

    // Unreachable: letters, digits, whitespace and everything else are covered
    // above. Advance anyway so a surprise cannot become an infinite loop.
    debug_assert!(false, "pre-tokenizer fell through on {:?}", ch(i));
    i + 1
}
