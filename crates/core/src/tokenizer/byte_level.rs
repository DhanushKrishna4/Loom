//! GPT-2's byte-to-unicode remapping.
//!
//! Byte-level BPE has a bootstrapping problem: the merge table is expressed over
//! *characters*, but the input is arbitrary bytes, and many bytes (control
//! characters, space, 0x7F..0xA0) either have no printable form or would be
//! mangled by the whitespace handling in the pre-tokenizer.
//!
//! GPT-2's fix is a bijection from all 256 byte values onto 256 printable,
//! non-whitespace codepoints. Bytes that are already printable ASCII or Latin-1
//! map to themselves; the remaining 68 are pushed up into U+0100..U+0143.
//!
//! This is why a GGUF vocab is full of tokens like `Ġthe` — that is not a typo,
//! it is byte 0x20 (space) rendered as U+0120. Every token string in the vocab is
//! in this mapped alphabet, so text must be mapped *into* it before any lookup
//! and mapped back *out* of it after decoding.

/// The 68 bytes with no printable Latin-1 form, in ascending order:
/// 0x00..=0x20, 0x7F..=0xA0, and 0xAD (soft hyphen).
const fn is_printable_passthrough(b: u8) -> bool {
    matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF)
}

const fn build_byte_to_unicode() -> [u32; 256] {
    let mut map = [0u32; 256];
    let mut next = 0u32;
    let mut b = 0usize;
    while b < 256 {
        if is_printable_passthrough(b as u8) {
            map[b] = b as u32;
        } else {
            // 0x100 onwards, assigned in ascending byte order.
            map[b] = 0x100 + next;
            next += 1;
        }
        b += 1;
    }
    map
}

/// Byte -> the codepoint that stands for it in the vocab.
pub static BYTE_TO_UNICODE: [u32; 256] = build_byte_to_unicode();

/// Highest codepoint the mapping produces: 0x100 + 68 - 1.
const MAX_MAPPED: u32 = 0x143;

const fn build_unicode_to_byte() -> [i16; (MAX_MAPPED + 1) as usize] {
    let mut rev = [-1i16; (MAX_MAPPED + 1) as usize];
    let mut b = 0usize;
    while b < 256 {
        rev[BYTE_TO_UNICODE_CONST[b] as usize] = b as i16;
        b += 1;
    }
    rev
}

// `build_unicode_to_byte` cannot read the static above in const context, so the
// table is built twice from the same const fn. They are identical by construction.
const BYTE_TO_UNICODE_CONST: [u32; 256] = build_byte_to_unicode();

/// Codepoint -> byte, or -1 if that codepoint is not part of the mapping.
pub static UNICODE_TO_BYTE: [i16; (MAX_MAPPED + 1) as usize] = build_unicode_to_byte();

/// Map one byte to its stand-in character.
#[inline]
pub fn byte_to_char(b: u8) -> char {
    // Every entry is a valid scalar value well below the surrogate range.
    char::from_u32(BYTE_TO_UNICODE[b as usize]).expect("mapping is total and valid")
}

/// Map a stand-in character back to its byte, if it is one.
#[inline]
pub fn char_to_byte(c: char) -> Option<u8> {
    let cp = c as u32;
    if cp > MAX_MAPPED {
        return None;
    }
    match UNICODE_TO_BYTE[cp as usize] {
        -1 => None,
        b => Some(b as u8),
    }
}

/// Encode raw bytes into the mapped alphabet, appending to `out`.
pub fn encode_into(bytes: &[u8], out: &mut alloc::string::String) {
    for &b in bytes {
        out.push(byte_to_char(b));
    }
}

/// Decode mapped characters back to raw bytes, appending to `out`.
///
/// Returns `false` if any character was not part of the mapping, which means the
/// vocab contained something that did not come from this alphabet.
pub fn decode_into(s: &str, out: &mut alloc::vec::Vec<u8>) -> bool {
    let mut ok = true;
    for c in s.chars() {
        match char_to_byte(c) {
            Some(b) => out.push(b),
            None => ok = false,
        }
    }
    ok
}
