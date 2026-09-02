//! A bounds-checked little-endian cursor over the raw file bytes.
//!
//! GGUF is defined little-endian.  A big-endian variant exists (produced by
//! `convert_hf_to_gguf.py --bigendian` for s390x) but it is vanishingly rare and
//! supporting it would mean threading an endianness parameter through every read
//! for no benefit here, so every read below is unconditionally LE.
//!
//! Two rules this file exists to enforce:
//!   1. No read ever indexes past the buffer -- corrupt files return `Err`.
//!   2. File-supplied `u64` lengths are compared against `remaining()` *before*
//!      being narrowed to `usize`.  On wasm32 `usize` is 32 bits, so casting
//!      first would silently truncate a hostile length into a plausible one.

use super::error::GgufError;

pub(crate) struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

macro_rules! read_le {
    ($name:ident, $t:ty) => {
        #[inline]
        pub(crate) fn $name(&mut self) -> Result<$t, GgufError> {
            const N: usize = core::mem::size_of::<$t>();
            let bytes = self.take(N)?;
            let mut a = [0u8; N];
            a.copy_from_slice(bytes);
            Ok(<$t>::from_le_bytes(a))
        }
    };
}

impl<'a> Reader<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    #[inline]
    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    #[inline]
    pub(crate) fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    #[inline]
    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        if n > self.remaining() {
            return Err(GgufError::UnexpectedEof {
                at: self.pos,
                needed: n,
                remaining: self.remaining(),
            });
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    read_le!(read_u8, u8);
    read_le!(read_i8, i8);
    read_le!(read_u16, u16);
    read_le!(read_i16, i16);
    read_le!(read_u32, u32);
    read_le!(read_i32, i32);
    read_le!(read_u64, u64);
    read_le!(read_i64, i64);
    read_le!(read_f32, f32);
    read_le!(read_f64, f64);

    /// GGUF bools are one byte; anything non-zero is true (llama.cpp writes 0/1).
    pub(crate) fn read_bool(&mut self) -> Result<bool, GgufError> {
        Ok(self.read_u8()? != 0)
    }

    /// A `u64` length followed by that many UTF-8 bytes.  Borrowed, never copied.
    pub(crate) fn read_str(&mut self) -> Result<&'a str, GgufError> {
        let len = self.read_len("string length")?;
        let at = self.pos;
        let bytes = self.take(len)?;
        core::str::from_utf8(bytes).map_err(|_| GgufError::InvalidUtf8 { at })
    }

    /// Read a `u64` count and reject it if it could not possibly fit in the rest
    /// of the file.  This is the allocation guard: every `Vec::with_capacity` in
    /// the parser is fed by a value that has passed through here.
    pub(crate) fn read_len(&mut self, what: &'static str) -> Result<usize, GgufError> {
        let n = self.read_u64()?;
        if n > self.remaining() as u64 {
            return Err(GgufError::ImplausibleCount {
                what,
                count: n,
                remaining: self.remaining(),
            });
        }
        Ok(n as usize)
    }

    /// As [`read_len`], but for arrays whose elements are `elem_size` bytes each,
    /// so we can bound `count * elem_size` rather than just `count`.
    pub(crate) fn read_len_scaled(
        &mut self,
        what: &'static str,
        elem_size: usize,
    ) -> Result<usize, GgufError> {
        let n = self.read_u64()?;
        let need = n.checked_mul(elem_size as u64);
        match need {
            Some(need) if need <= self.remaining() as u64 => Ok(n as usize),
            _ => Err(GgufError::ImplausibleCount {
                what,
                count: n,
                remaining: self.remaining(),
            }),
        }
    }
}
