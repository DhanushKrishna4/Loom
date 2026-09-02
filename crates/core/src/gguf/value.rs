//! Typed metadata values.
//!
//! # Why these borrow instead of copying
//!
//! Strings are `&'a str` pointing straight into the mapped/downloaded file
//! buffer.  Qwen2.5's vocab is 151936 tokens; storing them as owned `String`s
//! would cost ~2 MB of copies plus 151936 allocations at load time for no
//! benefit, because the file buffer outlives the parse anyway.
//!
//! Numeric arrays *are* materialised into `Vec`s.  They cannot be borrowed as
//! typed slices because nothing guarantees the file offset is 4- or 8-byte
//! aligned, and a misaligned `&[f32]` is UB.  The cost is bounded: for
//! Qwen2.5-0.5B the eager metadata (token strings + scores + token types +
//! merges) comes to roughly 6 MB against a ~400 MB model, i.e. ~1.5%.
//!
//! If that ever stops being acceptable the fix is a lazy `Array` that keeps the
//! byte range and decodes on index; the accessor API below would not change.

use alloc::vec::Vec;

use super::error::GgufError;
use super::reader::Reader;

/// The metadata value type tags, as written in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MetaType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl MetaType {
    #[rustfmt::skip] // the alignment is the point: this is a lookup table
    pub fn from_u32(v: u32) -> Result<Self, GgufError> {
        use MetaType::*;
        Ok(match v {
            0 => U8, 1 => I8, 2 => U16, 3 => I16, 4 => U32, 5 => I32,
            6 => F32, 7 => Bool, 8 => String, 9 => Array,
            10 => U64, 11 => I64, 12 => F64,
            other => return Err(GgufError::UnknownValueType(other)),
        })
    }

    /// Fixed byte width, or `None` for the variable-length types.  Used to bound
    /// array allocations before reserving.
    const fn fixed_size(self) -> Option<usize> {
        use MetaType::*;
        Some(match self {
            U8 | I8 | Bool => 1,
            U16 | I16 => 2,
            U32 | I32 | F32 => 4,
            U64 | I64 | F64 => 8,
            String | Array => return None,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value<'a> {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(&'a str),
    Array(Array<'a>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Array<'a> {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Bool(Vec<bool>),
    Str(Vec<&'a str>),
    /// Array of arrays.  Legal per the spec, unused by any model we target.
    Nested(Vec<Value<'a>>),
}

/// Deepest array-of-array nesting we will follow.  Real files use 0; the limit
/// exists so a crafted file cannot recurse us into a stack overflow.
const MAX_ARRAY_DEPTH: u32 = 4;

impl<'a> Value<'a> {
    pub(crate) fn read(r: &mut Reader<'a>, ty: MetaType, depth: u32) -> Result<Self, GgufError> {
        Ok(match ty {
            MetaType::U8 => Value::U8(r.read_u8()?),
            MetaType::I8 => Value::I8(r.read_i8()?),
            MetaType::U16 => Value::U16(r.read_u16()?),
            MetaType::I16 => Value::I16(r.read_i16()?),
            MetaType::U32 => Value::U32(r.read_u32()?),
            MetaType::I32 => Value::I32(r.read_i32()?),
            MetaType::U64 => Value::U64(r.read_u64()?),
            MetaType::I64 => Value::I64(r.read_i64()?),
            MetaType::F32 => Value::F32(r.read_f32()?),
            MetaType::F64 => Value::F64(r.read_f64()?),
            MetaType::Bool => Value::Bool(r.read_bool()?),
            MetaType::String => Value::Str(r.read_str()?),
            MetaType::Array => Value::Array(Array::read(r, depth)?),
        })
    }

    #[rustfmt::skip] // the alignment is the point: this is a lookup table
    pub const fn type_name(&self) -> &'static str {
        match self {
            Value::U8(_) => "u8", Value::I8(_) => "i8",
            Value::U16(_) => "u16", Value::I16(_) => "i16",
            Value::U32(_) => "u32", Value::I32(_) => "i32",
            Value::U64(_) => "u64", Value::I64(_) => "i64",
            Value::F32(_) => "f32", Value::F64(_) => "f64",
            Value::Bool(_) => "bool", Value::Str(_) => "string",
            Value::Array(_) => "array",
        }
    }

    /// Coerce any non-negative integer variant to `u64`.
    ///
    /// Writers are inconsistent about integer widths -- `general.alignment` is
    /// u32 in llama.cpp's writer but i32 in some converters -- so callers should
    /// never match on an exact width.
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64,
            Value::U16(v) => v as u64,
            Value::U32(v) => v as u64,
            Value::U64(v) => v,
            Value::I8(v) => u64::try_from(v).ok()?,
            Value::I16(v) => u64::try_from(v).ok()?,
            Value::I32(v) => u64::try_from(v).ok()?,
            Value::I64(v) => u64::try_from(v).ok()?,
            Value::Bool(v) => v as u64,
            _ => return None,
        })
    }

    pub fn as_i64(&self) -> Option<i64> {
        Some(match *self {
            Value::U8(v) => v as i64,
            Value::U16(v) => v as i64,
            Value::U32(v) => v as i64,
            Value::U64(v) => i64::try_from(v).ok()?,
            Value::I8(v) => v as i64,
            Value::I16(v) => v as i64,
            Value::I32(v) => v as i64,
            Value::I64(v) => v,
            _ => return None,
        })
    }

    /// Floats, plus integers widened to float (some writers store eps as f64).
    pub fn as_f32(&self) -> Option<f32> {
        Some(match *self {
            Value::F32(v) => v,
            Value::F64(v) => v as f32,
            _ => self.as_i64()? as f32,
        })
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(v) => Some(v),
            Value::U8(v) => Some(v != 0),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&'a str> {
        match *self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Array<'a>> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

impl<'a> Array<'a> {
    fn read(r: &mut Reader<'a>, depth: u32) -> Result<Self, GgufError> {
        if depth >= MAX_ARRAY_DEPTH {
            return Err(GgufError::ArrayNestingTooDeep);
        }
        let elem_ty = MetaType::from_u32(r.read_u32()?)?;

        // Bound the allocation by the bytes actually left in the file.  For
        // variable-width elements we can still say every element costs >= 1 byte
        // (a string is at least its 8-byte length prefix, but 1 is a safe floor).
        let n = match elem_ty.fixed_size() {
            Some(sz) => r.read_len_scaled("array element", sz)?,
            None => r.read_len("array element")?,
        };

        macro_rules! collect {
            ($variant:ident, $read:ident) => {{
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(r.$read()?);
                }
                Array::$variant(v)
            }};
        }

        Ok(match elem_ty {
            MetaType::U8 => collect!(U8, read_u8),
            MetaType::I8 => collect!(I8, read_i8),
            MetaType::U16 => collect!(U16, read_u16),
            MetaType::I16 => collect!(I16, read_i16),
            MetaType::U32 => collect!(U32, read_u32),
            MetaType::I32 => collect!(I32, read_i32),
            MetaType::U64 => collect!(U64, read_u64),
            MetaType::I64 => collect!(I64, read_i64),
            MetaType::F32 => collect!(F32, read_f32),
            MetaType::F64 => collect!(F64, read_f64),
            MetaType::Bool => collect!(Bool, read_bool),
            MetaType::String => collect!(Str, read_str),
            MetaType::Array => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(Value::Array(Array::read(r, depth + 1)?));
                }
                Array::Nested(v)
            }
        })
    }

    #[rustfmt::skip] // the alignment is the point: this is a lookup table
    pub fn len(&self) -> usize {
        match self {
            Array::U8(v) => v.len(), Array::I8(v) => v.len(),
            Array::U16(v) => v.len(), Array::I16(v) => v.len(),
            Array::U32(v) => v.len(), Array::I32(v) => v.len(),
            Array::U64(v) => v.len(), Array::I64(v) => v.len(),
            Array::F32(v) => v.len(), Array::F64(v) => v.len(),
            Array::Bool(v) => v.len(), Array::Str(v) => v.len(),
            Array::Nested(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[rustfmt::skip] // the alignment is the point: this is a lookup table
    pub const fn elem_type_name(&self) -> &'static str {
        match self {
            Array::U8(_) => "u8", Array::I8(_) => "i8",
            Array::U16(_) => "u16", Array::I16(_) => "i16",
            Array::U32(_) => "u32", Array::I32(_) => "i32",
            Array::U64(_) => "u64", Array::I64(_) => "i64",
            Array::F32(_) => "f32", Array::F64(_) => "f64",
            Array::Bool(_) => "bool", Array::Str(_) => "string",
            Array::Nested(_) => "array",
        }
    }

    pub fn as_str_slice(&self) -> Option<&[&'a str]> {
        match self {
            Array::Str(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            Array::F32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_i32_slice(&self) -> Option<&[i32]> {
        match self {
            Array::I32(v) => Some(v),
            _ => None,
        }
    }
}
