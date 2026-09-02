//! Parser tests.
//!
//! Everything here builds its own GGUF bytes with [`super::builder`], so the
//! suite runs with no model file present. The counterpart -- checking that we
//! agree with a *real* Qwen2.5-0.5B file -- is `nano-infer gguf-dump`, which is
//! why that CLI subcommand exists.

use super::builder::{dummy_vocab, qwen2_5_0_5b, Builder};
use super::*;
use crate::quant;

fn strs(v: &[String]) -> Vec<&str> {
    v.iter().map(|s| s.as_str()).collect()
}

// ---------------------------------------------------------------- header ----

#[test]
fn parses_an_empty_file() {
    let bytes = Builder::new().build();
    let g = Gguf::parse(&bytes).unwrap();
    assert_eq!(g.version, 3);
    assert_eq!(g.alignment, DEFAULT_ALIGNMENT);
    assert!(g.tensors.is_empty());
    assert!(g.metadata.is_empty());
    // 4 magic + 4 version + 8 + 8 = 24, padded up to 32.
    assert_eq!(g.tensor_data_offset, 32);
}

#[test]
fn accepts_v2_and_v3_rejects_others() {
    for v in [2u32, 3] {
        let bytes = Builder::new().version(v).build();
        assert_eq!(Gguf::parse(&bytes).unwrap().version, v);
    }
    for v in [0u32, 1, 4, 999] {
        let bytes = Builder::new().version(v).build();
        assert_eq!(
            Gguf::parse(&bytes).unwrap_err(),
            GgufError::UnsupportedVersion(v)
        );
    }
}

#[test]
fn rejects_bad_magic() {
    let bytes = Builder::new().magic(*b"GGML").build();
    assert_eq!(
        Gguf::parse(&bytes).unwrap_err(),
        GgufError::BadMagic(*b"GGML")
    );
}

#[test]
fn rejects_truncation_at_every_prefix() {
    let vocab = dummy_vocab(4);
    let bytes = qwen2_5_0_5b(&strs(&vocab))
        .tensor("blk.0.attn_q.weight", &[8, 8], GgmlType::F32, &[0u8; 256])
        .build();

    // Every strict prefix must fail cleanly -- never panic, never succeed.
    for cut in 0..bytes.len() {
        match Gguf::parse(&bytes[..cut]) {
            Err(_) => {}
            Ok(_) => panic!("prefix of length {cut} parsed as a complete file"),
        }
    }
    assert!(Gguf::parse(&bytes).is_ok());
}

// -------------------------------------------------------------- metadata ----

#[test]
fn every_value_type_round_trips() {
    let bytes = Builder::new()
        .u8("a.u8", 200)
        .i16("a.i16", -300)
        .u32("a.u32", 70_000)
        .i32("a.i32", -70_000)
        .u64("a.u64", 1 << 40)
        .f32("a.f32", 1.5)
        .f64("a.f64", -2.25)
        .bool("a.true", true)
        .bool("a.false", false)
        .string("a.str", "hello \u{1f600}")
        .str_array("a.strs", &["x", "yy", ""])
        .f32_array("a.f32s", &[1.0, -0.5, 3.25])
        .i32_array("a.i32s", &[-1, 0, 7])
        .build();
    let g = Gguf::parse(&bytes).unwrap();

    assert_eq!(g.get("a.u8"), Some(&Value::U8(200)));
    assert_eq!(g.get("a.i16"), Some(&Value::I16(-300)));
    assert_eq!(g.get("a.u32"), Some(&Value::U32(70_000)));
    assert_eq!(g.get("a.i32"), Some(&Value::I32(-70_000)));
    assert_eq!(g.get("a.u64"), Some(&Value::U64(1 << 40)));
    assert_eq!(g.get_f32("a.f32"), Some(1.5));
    assert_eq!(g.get("a.f64"), Some(&Value::F64(-2.25)));
    assert_eq!(g.get_bool("a.true"), Some(true));
    assert_eq!(g.get_bool("a.false"), Some(false));
    assert_eq!(g.get_str("a.str"), Some("hello \u{1f600}"));

    assert_eq!(
        g.get_array("a.strs").unwrap().as_str_slice(),
        Some(&["x", "yy", ""][..])
    );
    assert_eq!(
        g.get_array("a.f32s").unwrap().as_f32_slice(),
        Some(&[1.0, -0.5, 3.25][..])
    );
    assert_eq!(
        g.get_array("a.i32s").unwrap().as_i32_slice(),
        Some(&[-1, 0, 7][..])
    );
}

#[test]
fn integer_getters_ignore_the_stored_width() {
    // Writers disagree about whether these are u32 or i32; callers must not care.
    let bytes = Builder::new()
        .i32("x.a", 24)
        .u8("x.b", 7)
        .u64("x.c", 32768)
        .i32("x.neg", -1)
        .build();
    let g = Gguf::parse(&bytes).unwrap();
    assert_eq!(g.get_usize("x.a"), Some(24));
    assert_eq!(g.get_usize("x.b"), Some(7));
    assert_eq!(g.get_usize("x.c"), Some(32768));
    // Negative values are not silently wrapped into a huge unsigned.
    assert_eq!(g.get_u64("x.neg"), None);
    assert_eq!(g.get("x.neg").unwrap().as_i64(), Some(-1));
}

#[test]
fn rejects_invalid_utf8_in_strings() {
    let bytes = Builder::new().raw_string("bad", &[0xff, 0xfe]).build();
    assert!(matches!(
        Gguf::parse(&bytes),
        Err(GgufError::InvalidUtf8 { .. })
    ));
}

#[test]
fn rejects_array_length_larger_than_the_file() {
    // The allocation guard: a corrupt u64 must not become a 16 EiB reserve.
    let bytes = Builder::new()
        .lying_array("boom", MetaType::F32, u64::MAX / 8)
        .build();
    assert!(matches!(
        Gguf::parse(&bytes),
        Err(GgufError::ImplausibleCount { .. })
    ));
}

#[test]
fn rejects_implausible_counts_in_the_header() {
    let mut bytes = Builder::new().build();
    bytes[8..16].copy_from_slice(&u64::MAX.to_le_bytes()); // tensor_count
    assert!(matches!(
        Gguf::parse(&bytes),
        Err(GgufError::ImplausibleCount { what: "tensor", .. })
    ));
}

#[test]
fn rejects_unknown_metadata_type_tag() {
    let mut bytes = Builder::new().u32("k", 1).build();
    // The type tag sits right after the 8-byte length and 1-byte key.
    let tag_at = 24 + 8 + 1;
    bytes[tag_at..tag_at + 4].copy_from_slice(&99u32.to_le_bytes());
    assert_eq!(
        Gguf::parse(&bytes).unwrap_err(),
        GgufError::UnknownValueType(99)
    );
}

// --------------------------------------------------------------- tensors ----

#[test]
fn tensor_data_is_borrowed_not_copied() {
    let payload: Vec<u8> = (0..=255u8).collect();
    let bytes = Builder::new()
        .tensor("w", &[64], GgmlType::F32, &payload)
        .build();
    let g = Gguf::parse(&bytes).unwrap();

    let t = g.find_tensor("w").unwrap();
    assert_eq!(t.ggml_type, GgmlType::F32);
    assert_eq!(t.n_elements(), 64);
    assert_eq!(t.byte_size, 256);

    let data = g.tensor_data(t);
    assert_eq!(data, &payload[..]);

    // The returned slice must point *into* the caller's buffer. If this ever
    // fails we have started copying 400 MB of weights somewhere.
    let file_start = bytes.as_ptr() as usize;
    let slice_start = data.as_ptr() as usize;
    assert!(slice_start >= file_start);
    assert!(slice_start + data.len() <= file_start + bytes.len());
}

#[test]
fn dims_are_stored_in_ggml_order() {
    // ffn_gate for Qwen2.5-0.5B: torch shape (4864, 896) = (out, in).
    // ggml stores the contiguous axis first, so dims = [896, 4864].
    let n = 896 * 4864;
    let bytes = Builder::new()
        .tensor_info_only("blk.0.ffn_gate.weight", &[896, 4864], GgmlType::Q4_K, 0)
        .build();
    let g = Gguf::parse(&bytes[..]);
    // No data written, so this must fail bounds -- but the shape maths is what
    // we care about, so re-check it via a sized file below.
    assert!(matches!(g, Err(GgufError::TensorOutOfBounds { .. })));

    let data = vec![0u8; (n / 256) * GgmlType::Q4_K.block_bytes()];
    let bytes = Builder::new()
        .tensor("blk.0.ffn_gate.weight", &[896, 4864], GgmlType::Q4_K, &data)
        .build();
    let g = Gguf::parse(&bytes).unwrap();
    let t = g.find_tensor("blk.0.ffn_gate.weight").unwrap();

    assert_eq!(t.dims[..2], [896, 4864]);
    assert_eq!(t.shape_row_major(), vec![4864, 896]);
    assert_eq!(t.row_len(), 896);
    assert_eq!(t.n_rows(), 4864);
    assert_eq!(t.n_elements(), n as u64);
    assert_eq!(t.byte_size, (n as u64 / 256) * 144);
}

#[test]
fn honours_a_custom_alignment() {
    let bytes = Builder::new()
        .alignment(64)
        .u32("general.alignment", 64)
        .tensor("a", &[16], GgmlType::F32, &[1u8; 64])
        .tensor("b", &[16], GgmlType::F32, &[2u8; 64])
        .build();
    let g = Gguf::parse(&bytes).unwrap();
    assert_eq!(g.alignment, 64);
    assert_eq!(g.tensor_data_offset % 64, 0);
    assert_eq!(g.tensor_data(g.find_tensor("a").unwrap()), &[1u8; 64][..]);
    assert_eq!(g.tensor_data(g.find_tensor("b").unwrap()), &[2u8; 64][..]);
}

#[test]
fn rejects_non_power_of_two_alignment() {
    let bytes = Builder::new().u32("general.alignment", 24).build();
    assert_eq!(
        Gguf::parse(&bytes).unwrap_err(),
        GgufError::BadAlignment(24)
    );
    let bytes = Builder::new().u32("general.alignment", 0).build();
    assert_eq!(Gguf::parse(&bytes).unwrap_err(), GgufError::BadAlignment(0));
}

#[test]
fn rejects_misaligned_tensor_offset() {
    let bytes = Builder::new()
        .tensor_info_only("w", &[8], GgmlType::F32, 4) // 4 is not a multiple of 32
        .build();
    assert!(matches!(
        Gguf::parse(&bytes),
        Err(GgufError::MisalignedTensor { .. })
    ));
}

#[test]
fn rejects_tensor_past_the_end_of_the_data_section() {
    let bytes = Builder::new()
        .tensor_info_only("w", &[1024], GgmlType::F32, 0)
        .build();
    assert!(matches!(
        Gguf::parse(&bytes),
        Err(GgufError::TensorOutOfBounds { .. })
    ));
}

#[test]
fn rejects_element_count_that_is_not_a_whole_number_of_blocks() {
    // 100 elements cannot be tiled by 256-element Q4_K super-blocks.
    let bytes = Builder::new()
        .tensor_info_only("w", &[100], GgmlType::Q4_K, 0)
        .build();
    assert!(matches!(
        Gguf::parse(&bytes),
        Err(GgufError::NotBlockAligned {
            elements: 100,
            block: 256,
            ..
        })
    ));
}

#[test]
fn rejects_unknown_ggml_type_rather_than_guessing() {
    let bytes = Builder::new()
        .raw_tensor_info("w", 1, &[32], 1234, 0)
        .build();
    assert_eq!(
        Gguf::parse(&bytes).unwrap_err(),
        GgufError::UnknownGgmlType(1234)
    );
}

#[test]
fn rejects_more_than_four_dims() {
    let bytes = Builder::new()
        .raw_tensor_info("w", 5, &[2, 2, 2, 2, 2], GgmlType::F32 as u32, 0)
        .build();
    assert!(matches!(
        Gguf::parse(&bytes),
        Err(GgufError::TooManyDims { n_dims: 5, .. })
    ));
}

// ---------------------------------------------------------------- config ----

#[test]
fn extracts_qwen2_5_0_5b_config() {
    // These are the published hyper-parameters for Qwen2.5-0.5B-Instruct. If a
    // real GGUF ever disagrees with this test, the real GGUF wins and this test
    // gets updated -- but a silent drift here would poison every later step.
    let vocab = dummy_vocab(151_936);
    let bytes = qwen2_5_0_5b(&strs(&vocab)).build();
    let cfg = Gguf::parse(&bytes).unwrap().config().unwrap();

    assert_eq!(cfg.architecture, "qwen2");
    assert_eq!(cfg.name.as_deref(), Some("Qwen2.5-0.5B-Instruct"));
    assert_eq!(cfg.block_count, 24);
    assert_eq!(cfg.embedding_length, 896);
    assert_eq!(cfg.feed_forward_length, 4864);
    assert_eq!(cfg.head_count, 14);
    assert_eq!(cfg.head_count_kv, 2);
    assert_eq!(cfg.head_dim, 64); // 896 / 14
    assert_eq!(cfg.context_length, 32768);
    assert_eq!(cfg.rms_norm_eps, 1e-6);
    assert_eq!(cfg.rope_freq_base, 1_000_000.0);
    // Absent from the real file: every head dimension gets rotated.
    assert_eq!(cfg.rope_dimension_count, None);
    assert_eq!(cfg.vocab_size, 151_936);
    assert_eq!(cfg.eos_token_id, Some(151_645)); // <|im_end|>
    assert_eq!(cfg.pad_token_id, Some(151_643)); // <|endoftext|>
    assert_eq!(cfg.bos_token_id, Some(151_643)); // <|endoftext|>
    assert_eq!(cfg.add_bos_token, Some(false));
    assert_eq!(cfg.tokenizer_model.as_deref(), Some("gpt2"));

    // Derived quantities the attention code will lean on.
    assert_eq!(cfg.kv_group_size(), 7); // 14 query heads share 2 KV heads
    assert_eq!(cfg.q_dim(), 896);
    assert_eq!(cfg.kv_dim(), 128); // 2 * 64, not 896 -- this is GQA working
                                   // Full-context f32 KV cache: 2 * 24 * 2 * 32768 * 64 * 4 B = 768 MiB.
    assert_eq!(cfg.kv_cache_bytes(32768), 805_306_368);
    // ...which is why the engine will cap max_seq well below context_length.
    assert_eq!(cfg.kv_cache_bytes(2048), 50_331_648); // 48 MiB at 2k
}

#[test]
fn config_defaults_match_the_gguf_spec() {
    let bytes = Builder::new()
        .string("general.architecture", "llama")
        .u32("llama.block_count", 2)
        .u32("llama.context_length", 128)
        .u32("llama.embedding_length", 64)
        .u32("llama.feed_forward_length", 128)
        .u32("llama.attention.head_count", 4)
        // head_count_kv, rms eps and rope base all omitted on purpose.
        .str_array("tokenizer.ggml.tokens", &["a", "b"])
        .build();
    let cfg = Gguf::parse(&bytes).unwrap().config().unwrap();
    assert_eq!(cfg.head_count_kv, 4, "absent head_count_kv means plain MHA");
    assert_eq!(cfg.kv_group_size(), 1);
    assert_eq!(cfg.rms_norm_eps, 1e-5);
    assert_eq!(cfg.rope_freq_base, 10_000.0);
    assert_eq!(cfg.head_dim, 16);
    assert_eq!(cfg.vocab_size, 2);
}

#[test]
fn config_rejects_gqa_grouping_that_cannot_work() {
    // 14 query heads over 4 KV heads is not an integer grouping; letting this
    // through produces plausible-looking garbage instead of a crash.
    let bytes = Builder::new()
        .string("general.architecture", "qwen2")
        .u32("qwen2.block_count", 2)
        .u32("qwen2.context_length", 128)
        .u32("qwen2.embedding_length", 896)
        .u32("qwen2.feed_forward_length", 128)
        .u32("qwen2.attention.head_count", 14)
        .u32("qwen2.attention.head_count_kv", 4)
        .str_array("tokenizer.ggml.tokens", &["a"])
        .build();
    assert!(matches!(
        Gguf::parse(&bytes).unwrap().config(),
        Err(GgufError::InconsistentConfig(_))
    ));
}

#[test]
fn config_reports_the_missing_key_by_name() {
    let bytes = Builder::new()
        .string("general.architecture", "qwen2")
        .u32("qwen2.embedding_length", 896)
        .u32("qwen2.attention.head_count", 14)
        .str_array("tokenizer.ggml.tokens", &["a"])
        .build();
    let err = Gguf::parse(&bytes).unwrap().config().unwrap_err();
    assert_eq!(err, GgufError::MissingKey("qwen2.block_count".into()));
}

// ------------------------------------------------------- block geometry ----

#[test]
fn block_geometry_matches_ggml_common_h() {
    // Transcribed from the block_* struct definitions. A wrong entry here shifts
    // every tensor offset after the first tensor of that type, so it is worth
    // pinning explicitly rather than trusting the table to stay right.
    let expect: &[(GgmlType, usize, usize)] = &[
        (GgmlType::F32, 1, 4),
        (GgmlType::F16, 1, 2),
        (GgmlType::Q4_0, 32, 18),
        (GgmlType::Q4_1, 32, 20),
        (GgmlType::Q5_0, 32, 22),
        (GgmlType::Q5_1, 32, 24),
        (GgmlType::Q8_0, 32, 34),
        (GgmlType::Q2_K, 256, 84),
        (GgmlType::Q3_K, 256, 110),
        (GgmlType::Q4_K, 256, 144),
        (GgmlType::Q5_K, 256, 176),
        (GgmlType::Q6_K, 256, 210),
        (GgmlType::Q8_K, 256, 292),
    ];
    for &(ty, elems, bytes) in expect {
        assert_eq!(ty.block_elems(), elems, "{ty} block_elems");
        assert_eq!(ty.block_bytes(), bytes, "{ty} block_bytes");
        assert_eq!(GgmlType::from_u32(ty as u32), Ok(ty));
    }

    // Compression ratios worth sanity-checking: Q4_K should be ~4.5 bits/weight.
    let bits_per_weight = |t: GgmlType| t.block_bytes() as f64 * 8.0 / t.block_elems() as f64;
    assert!((bits_per_weight(GgmlType::Q4_K) - 4.5).abs() < 1e-9);
    assert!((bits_per_weight(GgmlType::Q6_K) - 6.5625).abs() < 1e-9);
    assert!((bits_per_weight(GgmlType::Q8_0) - 8.5).abs() < 1e-9);
    assert!((bits_per_weight(GgmlType::Q4_0) - 4.5).abs() < 1e-9);
}

#[test]
fn bytes_for_rejects_partial_blocks() {
    assert_eq!(GgmlType::Q4_K.bytes_for(256), Some(144));
    assert_eq!(GgmlType::Q4_K.bytes_for(512), Some(288));
    assert_eq!(GgmlType::Q4_K.bytes_for(255), None);
    assert_eq!(GgmlType::F32.bytes_for(7), Some(28));
}

// ------------------------------------------- end-to-end parse + dequant ----

#[test]
fn parses_a_file_and_dequantises_a_tensor_out_of_it() {
    // The whole point of step 1+2 in one test: bytes in, floats out, no copies
    // of the weight tensor along the way.
    let mut block = [0u8; 34];
    block[0..2].copy_from_slice(&quant::f32_to_f16(0.5).to_le_bytes());
    for (i, b) in block[2..].iter_mut().enumerate() {
        *b = (i as i8 - 64) as u8;
    }

    let bytes = Builder::new()
        .string("general.architecture", "test")
        .tensor("w", &[32], GgmlType::Q8_0, &block)
        .build();
    let g = Gguf::parse(&bytes).unwrap();
    let t = g.find_tensor("w").unwrap();

    let mut out = vec![0f32; t.n_elements() as usize];
    quant::dequantize_row(t.ggml_type, g.tensor_data(t), &mut out).unwrap();

    for (i, v) in out.iter().enumerate() {
        assert_eq!(*v, 0.5 * (i as f32 - 64.0), "element {i}");
    }
}
