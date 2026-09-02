//! Qwen2's chat format.
//!
//! ```text
//! <|im_start|>system
//! {content}<|im_end|>
//! <|im_start|>user
//! {content}<|im_end|>
//! <|im_start|>assistant
//! ```
//!
//! Note the trailing `<|im_start|>assistant\n` with no closing tag: that is the
//! generation prompt, and leaving it off is why a model sometimes answers by
//! writing the *user's* next turn.
//!
//! Qwen2.5 injects a default system message when none is supplied. That is not
//! cosmetic -- the model was tuned with it present, and omitting it measurably
//! changes behaviour -- so it is reproduced here rather than left to the caller.
//!
//! The rendered string must be encoded with `parse_special: true` so the
//! `<|im_*|>` markers become single tokens. Untrusted message *content* should
//! not be: see [`super::EncodeOptions::parse_special`].

use alloc::string::String;

/// Qwen2.5's default system prompt, verbatim from its chat template.
pub const DEFAULT_SYSTEM_PROMPT: &str =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatMessage<'m> {
    pub role: &'m str,
    pub content: &'m str,
}

impl<'m> ChatMessage<'m> {
    pub fn system(content: &'m str) -> Self {
        ChatMessage {
            role: "system",
            content,
        }
    }
    pub fn user(content: &'m str) -> Self {
        ChatMessage {
            role: "user",
            content,
        }
    }
    pub fn assistant(content: &'m str) -> Self {
        ChatMessage {
            role: "assistant",
            content,
        }
    }
}

/// Render messages into Qwen2's chat format.
///
/// If no system message is present one is inserted, matching the reference
/// template. `add_generation_prompt` appends the open assistant turn.
pub fn apply_chat_template(messages: &[ChatMessage<'_>], add_generation_prompt: bool) -> String {
    let mut out = String::new();

    // The reference template inspects only `messages[0]`: a system message there
    // becomes the header, anything else means the default header is inserted.
    // A system message *later* in the list is not special and is emitted inline
    // by the loop -- so testing "is there a system message anywhere" is wrong in
    // both directions (it would drop the default header, and it would emit a
    // leading system message twice).
    let first_is_system = messages.first().is_some_and(|m| m.role == "system");
    let header = if first_is_system {
        messages[0].content
    } else {
        DEFAULT_SYSTEM_PROMPT
    };
    push_turn(&mut out, "system", header);

    for (i, m) in messages.iter().enumerate() {
        if i == 0 && first_is_system {
            continue; // already emitted as the header
        }
        match m.role {
            "user" | "assistant" | "system" => push_turn(&mut out, m.role, m.content),
            // `tool` turns and assistant `tool_calls` have their own shape in the
            // reference template. Tool use is out of scope, so those are dropped
            // rather than rendered wrongly.
            _ => {}
        }
    }

    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
    }
    out
}

fn push_turn(out: &mut String, role: &str, content: &str) {
    out.push_str("<|im_start|>");
    out.push_str(role);
    out.push('\n');
    out.push_str(content);
    out.push_str("<|im_end|>\n");
}
