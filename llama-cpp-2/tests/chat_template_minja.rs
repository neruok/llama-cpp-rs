//! Renders a literal Jinja chat template with llama.cpp's Minja engine.
//!
//! The template uses a `for` loop over `messages` and an
//! `add_generation_prompt` branch. The heuristic template detection behind
//! `llama_chat_apply_template` recognizes only a fixed list of known
//! templates, so it cannot render this one; it needs a model to fail at all.
//! This test therefore needs no model, no vocabulary file and no fixture.

#![cfg(feature = "common")]

use llama_cpp_2::chat::LlamaMinjaChatTemplate;
use llama_cpp_2::model::{LlamaChatMessage, LlamaChatTemplate};

/// A loop over the messages and a generation-prompt branch, with no marker
/// that the legacy template detection matches.
const TEMPLATE: &str = concat!(
    "{% for message in messages %}",
    "<|{{ message['role'] }}|>{{ message['content'] }}<|end|>",
    "{% endfor %}",
    "{% if add_generation_prompt %}<|assistant|>{% endif %}",
);

fn messages() -> Vec<LlamaChatMessage> {
    vec![
        LlamaChatMessage::new("user".to_string(), "hello".to_string()).expect("valid message"),
        LlamaChatMessage::new("assistant".to_string(), "hi there".to_string())
            .expect("valid message"),
    ]
}

fn template() -> LlamaMinjaChatTemplate {
    let literal = LlamaChatTemplate::new(TEMPLATE).expect("template has no null byte");
    LlamaMinjaChatTemplate::new(&literal).expect("template is valid jinja")
}

#[test]
fn renders_loop_and_generation_prompt() {
    let template = template();
    let messages = messages();

    let with_prompt = template
        .render(&messages, true, true)
        .expect("rendering should succeed");
    assert_eq!(
        with_prompt,
        "<|user|>hello<|end|><|assistant|>hi there<|end|><|assistant|>"
    );

    let without_prompt = template
        .render(&messages, false, true)
        .expect("rendering should succeed");
    assert_eq!(
        without_prompt,
        "<|user|>hello<|end|><|assistant|>hi there<|end|>"
    );
}
