//! Renders a literal Jinja chat template with llama.cpp's Minja engine, and
//! rejects a model that embeds no default template.
//!
//! The literal template uses a `for` loop over `messages` and an
//! `add_generation_prompt` branch. The heuristic template detection behind
//! `llama_chat_apply_template` recognizes only a fixed list of known
//! templates, so it cannot render this one; it needs a model to fail at all.
//! Rendering the literal template therefore needs no model, no vocabulary file
//! and no fixture. Rejecting a model without an embedded template loads a
//! vocab-only GGUF that the llama.cpp submodule already checks in.

#![cfg(feature = "common")]

use std::path::Path;
use std::sync::OnceLock;

use llama_cpp_2::chat::{LlamaMinjaChatTemplate, MinjaChatTemplateError};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel};

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

/// A vocab-only GGUF checked into the llama.cpp submodule. It carries no
/// `tokenizer.chat_template` metadata, which is the case that must be rejected
/// instead of letting llama.cpp substitute its built-in ChatML template.
const VOCAB_ONLY_MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../llama-cpp-sys-2/llama.cpp/models/ggml-vocab-llama-spm.gguf"
);

/// Loads [`VOCAB_ONLY_MODEL`], or `None` when the submodule fixture is absent.
///
/// A missing fixture skips the test, but a backend or load failure panics: a
/// silently skipped load would pass the test without exercising the guard.
fn vocab_only_model() -> Option<(&'static LlamaBackend, &'static LlamaModel)> {
    static MODEL: OnceLock<Option<(LlamaBackend, LlamaModel)>> = OnceLock::new();
    if !Path::new(VOCAB_ONLY_MODEL).exists() {
        return None;
    }
    MODEL
        .get_or_init(|| {
            let backend = LlamaBackend::init().expect("backend should initialize");
            let params = LlamaModelParams::default().with_vocab_only(true);
            let model = LlamaModel::load_from_file(&backend, VOCAB_ONLY_MODEL, &params)
                .expect("vocab-only fixture should load");
            Some((backend, model))
        })
        .as_ref()
        .map(|(backend, model)| (backend, model))
}

#[test]
fn model_without_embedded_template_is_rejected() {
    let Some((_backend, model)) = vocab_only_model() else {
        return;
    };

    // The guard reads the raw metadata rather than llama_model_chat_template,
    // which synthesizes a template for some template-less models. Establish
    // that the fixture really has no metadata value to embed.
    assert!(
        model.meta_val_str("tokenizer.chat_template").is_err(),
        "fixture unexpectedly embeds a chat template"
    );

    assert_eq!(
        LlamaMinjaChatTemplate::from_model(model).expect_err("model has no embedded template"),
        MinjaChatTemplateError::InitFailed
    );
}

#[test]
fn empty_template_without_model_is_rejected() {
    let literal = LlamaChatTemplate::new("").expect("empty template has no null byte");

    assert_eq!(
        LlamaMinjaChatTemplate::new(&literal).expect_err("no model and no template"),
        MinjaChatTemplateError::InitFailed
    );
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
