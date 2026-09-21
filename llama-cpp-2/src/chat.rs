//! A safe wrapper around llama.cpp's common chat template engine (Minja).
//!
//! Unlike [`crate::model::LlamaModel::apply_chat_template`], which detects a
//! template from a fixed list of known templates, this module renders the
//! chat template with llama.cpp's full Jinja engine. It therefore handles
//! arbitrary templates, including loops and the `add_generation_prompt`
//! branch, and it can render a literal template without loading a model.
//!
//! This module requires the `common` feature.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr::{self, NonNull};

use crate::model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use crate::status_is_ok;

/// A chat template prepared by llama.cpp's Minja engine.
///
/// The handle owns the template state returned by llama.cpp and releases it on
/// drop.
#[derive(Debug)]
pub struct LlamaMinjaChatTemplate {
    templates: NonNull<llama_cpp_sys_2::llama_rs_chat_template>,
}

impl LlamaMinjaChatTemplate {
    /// Prepare the chat template embedded in `model`.
    ///
    /// # Errors
    ///
    /// Returns [`MinjaChatTemplateError::InitFailed`] when the model has no
    /// usable template or llama.cpp rejects it.
    pub fn from_model(model: &LlamaModel) -> Result<Self, MinjaChatTemplateError> {
        // Safety: `model` is a valid model for as long as the borrow lives and
        // llama.cpp keeps its own copy of the template.
        let handle = unsafe {
            llama_cpp_sys_2::llama_rs_chat_template_init(model.model.as_ptr(), ptr::null())
        };
        NonNull::new(handle)
            .map(|templates| Self { templates })
            .ok_or(MinjaChatTemplateError::InitFailed)
    }

    /// Prepare a literal Jinja `template` without a model.
    ///
    /// llama.cpp resolves the `bos_token` and `eos_token` variables to empty
    /// strings in that case.
    ///
    /// # Errors
    ///
    /// Returns [`MinjaChatTemplateError::InitFailed`] when llama.cpp cannot
    /// parse the template.
    pub fn new(template: &LlamaChatTemplate) -> Result<Self, MinjaChatTemplateError> {
        // Safety: `template` outlives the call and llama.cpp keeps its own copy.
        let handle = unsafe {
            llama_cpp_sys_2::llama_rs_chat_template_init(ptr::null(), template.as_c_str().as_ptr())
        };
        NonNull::new(handle)
            .map(|templates| Self { templates })
            .ok_or(MinjaChatTemplateError::InitFailed)
    }

    /// Render `messages` into a prompt, with no extra template variables.
    ///
    /// `enable_thinking` is forwarded to the template as its
    /// `enable_thinking` variable and is ignored by templates that do not use
    /// it.
    ///
    /// # Errors
    ///
    /// See [`MinjaChatTemplateError`] for the possible failures.
    pub fn render(
        &self,
        messages: &[LlamaChatMessage],
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<String, MinjaChatTemplateError> {
        self.render_with_kwargs(messages, add_generation_prompt, enable_thinking, &[])
    }

    /// Render `messages` into a prompt, forwarding `kwargs` to the template as
    /// llama.cpp's `chat_template_kwargs`.
    ///
    /// Each value is JSON text, which is the type llama.cpp stores, so a
    /// caller passes `("reasoning_effort", "\"high\"")` for the string
    /// `high`. A template that does not read a key ignores it, and an absent
    /// key leaves the template's own default in place.
    ///
    /// # Errors
    ///
    /// [`MinjaChatTemplateError::NulByte`] when a name or a value contains an
    /// interior null byte, and otherwise as for
    /// [`LlamaMinjaChatTemplate::render`].
    pub fn render_with_kwargs(
        &self,
        messages: &[LlamaChatMessage],
        add_generation_prompt: bool,
        enable_thinking: bool,
        kwargs: &[(&str, &str)],
    ) -> Result<String, MinjaChatTemplateError> {
        let messages: Vec<llama_cpp_sys_2::llama_chat_message> = messages
            .iter()
            .map(|message| llama_cpp_sys_2::llama_chat_message {
                role: message.role_ptr(),
                content: message.content_ptr(),
            })
            .collect();

        let keys: Vec<CString> = kwargs
            .iter()
            .map(|(key, _)| CString::new(*key))
            .collect::<Result<_, _>>()?;
        let values: Vec<CString> = kwargs
            .iter()
            .map(|(_, value)| CString::new(*value))
            .collect::<Result<_, _>>()?;
        let key_ptrs: Vec<*const c_char> = keys.iter().map(|key| key.as_ptr()).collect();
        let value_ptrs: Vec<*const c_char> = values.iter().map(|value| value.as_ptr()).collect();

        let mut out_prompt: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_chat_template_apply(
                self.templates.as_ptr(),
                messages.as_ptr(),
                messages.len(),
                add_generation_prompt,
                enable_thinking,
                key_ptrs.as_ptr(),
                value_ptrs.as_ptr(),
                key_ptrs.len(),
                ptr::from_mut(&mut out_prompt),
            )
        };

        if !status_is_ok(status) || out_prompt.is_null() {
            return Err(MinjaChatTemplateError::ApplyFailed(status as i32));
        }

        let prompt_bytes = unsafe { CStr::from_ptr(out_prompt) }.to_bytes().to_vec();
        let prompt = String::from_utf8(prompt_bytes).map_err(MinjaChatTemplateError::from);
        unsafe { llama_cpp_sys_2::llama_rs_string_free(out_prompt) };
        prompt
    }
}

impl Drop for LlamaMinjaChatTemplate {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::llama_rs_chat_template_free(self.templates.as_ptr()) }
    }
}

/// Errors from [`LlamaMinjaChatTemplate`].
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum MinjaChatTemplateError {
    /// llama.cpp failed to build the chat template.
    #[error("llama.cpp failed to initialize the chat template")]
    InitFailed,
    /// llama.cpp rejected a message or the render call failed. Contains the
    /// `llama_rs_status` value returned by the C API.
    #[error("llama.cpp failed to apply the chat template with status {0}")]
    ApplyFailed(i32),
    /// The rendered prompt was not valid utf8.
    #[error("{0}")]
    FromUtf8Error(#[from] std::string::FromUtf8Error),
    /// A keyword name or value contained an interior null byte.
    #[error("chat template keyword contained a null byte")]
    NulByte(#[from] std::ffi::NulError),
}
