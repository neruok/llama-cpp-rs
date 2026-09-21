#pragma once

#include "llama.cpp/include/llama.h"

#include <stdbool.h>
#include <stddef.h>

struct llama_model;
struct llama_sampler;
struct llama_rs_mtp_speculative;
struct llama_vocab;

#include "wrapper_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

llama_rs_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar);

struct llama_sampler * llama_rs_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root);

struct llama_sampler * llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_words,
    size_t num_trigger_words,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens);

struct llama_sampler * llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens);

llama_rs_status llama_rs_sampler_accept(struct llama_sampler * sampler, llama_token token);

// Fit model/context params to device memory (wraps llama.cpp's common_fit_params).
// Returns common_params_fit_status as an int: 0 = success, 1 = failure, 2 = error.
int llama_rs_fit_params(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level);

void llama_rs_memory_breakdown_print(const struct llama_context * ctx);

struct llama_rs_mtp_speculative * llama_rs_mtp_speculative_init(
    struct llama_context * ctx_tgt,
    struct llama_context * ctx_dft,
    int32_t n_max,
    int32_t n_min,
    float p_min);

void llama_rs_mtp_speculative_free(struct llama_rs_mtp_speculative * spec);

llama_rs_status llama_rs_mtp_speculative_begin(
    struct llama_rs_mtp_speculative * spec,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count);

llama_rs_status llama_rs_mtp_speculative_process(
    struct llama_rs_mtp_speculative * spec,
    const struct llama_batch * batch);

llama_rs_status llama_rs_mtp_speculative_draft(
    struct llama_rs_mtp_speculative * spec,
    llama_pos n_past,
    llama_token id_last,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count,
    llama_token * out_tokens,
    size_t out_tokens_capacity,
    size_t * out_tokens_count);

llama_rs_status llama_rs_mtp_speculative_accept(
    struct llama_rs_mtp_speculative * spec,
    uint16_t n_accepted);

// Opaque handle for llama.cpp's common chat template engine (Minja).
struct llama_rs_chat_template;

// Initialize a chat template. A non-empty `tmpl` is used as a literal Jinja
// template; when `tmpl` is NULL or empty the default template embedded in
// `model` is used. That requires `model` to be non-NULL and to carry a non-empty
// `tokenizer.chat_template` metadata value; no built-in fallback template is
// substituted. Returns NULL on failure.
struct llama_rs_chat_template * llama_rs_chat_template_init(
    const struct llama_model * model,
    const char * tmpl);

// Release a handle returned by llama_rs_chat_template_init. NULL is ignored.
void llama_rs_chat_template_free(struct llama_rs_chat_template * tmpls);

// Render `messages` with `tmpls` using the Minja engine and store the prompt in
// `*out_prompt` (release it with llama_rs_string_free).
//
// `kwarg_keys` and `kwarg_values` carry `n_kwargs` template variable names and
// their JSON-encoded values, forwarded to the template as llama.cpp's
// `chat_template_kwargs`. Both arrays must be non-NULL when `n_kwargs` is
// greater than zero. A value that is not valid JSON fails the call.
llama_rs_status llama_rs_chat_template_apply(
    const struct llama_rs_chat_template * tmpls,
    const struct llama_chat_message * messages,
    size_t n_messages,
    bool add_generation_prompt,
    bool enable_thinking,
    const char * const * kwarg_keys,
    const char * const * kwarg_values,
    size_t n_kwargs,
    char ** out_prompt);

void llama_rs_string_free(char * ptr);

#ifdef __cplusplus
}
#endif
