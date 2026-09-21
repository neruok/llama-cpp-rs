#include "wrapper_common.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <stdint.h>
#include <vector>

#include "llama.cpp/common/chat.h"
#include "llama.cpp/common/common.h"
#include "llama.cpp/common/fit.h"
#include "llama.cpp/common/json-schema-to-grammar.h"
#include "llama.cpp/common/speculative.h"
#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <nlohmann/json.hpp>

extern "C" llama_rs_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar) {
    if (!schema_json || !out_grammar) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    *out_grammar = nullptr;
    try {
        const auto schema = nlohmann::ordered_json::parse(schema_json);
        const auto grammar = json_schema_to_grammar(schema, force_gbnf);
        *out_grammar = llama_rs_dup_string(grammar);
        return *out_grammar ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" void llama_rs_string_free(char * ptr) {
    if (ptr) {
        std::free(ptr);
    }
}

struct llama_rs_chat_template {
    common_chat_templates_ptr tmpls;
};

extern "C" struct llama_rs_chat_template * llama_rs_chat_template_init(
    const struct llama_model * model,
    const char * tmpl) {
    try {
        const std::string tmpl_override = tmpl ? tmpl : "";
        if (tmpl_override.empty()) {
            // Without an explicit template the model must embed a default one.
            // llama_model_chat_template cannot decide this: for a 40-layer
            // TEKKEN model it synthesizes "mistral-v7-tekken" even when the
            // model embeds no template, and common_chat_templates_init
            // substitutes a built-in ChatML template for an absent or empty
            // one. Read the raw metadata instead. The getter returns -1 when
            // the key is absent and the value length when it is present, so
            // both a missing and an empty template are rejected.
            char probe = '\0';
            const int32_t meta_len = model
                ? llama_model_meta_val_str(model, "tokenizer.chat_template", &probe, sizeof(probe))
                : -1;
            if (meta_len <= 0) {
                return nullptr;
            }
        }
        common_chat_templates_ptr tmpls = common_chat_templates_init(model, tmpl_override);
        if (!tmpls) {
            return nullptr;
        }
        return new llama_rs_chat_template{ std::move(tmpls) };
    } catch (...) {
        return nullptr;
    }
}

extern "C" void llama_rs_chat_template_free(struct llama_rs_chat_template * tmpls) {
    try {
        delete tmpls;
    } catch (...) {
        // nothing to report from a void destructor path
    }
}

extern "C" llama_rs_status llama_rs_chat_template_apply(
    const struct llama_rs_chat_template * tmpls,
    const struct llama_chat_message * messages,
    size_t n_messages,
    bool add_generation_prompt,
    bool enable_thinking,
    const char * const * kwarg_keys,
    const char * const * kwarg_values,
    size_t n_kwargs,
    char ** out_prompt) {
    if (!tmpls || !tmpls->tmpls || !out_prompt || (!messages && n_messages > 0) ||
        (n_kwargs > 0 && (!kwarg_keys || !kwarg_values))) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    *out_prompt = nullptr;
    try {
        common_chat_templates_inputs inputs;
        inputs.use_jinja             = true;
        inputs.add_generation_prompt = add_generation_prompt;
        inputs.enable_thinking       = enable_thinking;
        inputs.messages.reserve(n_messages);
        for (size_t i = 0; i < n_messages; ++i) {
            if (!messages[i].role || !messages[i].content) {
                return LLAMA_RS_STATUS_INVALID_ARGUMENT;
            }
            common_chat_msg msg;
            msg.role    = messages[i].role;
            msg.content = messages[i].content;
            inputs.messages.push_back(std::move(msg));
        }

        // llama.cpp stores each value as JSON text and decodes it into the
        // template's variable set, so the caller supplies `"high"` for the
        // string high. A name that is absent leaves the template default.
        for (size_t i = 0; i < n_kwargs; ++i) {
            if (!kwarg_keys[i] || !kwarg_values[i]) {
                return LLAMA_RS_STATUS_INVALID_ARGUMENT;
            }
            inputs.chat_template_kwargs[kwarg_keys[i]] = kwarg_values[i];
        }

        const common_chat_params params = common_chat_templates_apply(tmpls->tmpls.get(), inputs);
        *out_prompt                     = llama_rs_dup_string(params.prompt);
        return *out_prompt ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root) {
    try {
        return llama_sampler_init_grammar(vocab, grammar_str, grammar_root);
    } catch (...) {
        return nullptr;
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_words,
    size_t num_trigger_words,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens) {
    try {
        std::vector<std::string> trigger_patterns;
        trigger_patterns.reserve(num_trigger_words);
        for (size_t i = 0; i < num_trigger_words; ++i) {
            const char * word = trigger_words ? trigger_words[i] : nullptr;
            if (word && word[0] != '\0') {
                trigger_patterns.push_back(regex_escape(word));
            }
        }
        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & pattern : trigger_patterns) {
            trigger_patterns_c.push_back(pattern.c_str());
        }
        return llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammar_str,
            grammar_root,
            trigger_patterns_c.data(),
            trigger_patterns_c.size(),
            trigger_tokens,
            num_trigger_tokens);
    } catch (...) {
        return nullptr;
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens) {
    try {
        return llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammar_str,
            grammar_root,
            trigger_patterns,
            num_trigger_patterns,
            trigger_tokens,
            num_trigger_tokens);
    } catch (...) {
        return nullptr;
    }
}

extern "C" llama_rs_status llama_rs_sampler_accept(struct llama_sampler * sampler, llama_token token) {
    if (!sampler) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    try {
        llama_sampler_accept(sampler, token);
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

// Thin pass-through to llama.cpp's common_fit_params (a C++ symbol in libcommon).
// Returns common_params_fit_status as an int: 0 = success, 1 = failure, 2 = error.
extern "C" int llama_rs_fit_params(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level) {
    return static_cast<int>(common_fit_params(
        path_model,
        mparams,
        cparams,
        tensor_split,
        tensor_buft_overrides,
        margins,
        n_ctx_min,
        log_level));
}

extern "C" void llama_rs_memory_breakdown_print(const struct llama_context * ctx) {
    common_memory_breakdown_print(ctx);
}

struct llama_rs_mtp_speculative {
    common_params_speculative params;
    common_speculative * spec = nullptr;
    std::vector<llama_token> prompt;
    std::vector<llama_token> draft;
    size_t last_draft_len = 0;
    bool draft_pending = false;
};

static constexpr llama_seq_id LLAMA_RS_MTP_SEQ_ID = 0;

static bool llama_rs_mtp_batch_compatible(const struct llama_batch & batch) {
    if (batch.n_tokens <= 0 || !batch.token || batch.embd || !batch.pos || !batch.n_seq_id ||
        !batch.seq_id) {
        return false;
    }
    for (int32_t k = 0; k < batch.n_tokens; ++k) {
        if (batch.n_seq_id[k] != 1 || !batch.seq_id[k] ||
            batch.seq_id[k][0] != LLAMA_RS_MTP_SEQ_ID) {
            return false;
        }
    }
    return true;
}

static void llama_rs_assign_tokens(
    std::vector<llama_token> & dst,
    const llama_token * tokens,
    size_t count) {
    if (count == 0) {
        dst.clear();
        return;
    }
    dst.assign(tokens, tokens + count);
}

extern "C" struct llama_rs_mtp_speculative * llama_rs_mtp_speculative_init(
    struct llama_context * ctx_tgt,
    struct llama_context * ctx_dft,
    int32_t n_max,
    int32_t n_min,
    float p_min) {
    if (!ctx_tgt || !ctx_dft || n_max <= 0 || n_min < 0 || n_min > n_max) {
        return nullptr;
    }

    try {
        auto wrapper = std::make_unique<llama_rs_mtp_speculative>();
        wrapper->params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        wrapper->params.draft.ctx_tgt = ctx_tgt;
        wrapper->params.draft.ctx_dft = ctx_dft;
        wrapper->params.draft.n_max = n_max;
        wrapper->params.draft.n_min = n_min;
        wrapper->params.draft.p_min = p_min;

        wrapper->spec = common_speculative_init(wrapper->params, 1);
        if (!wrapper->spec) {
            return nullptr;
        }

        return wrapper.release();
    } catch (...) {
        return nullptr;
    }
}

extern "C" void llama_rs_mtp_speculative_free(struct llama_rs_mtp_speculative * spec) {
    if (!spec) {
        return;
    }
    if (spec->spec) {
        common_speculative_free(spec->spec);
        spec->spec = nullptr;
    }
    delete spec;
}

extern "C" llama_rs_status llama_rs_mtp_speculative_begin(
    struct llama_rs_mtp_speculative * spec,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count) {
    if (!spec || !spec->spec || (!prompt_tokens && prompt_tokens_count > 0)) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        llama_rs_assign_tokens(spec->prompt, prompt_tokens, prompt_tokens_count);
        spec->last_draft_len = 0;
        spec->draft_pending = false;
        common_speculative_begin(spec->spec, LLAMA_RS_MTP_SEQ_ID, spec->prompt);
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_mtp_speculative_process(
    struct llama_rs_mtp_speculative * spec,
    const struct llama_batch * batch) {
    if (!spec || !spec->spec || !batch) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    if (!llama_rs_mtp_batch_compatible(*batch)) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        return common_speculative_process(spec->spec, *batch)
            ? LLAMA_RS_STATUS_OK
            : LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_mtp_speculative_draft(
    struct llama_rs_mtp_speculative * spec,
    llama_pos n_past,
    llama_token id_last,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count,
    llama_token * out_tokens,
    size_t out_tokens_capacity,
    size_t * out_tokens_count) {
    if (!spec || !spec->spec || (!prompt_tokens && prompt_tokens_count > 0) ||
        !out_tokens_count || n_past < 0) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        if (spec->draft_pending) {
            return LLAMA_RS_STATUS_INVALID_ARGUMENT;
        }
        llama_rs_assign_tokens(spec->prompt, prompt_tokens, prompt_tokens_count);
        spec->draft.clear();
        spec->last_draft_len = 0;

        auto & params = common_speculative_get_draft_params(spec->spec, LLAMA_RS_MTP_SEQ_ID);
        params = {
            true,
            spec->params.draft.n_max,
            n_past,
            id_last,
            &spec->prompt,
            &spec->draft,
        };

        common_speculative_draft(spec->spec);

        *out_tokens_count = spec->draft.size();
        if (spec->draft.size() > out_tokens_capacity) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
        if (!spec->draft.empty() && !out_tokens) {
            return LLAMA_RS_STATUS_INVALID_ARGUMENT;
        }
        if (!spec->draft.empty()) {
            std::memcpy(out_tokens, spec->draft.data(), spec->draft.size() * sizeof(llama_token));
        }
        spec->last_draft_len = spec->draft.size();
        spec->draft_pending = !spec->draft.empty();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_mtp_speculative_accept(
    struct llama_rs_mtp_speculative * spec,
    uint16_t n_accepted) {
    if (!spec || !spec->spec) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    if (!spec->draft_pending || n_accepted > spec->last_draft_len) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        common_speculative_accept(spec->spec, LLAMA_RS_MTP_SEQ_ID, n_accepted);
        spec->last_draft_len = 0;
        spec->draft_pending = false;
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}
