#include "diffusion.h"

#include "log.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

static float calculate_confidence(const llama_token_data_array & cur_p,
                                  diffusion_algorithm            algorithm,
                                  std::mt19937 &                 rng) {
    switch (algorithm) {
        case DIFFUSION_ALGORITHM_CONFIDENCE_BASED:
            return cur_p.data[cur_p.selected].p;  // Selected token probability

        case DIFFUSION_ALGORITHM_ENTROPY_BASED:
            {
                float       entropy = 0.0f;
                const float epsilon = 1e-10f;
                for (size_t i = 0; i < cur_p.size; i++) {
                    float prob = cur_p.data[i].p;
                    entropy += prob * logf(prob + epsilon);
                }
                return -entropy;  // Higher entropy = lower confidence
            }

        case DIFFUSION_ALGORITHM_MARGIN_BASED:
            return (cur_p.size > 1) ? cur_p.data[0].p - cur_p.data[1].p : cur_p.data[0].p;

        case DIFFUSION_ALGORITHM_RANDOM:
            {
                std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
                return uniform(rng);  // Random confidence
            }

        case DIFFUSION_ALGORITHM_ORIGIN:
            return cur_p.data[cur_p.selected].p;

        default:
            return 0.0f;
    }
}

// Unified transfer count calculation function
static int32_t calculate_transfer_count(int32_t                      step,
                                        int32_t                      total_steps,
                                        int32_t                      remaining_masked,
                                        diffusion_transfer_schedule  schedule,
                                        float                        eps,
                                        const std::vector<int32_t> & num_transfer_tokens = {}) {
    switch (schedule) {
        case DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED:
            {
                float t          = 1.0f - (float) step / total_steps * (1.0f - eps);
                float s          = 1.0f - (float) (step + 1) / total_steps * (1.0f - eps);
                float p_transfer = (step < total_steps - 1) ? (1.0f - s / t) : 1.0f;
                return (int32_t) (remaining_masked * p_transfer);
            }

        case DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED:
            if (!num_transfer_tokens.empty() && step < (int32_t) num_transfer_tokens.size()) {
                return num_transfer_tokens[step];
            }
            return remaining_masked / (total_steps - step);  // Fallback

        default:
            return remaining_masked / (total_steps - step);
    }
}

static void add_gumbel_noise(float * logits, int32_t n_vocab, float temperature, std::mt19937 & rng) {
    if (temperature == 0.0f) {
        return;
    }

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (int32_t i = 0; i < n_vocab; i++) {
        double noise        = uniform(rng);
        // Prevent log(0)
        noise               = std::max(noise, 1e-20);
        double gumbel_noise = std::pow(-std::log(noise), temperature);
        logits[i]           = std::exp(logits[i]) / gumbel_noise;
    }
}

static std::vector<int32_t> get_num_transfer_tokens(int32_t mask_count, int32_t steps) {
    std::vector<int32_t> num_transfer_tokens(steps);

    int32_t base      = mask_count / steps;
    int32_t remainder = mask_count % steps;

    for (int32_t i = 0; i < steps; i++) {
        num_transfer_tokens[i] = base + (i < remainder ? 1 : 0);
    }

    return num_transfer_tokens;
}

void diffusion_generate(llama_context *          ctx,
                        const llama_token *      input_tokens,
                        llama_token *            output_tokens,
                        int32_t                  n_input,
                        const diffusion_params & params,
                        int32_t &                n_generated) {
    n_generated = 0;
    if (!ctx || !input_tokens || !output_tokens || n_input <= 0 || params.max_length <= n_input) {
        return;
    }

    const llama_model * model = llama_get_model(ctx);

    // Initialize with input and pad with mask tokens
    std::copy(input_tokens, input_tokens + n_input, output_tokens);
    std::fill(output_tokens + n_input, output_tokens + params.max_length, params.mask_token_id);

    std::mt19937 rng(params.seed);

    llama_set_causal_attn(ctx, false);

    int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<llama_token_data> candidates(n_vocab);
    std::vector<llama_token_data> conf_candidates;
    conf_candidates.reserve(params.max_length);
    std::vector<int32_t> mask_positions;
    mask_positions.reserve(params.max_length);

    // Setup sampler chain
    struct llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (params.top_k > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(params.top_k));
    }
    if (params.top_p < 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(params.top_p, 1));
    }
    if (params.temperature > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(params.temperature));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(params.seed));

    struct llama_sampler * dist_sampler = llama_sampler_init_dist(params.seed);

    llama_batch batch = llama_batch_init(params.max_length, 0, 1);
    batch.n_tokens    = params.max_length;

    // Pre-allocate buffers for CFG if needed
    int32_t                  logits_size = n_vocab * params.max_length;
    std::vector<float>       cond_logits_buffer;
    std::vector<llama_token> un_x_buffer;
    if (params.cfg_scale > 0.0f) {
        cond_logits_buffer.resize(logits_size);
        un_x_buffer.resize(params.max_length);
    }

    // For block-based processing
    std::vector<int32_t> num_transfer_tokens;
    int32_t              num_blocks      = 1;
    int32_t              steps_per_block = params.steps;

    if (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) {
        GGML_ASSERT(params.max_length % params.block_length == 0);
        num_blocks = params.max_length / params.block_length;
        GGML_ASSERT(params.steps % num_blocks == 0);
        steps_per_block = params.steps / num_blocks;
    }

    std::vector<float> confidence(params.max_length);

    int64_t total_sampling_time = 0;
    int64_t total_time          = 0;
    int64_t time_start          = ggml_time_us();

    for (int block_num = 0; block_num < num_blocks; block_num++) {
        int32_t block_start = (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) ? n_input + block_num * params.block_length : 0;
        int32_t block_end   = (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) ?
                                  std::min(n_input + (block_num + 1) * params.block_length, params.max_length) :
                                  params.max_length;

        // Count masked tokens in current block for block-based processing
        if (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) {
            int32_t block_mask_count = 0;
            for (int i = block_start; i < block_end; i++) {
                if (output_tokens[i] == params.mask_token_id) {
                    block_mask_count++;
                }
            }
            num_transfer_tokens = get_num_transfer_tokens(block_mask_count, steps_per_block);
        }

        for (int32_t step = 0; step < steps_per_block; step++) {
            int32_t global_step = block_num * steps_per_block + step;

            if (params.step_callback) {
                if (!params.step_callback(
                        global_step, params.steps, output_tokens, params.max_length, params.step_callback_user_data)) {
                    break;
                }
            }

            // Setup batch
            for (int32_t i = 0; i < params.max_length; i++) {
                batch.token[i]     = output_tokens[i];
                batch.pos[i]       = i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 1;
            }

            float * logits = nullptr;

            if (params.cfg_scale > 0.0f) {
                int ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    LOG_ERR("Failed to generate conditional");
                    break;
                }
                float * cond_logits_ptr = llama_get_logits(ctx);
                std::memcpy(cond_logits_buffer.data(), cond_logits_ptr, logits_size * sizeof(float));

                // Unconditional generation (mask input)
                std::copy(output_tokens, output_tokens + params.max_length, un_x_buffer.begin());
                for (int32_t i = 0; i < n_input; i++) {
                    un_x_buffer[i] = params.mask_token_id;
                }

                for (int32_t i = 0; i < params.max_length; i++) {
                    batch.token[i] = un_x_buffer[i];
                }
                ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    LOG_ERR("Failed to generate unconditional");
                    break;
                }
                float * uncond_logits = llama_get_logits(ctx);

                // Apply CFG
                for (int32_t i = 0; i < logits_size; i++) {
                    cond_logits_buffer[i] =
                        uncond_logits[i] + (params.cfg_scale + 1.0f) * (cond_logits_buffer[i] - uncond_logits[i]);
                }
                logits = cond_logits_buffer.data();
            } else {
                int ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    LOG_ERR("%s: failed to decode at step %d, ret = %d\n", __func__, global_step, ret);
                    break;
                }
                logits = llama_get_logits(ctx);
            }

            if (!logits) {
                LOG_ERR("%s: failed to get logits at step %d\n", __func__, global_step);
                break;
            }

            auto get_logits_for_pos = [&](int32_t pos) -> const float * {
                if (params.shift_logits) {
                    return pos == 0 ? logits : logits + (pos - 1) * n_vocab;
                }
                return logits + pos * n_vocab;
            };

            int64_t time_start_sampling = ggml_time_us();

            mask_positions.clear();
            for (int32_t i = 0; i < params.max_length; i++) {
                if (output_tokens[i] == params.mask_token_id) {
                    // For block-based, only consider current block
                    if (params.schedule != DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED || (i >= block_start && i < block_end)) {
                        mask_positions.push_back(i);
                    }
                }
            }

            if (mask_positions.empty()) {
                break;
            }

            if (params.add_gumbel_noise && params.temperature > 0.0f) {
                add_gumbel_noise(logits, n_vocab, params.temperature, rng);
            }

            if (params.algorithm == DIFFUSION_ALGORITHM_ORIGIN) {
                int32_t transfer_count = calculate_transfer_count(
                    step, steps_per_block, mask_positions.size(), params.schedule, params.eps, num_transfer_tokens);
                float p_transfer = (float) transfer_count / mask_positions.size();

                for (int32_t pos : mask_positions) {
                    if (std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < p_transfer) {
                        const float * pos_logits = get_logits_for_pos(pos);
                        for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
                            candidates[token_id].id    = token_id;
                            candidates[token_id].logit = pos_logits[token_id];
                            candidates[token_id].p     = 0.0f;
                        }

                        llama_token_data_array cur_p = {
                            candidates.data(),
                            (size_t) n_vocab,
                            -1,
                            false,
                        };

                        llama_sampler_apply(sampler, &cur_p);
                        output_tokens[pos] = cur_p.data[cur_p.selected].id;
                    }
                }
            } else {
                std::vector<std::pair<float, int32_t>> confidences;
                std::vector<llama_token>               sampled_tokens(mask_positions.size());

                for (size_t i = 0; i < mask_positions.size(); i++) {
                    int32_t       pos        = mask_positions[i];
                    const float * pos_logits = get_logits_for_pos(pos);

                    for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
                        candidates[token_id].logit = pos_logits[token_id];
                        candidates[token_id].p     = 0.0f;
                        candidates[token_id].id    = token_id;
                    }

                    llama_token_data_array cur_p = {
                        candidates.data(),
                        candidates.size(),
                        -1,
                        false,
                    };

                    llama_sampler_apply(sampler, &cur_p);
                    llama_token sampled_token = cur_p.data[cur_p.selected].id;

                    float conf = calculate_confidence(cur_p, params.algorithm, rng);

                    sampled_tokens[i] = sampled_token;
                    confidences.emplace_back(conf, i);
                }

                int32_t transfer_count = calculate_transfer_count(
                    step, steps_per_block, mask_positions.size(), params.schedule, params.eps, num_transfer_tokens);

                if (transfer_count > 0) {
                    if (params.alg_temp == 0.0f) {
                        std::partial_sort(confidences.begin(),
                                          confidences.begin() + std::min(transfer_count, (int32_t) confidences.size()),
                                          confidences.end(),
                                          [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                                              if (a.first != b.first) {
                                                  return a.first > b.first;
                                              }
                                              return a.second < b.second;
                                          });

                        for (int32_t i = 0; i < std::min(transfer_count, (int32_t) confidences.size()); i++) {
                            int32_t mask_idx   = confidences[i].second;
                            int32_t pos        = mask_positions[mask_idx];
                            output_tokens[pos] = sampled_tokens[mask_idx];
                        }
                    } else {
                        conf_candidates.clear();
                        for (size_t i = 0; i < confidences.size(); i++) {
                            float conf_logit = confidences[i].first / params.alg_temp;
                            conf_candidates.emplace_back(llama_token_data{ (int32_t) i, conf_logit, 0.0f });
                        }

                        llama_token_data_array conf_array = {
                            conf_candidates.data(),
                            conf_candidates.size(),
                            -1,
                            false,
                        };

                        for (int32_t i = 0; i < std::min(transfer_count, (int32_t) confidences.size()); i++) {
                            llama_sampler_apply(dist_sampler, &conf_array);
                            int32_t selected_idx = conf_array.selected;
                            int32_t mask_idx     = selected_idx;
                            int32_t pos          = mask_positions[mask_idx];
                            output_tokens[pos]   = sampled_tokens[mask_idx];

                            conf_candidates[selected_idx].p = 0.0f;
                            conf_array.selected             = -1;
                        }
                    }
                }
            }

            int64_t time_end_sampling = ggml_time_us();
            total_sampling_time += time_end_sampling - time_start_sampling;
        }
    }

    int64_t time_end = ggml_time_us();
    total_time += time_end - time_start;

    LOG_INF("\ntotal time: %0.2fms, time per step: %0.2fms, sampling time per step: %0.2fms\n",
            total_time / 1000.0,
            total_time / 1000.0 / params.steps,
            total_sampling_time / 1000.0 / params.steps);

    llama_batch_free(batch);
    llama_sampler_free(sampler);
    llama_sampler_free(dist_sampler);

    n_generated = params.max_length;
}

int32_t diffusion_generate_blocks(llama_context *               ctx,
                                  const llama_token *           input_tokens,
                                  int32_t                       n_input,
                                  llama_token *                 output_tokens,
                                  int32_t                       max_output,
                                  const diffusion_block_params & params) {
    if (!ctx || !input_tokens || !output_tokens || n_input <= 0 || max_output <= 0) {
        return 0;
    }

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_memory_t      mem   = llama_get_memory(ctx);
    int32_t             n_vocab = llama_vocab_n_tokens(vocab);
    llama_token         eos     = llama_vocab_eos(vocab);

    std::mt19937 rng(params.seed);

    struct llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (params.top_k > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(params.top_k));
    }
    if (params.top_p > 0.0f && params.top_p < 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(params.top_p, 1));
    }
    if (params.temperature > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(params.temperature));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(params.seed));

    std::vector<llama_token_data> candidates(n_vocab);

    llama_memory_clear(mem, true);

    // Phase 1: Causal prefill — populate KV cache with input
    llama_set_causal_attn(ctx, true);
    {
        llama_batch batch = llama_batch_init(n_input, 0, 1);
        batch.n_tokens = n_input;
        for (int32_t i = 0; i < n_input; i++) {
            batch.token[i]     = input_tokens[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 0;
        }
        batch.logits[n_input - 1] = 1;

        int ret = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (ret != 0) {
            LOG_ERR("diffusion_generate_blocks: prefill failed (%d)\n", ret);
            llama_sampler_free(sampler);
            return 0;
        }
    }

    int32_t committed   = n_input;
    int32_t n_generated = 0;

    while (n_generated < max_output) {
        int32_t block_len = std::min(params.block_length, max_output - n_generated);
        if (block_len <= 0) break;

        std::vector<llama_token> block_tokens(block_len, params.mask_token_id);

        int32_t steps = std::min(params.steps_per_block, block_len);

        // Pre-compute even transfer schedule (Nvidia dLM style)
        std::vector<int32_t> transfer_schedule = get_num_transfer_tokens(block_len, steps);

        // Phase 2: Bidirectional denoising of the block
        llama_set_causal_attn(ctx, false);

        for (int32_t step = 0; step < steps; step++) {
            std::vector<int32_t> mask_positions;
            for (int32_t i = 0; i < block_len; i++) {
                if (block_tokens[i] == params.mask_token_id) {
                    mask_positions.push_back(i);
                }
            }
            if (mask_positions.empty()) break;

            llama_batch batch = llama_batch_init(block_len, 0, 1);
            batch.n_tokens = block_len;
            for (int32_t i = 0; i < block_len; i++) {
                batch.token[i]     = block_tokens[i];
                batch.pos[i]       = committed + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 1;
            }

            int ret = llama_decode(ctx, batch);
            llama_batch_free(batch);
            if (ret != 0) {
                LOG_ERR("diffusion_generate_blocks: decode failed at step %d (%d)\n", step, ret);
                llama_sampler_free(sampler);
                return n_generated;
            }

            float * logits = llama_get_logits(ctx);

            // Sample all mask positions and compute confidences
            std::vector<std::pair<float, int32_t>> confidences;
            std::vector<llama_token>               sampled(mask_positions.size());

            for (size_t mi = 0; mi < mask_positions.size(); mi++) {
                int32_t pos = mask_positions[mi];
                const float * pos_logits = logits + pos * n_vocab;

                for (int32_t v = 0; v < n_vocab; v++) {
                    candidates[v].id    = v;
                    candidates[v].logit = pos_logits[v];
                    candidates[v].p     = 0.0f;
                }

                llama_token_data_array cur_p = { candidates.data(), (size_t) n_vocab, -1, false };
                llama_sampler_apply(sampler, &cur_p);

                sampled[mi] = cur_p.data[cur_p.selected].id;
                confidences.emplace_back(cur_p.data[cur_p.selected].p, (int32_t) mi);
            }

            int32_t n_unmask = (step < steps - 1) ? transfer_schedule[step] : (int32_t) mask_positions.size();
            n_unmask = std::min(n_unmask, (int32_t) mask_positions.size());
            if (n_unmask <= 0) n_unmask = 1;

            // Sort by confidence, unmask the top-N (with optional threshold filter)
            std::partial_sort(confidences.begin(),
                              confidences.begin() + std::min(n_unmask, (int32_t) confidences.size()),
                              confidences.end(),
                              [](const auto & a, const auto & b) { return a.first > b.first; });

            for (int32_t i = 0; i < std::min(n_unmask, (int32_t) confidences.size()); i++) {
                // Nvidia dLM: always commit most confident token, threshold filters rest
                if (i > 0 && params.threshold > 0.0f && confidences[i].first < params.threshold) {
                    break;
                }
                int32_t mi  = confidences[i].second;
                int32_t pos = mask_positions[mi];
                block_tokens[pos] = sampled[mi];
            }

            // Clear block K/V from cache before next step
            llama_memory_seq_rm(mem, 0, committed, committed + block_len);
        }

        // Check for EOS and copy to output
        bool found_eos = false;
        for (int32_t i = 0; i < block_len; i++) {
            if (block_tokens[i] == eos) {
                block_len = i;
                found_eos = true;
                break;
            }
            output_tokens[n_generated + i] = block_tokens[i];
        }
        n_generated += block_len;

        // Phase 3: Causal commit — add block to KV cache
        llama_set_causal_attn(ctx, true);
        {
            llama_batch batch = llama_batch_init(block_len, 0, 1);
            batch.n_tokens = block_len;
            for (int32_t i = 0; i < block_len; i++) {
                batch.token[i]     = block_tokens[i];
                batch.pos[i]       = committed + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 0;
            }

            int ret = llama_decode(ctx, batch);
            llama_batch_free(batch);
            if (ret != 0) {
                LOG_ERR("diffusion_generate_blocks: commit failed (%d)\n", ret);
                break;
            }
        }

        committed += block_len;
        if (found_eos) break;
    }

    llama_sampler_free(sampler);
    return n_generated;
}

static llama_token argmax_logits(const float * logits, int32_t n_vocab) {
    llama_token best = 0;
    float best_val = logits[0];
    for (int32_t v = 1; v < n_vocab; v++) {
        if (logits[v] > best_val) {
            best_val = logits[v];
            best = v;
        }
    }
    return best;
}

int32_t diffusion_self_spec_generate(llama_context *                    ctx,
                                     const llama_token *                input_tokens,
                                     int32_t                            n_input,
                                     llama_token *                      output_tokens,
                                     int32_t                            max_output,
                                     const diffusion_self_spec_params & params) {
    if (!ctx || !input_tokens || !output_tokens || n_input <= 0 || max_output <= 0) {
        return 0;
    }

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_memory_t      mem   = llama_get_memory(ctx);
    int32_t             n_vocab = llama_vocab_n_tokens(vocab);
    llama_token         eos     = llama_vocab_eos(vocab);

    llama_memory_clear(mem, true);

    // Phase 1: Causal prefill
    llama_set_causal_attn(ctx, true);
    {
        llama_batch batch = llama_batch_init(n_input, 0, 1);
        batch.n_tokens = n_input;
        for (int32_t i = 0; i < n_input; i++) {
            batch.token[i]     = input_tokens[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 0;
        }
        batch.logits[n_input - 1] = 1;

        int ret = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (ret != 0) {
            LOG_ERR("self_spec: prefill failed (%d)\n", ret);
            return 0;
        }
    }

    // prev_logits: causal logits from the last committed position, predicts next token
    std::vector<float> prev_logits(n_vocab);
    std::memcpy(prev_logits.data(), llama_get_logits(ctx), n_vocab * sizeof(float));

    int32_t committed   = n_input;
    int32_t n_generated = 0;
    int32_t k           = params.draft_length;

    int64_t total_accepted = 0;
    int64_t total_drafted  = 0;
    int64_t total_cycles   = 0;

    std::vector<llama_token> draft(k);

    while (n_generated < max_output) {
        int32_t draft_len = std::min(k, max_output - n_generated);
        if (draft_len <= 0) break;

        total_cycles++;
        total_drafted += draft_len;

        // DRAFT: bidirectional — single forward pass, all masks denoised in parallel
        llama_set_causal_attn(ctx, false);
        {
            llama_batch batch = llama_batch_init(draft_len, 0, 1);
            batch.n_tokens = draft_len;
            for (int32_t i = 0; i < draft_len; i++) {
                batch.token[i]     = params.mask_token_id;
                batch.pos[i]       = committed + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 1;
            }

            int ret = llama_decode(ctx, batch);
            llama_batch_free(batch);
            if (ret != 0) {
                LOG_ERR("self_spec: draft decode failed (%d)\n", ret);
                break;
            }
        }

        // Sample draft tokens (greedy argmax — bidirectional predicts AT position)
        {
            float * logits = llama_get_logits(ctx);
            for (int32_t i = 0; i < draft_len; i++) {
                draft[i] = argmax_logits(logits + i * n_vocab, n_vocab);
            }
        }

        // Clear bidirectional KV entries (not valid for causal verify)
        llama_memory_seq_rm(mem, 0, committed, committed + draft_len);

        // VERIFY: causal — single forward pass with draft tokens
        llama_set_causal_attn(ctx, true);
        {
            llama_batch batch = llama_batch_init(draft_len, 0, 1);
            batch.n_tokens = draft_len;
            for (int32_t i = 0; i < draft_len; i++) {
                batch.token[i]     = draft[i];
                batch.pos[i]       = committed + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 1;
            }

            int ret = llama_decode(ctx, batch);
            llama_batch_free(batch);
            if (ret != 0) {
                LOG_ERR("self_spec: verify decode failed (%d)\n", ret);
                break;
            }
        }

        float * verify_logits = llama_get_logits(ctx);

        // ACCEPT: longest contiguous prefix where AR agrees with draft
        // prev_logits (causal, from last committed pos) predicts draft[0]
        // verify_logits[j] (causal, at committed+j) predicts draft[j+1]
        int32_t n_accept = 0;

        if (argmax_logits(prev_logits.data(), n_vocab) == draft[0]) {
            n_accept = 1;
            for (int32_t j = 0; j < draft_len - 1; j++) {
                if (argmax_logits(verify_logits + j * n_vocab, n_vocab) == draft[j + 1]) {
                    n_accept++;
                } else {
                    break;
                }
            }
        }

        // Bonus token from the AR logits at the rejection point
        llama_token bonus;
        if (n_accept == 0) {
            bonus = argmax_logits(prev_logits.data(), n_vocab);
        } else {
            bonus = argmax_logits(verify_logits + (n_accept - 1) * n_vocab, n_vocab);
        }

        // Clear rejected KV positions (keep 0..committed+n_accept-1)
        if (n_accept < draft_len) {
            llama_memory_seq_rm(mem, 0, committed + n_accept, committed + draft_len);
        }

        // Decode bonus token — updates KV cache and produces new prev_logits
        {
            llama_batch batch = llama_batch_init(1, 0, 1);
            batch.n_tokens     = 1;
            batch.token[0]     = bonus;
            batch.pos[0]       = committed + n_accept;
            batch.n_seq_id[0]  = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0]    = 1;

            int ret = llama_decode(ctx, batch);
            llama_batch_free(batch);
            if (ret != 0) {
                LOG_ERR("self_spec: bonus decode failed (%d)\n", ret);
                break;
            }

            std::memcpy(prev_logits.data(), llama_get_logits(ctx), n_vocab * sizeof(float));
        }

        // Output accepted draft tokens + bonus
        bool found_eos = false;
        for (int32_t i = 0; i < n_accept && n_generated < max_output; i++) {
            if (draft[i] == eos) {
                found_eos = true;
                break;
            }
            output_tokens[n_generated++] = draft[i];
        }
        if (!found_eos && n_generated < max_output) {
            if (bonus == eos) {
                found_eos = true;
            } else {
                output_tokens[n_generated++] = bonus;
            }
        }

        total_accepted += n_accept + 1;
        committed += n_accept + 1;

        if (found_eos) break;
    }

    if (total_cycles > 0) {
        LOG_INF("self_spec: %lld cycles, avg %.1f tokens/cycle, %.1f%% draft accept rate\n",
                (long long) total_cycles,
                (double) total_accepted / total_cycles,
                total_drafted > 0 ? (double) (total_accepted - total_cycles) / total_drafted * 100.0 : 0.0);
    }

    return n_generated;
}
