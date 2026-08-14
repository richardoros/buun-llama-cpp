#pragma once

// fattn-mma-f16.cuh must be included before this header (provides flash_attn_ext_f16,
// launch_fattn, fattn_kernel_t, and the MMA helper functions).

// Fused MMA-native turbo flash attention: reads raw turbo-quantized K/V directly,
// dequants into half2 shmem tiles inside the attention loop. No intermediate fp16 buffers.
// Uses the same flash_attn_ext_f16 kernel with type_K/type_V template params.

template <int DKQ, int DV, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_mma_turbo_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr int ncols = ncols1 * ncols2;

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);

    // Turbo forces nstages=0: cp.async can't do ALU dequant, so tiles load synchronously.
    // With nstages=0, tile_K and tile_V share the same shmem region (overlap).
    constexpr int nstages = 0;

    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps         = nthreads / warp_size_host;

    constexpr bool V_is_K_view = false; // Turbo K/V are separate tensors.

    const size_t nbytes_shared_KV_1stage = nbatch_fa            * std::max(nbatch_K2 + 4,  nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_Q         = ncols                * (DKQ/2 + 4)                             * sizeof(half2);
    const size_t nbytes_shared_mask      = ncols1               * (nbatch_fa/2 + 4)                       * sizeof(half2);
    const size_t nbytes_shared_combine   = nwarps*cols_per_warp * (nbatch_combine + 4)                    * sizeof(half2);

    const size_t nbytes_shared_KV = nbytes_shared_KV_1stage; // nstages=0 → 1-stage layout

    const size_t nbytes_shared_total = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q,  nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

#if defined(GGML_USE_HIP)
    using fattn_kernel_ptr_t = const void*;
#else
    using fattn_kernel_ptr_t = fattn_kernel_t;
#endif // defined(GGML_USE_HIP)
    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, type_K, type_V>;

#if !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, type_K, type_V>;

#if !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)
    }

    // Set TCQ constants in THIS compilation unit's __constant__ memory before kernel launch.
    // Each template instance .cu file has its own static __constant__ copies.
    if constexpr (type_K == GGML_TYPE_TURBO3_TCQ || type_V == GGML_TYPE_TURBO3_TCQ) {
        static bool cb3_loaded = false;
        if (!cb3_loaded) {
            cb3_loaded = true;
            const char * cb_path = getenv("TURBO_TCQ_CB");
            if (cb_path) {
                float cb[512];
                FILE * f = fopen(cb_path, "rb");
                if (f && fread(cb, sizeof(float), 512, f) == 512) {
                    fclose(f);
                    CUDA_CHECK(cudaMemcpyToSymbol(d_turbo3_tcq_codebook_fattn, cb, 512 * sizeof(float)));
                } else { if (f) fclose(f); }
            }
        }
    }
    if constexpr (type_K == GGML_TYPE_TURBO2_TCQ || type_V == GGML_TYPE_TURBO2_TCQ) {
        static bool cb2_loaded = false;
        if (!cb2_loaded) {
            cb2_loaded = true;
            const char * cb_path = getenv("TURBO_TCQ_CB2");
            if (cb_path) {
                float cb[256];
                FILE * f = fopen(cb_path, "rb");
                if (f && fread(cb, sizeof(float), 256, f) == 256) {
                    fclose(f);
                    CUDA_CHECK(cudaMemcpyToSymbol(d_turbo2_tcq_codebook_fattn, cb, 256 * sizeof(float)));
                } else { if (f) fclose(f); }
            }
        }
    }
    if constexpr (type_K == GGML_TYPE_TURBO3_TCQ || type_K == GGML_TYPE_TURBO2_TCQ ||
                  type_V == GGML_TYPE_TURBO3_TCQ || type_V == GGML_TYPE_TURBO2_TCQ) {
        const ggml_tensor * V = dst->src[2];
        const int64_t n_kv = V->ne[1] > 0 ? V->ne[1] : 1;
        const float ln_ctx = logf((float)n_kv);

        // V alpha: context-adaptive unless env var override
        float alpha_v = 1.0f;
        static float alpha_v_static = -1.0f;
        if (alpha_v_static < 0.0f) {
            alpha_v_static = 0.0f;
            const char * s = getenv("TURBO_TCQ_DECODE_ALPHA_V");
            if (s) {
                char * end;
                float a = strtof(s, &end);
                if (end != s && a > 0.0f && a < 10.0f) alpha_v_static = a;
            }
        }
        if (alpha_v_static > 0.0f) {
            alpha_v = alpha_v_static;
        } else if constexpr (type_V == GGML_TYPE_TURBO3_TCQ) {
            alpha_v = fmaxf(0.98f, fminf(1.06f, 1.1484f - 0.01443f * ln_ctx));
        } else if constexpr (type_V == GGML_TYPE_TURBO2_TCQ) {
            alpha_v = fmaxf(1.00f, fminf(1.12f, 0.8865f + 0.0195f * ln_ctx));
        }
        CUDA_CHECK(cudaMemcpyToSymbol(d_tcq_decode_alpha_v_fattn, &alpha_v, sizeof(float)));

        // K alpha: static (default 1.0, env var override)
        static float alpha_k = -1.0f;
        if (alpha_k < 0.0f) {
            alpha_k = 1.0f;
            const char * s = getenv("TURBO_TCQ_DECODE_ALPHA_K");
            if (s) {
                char * end;
                float a = strtof(s, &end);
                if (end != s && a > 0.0f && a < 10.0f) alpha_k = a;
            }
        }
        CUDA_CHECK(cudaMemcpyToSymbol(d_tcq_decode_alpha_k_fattn, &alpha_k, sizeof(float)));
    }

    // need_f16_K=false, need_f16_V=false: raw turbo data passes through to kernel.
    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, false, false, true, warp_size_host);
}


#define DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, ncols1, ncols2, tK, tV)                                  \
    template void ggml_cuda_flash_attn_ext_mma_turbo_case                                            \
    <DKQ, DV, ncols1, ncols2, tK, tV>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)            \

// Matched K/V at D=128 and D=256. ncols2 ≤ 8.
#define DECL_FATTN_MMA_TURBO_CASES_ALL_NCOLS2(DKQ, DV, ncols, tK, tV)           \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, (ncols)/1, 1, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, (ncols)/2, 2, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, (ncols)/4, 4, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, (ncols)/8, 8, tK, tV); \

#define DECL_FATTN_MMA_TURBO_ALL(DKQ, DV, tK, tV) \
    DECL_FATTN_MMA_TURBO_CASES_ALL_NCOLS2(DKQ, DV,  8, tK, tV) \
    DECL_FATTN_MMA_TURBO_CASES_ALL_NCOLS2(DKQ, DV, 16, tK, tV) \
    DECL_FATTN_MMA_TURBO_CASES_ALL_NCOLS2(DKQ, DV, 32, tK, tV) \
    DECL_FATTN_MMA_TURBO_CASES_ALL_NCOLS2(DKQ, DV, 64, tK, tV) \

DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO4_0,   GGML_TYPE_TURBO4_0)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO4_0,   GGML_TYPE_TURBO4_0)
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO8_0,   GGML_TYPE_TURBO8_0)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO8_0,   GGML_TYPE_TURBO8_0)
// Asymmetric "q6 sweet spot" (6.124 bpw): turbo8 K + turbo4 V, D=256 only.
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO8_0,   GGML_TYPE_TURBO4_0)
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_TCQ)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_TCQ)
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO2_TCQ)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO2_TCQ)
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO3_0,   GGML_TYPE_TURBO3_0)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO3_0,   GGML_TYPE_TURBO3_0)
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO2_0,   GGML_TYPE_TURBO2_0)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO2_0,   GGML_TYPE_TURBO2_0)
