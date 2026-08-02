/*  FMSynthMetal.mm  –  Objective-C++ Metal implementation of FMSynth.
 *
 *  Build requirements (macOS 13+, Apple Silicon or AMD/Intel with Metal):
 *    clang++ -std=c++17 -fobjc-arc -framework Metal -framework Foundation
 *
 *  The .metal shader is compiled at run-time on first construction
 *  (MTLDevice::newLibraryWithSource).  On Apple Silicon this takes ~5 ms
 *  and is cached by the driver on subsequent runs.
 */

#import  <Metal/Metal.h>
#import  <Foundation/Foundation.h>
#include "FMSynth.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

// ── Metal shader source (embedded) ────────────────────────────────────────────
static constexpr const char* SHADER_SRC = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct PairParams {
    float freq_c, freq_m;
    float level_start, level_end;
    float mi_start, mi_end;
    float phase_c, phase_m;
};
struct VoiceParams {
    PairParams pairs[4];
    float  gain_start, gain_end;
    uint   n_pairs;
    float  _pad;
};
struct GenParams {
    float inv_sr, master_volume;
    uint  n_frames, _pad;
};

kernel void fm_generate(
    device       float*       output  [[ buffer(0) ]],
    device const VoiceParams* voices  [[ buffer(1) ]],
    constant     GenParams&   gp      [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid >= gp.n_frames) return;
    const float t      = float(gid);
    const float t_norm = (gp.n_frames > 1u) ? t / float(gp.n_frames - 1u) : 0.0f;
    const float TWO_PI = 2.0f * M_PI_F;
    float sample = 0.0f;

    for (uint v = 0u; v < 16u; v++) {
        const device VoiceParams& vp = voices[v];
        if (vp.n_pairs == 0u) continue;
        float gain = mix(vp.gain_start, vp.gain_end, t_norm);
        if (gain <= 0.0f) continue;
        float voice = 0.0f;
        for (uint p = 0u; p < vp.n_pairs; p++) {
            const device PairParams& pp = vp.pairs[p];
            float phi_c  = pp.phase_c + TWO_PI * pp.freq_c * gp.inv_sr * t;
            float phi_m  = pp.phase_m + TWO_PI * pp.freq_m * gp.inv_sr * t;
            float level  = mix(pp.level_start, pp.level_end, t_norm);
            float mi     = mix(pp.mi_start,    pp.mi_end,    t_norm);
            float mod    = (pp.freq_m > 0.0f) ? sin(phi_m) : 0.0f;
            voice += level * sin(phi_c + mi * mod);
        }
        voice /= float(vp.n_pairs);
        sample += voice * gain;
    }
    output[gid] = tanh(sample * 0.7f) * gp.master_volume;
}
)MSL";

// ── Impl ──────────────────────────────────────────────────────────────────────
struct FMSynth::Impl {
    // Metal objects (ARC-managed)
    id<MTLDevice>               device     = nil;
    id<MTLCommandQueue>         cmdQueue   = nil;
    id<MTLComputePipelineState> pipeline   = nil;
    id<MTLBuffer>               outputBuf  = nil;
    id<MTLBuffer>               voicesBuf  = nil;
    id<MTLBuffer>               genParBuf  = nil;

    int   sr;
    int   max_frames;

    // ── synthesis state ───────────────────────────────────────────────────────
    float phases[COLS][NUM_OPS]{};    // [col][op]  current phase (radians)
    float env[NUM_OPS][COLS]{};       // [op][col]  amplitude envelope 0-1
    float voice_gain[COLS]{};         // per-voice crossfade gain
    bool  voice_active[COLS]{};

    // ── preset tables ─────────────────────────────────────────────────────────
    float ratios[COLS][NUM_OPS]{};
    float levels[COLS][NUM_OPS]{};
    float mod_idx[COLS][NUM_OPS]{};
    float base_freqs[COLS]{};
    float frequencies[COLS][NUM_OPS]{};  // = base_freqs * ratios * ratio_scale
    float ratio_scale = 1.0f;

    // ── runtime params ────────────────────────────────────────────────────────
    float mod_index_scale = 0.25f;
    float per_buf_carrier = 0.9286f;
    float per_buf_mod     = 0.8700f;
    int   active_pairs    = 2;
    float master_volume   = 0.5f;

    void init_metal();
    void recompute_freqs();
};

void FMSynth::Impl::init_metal() {
    @autoreleasepool {
        device = MTLCreateSystemDefaultDevice();
        if (!device)
            throw std::runtime_error("[FMSynth] No Metal device available");

        cmdQueue = [device newCommandQueue];

        NSError*  err = nil;
        NSString* src = [NSString stringWithUTF8String:SHADER_SRC];
        id<MTLLibrary> lib = [device newLibraryWithSource:src
                                                  options:nil
                                                    error:&err];
        if (!lib)
            throw std::runtime_error(
                std::string("[FMSynth] Shader compile error: ")
                + err.localizedDescription.UTF8String);

        id<MTLFunction> fn = [lib newFunctionWithName:@"fm_generate"];
        if (!fn)
            throw std::runtime_error("[FMSynth] fm_generate not found in library");

        pipeline = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pipeline)
            throw std::runtime_error(
                std::string("[FMSynth] Pipeline error: ")
                + err.localizedDescription.UTF8String);

        // Shared-memory buffers (unified memory on Apple Silicon = zero-copy)
        const NSUInteger outBytes    = (NSUInteger)(max_frames * sizeof(float));
        const NSUInteger voicesBytes = 16 * sizeof(VoiceParams);
        const NSUInteger parBytes    = sizeof(GenParams);

        const MTLResourceOptions shm = MTLResourceStorageModeShared;
        outputBuf = [device newBufferWithLength:outBytes    options:shm];
        voicesBuf = [device newBufferWithLength:voicesBytes options:shm];
        genParBuf = [device newBufferWithLength:parBytes    options:shm];
    }
}

void FMSynth::Impl::recompute_freqs() {
    for (int col = 0; col < COLS; ++col)
        for (int op = 0; op < NUM_OPS; ++op)
            frequencies[col][op] = base_freqs[col] * ratios[col][op] * ratio_scale;
}

// ── FMSynth public API ────────────────────────────────────────────────────────
FMSynth::FMSynth(int sample_rate, int buffer_size)
    : impl_(std::make_unique<Impl>())
{
    impl_->sr         = sample_rate;
    impl_->max_frames = buffer_size;
    impl_->init_metal();

    // Default: chromatic scale from C4
    constexpr float C4 = 261.63f;
    for (int i = 0; i < COLS; ++i)
        impl_->base_freqs[i] = C4 * std::pow(2.0f, i / 12.0f);
}

FMSynth::~FMSynth() {}

void FMSynth::load_preset(int col,
                           const float* ratios,
                           const float* levels,
                           const float* mod_indices) {
    for (int op = 0; op < NUM_OPS; ++op) {
        impl_->ratios[col][op]  = ratios[op];
        impl_->levels[col][op]  = levels[op];
        impl_->mod_idx[col][op] = mod_indices[op];
        impl_->frequencies[col][op] =
            impl_->base_freqs[col] * ratios[op] * impl_->ratio_scale;
    }
}

void FMSynth::set_base_freq(int col, float freq) {
    impl_->base_freqs[col] = freq;
    for (int op = 0; op < NUM_OPS; ++op)
        impl_->frequencies[col][op] =
            freq * impl_->ratios[col][op] * impl_->ratio_scale;
}

void FMSynth::set_all_base_freqs(const float* freqs) {
    for (int col = 0; col < COLS; ++col) {
        impl_->base_freqs[col] = freqs[col];
        for (int op = 0; op < NUM_OPS; ++op)
            impl_->frequencies[col][op] =
                freqs[col] * impl_->ratios[col][op] * impl_->ratio_scale;
    }
}

void FMSynth::trigger_and_activate(const float* spikes_float, int col) {
    for (int i = 0; i < COLS; ++i)
        impl_->voice_active[i] = (i == col);
    for (int op = 0; op < NUM_OPS; ++op)
        impl_->env[op][col] = spikes_float[op * COLS + col];
}

void FMSynth::set_active_step(int step) {
    for (int i = 0; i < COLS; ++i)
        impl_->voice_active[i] = (i == step);
}

void FMSynth::set_mod_index_scale(float s)  { impl_->mod_index_scale = s; }
void FMSynth::set_master_volume(float v)    { impl_->master_volume   = v; }
float FMSynth::get_master_volume() const    { return impl_->master_volume; }

void FMSynth::set_active_pairs(int n) {
    impl_->active_pairs = std::max(1, std::min(MAX_PAIRS, n));
}

void FMSynth::set_decay_speed(float speed) {
    float s = std::max(0.0f, std::min(1.0f, speed));
    impl_->per_buf_carrier = 0.9953f - s * (0.9953f - 0.862f);
    impl_->per_buf_mod     = 0.9900f - s * (0.9900f - 0.750f);
}

void FMSynth::set_ratio_scale(float scale) {
    impl_->ratio_scale = scale;
    impl_->recompute_freqs();
}

// ── generate  (called from the real-time audio callback) ──────────────────────
void FMSynth::generate(float* output, int frames) {
    @autoreleasepool {
        auto& d = *impl_;

        // ── 1. Build VoiceParams for the GPU (CPU, ~5 µs) ────────────────────
        auto* vp_arr = static_cast<VoiceParams*>(d.voicesBuf.contents);

        for (int col = 0; col < COLS; ++col) {
            VoiceParams& vp = vp_arr[col];

            // Voice crossfade gain ramp
            float gs     = d.voice_gain[col];
            float target = d.voice_active[col] ? 1.0f : 0.0f;
            float ge     = (target > gs)
                           ? std::min(1.0f, gs + 1.0f)
                           : std::max(0.0f, gs - 1.0f);
            d.voice_gain[col] = ge;

            vp.gain_start = gs;
            vp.gain_end   = ge;
            vp.n_pairs    = (gs == 0.0f && ge == 0.0f) ? 0u
                            : static_cast<uint32_t>(d.active_pairs);
            vp._pad       = 0.0f;

            for (int pair = 0; pair < d.active_pairs; ++pair) {
                const int c_op = pair * 2;
                const int m_op = pair * 2 + 1;
                PairParams& pp = vp.pairs[pair];

                pp.freq_c = d.frequencies[col][c_op];
                pp.freq_m = d.frequencies[col][m_op];

                const float ec0 = d.env[c_op][col];
                const float em0 = d.env[m_op][col];
                const float ec1 = ec0 * d.per_buf_carrier;
                const float em1 = em0 * d.per_buf_mod;

                pp.level_start = d.levels[col][c_op] * ec0;
                pp.level_end   = d.levels[col][c_op] * ec1;
                pp.mi_start    = d.mod_idx[col][m_op] * d.mod_index_scale * em0;
                pp.mi_end      = d.mod_idx[col][m_op] * d.mod_index_scale * em1;
                pp.phase_c     = d.phases[col][c_op];
                pp.phase_m     = d.phases[col][m_op];

                // Commit decayed env + advanced phases (CPU, used next buffer)
                d.env[c_op][col] = ec1;
                d.env[m_op][col] = em1;

                constexpr float TWO_PI = 6.283185307f;
                d.phases[col][c_op] =
                    std::fmod(d.phases[col][c_op]
                              + TWO_PI * pp.freq_c / d.sr * frames, TWO_PI);
                d.phases[col][m_op] =
                    std::fmod(d.phases[col][m_op]
                              + TWO_PI * pp.freq_m / d.sr * frames, TWO_PI);
            }
        }

        // ── 2. GenParams ──────────────────────────────────────────────────────
        auto* gp   = static_cast<GenParams*>(d.genParBuf.contents);
        gp->inv_sr         = 1.0f / static_cast<float>(d.sr);
        gp->master_volume  = d.master_volume;
        gp->n_frames       = static_cast<uint32_t>(frames);
        gp->_pad           = 0;

        // ── 3. Dispatch Metal compute shader (GPU, ~20-50 µs on M-series) ─────
        id<MTLCommandBuffer>         cmdbuf  = [d.cmdQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdbuf computeCommandEncoder];

        [encoder setComputePipelineState:d.pipeline];
        [encoder setBuffer:d.outputBuf  offset:0 atIndex:0];
        [encoder setBuffer:d.voicesBuf  offset:0 atIndex:1];
        [encoder setBuffer:d.genParBuf  offset:0 atIndex:2];

        const NSUInteger maxTG = d.pipeline.maxTotalThreadsPerThreadgroup;
        const NSUInteger tgSz  = std::min(maxTG, static_cast<NSUInteger>(frames));
        [encoder dispatchThreads:MTLSizeMake(frames, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tgSz, 1, 1)];
        [encoder endEncoding];

        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];   // ~30-80 µs total on Apple Silicon

        // ── 4. Copy result (shared memory = no actual copy on Apple Silicon) ──
        std::memcpy(output,
                    d.outputBuf.contents,
                    static_cast<std::size_t>(frames) * sizeof(float));
    }
}

void FMSynth::get_operator_frequencies(float* out) const {
    // Output layout: (NUM_OPS × COLS) row-major
    for (int op = 0; op < NUM_OPS; ++op)
        for (int col = 0; col < COLS; ++col)
            out[op * COLS + col] = impl_->frequencies[col][op];
}
