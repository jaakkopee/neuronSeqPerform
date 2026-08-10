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
#include <vector>

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
    id<MTLBuffer>               outputBuf[2]  = {nil, nil};
    id<MTLBuffer>               voicesBuf[2]  = {nil, nil};
    id<MTLBuffer>               genParBuf[2]  = {nil, nil};
    id<MTLCommandBuffer>        inFlightCmd[2] = {nil, nil};

    int   sr;
    int   max_frames;

    // ── synthesis state ───────────────────────────────────────────────────────
    float phases[COLS][NUM_OPS]{};    // [col][op]  current phase (radians)
    float env[NUM_OPS][COLS]{};       // [op][col]  amplitude envelope 0-1
    float env_target[NUM_OPS][COLS]{}; // [op][col] trigger target 0-1
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
    float env_attack_carrier = 0.08f;  // ~280ms attack to prevent crackles from many neurons
    float env_attack_mod     = 0.08f;  // ~280ms attack to prevent crackles from many neurons
    int   active_pairs    = 2;
    float master_volume   = 0.5f;
    float last_output_sample = 0.0f;
    int   output_declick_samples = 24;
    bool  slotBusy[2] = {false, false};
    bool  slotHadError[2] = {false, false};
    int   inFlightFrames[2] = {0, 0};
    int   pendingSlots[2] = {-1, -1};
    int   pendingCount = 0;
    std::vector<float> last_good_output;

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
        for (int i = 0; i < 2; ++i) {
            outputBuf[i] = [device newBufferWithLength:outBytes    options:shm];
            voicesBuf[i] = [device newBufferWithLength:voicesBytes options:shm];
            genParBuf[i] = [device newBufferWithLength:parBytes    options:shm];
            inFlightCmd[i] = nil;
            slotBusy[i] = false;
            slotHadError[i] = false;
            inFlightFrames[i] = 0;
        }
        pendingSlots[0] = -1;
        pendingSlots[1] = -1;
        pendingCount = 0;
        last_good_output.clear();
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
        impl_->env_target[op][col] = std::clamp(spikes_float[op * COLS + col], 0.0f, 1.0f);
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

        if (frames <= 0) {
            return;
        }

        const int req_frames = frames;
        const int render_frames = std::min(req_frames, d.max_frames);

        auto is_pending = [&](int slot) -> bool {
            for (int i = 0; i < d.pendingCount; ++i) {
                if (d.pendingSlots[i] == slot) return true;
            }
            return false;
        };

        auto pop_pending_front = [&]() {
            if (d.pendingCount <= 0) return;
            d.pendingSlots[0] = d.pendingSlots[1];
            d.pendingSlots[1] = -1;
            d.pendingCount -= 1;
        };

        // Poll completion state for submitted renders without blocking.
        for (int slot = 0; slot < 2; ++slot) {
            if (!d.slotBusy[slot]) continue;
            id<MTLCommandBuffer> cmd = d.inFlightCmd[slot];
            if (!cmd) {
                d.slotBusy[slot] = false;
                d.slotHadError[slot] = true;
                continue;
            }
            const MTLCommandBufferStatus st = cmd.status;
            if (st == MTLCommandBufferStatusCompleted) {
                d.slotBusy[slot] = false;
                d.slotHadError[slot] = false;
                d.inFlightCmd[slot] = nil;
            } else if (st == MTLCommandBufferStatusError) {
                d.slotBusy[slot] = false;
                d.slotHadError[slot] = true;
                d.inFlightCmd[slot] = nil;
            }
        }

        bool produced = false;

        // Consume the oldest finished slot (one-buffer latency pipeline).
        if (d.pendingCount > 0) {
            const int readSlot = d.pendingSlots[0];
            if (!d.slotBusy[readSlot]) {
                const bool badSlot = d.slotHadError[readSlot];
                const int rendered = std::max(0, d.inFlightFrames[readSlot]);
                if (!badSlot && rendered > 0) {
                    const int ncopy = std::min(req_frames, rendered);
                    std::memcpy(output,
                                [d.outputBuf[readSlot] contents],
                                static_cast<std::size_t>(ncopy) * sizeof(float));
                    if (ncopy < req_frames) {
                        std::memset(output + ncopy,
                                    0,
                                    static_cast<std::size_t>(req_frames - ncopy) * sizeof(float));
                    }
                    produced = true;
                }
                d.slotHadError[readSlot] = false;
                pop_pending_front();
            }
        }

        // Choose a free slot for next async submission.
        int submitSlot = -1;
        for (int slot = 0; slot < 2; ++slot) {
            if (!d.slotBusy[slot] && !is_pending(slot)) {
                submitSlot = slot;
                break;
            }
        }

        if (submitSlot >= 0 && render_frames > 0) {
            // ── 1. Build VoiceParams for the GPU (CPU, ~5 µs) ───────────────
            auto* vp_arr = static_cast<VoiceParams*>([d.voicesBuf[submitSlot] contents]);
            const float env_release_carrier = std::clamp(1.0f - d.per_buf_carrier, 0.001f, 1.0f);
            const float env_release_mod     = std::clamp(1.0f - d.per_buf_mod, 0.001f, 1.0f);

            for (int col = 0; col < COLS; ++col) {
                VoiceParams& vp = vp_arr[col];

                float gs     = d.voice_gain[col];
                float target = d.voice_active[col] ? 1.0f : 0.0f;
                // Smooth voice crossfade: ramp at 0.2 per buffer (200ms to fully switch)
                // This prevents zipper clicks when switching between voices.
                const float voice_ramp_rate = 0.2f;
                float ge = gs + (target - gs) * voice_ramp_rate;
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
                    const float tc  = d.env_target[c_op][col];
                    const float tm  = d.env_target[m_op][col];
                    const float ec_alpha = (tc > ec0) ? d.env_attack_carrier : env_release_carrier;
                    const float em_alpha = (tm > em0) ? d.env_attack_mod : env_release_mod;
                    const float ec1 = ec0 + (tc - ec0) * ec_alpha;
                    const float em1 = em0 + (tm - em0) * em_alpha;

                    pp.level_start = d.levels[col][c_op] * ec0;
                    pp.level_end   = d.levels[col][c_op] * ec1;
                    pp.mi_start    = d.mod_idx[col][m_op] * d.mod_index_scale * em0;
                    pp.mi_end      = d.mod_idx[col][m_op] * d.mod_index_scale * em1;
                    pp.phase_c     = d.phases[col][c_op];
                    pp.phase_m     = d.phases[col][m_op];

                    d.env[c_op][col] = ec1;
                    d.env[m_op][col] = em1;

                    constexpr float TWO_PI = 6.283185307f;
                    d.phases[col][c_op] =
                        std::fmod(d.phases[col][c_op]
                                  + TWO_PI * pp.freq_c / d.sr * render_frames, TWO_PI);
                    d.phases[col][m_op] =
                        std::fmod(d.phases[col][m_op]
                                  + TWO_PI * pp.freq_m / d.sr * render_frames, TWO_PI);
                }
            }

            // ── 2. GenParams ────────────────────────────────────────────────
            auto* gp = static_cast<GenParams*>([d.genParBuf[submitSlot] contents]);
            gp->inv_sr         = 1.0f / static_cast<float>(d.sr);
            gp->master_volume  = d.master_volume;
            gp->n_frames       = static_cast<uint32_t>(render_frames);
            gp->_pad           = 0;

            // ── 3. Dispatch asynchronously (do not wait here) ───────────────
            id<MTLCommandBuffer> cmdbuf = [d.cmdQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmdbuf computeCommandEncoder];

            [encoder setComputePipelineState:d.pipeline];
            [encoder setBuffer:d.outputBuf[submitSlot] offset:0 atIndex:0];
            [encoder setBuffer:d.voicesBuf[submitSlot] offset:0 atIndex:1];
            [encoder setBuffer:d.genParBuf[submitSlot] offset:0 atIndex:2];

            const NSUInteger maxTG = d.pipeline.maxTotalThreadsPerThreadgroup;
            const NSUInteger tgSz  = std::min(maxTG, static_cast<NSUInteger>(render_frames));
            [encoder dispatchThreads:MTLSizeMake(render_frames, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(tgSz, 1, 1)];
            [encoder endEncoding];

            [cmdbuf commit];

            d.inFlightCmd[submitSlot] = cmdbuf;
            d.slotBusy[submitSlot] = true;
            d.slotHadError[submitSlot] = false;
            d.inFlightFrames[submitSlot] = render_frames;
            if (d.pendingCount < 2) {
                d.pendingSlots[d.pendingCount++] = submitSlot;
            }
        }

        if (!produced) {
            if (static_cast<int>(d.last_good_output.size()) == req_frames) {
                std::memcpy(output,
                            d.last_good_output.data(),
                            static_cast<std::size_t>(req_frames) * sizeof(float));
            } else {
                std::memset(output, 0, static_cast<std::size_t>(req_frames) * sizeof(float));
            }
            return;
        }

        // De-click envelope at audio block boundaries to suppress zipper clicks.
        if (req_frames > 0) {
            const int n = std::min(req_frames, std::max(1, d.output_declick_samples));
            if (n > 1) {
                const float prev = d.last_output_sample;
                for (int i = 0; i < n; ++i) {
                    const float t = static_cast<float>(i) / static_cast<float>(n - 1);
                    output[i] = prev + (output[i] - prev) * t;
                }
            } else {
                output[0] = 0.5f * output[0] + 0.5f * d.last_output_sample;
            }
            d.last_output_sample = output[req_frames - 1];
            d.last_good_output.assign(output, output + req_frames);
        }
    }
}

void FMSynth::get_operator_frequencies(float* out) const {
    // Output layout: (NUM_OPS × COLS) row-major
    for (int op = 0; op < NUM_OPS; ++op)
        for (int col = 0; col < COLS; ++col)
            out[op * COLS + col] = impl_->frequencies[col][op];
}
