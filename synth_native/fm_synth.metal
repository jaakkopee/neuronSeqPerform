/*  fm_synth.metal  –  Metal compute shader for the FM synthesiser
 *
 *  Each GPU thread computes one output sample.
 *  16 voices (presets), up to 4 carrier-modulator pairs per voice.
 *  Linear gain ramp + linear env ramp within every buffer → click-free.
 *
 *  Parallelism:   samples   (1024 threads for BUFFER_SIZE=1024)
 *  Loop overhead: voices×pairs  (at most 16×4 = 64 iterations per thread)
 */

#include <metal_stdlib>
using namespace metal;

// ── data layout  (must match CPU structs in FMSynth.h) ────────────────────────
struct PairParams {
    float freq_c;        // carrier frequency  (Hz)
    float freq_m;        // modulator frequency (Hz)
    float level_start;   // preset_level × env_carrier  at buffer start
    float level_end;     // preset_level × env_carrier  at buffer end
    float mi_start;      // preset_MI × env_mod × mi_scale  at buffer start
    float mi_end;        // ditto at buffer end
    float phase_c;       // accumulated carrier  phase (rad)
    float phase_m;       // accumulated modulator phase (rad)
};  // 8 × 4 = 32 bytes

struct VoiceParams {
    PairParams pairs[4]; // 4 × 32 = 128 bytes
    float  gain_start;
    float  gain_end;
    uint   n_pairs;      // 0 → skip this voice entirely
    float  _pad;
};  // 128 + 16 = 144 bytes

struct GenParams {
    float inv_sr;         // 1.0 / sample_rate
    float master_volume;
    uint  n_frames;
    uint  _pad;
};  // 16 bytes

// ── kernel ────────────────────────────────────────────────────────────────────
kernel void fm_generate(
    device       float*       output  [[ buffer(0) ]],
    device const VoiceParams* voices  [[ buffer(1) ]],
    constant     GenParams&   gp      [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid >= gp.n_frames) return;

    const float t      = float(gid);
    const float t_norm = (gp.n_frames > 1u)
                         ? t / float(gp.n_frames - 1u) : 0.0f;
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

            float phi_c = pp.phase_c + TWO_PI * pp.freq_c * gp.inv_sr * t;
            float phi_m = pp.phase_m + TWO_PI * pp.freq_m * gp.inv_sr * t;

            float level = mix(pp.level_start, pp.level_end, t_norm);
            float mi    = mix(pp.mi_start,    pp.mi_end,    t_norm);

            float mod_sig = (pp.freq_m > 0.0f) ? sin(phi_m) : 0.0f;
            voice += level * sin(phi_c + mi * mod_sig);
        }

        voice /= float(vp.n_pairs);
        sample += voice * gain;
    }

    // soft limiting + master volume
    output[gid] = tanh(sample * 0.7f) * gp.master_volume;
}
