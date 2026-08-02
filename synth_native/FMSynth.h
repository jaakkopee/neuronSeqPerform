#pragma once
/*  FMSynth.h  –  C++ interface to the Metal-backed FM synthesiser.
 *
 *  This header is included by both FMSynthMetal.mm (implementation)
 *  and bindings.cpp (pybind11 glue).  Keep it dependency-free.
 */

#include <cstdint>
#include <memory>

class FMSynth {
public:
    static constexpr int COLS      = 16;  // presets / sequencer columns
    static constexpr int NUM_OPS   = 8;   // operators per preset
    static constexpr int MAX_PAIRS = 4;   // carrier-mod pairs per preset

    // ── lifetime ──────────────────────────────────────────────────────────────
    explicit FMSynth(int sample_rate = 44100, int buffer_size = 1024);
    ~FMSynth();

    FMSynth(const FMSynth&)            = delete;
    FMSynth& operator=(const FMSynth&) = delete;

    // ── preset / tuning ───────────────────────────────────────────────────────
    /** Load one preset column.  All arrays have length NUM_OPS. */
    void load_preset(int col,
                     const float* ratios,
                     const float* levels,
                     const float* mod_indices);

    /** Set base frequency for a single column. */
    void set_base_freq(int col, float freq);

    /** Set all 16 base frequencies at once.  `freqs` has length COLS. */
    void set_all_base_freqs(const float* freqs);

    // ── step control ──────────────────────────────────────────────────────────
    /**
     * Atomically switch the active voice to `col` and gate its operator
     * envelopes from the spike array.
     *
     * spikes_float : (NUM_OPS × COLS) row-major float32
     *                1.0 = fired this tick, 0.0 = silent
     */
    void trigger_and_activate(const float* spikes_float, int col);

    /** Switch active voice without updating envelopes. */
    void set_active_step(int step);

    // ── parameter control ─────────────────────────────────────────────────────
    void  set_mod_index_scale(float scale);
    void  set_active_pairs(int n);
    void  set_decay_speed(float speed);   // 0 = sustain, 1 = staccato
    void  set_ratio_scale(float scale);
    void  set_master_volume(float vol);
    float get_master_volume() const;

    // ── audio generation ──────────────────────────────────────────────────────
    /**
     * Render `frames` samples into `output` (float32, mono).
     * Dispatches a Metal compute shader; returns when GPU is done.
     * Lock-free: safe to call from a real-time audio callback.
     */
    void generate(float* output, int frames);

    // ── display ───────────────────────────────────────────────────────────────
    /**
     * Fill `out` with operator frequencies.
     * Layout: (NUM_OPS × COLS) row-major float32.
     */
    void get_operator_frequencies(float* out) const;

private:
    // CPU-side struct layout – must exactly mirror the Metal PairParams struct.
    // Declared here so bindings.cpp can see the constants.
    struct PairParams {
        float freq_c, freq_m;
        float level_start, level_end;
        float mi_start, mi_end;
        float phase_c, phase_m;
    };  // 32 bytes

    struct VoiceParams {
        PairParams pairs[4];   // 128 bytes
        float   gain_start;
        float   gain_end;
        uint32_t n_pairs;
        float   _pad;
    };  // 144 bytes

    struct GenParams {
        float    inv_sr;
        float    master_volume;
        uint32_t n_frames;
        uint32_t _pad;
    };  // 16 bytes

    struct Impl;
    std::unique_ptr<Impl> impl_;
};
