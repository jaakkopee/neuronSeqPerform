#pragma once

#include <cstdint>
#include <memory>

class LIFNetworkNative {
public:
    enum class Topology {
        Ring = 0,
        FullyConnected = 1,
        Feedforward = 2,
        SparseRandom = 3,
        SmallWorld = 4,
    };

    explicit LIFNetworkNative(int neuron_count = 512, int cols = 16);
    ~LIFNetworkNative();

    LIFNetworkNative(const LIFNetworkNative&) = delete;
    LIFNetworkNative& operator=(const LIFNetworkNative&) = delete;

    void set_topology(int topology_index);
    int topology() const;

    void set_neuron_count(int neuron_count);
    int neuron_count() const;

    int cols() const;
    int rows() const;

    void set_threshold(float threshold);
    void set_tau(float tau);
    void set_refractory_ms(float refractory_ms);
    void set_weight_scale(float scale);
    void set_global_drive(float drive);

    void set_external_drive(const float* values, int len);
    void set_neuron_drive(int row, int col, float value);

    void randomize_weights();
    void reset_state();

    void step();

    // Output layout is row-major (rows * cols), padded with zeros past neuron_count.
    void get_spikes(float* out) const;
    void get_potentials(float* out) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
