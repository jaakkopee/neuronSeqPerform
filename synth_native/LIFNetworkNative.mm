#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "LIFNetworkNative.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <simd/simd.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct alignas(16) LIFStepParams {
    uint32_t neuronCount = 0;
    float dt = 1.0f;
    float tau = 20.0f;
    float threshold = 1.0f;
    float reset = 0.0f;
    float refractoryMs = 5.0f;
    float globalDrive = 0.8f;
    float _pad = 0.0f;
};

static constexpr const char* LIF_SHADER_SRC = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct LIFStepParams {
    uint neuronCount;
    float dt;
    float tau;
    float threshold;
    float reset;
    float refractoryMs;
    float globalDrive;
    float _pad;
};

kernel void lif_step_native(
    device const float4* prevState [[ buffer(0) ]],
    device float4*       nextState [[ buffer(1) ]],
    device const float*  weights   [[ buffer(2) ]],
    device const float*  drive     [[ buffer(3) ]],
    constant LIFStepParams& p      [[ buffer(4) ]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.neuronCount) return;

    const float4 prev = prevState[i];
    float membrane = prev.x;
    float refractoryLeft = max(0.0f, prev.z - p.dt);

    float syn = 0.0f;
    const device float* row = weights + i * p.neuronCount;
    for (uint j = 0; j < p.neuronCount; ++j)
        syn += row[j] * prevState[j].y;

    float spike = 0.0f;
    if (refractoryLeft <= 0.0f) {
        const float inputI = drive[i] + p.globalDrive + syn;
        membrane += (-(membrane) + inputI) * (p.dt / max(p.tau, 0.001f));
        membrane = clamp(membrane, -0.5f, 2.5f);
        if (membrane >= p.threshold) {
            spike = 1.0f;
            membrane = p.reset;
            refractoryLeft = p.refractoryMs;
        }
    } else {
        membrane = max(p.reset, membrane - (p.dt / max(p.tau, 0.001f)) * 0.5f);
    }

    nextState[i] = float4(membrane, spike, refractoryLeft, 0.0f);
}
)MSL";

inline LIFNetworkNative::Topology clamp_topology(int idx) {
    idx = std::clamp(idx, 0, 4);
    return static_cast<LIFNetworkNative::Topology>(idx);
}

} // namespace

struct LIFNetworkNative::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> cmdQueue = nil;
    id<MTLComputePipelineState> pipeline = nil;

    id<MTLBuffer> stateBuf[2] = {nil, nil}; // float4 membrane/spike/refractory/pad
    id<MTLBuffer> weightBuf = nil;          // float neuronCount*neuronCount
    id<MTLBuffer> driveBuf = nil;           // float neuronCount
    id<MTLBuffer> paramBuf = nil;           // LIFStepParams

    int neuronCount = 512;
    int cols = 16;
    int rows = 32;
    int readIndex = 0;

    Topology topology = Topology::Ring;
    float weightScale = 1.0f;

    std::mt19937 rng{1337};

    void init_metal() {
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (!device) {
                throw std::runtime_error("[LIF] No Metal device available");
            }
            cmdQueue = [device newCommandQueue];

            NSError* err = nil;
            NSString* src = [NSString stringWithUTF8String:LIF_SHADER_SRC];
            id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
            if (!lib) {
                throw std::runtime_error(
                    std::string("[LIF] Shader compile failed: ") + err.localizedDescription.UTF8String);
            }

            id<MTLFunction> fn = [lib newFunctionWithName:@"lif_step_native"];
            if (!fn) {
                throw std::runtime_error("[LIF] lif_step_native kernel not found");
            }

            pipeline = [device newComputePipelineStateWithFunction:fn error:&err];
            if (!pipeline) {
                throw std::runtime_error(
                    std::string("[LIF] Pipeline creation failed: ") + err.localizedDescription.UTF8String);
            }
        }
    }

    void allocate_resources() {
        rows = (neuronCount + cols - 1) / cols;

        const NSUInteger stateBytes = static_cast<NSUInteger>(neuronCount) * sizeof(simd_float4);
        const NSUInteger weightsBytes = static_cast<NSUInteger>(neuronCount) * static_cast<NSUInteger>(neuronCount) * sizeof(float);
        const NSUInteger driveBytes = static_cast<NSUInteger>(neuronCount) * sizeof(float);

        const MTLResourceOptions shm = MTLResourceStorageModeShared;
        stateBuf[0] = [device newBufferWithLength:stateBytes options:shm];
        stateBuf[1] = [device newBufferWithLength:stateBytes options:shm];
        weightBuf = [device newBufferWithLength:weightsBytes options:shm];
        driveBuf = [device newBufferWithLength:driveBytes options:shm];
        paramBuf = [device newBufferWithLength:sizeof(LIFStepParams) options:shm];
        readIndex = 0;

        if (driveBuf) {
            std::memset([driveBuf contents], 0, driveBytes);
        }
    }

    void seed_state() {
        if (!stateBuf[0] || !stateBuf[1]) return;

        std::uniform_real_distribution<float> dist(0.02f, 0.22f);
        auto* a = static_cast<simd_float4*>([stateBuf[0] contents]);
        auto* b = static_cast<simd_float4*>([stateBuf[1] contents]);
        for (int i = 0; i < neuronCount; ++i) {
            simd_float4 v{dist(rng), 0.0f, 0.0f, 0.0f};
            a[i] = v;
            b[i] = v;
        }
    }

    void rebuild_weights() {
        if (!weightBuf) return;

        std::vector<float> w(static_cast<size_t>(neuronCount) * static_cast<size_t>(neuronCount), 0.0f);
        auto at = [&](int from, int to) -> float& {
            return w[static_cast<size_t>(from) * static_cast<size_t>(neuronCount) + static_cast<size_t>(to)];
        };

        std::uniform_real_distribution<float> weightDist(0.05f, 0.22f);
        std::uniform_real_distribution<float> rewireDist(0.0f, 1.0f);
        std::uniform_int_distribution<int> pickNeuron(0, neuronCount - 1);

        switch (topology) {
            case Topology::Ring:
                for (int i = 0; i < neuronCount; ++i) {
                    for (int hop : {1, 2, 4}) {
                        at(i, (i + hop) % neuronCount) = 0.18f / static_cast<float>(hop);
                        at(i, (i - hop + neuronCount) % neuronCount) = 0.18f / static_cast<float>(hop);
                    }
                }
                break;
            case Topology::FullyConnected:
                for (int i = 0; i < neuronCount; ++i) {
                    for (int j = 0; j < neuronCount; ++j) {
                        if (i != j) {
                            at(i, j) = 0.12f / std::sqrt(static_cast<float>(neuronCount));
                        }
                    }
                }
                break;
            case Topology::Feedforward: {
                const int layers = 4;
                const int layerSize = std::max(1, neuronCount / layers);
                for (int i = 0; i < neuronCount; ++i) {
                    int srcLayer = std::min(i / layerSize, layers - 1);
                    for (int j = 0; j < neuronCount; ++j) {
                        int dstLayer = std::min(j / layerSize, layers - 1);
                        if (dstLayer == srcLayer + 1) {
                            at(i, j) = 0.20f;
                        }
                    }
                }
                break;
            }
            case Topology::SparseRandom:
                for (int i = 0; i < neuronCount; ++i) {
                    for (int j = 0; j < neuronCount; ++j) {
                        if (i != j && rewireDist(rng) < 0.10f) {
                            at(i, j) = weightDist(rng);
                        }
                    }
                }
                break;
            case Topology::SmallWorld:
                for (int i = 0; i < neuronCount; ++i) {
                    for (int hop : {1, 2, 3}) {
                        int target = (i + hop) % neuronCount;
                        if (rewireDist(rng) < 0.05f) {
                            target = pickNeuron(rng);
                        }
                        at(i, target) = 0.16f / static_cast<float>(hop);
                    }
                    for (int k = 0; k < 2; ++k) {
                        at(i, pickNeuron(rng)) = weightDist(rng);
                    }
                }
                break;
        }

        for (float& v : w) {
            v *= weightScale;
        }

        std::memcpy([weightBuf contents], w.data(), w.size() * sizeof(float));
    }

    void write_default_params() {
        auto* p = static_cast<LIFStepParams*>([paramBuf contents]);
        p->neuronCount = static_cast<uint32_t>(neuronCount);
        p->dt = 1.0f;
        p->tau = 20.0f;
        p->threshold = 1.0f;
        p->reset = 0.0f;
        p->refractoryMs = 5.0f;
        p->globalDrive = 0.8f;
        p->_pad = 0.0f;
    }
};

LIFNetworkNative::LIFNetworkNative(int neuron_count, int cols)
    : impl_(std::make_unique<Impl>()) {
    impl_->cols = std::max(4, cols);
    impl_->neuronCount = std::max(64, neuron_count);
    impl_->init_metal();
    impl_->allocate_resources();
    impl_->write_default_params();
    impl_->rebuild_weights();
    impl_->seed_state();
}

LIFNetworkNative::~LIFNetworkNative() = default;

void LIFNetworkNative::set_topology(int topology_index) {
    Topology t = clamp_topology(topology_index);
    if (impl_->topology == t) return;
    impl_->topology = t;
    impl_->rebuild_weights();
}

int LIFNetworkNative::topology() const {
    return static_cast<int>(impl_->topology);
}

void LIFNetworkNative::set_neuron_count(int neuron_count) {
    neuron_count = std::max(64, neuron_count);
    if (impl_->neuronCount == neuron_count) return;
    impl_->neuronCount = neuron_count;
    impl_->allocate_resources();
    impl_->write_default_params();
    impl_->rebuild_weights();
    impl_->seed_state();
}

int LIFNetworkNative::neuron_count() const { return impl_->neuronCount; }
int LIFNetworkNative::cols() const { return impl_->cols; }
int LIFNetworkNative::rows() const { return impl_->rows; }

void LIFNetworkNative::set_threshold(float threshold) {
    auto* p = static_cast<LIFStepParams*>([impl_->paramBuf contents]);
    p->threshold = std::clamp(threshold, 0.05f, 3.0f);
}

void LIFNetworkNative::set_tau(float tau) {
    auto* p = static_cast<LIFStepParams*>([impl_->paramBuf contents]);
    p->tau = std::clamp(tau, 1.0f, 200.0f);
}

void LIFNetworkNative::set_refractory_ms(float refractory_ms) {
    auto* p = static_cast<LIFStepParams*>([impl_->paramBuf contents]);
    p->refractoryMs = std::clamp(refractory_ms, 0.0f, 40.0f);
}

void LIFNetworkNative::set_weight_scale(float scale) {
    impl_->weightScale = std::clamp(scale, 0.0f, 6.0f);
    impl_->rebuild_weights();
}

void LIFNetworkNative::set_global_drive(float drive) {
    auto* p = static_cast<LIFStepParams*>([impl_->paramBuf contents]);
    p->globalDrive = std::clamp(drive, 0.0f, 3.0f);
}

void LIFNetworkNative::set_external_drive(const float* values, int len) {
    if (!values || !impl_->driveBuf) return;
    auto* dst = static_cast<float*>([impl_->driveBuf contents]);
    const int n = std::min(len, impl_->neuronCount);
    std::memcpy(dst, values, static_cast<size_t>(n) * sizeof(float));
    if (n < impl_->neuronCount) {
        std::memset(dst + n, 0, static_cast<size_t>(impl_->neuronCount - n) * sizeof(float));
    }
}

void LIFNetworkNative::set_neuron_drive(int row, int col, float value) {
    const int idx = row * impl_->cols + col;
    if (idx < 0 || idx >= impl_->neuronCount || !impl_->driveBuf) return;
    auto* dst = static_cast<float*>([impl_->driveBuf contents]);
    dst[idx] = value;
}

void LIFNetworkNative::randomize_weights() {
    impl_->rng.seed(std::random_device{}());
    impl_->rebuild_weights();
}

void LIFNetworkNative::reset_state() {
    impl_->seed_state();
}

void LIFNetworkNative::step() {
    @autoreleasepool {
        auto& d = *impl_;
        auto* p = static_cast<LIFStepParams*>([d.paramBuf contents]);
        p->neuronCount = static_cast<uint32_t>(d.neuronCount);

        const int writeIndex = 1 - d.readIndex;

        id<MTLCommandBuffer> cmdbuf = [d.cmdQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];

        [enc setComputePipelineState:d.pipeline];
        [enc setBuffer:d.stateBuf[d.readIndex] offset:0 atIndex:0];
        [enc setBuffer:d.stateBuf[writeIndex] offset:0 atIndex:1];
        [enc setBuffer:d.weightBuf offset:0 atIndex:2];
        [enc setBuffer:d.driveBuf offset:0 atIndex:3];
        [enc setBuffer:d.paramBuf offset:0 atIndex:4];

        const NSUInteger maxTG = d.pipeline.maxTotalThreadsPerThreadgroup;
        const NSUInteger tgSize = std::min(maxTG, static_cast<NSUInteger>(d.neuronCount));

        [enc dispatchThreads:MTLSizeMake(d.neuronCount, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        [enc endEncoding];

        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        d.readIndex = writeIndex;
    }
}

void LIFNetworkNative::get_spikes(float* out) const {
    const int padded = impl_->rows * impl_->cols;
    std::fill(out, out + padded, 0.0f);

    const auto* state = static_cast<const simd_float4*>([impl_->stateBuf[impl_->readIndex] contents]);
    for (int i = 0; i < impl_->neuronCount; ++i) {
        out[i] = state[i].y > 0.5f ? 1.0f : 0.0f;
    }
}

void LIFNetworkNative::get_potentials(float* out) const {
    const int padded = impl_->rows * impl_->cols;
    std::fill(out, out + padded, 0.0f);

    const auto* p = static_cast<const LIFStepParams*>([impl_->paramBuf contents]);
    const float denom = std::max(p->threshold, 0.001f);
    const auto* state = static_cast<const simd_float4*>([impl_->stateBuf[impl_->readIndex] contents]);
    for (int i = 0; i < impl_->neuronCount; ++i) {
        out[i] = std::clamp(state[i].x / denom, 0.0f, 1.0f);
    }
}
