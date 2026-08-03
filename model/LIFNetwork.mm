#include "LIFNetwork.h"

#include "Constants.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <simd/simd.h>
#include <vector>

namespace {
struct alignas(16) LIFSimParams {
    uint32_t neuronCount = 0;
    uint32_t gridSize = 0;
    float dt = 0.016f;
    float leak = 0.42f;
    float threshold = 0.58f;
    float reset = 0.08f;
    float refractory = 0.06f;
    float rms = 0.0f;
    float timeSeconds = 0.0f;
};
}

LIFNetwork::LIFNetwork() = default;
LIFNetwork::~LIFNetwork() = default;

bool LIFNetwork::init(id<MTLDevice> device,
                      id<MTLCommandQueue> cmdQueue,
                      id<MTLLibrary> library,
                      Topology topology,
                      int neuronCount) {
    device_ = device;
    cmdQueue_ = cmdQueue;
    library_ = library;
    topology_ = topology;
    neuronCount_ = std::max(64, neuronCount);

    psoStep_ = makePSO(@"lif_step");
    psoToTexture_ = makePSO(@"lif_to_texture");
    if (!psoStep_ || !psoToTexture_)
        return false;

    allocateResources();
    rebuildWeights();
    seedState();
    return stateTex_ != nil;
}

void LIFNetwork::setTopology(Topology topology) {
    if (topology_ == topology)
        return;
    topology_ = topology;
    rebuildWeights();
}

void LIFNetwork::setNeuronCount(int neuronCount) {
    neuronCount = std::max(64, neuronCount);
    if (neuronCount_ == neuronCount)
        return;
    neuronCount_ = neuronCount;
    allocateResources();
    rebuildWeights();
    seedState();
}

void LIFNetwork::allocateResources() {
    gridSize_ = static_cast<int>(std::ceil(std::sqrt(static_cast<float>(neuronCount_))));

    const NSUInteger stateBytes = static_cast<NSUInteger>(neuronCount_) * sizeof(simd::float4);
    const NSUInteger weightBytes = static_cast<NSUInteger>(neuronCount_) * static_cast<NSUInteger>(neuronCount_) * sizeof(float);
    const NSUInteger inputBytes = static_cast<NSUInteger>(neuronCount_) * sizeof(float);

    stateBuf_[0] = [device_ newBufferWithLength:stateBytes options:MTLResourceStorageModeShared];
    stateBuf_[1] = [device_ newBufferWithLength:stateBytes options:MTLResourceStorageModeShared];
    weightBuf_ = [device_ newBufferWithLength:weightBytes options:MTLResourceStorageModeShared];
    inputBuf_ = [device_ newBufferWithLength:inputBytes options:MTLResourceStorageModeShared];
    stateTex_ = makeStateTexture();
    readIndex_ = 0;

    if (inputBuf_)
        std::memset([inputBuf_ contents], 0, inputBytes);
}

void LIFNetwork::seedState() {
    if (!stateBuf_[0] || !stateBuf_[1])
        return;

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(0.02f, 0.18f);

    auto* a = static_cast<simd::float4*>([stateBuf_[0] contents]);
    auto* b = static_cast<simd::float4*>([stateBuf_[1] contents]);
    for (int i = 0; i < neuronCount_; ++i) {
        simd::float4 v{dist(rng), 0.0f, 0.0f, -1000.0f};
        a[i] = v;
        b[i] = v;
    }
}

void LIFNetwork::rebuildWeights() {
    if (!weightBuf_)
        return;

    std::vector<float> weights(static_cast<size_t>(neuronCount_) * static_cast<size_t>(neuronCount_), 0.0f);
    auto at = [&](int from, int to) -> float& {
        return weights[static_cast<size_t>(from) * static_cast<size_t>(neuronCount_) + static_cast<size_t>(to)];
    };

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> weightDist(0.05f, 0.22f);
    std::uniform_real_distribution<float> rewireDist(0.0f, 1.0f);
    std::uniform_int_distribution<int> pickNeuron(0, neuronCount_ - 1);

    switch (topology_) {
        case Topology::Ring:
            for (int i = 0; i < neuronCount_; ++i) {
                for (int hop : {1, 2, 4}) {
                    at(i, (i + hop) % neuronCount_) = 0.18f / static_cast<float>(hop);
                    at(i, (i - hop + neuronCount_) % neuronCount_) = 0.18f / static_cast<float>(hop);
                }
            }
            break;
        case Topology::FullyConnected:
            for (int i = 0; i < neuronCount_; ++i)
                for (int j = 0; j < neuronCount_; ++j)
                    if (i != j)
                        at(i, j) = 0.12f / std::sqrt(static_cast<float>(neuronCount_));
            break;
        case Topology::Feedforward: {
            const int layers = 4;
            const int layerSize = std::max(1, neuronCount_ / layers);
            for (int i = 0; i < neuronCount_; ++i) {
                int srcLayer = std::min(i / layerSize, layers - 1);
                for (int j = 0; j < neuronCount_; ++j) {
                    int dstLayer = std::min(j / layerSize, layers - 1);
                    if (dstLayer == srcLayer + 1)
                        at(i, j) = 0.20f;
                }
            }
            break;
        }
        case Topology::SparseRandom:
            for (int i = 0; i < neuronCount_; ++i)
                for (int j = 0; j < neuronCount_; ++j)
                    if (i != j && rewireDist(rng) < 0.10f)
                        at(i, j) = weightDist(rng);
            break;
        case Topology::SmallWorld:
            for (int i = 0; i < neuronCount_; ++i) {
                for (int hop : {1, 2, 3}) {
                    int target = (i + hop) % neuronCount_;
                    if (rewireDist(rng) < 0.05f)
                        target = pickNeuron(rng);
                    at(i, target) = 0.16f / static_cast<float>(hop);
                }
                for (int k = 0; k < 2; ++k)
                    at(i, pickNeuron(rng)) = weightDist(rng);
            }
            break;
    }

    std::memcpy([weightBuf_ contents], weights.data(), weights.size() * sizeof(float));
}

id<MTLTexture> LIFNetwork::makeStateTexture() const {
    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                           width:WORK_W
                                                          height:WORK_H
                                                       mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModePrivate;
    return [device_ newTextureWithDescriptor:desc];
}

id<MTLComputePipelineState> LIFNetwork::makePSO(NSString* kernelName) const {
    id<MTLFunction> fn = [library_ newFunctionWithName:kernelName];
    if (!fn)
        return nil;
    NSError* err = nil;
    return [device_ newComputePipelineStateWithFunction:fn error:&err];
}

void LIFNetwork::step(id<MTLCommandBuffer> cmdBuffer,
                      id<MTLTexture> sourceTex,
                      const std::array<float, 8>& bands,
                      const std::array<float, NUM_TONE_BINS>& transientBins,
                      float rms,
                      float influence,
                      float dt,
                      float timeSeconds) {
    if (!cmdBuffer || !sourceTex || !stateTex_ || !psoStep_ || !psoToTexture_)
        return;

    influence = std::clamp(influence, 0.0f, 1.0f);
    auto* input = static_cast<float*>([inputBuf_ contents]);
    const float influenceDrive = 0.35f + influence * 1.65f;
    for (int i = 0; i < neuronCount_; ++i) {
        int group = (i * static_cast<int>(bands.size())) / std::max(neuronCount_, 1);
        group = std::clamp(group, 0, static_cast<int>(bands.size()) - 1);
        int transientGroup = (i * static_cast<int>(transientBins.size())) / std::max(neuronCount_, 1);
        transientGroup = std::clamp(transientGroup, 0, static_cast<int>(transientBins.size()) - 1);
        float bandDrive = bands[group] * (0.30f + rms * 1.70f);
        float cross = bands[(group + 3) % static_cast<int>(bands.size())] * 0.15f;
        float transientDrive = transientBins[transientGroup] * (0.55f + influence * 1.20f);
        input[i] = (bandDrive + cross + transientDrive) * influenceDrive;
    }

    LIFSimParams sim;
    sim.neuronCount = static_cast<uint32_t>(neuronCount_);
    sim.gridSize = static_cast<uint32_t>(gridSize_);
    sim.dt = std::max(0.001f, dt);
    sim.leak = 0.70f - influence * 0.45f;
    sim.threshold = 0.75f - influence * 0.35f;
    sim.reset = 0.04f + (1.0f - influence) * 0.10f;
    sim.refractory = 0.03f + (1.0f - influence) * 0.08f;
    sim.rms = rms;
    sim.timeSeconds = timeSeconds;

    const int writeIndex = 1 - readIndex_;

    {
        id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
        [enc setComputePipelineState:psoStep_];
        [enc setTexture:sourceTex atIndex:0];
        [enc setBuffer:stateBuf_[readIndex_] offset:0 atIndex:0];
        [enc setBuffer:stateBuf_[writeIndex] offset:0 atIndex:1];
        [enc setBuffer:weightBuf_ offset:0 atIndex:2];
        [enc setBuffer:inputBuf_ offset:0 atIndex:3];
        [enc setBytes:&sim length:sizeof(sim) atIndex:4];

        MTLSize threads = {psoStep_.threadExecutionWidth, 1, 1};
        MTLSize groups = {(static_cast<NSUInteger>(neuronCount_) + threads.width - 1) / threads.width, 1, 1};
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
        [enc endEncoding];
    }

    {
        id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
        [enc setComputePipelineState:psoToTexture_];
        [enc setBuffer:stateBuf_[writeIndex] offset:0 atIndex:0];
        [enc setTexture:stateTex_ atIndex:0];
        [enc setBytes:&sim length:sizeof(sim) atIndex:1];
        MTLSize threads = {psoToTexture_.threadExecutionWidth, 8, 1};
        MTLSize groups = {(WORK_W + threads.width - 1) / threads.width,
                          (WORK_H + threads.height - 1) / threads.height,
                          1};
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
        [enc endEncoding];
    }

    readIndex_ = writeIndex;
}

std::array<float, LIFNetwork::NUM_TONE_BINS> LIFNetwork::sampleColumn(float phase01) const {
    std::array<float, NUM_TONE_BINS> bins{};
    if (!stateBuf_[readIndex_] || neuronCount_ <= 0 || gridSize_ <= 0)
        return bins;

    const auto* state = static_cast<const simd::float4*>([stateBuf_[readIndex_] contents]);
    if (!state)
        return bins;

    const int x = std::clamp(static_cast<int>(phase01 * static_cast<float>(gridSize_ - 1)), 0, gridSize_ - 1);
    for (int b = 0; b < NUM_TONE_BINS; ++b) {
        const float normY = (static_cast<float>(b) + 0.5f) / static_cast<float>(NUM_TONE_BINS);
        const int y = std::clamp(static_cast<int>(normY * static_cast<float>(gridSize_ - 1)), 0, gridSize_ - 1);
        const int idx = y * gridSize_ + x;
        if (idx >= neuronCount_) {
            bins[b] = 0.0f;
            continue;
        }
        // x=membrane potential, y=spike indicator; combine for a richer tone envelope.
        const float membrane = std::clamp(state[idx].x, 0.0f, 1.0f);
        const float spike = std::clamp(state[idx].y, 0.0f, 1.0f);
        bins[b] = std::clamp(membrane * 0.8f + spike * 0.6f, 0.0f, 1.0f);
    }
    return bins;
}