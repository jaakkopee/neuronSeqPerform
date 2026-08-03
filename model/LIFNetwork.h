#pragma once

#include <array>
#include <cstddef>

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
using MetalDeviceRef = id<MTLDevice>;
using MetalQueueRef = id<MTLCommandQueue>;
using MetalLibraryRef = id<MTLLibrary>;
using MetalCommandBufferRef = id<MTLCommandBuffer>;
using MetalTextureRef = id<MTLTexture>;
using MetalBufferRef = id<MTLBuffer>;
using MetalComputePsoRef = id<MTLComputePipelineState>;
using MetalStringRef = NSString*;
constexpr auto kMetalNil = nil;
#else
using MetalDeviceRef = void*;
using MetalQueueRef = void*;
using MetalLibraryRef = void*;
using MetalCommandBufferRef = void*;
using MetalTextureRef = void*;
using MetalBufferRef = void*;
using MetalComputePsoRef = void*;
using MetalStringRef = const char*;
constexpr std::nullptr_t kMetalNil = nullptr;
#endif

class LIFNetwork {
public:
    static constexpr int NUM_TONE_BINS = 16;

    enum class Topology {
        Ring = 0,
        FullyConnected = 1,
        Feedforward = 2,
        SparseRandom = 3,
        SmallWorld = 4,
    };

    LIFNetwork();
    ~LIFNetwork();

    bool init(MetalDeviceRef device,
              MetalQueueRef cmdQueue,
              MetalLibraryRef library,
              Topology topology,
              int neuronCount);

    void step(MetalCommandBufferRef cmdBuffer,
              MetalTextureRef sourceTex,
              const std::array<float, 8>& bands,
              const std::array<float, NUM_TONE_BINS>& transientBins,
              float rms,
              float influence,
              float dt,
              float timeSeconds);

    MetalTextureRef stateTexture() const { return stateTex_; }

    void setTopology(Topology topology);
    void setNeuronCount(int neuronCount);

    Topology topology() const { return topology_; }
    int neuronCount() const { return neuronCount_; }

    // Sample a vertical column of network activity.
    // phase01 selects horizontal position (0..1) and output bins map top->bottom rows.
    std::array<float, NUM_TONE_BINS> sampleColumn(float phase01) const;

private:
    MetalDeviceRef device_ = kMetalNil;
    MetalQueueRef cmdQueue_ = kMetalNil;
    MetalLibraryRef library_ = kMetalNil;

    MetalComputePsoRef psoStep_ = kMetalNil;
    MetalComputePsoRef psoToTexture_ = kMetalNil;

    MetalBufferRef stateBuf_[2] = {kMetalNil, kMetalNil};
    MetalBufferRef weightBuf_ = kMetalNil;
    MetalBufferRef inputBuf_ = kMetalNil;
    MetalTextureRef stateTex_ = kMetalNil;

    Topology topology_ = Topology::Ring;
    int neuronCount_ = 0;
    int gridSize_ = 0;
    int readIndex_ = 0;

    void allocateResources();
    void rebuildWeights();
    void seedState();

    MetalTextureRef makeStateTexture() const;
    MetalComputePsoRef makePSO(MetalStringRef kernelName) const;
};