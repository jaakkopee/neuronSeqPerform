/*  MatrixViewNative.mm  –  Metal implementation for matrix view rendering
 */

#include "MatrixViewNative.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstring>
#include <stdexcept>

struct RenderParams {
    uint32_t rows;
    uint32_t cols;
    uint32_t cell_width;
    uint32_t cell_height;
    uint32_t gap_width;
    uint32_t texture_width;
    uint32_t texture_height;
    float threshold;
    float potential_scale;
    uint32_t step;
    float _pad;
};

MatrixViewNative::MatrixViewNative(uint32_t rows, uint32_t cols, uint32_t cell_width, uint32_t gap_width)
    : m_rows(rows), m_cols(cols), m_cell_width(cell_width), m_gap_width(gap_width),
      m_device(nullptr), m_command_queue(nullptr), m_pipeline(nullptr)
{
    // Calculate texture dimensions
    m_texture_width = cols * (cell_width + gap_width);
    m_texture_height = rows * (cell_width + gap_width);
    
    // Get Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        throw std::runtime_error("[MatrixViewNative] No Metal device available");
    }
    m_device = (__bridge_retained void*)device;
    
    // Create command queue
    id<MTLCommandQueue> cmd_queue = [device newCommandQueue];
    if (!cmd_queue) {
        throw std::runtime_error("[MatrixViewNative] Failed to create command queue");
    }
    m_command_queue = (__bridge_retained void*)cmd_queue;
    
    // Compile shader library
    NSError* error = nil;
    
    // Load shader source from bundle or embedded string
    NSString* shader_code = @R"(
#include <metal_stdlib>
using namespace metal;

struct RenderParams {
    uint   rows;
    uint   cols;
    uint   cell_width;
    uint   cell_height;
    uint   gap_width;
    uint   texture_width;
    uint   texture_height;
    float  threshold;
    float  potential_scale;
    uint   step;
    float  _pad;
};

inline float3 hsv_to_rgb(float h, float s, float v) {
    h = fmod(h, 1.0);
    if (h < 0.0) h += 1.0;
    
    float c = v * s;
    float hp = h * 6.0;
    float x = c * (1.0 - fabs(fmod(hp, 2.0) - 1.0));
    
    float3 rgb;
    if (hp < 1.0) rgb = float3(c, x, 0.0);
    else if (hp < 2.0) rgb = float3(x, c, 0.0);
    else if (hp < 3.0) rgb = float3(0.0, c, x);
    else if (hp < 4.0) rgb = float3(0.0, x, c);
    else if (hp < 5.0) rgb = float3(x, 0.0, c);
    else rgb = float3(c, 0.0, x);
    
    float m = v - c;
    return rgb + m;
}

kernel void render_matrix(
    device       uchar4*  output_texture  [[ buffer(0) ]],
    device const float*   potentials      [[ buffer(1) ]],
    device const uchar*   spikes          [[ buffer(2) ]],
    constant     RenderParams& params     [[ buffer(3) ]],
    uint2 gid [[ thread_position_in_grid ]])
{
    if (gid.x >= params.texture_width || gid.y >= params.texture_height) return;
    
    uint px = gid.x;
    uint py = gid.y;
    
    uint cell_x = px / (params.cell_width + params.gap_width);
    uint cell_y = py / (params.cell_height + params.gap_width);
    uint local_x = px % (params.cell_width + params.gap_width);
    uint local_y = py % (params.cell_height + params.gap_width);
    
    if (local_x >= params.cell_width || local_y >= params.cell_height) {
        output_texture[py * params.texture_width + px] = uchar4(20, 28, 50, 255);
        return;
    }
    
    if (cell_x >= params.cols || cell_y >= params.rows) {
        output_texture[py * params.texture_width + px] = uchar4(10, 14, 28, 255);
        return;
    }
    
    uint idx = cell_y * params.cols + cell_x;
    float potential = potentials[idx];
    bool is_spike = spikes[idx] > 0;
    
    float normalized = potential / params.threshold;
    normalized = clamp(normalized, 0.0, 1.0);
    
    float hue = 0.6 - (normalized * 0.3);
    float saturation = 0.7 + (normalized * 0.3);
    float value = 0.4 + (normalized * 0.5);
    
    if (is_spike) {
        hue = 0.1;
        saturation = 1.0;
        value = 1.0;
    }
    
    float3 rgb = hsv_to_rgb(hue, saturation, value);
    
    uchar4 color = uchar4(
        (uchar)(rgb.r * 255.0),
        (uchar)(rgb.g * 255.0),
        (uchar)(rgb.b * 255.0),
        255
    );
    
    output_texture[py * params.texture_width + px] = color;
}
    )";
    
    id<MTLLibrary> library = [device newLibraryWithSource:shader_code options:nil error:&error];
    if (!library) {
        NSString* error_msg = [NSString stringWithFormat:@"[MatrixViewNative] Shader compilation failed: %@", error.localizedDescription];
        throw std::runtime_error([error_msg UTF8String]);
    }
    
    // Get kernel function
    id<MTLFunction> kernel_fn = [library newFunctionWithName:@"render_matrix"];
    if (!kernel_fn) {
        throw std::runtime_error("[MatrixViewNative] Kernel function not found");
    }
    
    // Create pipeline
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernel_fn error:&error];
    if (!pipeline) {
        throw std::runtime_error("[MatrixViewNative] Pipeline creation failed");
    }
    m_pipeline = (__bridge_retained void*)pipeline;
    
    printf("[MatrixViewNative] Initialized (%ux%u grid, %ux%u texture)\n",
           m_cols, m_rows, m_texture_width, m_texture_height);
}

MatrixViewNative::~MatrixViewNative() {
    if (m_device) CFRelease(m_device);
    if (m_command_queue) CFRelease(m_command_queue);
    if (m_pipeline) CFRelease(m_pipeline);
}

std::vector<uint8_t> MatrixViewNative::render(
    const float* potentials,
    const uint8_t* spikes,
    float threshold)
{
    id<MTLDevice> device = (__bridge id<MTLDevice>)m_device;
    id<MTLCommandQueue> cmd_queue = (__bridge id<MTLCommandQueue>)m_command_queue;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)m_pipeline;
    
    // Create buffers
    size_t potential_size = m_rows * m_cols * sizeof(float);
    size_t spike_size = m_rows * m_cols * sizeof(uint8_t);
    size_t output_size = m_texture_width * m_texture_height * 4; // RGBA
    
    id<MTLBuffer> potential_buf = [device newBufferWithBytes:(void*)potentials
                                                      length:potential_size
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> spike_buf = [device newBufferWithBytes:(void*)spikes
                                                  length:spike_size
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buf = [device newBufferWithLength:output_size
                                                   options:MTLResourceStorageModeShared];
    
    // Params
    RenderParams params = {
        m_rows, m_cols, m_cell_width, m_cell_width, m_gap_width,
        m_texture_width, m_texture_height, threshold, 1.0, 0, 0.0
    };
    
    // Encode
    id<MTLCommandBuffer> cmd_buf = [cmd_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
    
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:output_buf offset:0 atIndex:0];
    [encoder setBuffer:potential_buf offset:0 atIndex:1];
    [encoder setBuffer:spike_buf offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    
    MTLSize grid = MTLSizeMake(m_texture_width, m_texture_height, 1);
    MTLSize threads = MTLSizeMake(pipeline.threadExecutionWidth, 
                                   pipeline.maxTotalThreadsPerThreadgroup / pipeline.threadExecutionWidth, 1);
    
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];
    
    // Copy output
    std::vector<uint8_t> result(output_size);
    std::memcpy(result.data(), [output_buf contents], output_size);
    
    return result;
}
