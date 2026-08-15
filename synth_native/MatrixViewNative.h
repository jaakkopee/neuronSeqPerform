/*  MatrixViewNative.h  –  Metal-accelerated matrix view renderer
 */

#ifndef MATRIX_VIEW_NATIVE_H
#define MATRIX_VIEW_NATIVE_H

#include <cstdint>
#include <vector>
#include <memory>

class MatrixViewNative {
public:
    MatrixViewNative(uint32_t rows, uint32_t cols, uint32_t cell_width = 24, uint32_t gap_width = 2);
    ~MatrixViewNative();
    
    // Render grid to texture
    // potentials: float array of size rows*cols
    // spikes: uint8 array of size rows*cols (0 or 1)
    // Returns RGBA8 texture data (4 bytes per pixel)
    std::vector<uint8_t> render(
        const float* potentials,
        const uint8_t* spikes,
        float threshold
    );
    
    // Get output texture dimensions
    uint32_t texture_width() const { return m_texture_width; }
    uint32_t texture_height() const { return m_texture_height; }
    
private:
    uint32_t m_rows;
    uint32_t m_cols;
    uint32_t m_cell_width;
    uint32_t m_gap_width;
    uint32_t m_texture_width;
    uint32_t m_texture_height;
    
    // Metal resources (opaque handle)
    void* m_device;        // id<MTLDevice>
    void* m_command_queue; // id<MTLCommandQueue>
    void* m_pipeline;      // id<MTLComputePipelineState>
};

#endif
