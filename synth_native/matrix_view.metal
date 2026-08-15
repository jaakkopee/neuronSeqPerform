/*  matrix_view.metal  –  Metal compute shader for efficient LIF grid rendering
 *
 *  Renders a rows×cols neuron grid to a texture:
 *  - Each thread processes one pixel
 *  - Uses potentials + spikes to compute HSV color
 *  - Outputs to RGBA8 texture for fast blit to screen
 */

#include <metal_stdlib>
using namespace metal;

// Shared structs
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

// HSV to RGB conversion
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
    
    // Determine which cell this pixel belongs to
    uint cell_x = px / (params.cell_width + params.gap_width);
    uint cell_y = py / (params.cell_height + params.gap_width);
    uint local_x = px % (params.cell_width + params.gap_width);
    uint local_y = py % (params.cell_height + params.gap_width);
    
    // Skip gap pixels
    if (local_x >= params.cell_width || local_y >= params.cell_height) {
        output_texture[py * params.texture_width + px] = uchar4(20, 28, 50, 255); // dark bg
        return;
    }
    
    // Skip if outside grid bounds
    if (cell_x >= params.cols || cell_y >= params.rows) {
        output_texture[py * params.texture_width + px] = uchar4(10, 14, 28, 255);
        return;
    }
    
    uint idx = cell_y * params.cols + cell_x;
    
    // Get potential and spike for this cell
    float potential = potentials[idx];
    bool is_spike = spikes[idx] > 0;
    
    // Normalize potential to 0-1
    float normalized = potential / params.threshold;
    normalized = clamp(normalized, 0.0, 1.0);
    
    // Compute HSV color based on potential
    float hue = 0.6 - (normalized * 0.3); // Blue to cyan based on potential
    float saturation = 0.7 + (normalized * 0.3); // Increase saturation with activity
    float value = 0.4 + (normalized * 0.5); // Increase brightness with activity
    
    // If spiking, flash bright
    if (is_spike) {
        hue = 0.1; // Yellow/orange
        saturation = 1.0;
        value = 1.0;
    }
    
    // Convert HSV to RGB
    float3 rgb = hsv_to_rgb(hue, saturation, value);
    
    // Convert to 8-bit
    uchar4 color = uchar4(
        (uchar)(rgb.r * 255.0),
        (uchar)(rgb.g * 255.0),
        (uchar)(rgb.b * 255.0),
        255
    );
    
    output_texture[py * params.texture_width + px] = color;
}
