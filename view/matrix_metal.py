"""Metal-accelerated matrix view renderer wrapper for Pygame display."""

import numpy as np

_native_ok = False
_native_renderer = None

try:
    from synth_native._fm_synth import MatrixViewNative
    _native_ok = True
except ImportError:
    _native_ok = False


class MatrixViewMetalRenderer:
    """Metal-accelerated grid renderer (falls back to NumPy if unavailable)."""
    
    def __init__(self, rows: int, cols: int, cell_width: int = 24, gap_width: int = 2):
        self.rows = rows
        self.cols = cols
        self.cell_width = cell_width
        self.gap_width = gap_width
        self._native = None
        self._native_texture = None
        
        if _native_ok:
            try:
                self._native = MatrixViewNative(rows, cols, cell_width, gap_width)
                self._native_texture = np.empty(
                    (self._native.texture_height(), self._native.texture_width(), 4),
                    dtype=np.uint8,
                )
                print(f"[MatrixView] Metal renderer initialized ({cols}×{rows} grid)")
            except Exception as e:
                print(f"[MatrixView] Metal renderer init failed ({e}); using NumPy fallback")
                self._native = None
                self._native_texture = None
        else:
            print("[MatrixView] Metal backend not available; using NumPy fallback")
    
    def render(self, potentials: np.ndarray, spikes: np.ndarray, threshold: float) -> np.ndarray:
        """
        Render grid to RGBA8 texture.
        
        Args:
            potentials: (rows, cols) float32 array
            spikes: (rows, cols) bool array
            threshold: firing threshold
        
        Returns:
            (texture_height, texture_width, 4) uint8 RGBA array
        """
        potentials = np.asarray(potentials, dtype=np.float32)
        spikes = np.asarray(spikes, dtype=np.uint8)
        
        if self._native is not None:
            # Use Metal renderer with reusable output buffer to avoid per-frame allocations.
            self._native.render_into(potentials, spikes, threshold, self._native_texture)
            return self._native_texture
        else:
            # NumPy fallback: simple colormap rendering
            return self._render_numpy_fallback(potentials, spikes, threshold)
    
    def _render_numpy_fallback(self, potentials: np.ndarray, spikes: np.ndarray, threshold: float) -> np.ndarray:
        """Fallback NumPy-based rendering (CPU)."""
        import colorsys
        
        # Create output texture
        tex_width = self.cols * (self.cell_width + self.gap_width)
        tex_height = self.rows * (self.cell_width + self.gap_width)
        texture = np.zeros((tex_height, tex_width, 4), dtype=np.uint8)
        
        # Fill with background
        texture[:, :] = [10, 14, 28, 255]
        
        # Render each cell
        for row in range(self.rows):
            for col in range(self.cols):
                potential = potentials[row, col]
                is_spike = spikes[row, col] > 0
                
                # Normalize potential
                normalized = min(1.0, max(0.0, potential / threshold))
                
                # Compute HSV
                if is_spike:
                    h, s, v = 0.1, 1.0, 1.0  # Yellow/orange
                else:
                    h = 0.6 - (normalized * 0.3)  # Blue to cyan
                    s = 0.7 + (normalized * 0.3)
                    v = 0.4 + (normalized * 0.5)
                
                # Convert to RGB
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                color = np.array([int(r * 255), int(g * 255), int(b * 255), 255], dtype=np.uint8)
                
                # Fill cell in texture
                y_start = row * (self.cell_width + self.gap_width)
                x_start = col * (self.cell_width + self.gap_width)
                y_end = y_start + self.cell_width
                x_end = x_start + self.cell_width
                
                texture[y_start:y_end, x_start:x_end] = color
        
        return texture
    
    def texture_width(self) -> int:
        """Get texture width in pixels."""
        if self._native:
            return self._native.texture_width()
        return self.cols * (self.cell_width + self.gap_width)
    
    def texture_height(self) -> int:
        """Get texture height in pixels."""
        if self._native:
            return self._native.texture_height()
        return self.rows * (self.cell_width + self.gap_width)
