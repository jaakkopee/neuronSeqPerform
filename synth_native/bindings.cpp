/*  bindings.cpp  –  pybind11 glue between Python and FMSynth.
 *
 *  Mirrors the Python FMSynth interface so model/fm_synth.py can use this
 *  as a drop-in backend (see NativeFMSynth wrapper in that file).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstddef>
#include <algorithm>
#include "FMSynth.h"
#include "LIFNetworkNative.h"
#include "MatrixViewNative.h"

namespace py = pybind11;

PYBIND11_MODULE(_fm_synth, m) {
    m.doc() = "Metal-accelerated FM synthesiser (Apple Silicon optimised)";

    py::class_<FMSynth>(m, "FMSynth")

        .def(py::init<int, int>(),
             py::arg("sample_rate") = 44100,
             py::arg("buffer_size") = 1024,
             "Create a new FMSynth instance. Compiles the Metal shader on first call.")

        // ── preset / tuning ───────────────────────────────────────────────────
        .def("load_preset",
             [](FMSynth& self, int col,
                py::array_t<float, py::array::c_style> ratios,
                py::array_t<float, py::array::c_style> levels,
                py::array_t<float, py::array::c_style> mod_indices) {
                 self.load_preset(col,
                                  ratios.data(),
                                  levels.data(),
                                  mod_indices.data());
             },
             py::arg("col"), py::arg("ratios"),
             py::arg("levels"), py::arg("mod_indices"),
             "Load preset parameters for one column.")

        .def("set_base_freq", &FMSynth::set_base_freq,
             py::arg("col"), py::arg("freq"))

        .def("set_all_base_freqs",
             [](FMSynth& self,
                py::array_t<float, py::array::c_style> freqs) {
                 self.set_all_base_freqs(freqs.data());
             },
             py::arg("freqs"),
             "Set all 16 voice base frequencies (length-16 float32 array).")

        // ── step control ──────────────────────────────────────────────────────
        .def("trigger_and_activate",
             [](FMSynth& self,
                py::array_t<float, py::array::c_style> spikes,
                int col) {
                 // spikes: (NUM_OPS, COLS) float32
                 self.trigger_and_activate(spikes.data(), col);
             },
             py::arg("spikes"), py::arg("col"),
             "Atomically switch active voice and gate envelope from spike array.")

        .def("set_active_step", &FMSynth::set_active_step,
             py::arg("step"))

        .def("set_operator_gains",
             [](FMSynth& self, int col,
                py::array_t<float, py::array::c_style> gains) {
                 self.set_operator_gains(col, gains.data());
             },
             py::arg("col"), py::arg("gains"),
             "Override operator gains for a column from neuron activation.")

        // ── parameter control ─────────────────────────────────────────────────
        .def("set_mod_index_scale", &FMSynth::set_mod_index_scale, py::arg("scale"))
        .def("set_active_pairs",    &FMSynth::set_active_pairs,    py::arg("n"))
        .def("set_decay_speed",     &FMSynth::set_decay_speed,     py::arg("speed"))
        .def("set_ratio_scale",     &FMSynth::set_ratio_scale,     py::arg("scale"))

        .def_property("master_volume",
                      &FMSynth::get_master_volume,
                      &FMSynth::set_master_volume)

        // ── audio generation ──────────────────────────────────────────────────
        .def("generate",
             [](FMSynth& self, int frames) -> py::array_t<float> {
                 py::array_t<float> out({frames});
                 self.generate(out.mutable_data(), frames);
                 return out;
             },
             py::arg("frames"),
             "Dispatch Metal shader and return float32 array of `frames` samples.")

        // ── display ───────────────────────────────────────────────────────────
        .def("get_operator_frequencies",
             [](FMSynth& self) -> py::array_t<float> {
                 py::array_t<float> out({FMSynth::NUM_OPS, FMSynth::COLS});
                 self.get_operator_frequencies(out.mutable_data());
                 return out;
             },
             "Return operator frequencies as (NUM_OPS × COLS) float32 array.");

    py::class_<LIFNetworkNative>(m, "LIFNetwork")
        .def(py::init<int, int>(),
             py::arg("neuron_count") = 512,
             py::arg("cols") = 16,
             "Create a Metal-backed LIF network.")

        .def("set_topology", &LIFNetworkNative::set_topology, py::arg("topology_index"))
        .def("topology", &LIFNetworkNative::topology)
        .def("set_neuron_count", &LIFNetworkNative::set_neuron_count, py::arg("neuron_count"))
        .def("neuron_count", &LIFNetworkNative::neuron_count)
        .def("rows", &LIFNetworkNative::rows)
        .def("cols", &LIFNetworkNative::cols)

        .def("set_threshold", &LIFNetworkNative::set_threshold, py::arg("threshold"))
        .def("set_tau", &LIFNetworkNative::set_tau, py::arg("tau"))
        .def("set_refractory_ms", &LIFNetworkNative::set_refractory_ms, py::arg("refractory_ms"))
        .def("set_weight_scale", &LIFNetworkNative::set_weight_scale, py::arg("scale"))
        .def("set_global_drive", &LIFNetworkNative::set_global_drive, py::arg("drive"))

        .def("set_external_drive",
             [](LIFNetworkNative& self, py::array_t<float, py::array::c_style> values) {
                 self.set_external_drive(values.data(), static_cast<int>(values.size()));
             },
             py::arg("values"))
        .def("set_neuron_drive", &LIFNetworkNative::set_neuron_drive,
             py::arg("row"), py::arg("col"), py::arg("value"))

        .def("randomize_weights", &LIFNetworkNative::randomize_weights)
        .def("reset_state", &LIFNetworkNative::reset_state)
        .def("step", &LIFNetworkNative::step)

        .def("get_spikes",
             [](const LIFNetworkNative& self) -> py::array_t<float> {
                 py::array_t<float> out({self.rows(), self.cols()});
                 self.get_spikes(out.mutable_data());
                 return out;
             })
        .def("get_potentials",
             [](const LIFNetworkNative& self) -> py::array_t<float> {
                 py::array_t<float> out({self.rows(), self.cols()});
                 self.get_potentials(out.mutable_data());
                 return out;
             });

    // ── MatrixViewNative: Metal-accelerated grid renderer ────────────────────
    py::class_<MatrixViewNative>(m, "MatrixViewNative")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
             py::arg("rows"), py::arg("cols"),
             py::arg("cell_width") = 24,
             py::arg("gap_width") = 2,
             "Create Metal-accelerated matrix view renderer")

          .def("render_into",
                [](MatrixViewNative& self,
                    py::array_t<float, py::array::c_style> potentials,
                    py::array_t<uint8_t, py::array::c_style> spikes,
                    float threshold,
                    py::array_t<uint8_t, py::array::c_style> out) {
                     auto out_buf = out.request();
                     if (out_buf.ndim != 3) {
                          throw std::runtime_error("render_into output must be 3D (H, W, 4)");
                     }
                     if (static_cast<size_t>(out_buf.shape[0]) != self.texture_height() ||
                          static_cast<size_t>(out_buf.shape[1]) != self.texture_width() ||
                          static_cast<size_t>(out_buf.shape[2]) != 4) {
                          throw std::runtime_error("render_into output shape mismatch");
                     }

                     self.render_into(
                          potentials.data(),
                          spikes.data(),
                          threshold,
                          static_cast<uint8_t*>(out_buf.ptr),
                          static_cast<size_t>(out_buf.size));
                },
                py::arg("potentials"), py::arg("spikes"), py::arg("threshold"), py::arg("out"),
                "Render grid into preallocated RGBA8 output array (height×width×4).")

        .def("render",
             [](MatrixViewNative& self,
                py::array_t<float, py::array::c_style> potentials,
                py::array_t<uint8_t, py::array::c_style> spikes,
                float threshold) {
                 auto result = self.render(potentials.data(), spikes.data(), threshold);
                 // Create shape for output array (height, width, 4)
                 std::vector<size_t> shape = {self.texture_height(), self.texture_width(), 4};
                 auto out = py::array_t<uint8_t>(shape);
                 auto buf = out.request();
                 std::copy(result.begin(), result.end(), (uint8_t*)buf.ptr);
                 return out;
             },
             py::arg("potentials"), py::arg("spikes"), py::arg("threshold"),
             "Render grid to RGBA8 texture (height×width×4 array)")

        .def("texture_width", &MatrixViewNative::texture_width)
        .def("texture_height", &MatrixViewNative::texture_height);
}
