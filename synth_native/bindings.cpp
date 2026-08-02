/*  bindings.cpp  –  pybind11 glue between Python and FMSynth.
 *
 *  Mirrors the Python FMSynth interface so model/fm_synth.py can use this
 *  as a drop-in backend (see NativeFMSynth wrapper in that file).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "FMSynth.h"

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
}
