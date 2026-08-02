"""
Build the Metal-accelerated FM synthesiser extension.

Usage (from the repo root, with the venv active):
    pip install pybind11
    pip install -e synth_native/

Or to build in-place and run immediately:
    cd synth_native && python setup.py build_ext --inplace && cd ..

The extension is named synth_native._fm_synth.
After building, model/fm_synth.py will automatically pick it up.
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import subprocess
import sys
import os


class ObjCppBuildExt(build_ext):
    """
    Custom build_ext that compiles .mm sources as Objective-C++.
    clang on macOS recognises the .mm extension natively, but setuptools
    may not pass the right flags.  We override compile to ensure it does.
    """

    def build_extension(self, ext):
        # Patch the compiler so .mm files are treated as Objective-C++
        orig_compile = self.compiler._compile

        def patched_compile(obj, src, ext_suffix, cc_args, extra_postargs, pp_opts):
            if src.endswith(".mm"):
                extra_postargs = list(extra_postargs) + ["-x", "objective-c++"]
            orig_compile(obj, src, ext_suffix, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = patched_compile
        super().build_extension(ext)


ext = Extension(
    "synth_native._fm_synth",
    sources=[
        "FMSynthMetal.mm",
        "bindings.cpp",
    ],
    include_dirs=[
        pybind11.get_include(),
        ".",  # for FMSynth.h
    ],
    extra_compile_args=[
        "-std=c++17",
        "-fobjc-arc",       # automatic reference counting for Metal objects
        "-O2",
        "-Wall",
    ],
    extra_link_args=[
        "-framework", "Metal",
        "-framework", "Foundation",
    ],
    language="c++",
)

setup(
    name="synth_native",
    version="0.1.0",
    description="Metal-accelerated FM synthesiser for neuronSeqPerform",
    ext_modules=[ext],
    cmdclass={"build_ext": ObjCppBuildExt},
    python_requires=">=3.9",
)
