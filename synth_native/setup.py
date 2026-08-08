"""
Build the Metal-accelerated FM synthesiser extension.

Usage (from the repo root, with the venv active):
    pip install pybind11 setuptools
    cd synth_native && python setup.py build_ext --inplace && cd ..

setuptools does not recognise .mm files, so we compile the Objective-C++ source
ourselves with clang and then let setuptools link the final .so.
"""

import os
import subprocess
import sys
import sysconfig
import tempfile

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ObjCppBuildExt(build_ext):
    """
    Custom builder that compiles every .mm source file with clang and adds the
    resulting object file to the sources before the normal link step.
    """

    def build_extension(self, ext):
        obj_files = []

        new_sources = []
        for src in ext.sources:
            if src.endswith(".mm"):
                obj = self._compile_mm(src, ext)
                obj_files.append(obj)
            else:
                new_sources.append(src)

        ext.sources = new_sources
        ext.extra_objects = list(getattr(ext, "extra_objects", [])) + obj_files
        super().build_extension(ext)

    def _compile_mm(self, src: str, ext: Extension) -> str:
        """Compile a .mm file with clang and return the path to the .o file."""
        build_temp = self.build_temp
        os.makedirs(build_temp, exist_ok=True)

        base   = os.path.splitext(os.path.basename(src))[0]
        obj    = os.path.join(build_temp, base + ".o")
        py_inc = sysconfig.get_path("include")

        cmd = [
            "clang++",
            "-x", "objective-c++",
            "-std=c++17",
            "-fobjc-arc",
            "-O2",
            "-fPIC",
            # Python headers
            f"-I{py_inc}",
            # pybind11 headers
            f"-I{pybind11.get_include()}",
        ]
        # Extension include dirs
        for inc in (ext.include_dirs or []):
            cmd.append(f"-I{inc}")

        src_path = src if os.path.isabs(src) else os.path.join(BASE_DIR, src)
        cmd += ["-c", src_path, "-o", obj]

        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        return obj


ext = Extension(
    "_fm_synth",
    sources=[
        os.path.join(BASE_DIR, "FMSynthMetal.mm"),   # compiled to .o by ObjCppBuildExt above
        os.path.join(BASE_DIR, "LIFNetworkNative.mm"),
        os.path.join(BASE_DIR, "bindings.cpp"),
    ],
    include_dirs=[
        pybind11.get_include(),
        BASE_DIR,
    ],
    extra_compile_args=["-std=c++17", "-O2"],
    extra_link_args=[
        "-framework", "Metal",
        "-framework", "Foundation",
    ],
    language="c++",
)

setup(
    name="synth_native",
    version="0.1.0",
    ext_modules=[ext],
    cmdclass={"build_ext": ObjCppBuildExt},
    python_requires=">=3.9",
)

