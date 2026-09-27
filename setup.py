import os
import platform
import sys
from setuptools import setup
from setuptools.command.build_ext import build_ext

# Detect operating system and architecture
is_windows = sys.platform == "win32"
machine = platform.machine().lower()
is_x86 = machine in ("x86_64", "amd64", "i386", "i686")

# Environment variables to control C++ compilation
force_cpp = os.environ.get("TRI_TIER_FORCE_CPP", "0") == "1"
disable_cpp = os.environ.get("TRI_TIER_DISABLE_CPP", "0") == "1"

ext_modules = []

if not disable_cpp and (is_x86 or force_cpp):
    try:
        from pybind11.setup_helpers import Pybind11Extension

        extra_compile_args = []
        extra_link_args = []

        if is_windows:
            # MSVC specific compiler flags
            extra_compile_args = ["/O2", "/openmp"]
            if is_x86:
                extra_compile_args.append("/arch:AVX2")
        elif sys.platform == "darwin":
            # macOS Clang
            extra_compile_args = ["-O3"]
            if is_x86:
                extra_compile_args.extend(["-mavx2", "-mbmi2", "-mfma", "-mf16c"])
        else:
            # Linux GCC / Clang
            extra_compile_args = ["-O3", "-fopenmp"]
            extra_link_args = ["-fopenmp"]
            if is_x86:
                extra_compile_args.extend(["-mavx2", "-mbmi2", "-mfma", "-mf16c"])

        ext_modules = [
            Pybind11Extension(
                "tri_tier._C",
                sources=[
                    "csrc/bindings.cpp",
                    "csrc/cache_engine.cpp",
                    "csrc/cpu/quantize_k_avx2.cpp",
                    "csrc/cpu/quantize_v_avx2.cpp",
                    "csrc/cpu/dequantize_avx2.cpp",
                    "csrc/cpu/fused_attn_avx2.cpp",
                ],
                include_dirs=[
                    "csrc",
                    "csrc/cpu",
                    "csrc/include",
                ],
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                cxx_std=17,
            ),
        ]
    except ImportError:
        ext_modules = []
elif not is_x86:
    print(
        f"[TriTierCache] Notice: Non-x86 architecture detected ({machine}). "
        "Skipping AVX2 C++ extension build; pure PyTorch reference path will be used.",
        file=sys.stderr,
    )


class OptionalBuildExt(build_ext):
    """
    Attempts to compile the native C++ extension. If compilation fails
    (e.g., missing compiler, missing OpenMP runtime), installation continues
    gracefully with the pure PyTorch reference path.
    """

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            if force_cpp:
                raise
            print(
                f"\n[TriTierCache] WARNING: Failed to compile native C++ extension '{ext.name}': {e}\n"
                f"[TriTierCache] Installation will continue in pure PyTorch reference mode.\n",
                file=sys.stderr,
            )


cmdclass = {}
if ext_modules:
    cmdclass["build_ext"] = OptionalBuildExt

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)