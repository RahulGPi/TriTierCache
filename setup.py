import platform
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Detect operating system and architecture
is_windows = sys.platform == "win32"
is_x86 = platform.machine().lower() in ("x86_64", "amd64", "i386", "i686")

extra_compile_args = []
extra_link_args = []

if is_windows:
    # MSVC specific compiler flags
    extra_compile_args = ["/O2", "/openmp"]
    if is_x86:
        extra_compile_args.append("/arch:AVX2")
else:
    # GCC / Clang compiler flags
    extra_compile_args = ["-O3", "-fopenmp"]
    extra_link_args = ["-fopenmp"]
    
    # AVX2 flags only apply to x86_64 processors
    if is_x86:
        extra_compile_args.extend(["-mavx2", "-mbmi2", "-mfma", "-mf16c"])
    else:
        # Prevent Apple Silicon / ARM builds from attempting AVX compilation
        print("Warning: Non-x86 architecture detected. AVX2 optimizations skipped.", file=sys.stderr)

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

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)