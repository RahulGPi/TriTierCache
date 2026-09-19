from setuptools import setup, Extension
import pybind11

ext = Extension(
    "tri_tier._C",
    sources=[
        "csrc/bindings.cpp",
        "csrc/cpu/quantize_k_avx2.cpp",
        "csrc/cpu/quantize_v_avx2.cpp",
        "csrc/cpu/dequantize_avx2.cpp",
        "csrc/cpu/fused_attn_avx2.cpp",
    ],
    include_dirs=[pybind11.get_include(), "csrc/cpu", "csrc/include", "csrc"],
    extra_compile_args=["-O3", "-mavx2", "-mbmi2", "-mfma", "-std=c++17"],
    language="c++",
)

setup(
    name="tri_tier",
    packages=["tri_tier", "tri_tier.integration"],
    ext_modules=[ext],
)
