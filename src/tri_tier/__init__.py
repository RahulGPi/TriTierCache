"""
TriTier Cache: Multi-tiered KV cache architecture with AVX2 acceleration.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys


def is_avx2_supported() -> bool:
    """
    Checks whether the host CPU supports the AVX2 instruction set.

    Returns:
        bool: True if AVX2 is supported, False otherwise.
    """
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64", "x86", "i386", "i686"):
        return False

    system = platform.system()

    # Linux: Query /proc/cpuinfo flags
    if system == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("flags") or line.startswith("Features"):
                        flags = line.split(":", 1)[1].strip().split()
                        return "avx2" in flags
        except Exception:
            pass

        # Fallback via lscpu command
        try:
            out = subprocess.check_output(
                ["lscpu"], stderr=subprocess.DEVNULL, text=True
            )
            return "avx2" in out.lower()
        except Exception:
            pass

    # macOS (Darwin): Query sysctl hw.optional.avx2_0
    elif system == "Darwin":
        for feature_key in ("hw.optional.avx2_0", "hw.optional.avx2"):
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", feature_key],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
                if out == "1":
                    return True
            except Exception:
                continue

    # Windows: Query via ctypes CPUID or WMIC
    elif system == "Windows":
        try:
            import ctypes

            # x86_64 machine code executing cpuid(eax=7, ecx=0) and inspecting EBX bit 5 (AVX2)
            # 53 (push rbx) -> b8 07 00 00 00 (mov eax, 7) -> 31 c9 (xor ecx, ecx)
            # 0f a2 (cpuid) -> 89 d8 (mov eax, ebx) -> c1 e8 05 (shr eax, 5)
            # 83 e0 01 (and eax, 1) -> 5b (pop rbx) -> c3 (ret)
            code = b"\x53\xb8\x07\x00\x00\x00\x31\xc9\x0f\xa2\x89\xd8\xc1\xe8\x05\x83\xe0\x01\x5b\xc3"
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            addr = kernel32.VirtualAlloc(0, len(code), 0x3000, 0x40)
            if addr:
                try:
                    ctypes.memmove(addr, code, len(code))
                    func = ctypes.CFUNCTYPE(ctypes.c_int)(addr)
                    res = func()
                    return bool(res)
                finally:
                    kernel32.VirtualFree(addr, 0, 0x8000)
        except Exception:
            pass

    return False


# ---------------------------------------------------------------------------
# Safety Gate: Validate AVX2 CPU compatibility before importing C++ extension
# ---------------------------------------------------------------------------
_disable_avx2_check = os.environ.get("TRI_TIER_DISABLE_AVX2_CHECK", "0") == "1"

if not is_avx2_supported() and not _disable_avx2_check:
    raise RuntimeError(
        f"TriTier requires an x86_64 CPU supporting the AVX2 instruction set. "
        f"The host CPU architecture '{platform.machine()}' on '{platform.system()}' "
        f"does not support AVX2 instructions. Execution halted to prevent low-level crashes."
    )

# ---------------------------------------------------------------------------
# Top-level Public Interface
# ---------------------------------------------------------------------------
from tri_tier.cache import TriTierCache
from tri_tier.constants import CHUNK_SIZE, DEVICE, SINK_SIZE, UPDATE_THRESHOLD

try:
    from tri_tier import _C

    HAS_CPP_EXT = True
except ImportError:
    _C = None  # type: ignore[assignment]
    HAS_CPP_EXT = False

__version__ = "0.1.0"

__all__ = [
    "TriTierCache",
    "DEVICE",
    "UPDATE_THRESHOLD",
    "CHUNK_SIZE",
    "SINK_SIZE",
    "is_avx2_supported",
    "HAS_CPP_EXT",
    "_C",
    "__version__",
]
