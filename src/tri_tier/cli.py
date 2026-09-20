"""
Command-line interface and diagnostic health-check utility for TriTier.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from typing import Any, Dict, List, Tuple


def _get_ansi_codes(enabled: bool) -> Dict[str, str]:
    if not enabled:
        return {
            "reset": "",
            "bold": "",
            "green": "",
            "red": "",
            "yellow": "",
            "cyan": "",
            "gray": "",
        }
    return {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "green": "\033[32m",
        "red": "\033[31m",
        "yellow": "\033[33m",
        "cyan": "\033[36m",
        "gray": "\033[90m",
    }


def get_environment_info() -> Dict[str, Any]:
    """Collects system, hardware, and runtime environment details."""
    info: Dict[str, Any] = {
        "os_name": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "Unknown",
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "tri_tier_version": "0.1.0",
        "torch_version": None,
        "torch_cuda_available": None,
        "numpy_version": None,
    }

    try:
        import tri_tier

        info["tri_tier_version"] = getattr(tri_tier, "__version__", "0.1.0")
    except Exception:
        pass

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass

    try:
        import numpy

        info["numpy_version"] = numpy.__version__
    except ImportError:
        pass

    return info


def run_diagnostics(run_smoke_test: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """
    Executes a comprehensive health check on CPU compatibility, C++ extension,
    and cache engine functionality.

    Returns:
        (all_passed, report_dictionary)
    """
    report: Dict[str, Any] = {
        "environment": get_environment_info(),
        "checks": {},
        "success": False,
    }

    checks: Dict[str, Dict[str, Any]] = {}
    all_ok = True

    # 1. Check AVX2 CPU Compatibility
    # We safely import is_avx2_supported from tri_tier.__init__
    avx2_ok = False
    try:
        from tri_tier import is_avx2_supported

        avx2_ok = is_avx2_supported()
    except Exception as e:
        checks["avx2_support"] = {
            "status": "FAIL",
            "message": f"Failed to execute CPU feature inspection: {e}",
            "passed": False,
        }
        all_ok = False
    else:
        if avx2_ok:
            checks["avx2_support"] = {
                "status": "PASS",
                "message": f"AVX2 instructions supported on {platform.machine()}",
                "passed": True,
            }
        else:
            checks["avx2_support"] = {
                "status": "FAIL",
                "message": (
                    f"Host architecture '{platform.machine()}' on '{platform.system()}' "
                    "lacks AVX2 vector instruction support."
                ),
                "passed": False,
            }
            all_ok = False

    # 2. Check C++ Extension (_C) Import & Required Symbols
    cpp_ok = False
    cpp_module = None
    if avx2_ok:
        try:
            from tri_tier import _C

            if _C is None:
                raise ImportError("tri_tier._C extension module is None.")
            cpp_module = _C
            cpp_ok = True
        except Exception as e:
            checks["cpp_extension"] = {
                "status": "FAIL",
                "message": f"Failed to load C++ extension 'tri_tier._C': {e}",
                "passed": False,
            }
            all_ok = False
    else:
        checks["cpp_extension"] = {
            "status": "SKIPPED",
            "message": "Skipped due to missing AVX2 CPU support.",
            "passed": False,
        }
        all_ok = False

    if cpp_ok and cpp_module is not None:
        required_symbols = [
            "TriTierCacheEngine",
            "fused_attention_decode",
            "quantize_k_block",
            "quantize_v_block",
            "dequantize_k",
            "dequantize_v",
        ]
        missing_symbols = [s for s in required_symbols if not hasattr(cpp_module, s)]
        if missing_symbols:
            checks["cpp_extension"] = {
                "status": "FAIL",
                "message": f"C++ module missing required symbols: {', '.join(missing_symbols)}",
                "passed": False,
            }
            all_ok = False
        else:
            ext_path = getattr(cpp_module, "__file__", "builtin")
            checks["cpp_extension"] = {
                "status": "PASS",
                "message": f"Loaded successfully ({ext_path})",
                "passed": True,
            }

    # 3. Functional Smoke Verification
    if run_smoke_test and cpp_ok and cpp_module is not None:
        try:
            import numpy as np

            num_q_heads = 4
            num_kv_heads = 4
            head_dim = 64
            max_seq_len = 128

            engine = cpp_module.TriTierCacheEngine(
                num_q_heads, num_kv_heads, head_dim, max_seq_len
            )

            # Validate basic engine properties
            assert engine.s_count == 0
            assert engine.total_processed_tokens == 0

            # Execute a single-token step
            q = np.random.randn(num_q_heads * head_dim).astype(np.float32)
            k = np.random.randn(num_kv_heads * head_dim).astype(np.float32)
            v = np.random.randn(num_kv_heads * head_dim).astype(np.float32)
            out = np.zeros(num_q_heads * head_dim, dtype=np.float32)

            engine.step(q, k, v, out)

            assert engine.s_count == 1
            assert engine.total_processed_tokens == 1
            assert engine.get_buffer_bytes() > 0

            checks["functional_smoke_test"] = {
                "status": "PASS",
                "message": (
                    f"TriTierCacheEngine initialized and executed step correctly "
                    f"({engine.get_buffer_bytes()} bytes allocated)."
                ),
                "passed": True,
            }
        except Exception as e:
            checks["functional_smoke_test"] = {
                "status": "FAIL",
                "message": f"Functional verification failed: {e}",
                "passed": False,
            }
            all_ok = False
    elif not run_smoke_test:
        checks["functional_smoke_test"] = {
            "status": "SKIPPED",
            "message": "Smoke test skipped by request.",
            "passed": True,
        }
    else:
        checks["functional_smoke_test"] = {
            "status": "SKIPPED",
            "message": "Skipped due to upstream extension failure.",
            "passed": False,
        }

    report["checks"] = checks
    report["success"] = all_ok
    return all_ok, report


def print_health_report(report: Dict[str, Any], use_color: bool = True) -> None:
    """Prints a formatted, human-readable terminal diagnostic report."""
    c = _get_ansi_codes(use_color)
    env = report["environment"]
    checks = report["checks"]
    success = report["success"]

    sep = "=" * 70
    thin_sep = "-" * 70

    print(f"\n{c['bold']}{c['cyan']}{sep}{c['reset']}")
    print(
        f"{c['bold']}{c['cyan']} TriTier Health Check & System Diagnostics{c['reset']}"
    )
    print(f"{c['bold']}{c['cyan']}{sep}{c['reset']}")

    print(f"\n{c['bold']}[Environment Details]{c['reset']}")
    print(f"  • TriTier Version   : {env.get('tri_tier_version')}")
    print(f"  • Platform / OS     : {env.get('os_name')} {env.get('os_release')} ({env.get('machine')})")
    print(f"  • Processor         : {env.get('processor')}")
    print(f"  • Python Runtime    : {env.get('python_version')} ({env.get('python_executable')})")
    if env.get("torch_version"):
        cuda_status = "CUDA available" if env.get("torch_cuda_available") else "CPU only"
        print(f"  • PyTorch Version   : {env.get('torch_version')} ({cuda_status})")
    if env.get("numpy_version"):
        print(f"  • NumPy Version     : {env.get('numpy_version')}")

    print(f"\n{c['bold']}[Diagnostic Checks]{c['reset']}")
    print(f"{thin_sep}")

    labels = {
        "avx2_support": "Host CPU AVX2 Support",
        "cpp_extension": "C++ Native Extension (_C)",
        "functional_smoke_test": "Engine Functional Sanity",
    }

    for key, label in labels.items():
        if key in checks:
            chk = checks[key]
            status = chk["status"]
            if status == "PASS":
                tag = f"{c['green']}[PASS]{c['reset']}"
            elif status == "FAIL":
                tag = f"{c['red']}[FAIL]{c['reset']}"
            else:
                tag = f"{c['yellow']}[SKIP]{c['reset']}"

            print(f"  {tag} {c['bold']}{label:<28}{c['reset']}: {chk['message']}")

    print(f"{thin_sep}")
    if success:
        print(
            f"{c['bold']}{c['green']}✓ All checks passed successfully. TriTier is fully operational!{c['reset']}\n"
        )
    else:
        print(
            f"{c['bold']}{c['red']}✗ Diagnostics failed. Please resolve the errors listed above.{c['reset']}\n"
        )


def main(argv: List[str] | None = None) -> int:
    """CLI entry point for TriTier diagnostic health check tool."""
    parser = argparse.ArgumentParser(
        prog="tri_tier",
        description="TriTier Cache CLI: Health check and diagnostic tool.",
    )
    parser.add_argument(
        "-c",
        "--check",
        action="store_true",
        help="Run comprehensive environment and compatibility health checks (default action).",
    )
    parser.add_argument(
        "-s",
        "--smoke",
        action="store_true",
        help="Run functional in-memory smoke test.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output diagnostics report in JSON format.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color codes in output.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="tri_tier 0.1.0",
        help="Show program version number and exit.",
    )

    args = parser.parse_args(argv)

    use_color = not args.no_color and sys.stdout.isatty()
    all_ok, report = run_diagnostics(run_smoke_test=True)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_health_report(report, use_color=use_color)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
