"""NumPy FP16 matmul through the open iwagumi C API, without Rockchip libraries.

This is a prototype primitive backend, not a complete language-model runtime.
The instruction encoder/driver transport comes from iwagumi (Apache-2.0).
"""

import ctypes as c
from pathlib import Path
import platform

import numpy as np


class Caps(c.Structure):
    _fields_ = [("present", c.c_bool), ("usable", c.c_bool), ("experimental", c.c_bool),
                ("backend", c.c_char * 32), ("soc", c.c_char * 32), ("board", c.c_char * 64),
                ("kernel", c.c_char * 65), ("driver", c.c_char * 64), ("uapi", c.c_char * 32)]
    _fields_ += [(name, c.c_uint32) for name in (
        "npu_core_count", "supported_core_mask", "max_regcmd_words", "dma_addr_bits",
        "single_submit_n_max", "single_submit_k_max", "a_cbuf_mk_max", "partial_k_tile", "int8_w8a8_n_max")]
    _fields_ += [(name, c.c_bool) for name in (
        "has_iommu", "has_domain_switch", "has_cache_sync", "has_fence", "has_bo_mmap",
        "has_raw_regcmd_submit", "sidecar_required_for_offload", "allows_w8a8", "allows_multicore")]
    _fields_ += [("reject_reason", c.c_char * 128), ("card_index", c.c_int32)]


class OpenNPU:
    def __init__(self, library: Path):
        if platform.machine() != "aarch64":
            raise ValueError("This backend requires RK3588 aarch64 Linux")
        self.library = Path(library).resolve(strict=True)
        self.lib = c.CDLL(str(self.library))
        self.lib.iwa_probe.argtypes = [c.POINTER(Caps)]
        self.lib.iwa_probe.restype = c.c_int
        self.lib.iwa_context_create.argtypes = [c.c_void_p, c.POINTER(c.c_void_p)]
        self.lib.iwa_context_create.restype = c.c_int
        self.lib.iwa_context_destroy.argtypes = [c.c_void_p]
        self.lib.iwa_context_destroy.restype = None
        self.lib.iwa_matmul.argtypes = [c.c_void_p, c.c_int, c.c_int32, c.c_int32, c.c_int32,
                                        c.c_void_p, c.c_void_p, c.c_void_p, c.c_void_p]
        self.lib.iwa_matmul.restype = c.c_int
        self.caps = Caps()
        status = self.lib.iwa_probe(c.byref(self.caps))
        if status or not self.caps.usable or self.caps.soc != b"rk3588":
            raise RuntimeError(f"NPU unavailable: {status}, {self.caps.reject_reason.decode()}")
        self.context = c.c_void_p()
        status = self.lib.iwa_context_create(None, c.byref(self.context))
        if status or not self.context:
            raise RuntimeError(f"Cannot create open NPU context: {status}")
        self.submits = 0

    def matmul(self, activations: np.ndarray, weights: np.ndarray, n_tile=2048) -> np.ndarray:
        """A[M,K] @ W[N,K].T -> FP32 C[M,N], splitting output channels.

        A and W round to FP16 before submission. No CPU fallback is performed;
        unsupported shapes or submit failures raise. No partial-K summation,
        multi-core scheduling, or persistent weight cache is implemented here.
        """
        if not self.context:
            raise RuntimeError("NPU context is closed")
        if activations.ndim != 2 or weights.ndim != 2:
            raise ValueError("Activations and weights must have rank two")
        m, k = activations.shape
        n, wk = weights.shape
        if (k != wk or min(m, k, n) <= 0 or k % 32 or n % 16 or n_tile % 16
                or not 16 <= n_tile <= self.caps.single_submit_n_max
                or k > self.caps.single_submit_k_max or m * k > self.caps.a_cbuf_mk_max):
            raise ValueError("Shape lies outside this backend's bounded FP16 envelope")
        a = np.ascontiguousarray(activations, dtype="<f2")
        if not np.isfinite(a).all():
            raise ValueError("Activations cannot be represented as finite FP16")
        output = np.empty((m, n), dtype="<f4")
        for start in range(0, n, n_tile):
            end = min(start + n_tile, n)
            b = np.ascontiguousarray(weights[start:end].T, dtype="<f2")
            if not np.isfinite(b).all():
                raise ValueError("Weights cannot be represented as finite FP16")
            tile = np.full((m, end - start), np.nan, dtype="<f4")
            # IWA_QMODE_FP16 = 2 in the pinned public 1.0.0 API.
            status = self.lib.iwa_matmul(self.context, 2, m, k, end - start,
                                         a.ctypes.data, b.ctypes.data, None, tile.ctypes.data)
            if status or not np.isfinite(tile).all():
                raise RuntimeError(f"Open NPU FP16 tile failed: status={status}, N={start}:{end}")
            output[:, start:end] = tile
            self.submits += 1
        return output

    def close(self):
        if getattr(self, "context", None):
            self.lib.iwa_context_destroy(self.context)
            self.context = c.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
