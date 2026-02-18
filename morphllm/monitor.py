"""Resource monitoring for adaptive growth decisions.

Reads system RAM via psutil and GPU VRAM via pynvml, with graceful
fallback when no GPU is available. Provides normalized metrics for
the GrowthController to make decisions.
"""

from dataclasses import dataclass
from typing import Optional

import psutil


@dataclass
class ResourceSnapshot:
    """A point-in-time snapshot of system resource utilization.

    All percentages are in [0, 100]. Byte counts are raw values.
    Fields are None when the corresponding hardware is unavailable.
    """

    # RAM
    ram_total_bytes: int
    ram_used_bytes: int
    ram_used_pct: float
    ram_free_bytes: int

    # VRAM (GPU)
    vram_total_bytes: Optional[int]
    vram_used_bytes: Optional[int]
    vram_used_pct: Optional[float]
    vram_free_bytes: Optional[int]

    # GPU utilization (0-100)
    gpu_utilization_pct: Optional[float]

    # Derived convenience flag
    has_gpu: bool


class ResourceMonitor:
    """Monitors system RAM and GPU VRAM utilization.

    Uses psutil for RAM and pynvml for NVIDIA GPU metrics.
    Gracefully degrades when no GPU is present.

    Args:
        gpu_index: Which GPU to monitor (default 0).
    """

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self._gpu_available = False
        self._pynvml = None
        self._gpu_handle = None

        try:
            import pynvml  # type: ignore[import]

            pynvml.nvmlInit()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self._pynvml = pynvml
            self._gpu_available = True
        except Exception:
            # No GPU, pynvml not installed, or init failed
            pass

    @property
    def has_gpu(self) -> bool:
        """Whether a GPU was successfully detected."""
        return self._gpu_available

    def snapshot(self) -> ResourceSnapshot:
        """Capture a point-in-time resource snapshot.

        Returns:
            ResourceSnapshot with current utilization metrics.
        """
        # --- RAM ---
        mem = psutil.virtual_memory()
        ram_total = mem.total
        ram_used = mem.used
        ram_pct = mem.percent
        ram_free = mem.available

        # --- VRAM ---
        vram_total: Optional[int] = None
        vram_used: Optional[int] = None
        vram_pct: Optional[float] = None
        vram_free: Optional[int] = None
        gpu_util: Optional[float] = None

        if self._gpu_available and self._pynvml is not None:
            try:
                info = self._pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                vram_total = info.total
                vram_used = info.used
                vram_free = info.free
                vram_pct = (vram_used / vram_total * 100) if vram_total > 0 else 0.0

                util = self._pynvml.nvmlDeviceGetUtilizationRates(
                    self._gpu_handle
                )
                gpu_util = float(util.gpu)
            except Exception:
                # GPU query failed at runtime (e.g., device reset)
                pass

        return ResourceSnapshot(
            ram_total_bytes=ram_total,
            ram_used_bytes=ram_used,
            ram_used_pct=ram_pct,
            ram_free_bytes=ram_free,
            vram_total_bytes=vram_total,
            vram_used_bytes=vram_used,
            vram_used_pct=vram_pct,
            vram_free_bytes=vram_free,
            gpu_utilization_pct=gpu_util,
            has_gpu=self._gpu_available,
        )

    def estimate_param_vram(self, num_params: int, dtype_bytes: int = 4) -> int:
        """Estimate VRAM needed for parameters + Adam optimizer state.

        Adam stores 2 additional tensors per parameter (exp_avg, exp_avg_sq),
        so total VRAM ≈ 3× parameter memory.

        Args:
            num_params: Number of model parameters.
            dtype_bytes: Bytes per parameter (default: 4 for float32).

        Returns:
            Estimated VRAM in bytes.
        """
        param_bytes = num_params * dtype_bytes
        return param_bytes * 3  # params + 2 optimizer states

    def can_fit_params(
        self,
        additional_params: int,
        safety_margin_pct: float = 10.0,
        dtype_bytes: int = 4,
    ) -> bool:
        """Check if the GPU can accommodate additional parameters.

        Args:
            additional_params: Number of new parameters to add.
            safety_margin_pct: Percentage of VRAM to keep free.
            dtype_bytes: Bytes per parameter.

        Returns:
            True if there's enough free VRAM after the safety margin.
        """
        if not self._gpu_available:
            return True  # CPU-only, assume RAM is sufficient

        snap = self.snapshot()
        if snap.vram_free_bytes is None or snap.vram_total_bytes is None:
            return True

        needed = self.estimate_param_vram(additional_params, dtype_bytes)
        safety_bytes = int(snap.vram_total_bytes * safety_margin_pct / 100)
        available = snap.vram_free_bytes - safety_bytes

        return needed <= available

    def __del__(self):
        """Shutdown pynvml if initialized."""
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
