"""Tests for ResourceMonitor."""

from morphllm.monitor import ResourceMonitor, ResourceSnapshot


class TestResourceSnapshot:
    """Tests for the ResourceSnapshot dataclass."""

    def test_ram_fields(self):
        snap = ResourceSnapshot(
            ram_total_bytes=16_000_000_000,
            ram_used_bytes=8_000_000_000,
            ram_used_pct=50.0,
            ram_free_bytes=8_000_000_000,
            vram_total_bytes=None,
            vram_used_bytes=None,
            vram_used_pct=None,
            vram_free_bytes=None,
            gpu_utilization_pct=None,
            has_gpu=False,
        )
        assert snap.ram_used_pct == 50.0
        assert not snap.has_gpu
        assert snap.vram_used_pct is None


class TestResourceMonitor:
    """Tests for ResourceMonitor (live system queries)."""

    def test_snapshot_returns_ram_data(self):
        """snapshot() always returns valid RAM data."""
        monitor = ResourceMonitor()
        snap = monitor.snapshot()
        assert snap.ram_total_bytes > 0
        assert snap.ram_used_bytes > 0
        assert 0 <= snap.ram_used_pct <= 100
        assert snap.ram_free_bytes >= 0

    def test_estimate_param_vram(self):
        """VRAM estimate = 3× param memory (params + 2 Adam states)."""
        monitor = ResourceMonitor()
        # 1M params × 4 bytes = 4MB → 12MB with optimizer
        est = monitor.estimate_param_vram(1_000_000)
        assert est == 12_000_000

    def test_estimate_param_vram_fp16(self):
        """FP16 estimate uses 2 bytes per param."""
        monitor = ResourceMonitor()
        est = monitor.estimate_param_vram(1_000_000, dtype_bytes=2)
        assert est == 6_000_000

    def test_can_fit_params_cpu(self):
        """On CPU (no GPU), can_fit_params always returns True."""
        monitor = ResourceMonitor()
        if not monitor.has_gpu:
            assert monitor.can_fit_params(100_000_000)  # 100M params

    def test_snapshot_consistency(self):
        """Two snapshots taken sequentially should be roughly consistent."""
        monitor = ResourceMonitor()
        snap1 = monitor.snapshot()
        snap2 = monitor.snapshot()
        # RAM total shouldn't change between snapshots
        assert snap1.ram_total_bytes == snap2.ram_total_bytes
