# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import torch

from megatron.core.ssm.ops import mamba_split_conv1d_scan


def test_mamba_split_conv1d_scan_forwards_state_dtype(monkeypatch):
    calls = {}
    original_fwd = object()
    fake_ssd_combined = SimpleNamespace(_mamba_chunk_scan_combined_fwd=original_fwd)

    def fake_mamba_split_conv1d_scan_combined(*args, **kwargs):
        return fake_ssd_combined._mamba_chunk_scan_combined_fwd(
            "x",
            "dt",
            "A",
            "B",
            "C",
            16,
            D=None,
            z=None,
            dt_bias=None,
            initial_states=None,
            seq_idx=None,
            cu_seqlens=None,
            dt_softplus=True,
        )

    def fake_fwd_with_state_dtype(*args, state_dtype=None, **kwargs):
        calls["state_dtype"] = state_dtype
        return "patched-result"

    fake_ssd_combined.mamba_split_conv1d_scan_combined = fake_mamba_split_conv1d_scan_combined
    monkeypatch.setattr(mamba_split_conv1d_scan, "_ssd_combined", fake_ssd_combined)
    monkeypatch.setattr(mamba_split_conv1d_scan, "HAVE_MAMBA_SSM", True)
    monkeypatch.setattr(
        mamba_split_conv1d_scan,
        "_mamba_chunk_scan_combined_fwd_with_state_dtype",
        fake_fwd_with_state_dtype,
    )

    result = mamba_split_conv1d_scan.mamba_split_conv1d_scan_combined(
        "zxbcdt",
        "conv1d_weight",
        "conv1d_bias",
        "dt_bias",
        "A",
        "D",
        16,
        state_dtype=torch.float32,
    )

    assert result == "patched-result"
    assert calls["state_dtype"] == torch.float32
    assert fake_ssd_combined._mamba_chunk_scan_combined_fwd is original_fwd
