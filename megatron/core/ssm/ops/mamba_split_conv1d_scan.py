# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Local adapter for mamba_ssm's fused split-conv training kernel.

The upstream `mamba_split_conv1d_scan_combined` API does not expose the dtype used for the
inter-chunk SSM states. Megatron's dynamic inference varlen path has this knob, and this adapter
adds the same control to the training path while delegating the rest of the fused op to mamba_ssm.
"""

from typing import Optional

import torch

try:
    from einops import rearrange

    HAVE_EINOPS = True
except ImportError:
    rearrange = None
    HAVE_EINOPS = False

try:
    from mamba_ssm.ops.triton import ssd_combined as _ssd_combined

    HAVE_MAMBA_SSM = True
except ImportError:
    _ssd_combined = None
    HAVE_MAMBA_SSM = False


def _mamba_chunk_scan_combined_fwd_with_state_dtype(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    cu_seqlens=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    state_dtype: Optional[torch.dtype] = None,
):
    """Upstream `_mamba_chunk_scan_combined_fwd` with configurable state output dtype."""

    if not HAVE_EINOPS:
        raise ImportError("einops is required by the Mamba training SSM dtype adapter")

    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if x.stride(-1) != 1 and x.stride(1) != 1:
        x = x.contiguous()
    if z is not None and z.stride(-1) != 1 and z.stride(1) != 1:
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)

    dA_cumsum, dt = _ssd_combined._chunk_cumsum_fwd(
        dt,
        A,
        chunk_size,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )
    states = _ssd_combined._chunk_state_fwd(
        B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True
    )
    states, final_states = _ssd_combined._state_passing_fwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1],
        initial_states=(
            rearrange(initial_states, "... p n -> ... (p n)")
            if initial_states is not None
            else None
        ),
        seq_idx=seq_idx,
        chunk_size=chunk_size,
        out_dtype=state_dtype if state_dtype is not None else C.dtype,
    )
    states, final_states = [
        rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]
    ]

    CB = _ssd_combined._bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    out, out_x = _ssd_combined._chunk_scan_fwd(
        CB, x, dt, dA_cumsum, C, states, D=D, z=z, seq_idx=seq_idx
    )
    if cu_seqlens is None:
        return out, out_x, dt, dA_cumsum, states, final_states

    assert batch == 1, "passing cu_seqlens to get varlen states is only supported if batch == 1"
    varlen_states = _ssd_combined.chunk_state_varlen(
        B.squeeze(0),
        x.squeeze(0),
        dt.squeeze(0),
        dA_cumsum.squeeze(0),
        cu_seqlens,
        states.squeeze(0),
    )
    return out, out_x, dt, dA_cumsum, states, final_states, varlen_states


def mamba_split_conv1d_scan_combined(
    zxbcdt,
    conv1d_weight,
    conv1d_bias,
    dt_bias,
    A,
    D,
    chunk_size,
    initial_states=None,
    seq_idx=None,
    dt_limit=(0.0, float("inf")),
    return_final_states=False,
    activation="silu",
    rmsnorm_weight=None,
    rmsnorm_eps=1e-6,
    outproj_weight=None,
    outproj_bias=None,
    headdim=None,
    ngroups=1,
    norm_before_gate=True,
    state_dtype: Optional[torch.dtype] = None,
):
    """Call mamba_ssm's fused training op with an optional SSM state dtype override."""

    if not HAVE_MAMBA_SSM:
        raise ImportError(
            "MambaSSM is not installed. Please install it with `pip install mamba-ssm`."
        )

    if state_dtype is None:
        return _ssd_combined.mamba_split_conv1d_scan_combined(
            zxbcdt,
            conv1d_weight,
            conv1d_bias,
            dt_bias,
            A,
            D,
            chunk_size,
            initial_states=initial_states,
            seq_idx=seq_idx,
            dt_limit=dt_limit,
            return_final_states=return_final_states,
            activation=activation,
            rmsnorm_weight=rmsnorm_weight,
            rmsnorm_eps=rmsnorm_eps,
            outproj_weight=outproj_weight,
            outproj_bias=outproj_bias,
            headdim=headdim,
            ngroups=ngroups,
            norm_before_gate=norm_before_gate,
        )

    original_fwd = _ssd_combined._mamba_chunk_scan_combined_fwd

    def _patched_mamba_chunk_scan_combined_fwd(*args, **kwargs):
        return _mamba_chunk_scan_combined_fwd_with_state_dtype(
            *args, state_dtype=state_dtype, **kwargs
        )

    _ssd_combined._mamba_chunk_scan_combined_fwd = _patched_mamba_chunk_scan_combined_fwd
    try:
        return _ssd_combined.mamba_split_conv1d_scan_combined(
            zxbcdt,
            conv1d_weight,
            conv1d_bias,
            dt_bias,
            A,
            D,
            chunk_size,
            initial_states=initial_states,
            seq_idx=seq_idx,
            dt_limit=dt_limit,
            return_final_states=return_final_states,
            activation=activation,
            rmsnorm_weight=rmsnorm_weight,
            rmsnorm_eps=rmsnorm_eps,
            outproj_weight=outproj_weight,
            outproj_bias=outproj_bias,
            headdim=headdim,
            ngroups=ngroups,
            norm_before_gate=norm_before_gate,
        )
    finally:
        _ssd_combined._mamba_chunk_scan_combined_fwd = original_fwd
