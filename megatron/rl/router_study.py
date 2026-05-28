# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
MoE router non-determinism study harness.

Entry point: run_router_study(model, args)
Called from megatron.training.training.pretrain() when --router-study-mode is set.
Model is fully loaded from checkpoint before this function is called.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
from megatron.training import get_args, print_rank_0


# ---------------------------------------------------------------------------
# Activation store
# ---------------------------------------------------------------------------

@dataclass
class RouterActivationRecord:
    layer_idx: int
    logits: torch.Tensor       # [S, num_experts_local], dtype=router_dtype (fp64)
    probs: torch.Tensor        # [S, num_experts_local], dense-sparse: non-zero at top-k slots only
    routing_map: torch.Tensor  # [S, num_experts_global], bool (True = routed; after EP gather)


class RouterActivationStore:
    """Accumulates one RouterActivationRecord per MoE layer per forward pass."""

    def __init__(self):
        self.records: List[RouterActivationRecord] = []

    def clear(self):
        self.records.clear()


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def gather_routing_map_global(routing_map: torch.Tensor) -> torch.Tensor:
    """Return the global routing_map.

    The router weight [num_experts, hidden_size] is *replicated* across all EP
    ranks, so routing decisions are already made over all experts on every rank.
    The routing_map is therefore already global — no all_gather is needed.
    (EP sharding applies only to which experts *execute* the MLP, not to the
    routing computation itself.)
    """
    return routing_map


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------

def _make_router_hook(layer_idx: int, store: RouterActivationStore):
    """Return a forward hook that captures one RouterActivationRecord."""
    def hook(module, input, output):
        probs, routing_map = output
        # Gather routing map across EP ranks so each rank sees the full picture.
        routing_map_global = gather_routing_map_global(routing_map)
        # Flatten any leading batch dimensions → [S, num_experts] / [S, top_k] / [S, E_global]
        store.records.append(RouterActivationRecord(
            layer_idx=layer_idx,
            logits=module._last_logits.detach().cpu().flatten(end_dim=-2),
            probs=probs.detach().cpu().flatten(end_dim=-2),
            routing_map=routing_map_global.detach().cpu().flatten(end_dim=-2),
        ))
    return hook


def _unwrap(model_chunk):
    """Unwrap Float16Module (used under --bf16/--fp16) to get the GPT model."""
    return getattr(model_chunk, 'module', model_chunk)


def register_router_hooks(model_chunk, store: RouterActivationStore) -> List:
    """Register forward hooks on every TopKRouter in the model.

    Returns the list of hook handles so they can be removed later.
    """
    handles = []
    for idx, layer in enumerate(_unwrap(model_chunk).decoder.layers):
        # Hybrid model: MambaLayers have no .mlp; dense Transformer layers have
        # .mlp but no .router.  Skip both.
        mlp = getattr(layer, 'mlp', None)
        if mlp is None:
            continue
        router = getattr(mlp, 'router', None)
        if router is None:
            continue
        handle = router.register_forward_hook(_make_router_hook(idx, store))
        handles.append(handle)
    return handles


def remove_hooks(handles: List):
    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# Validation (M2 milestone check)
# ---------------------------------------------------------------------------

def _count_moe_layers(model_chunk) -> int:
    count = 0
    for layer in _unwrap(model_chunk).decoder.layers:
        mlp = getattr(layer, 'mlp', None)
        if mlp is not None and getattr(mlp, 'router', None) is not None:
            count += 1
    return count


def validate_store(store: RouterActivationStore, model_chunk, args):
    """Assert shapes and dtypes are as expected; print a summary on rank 0."""
    num_moe_layers = _count_moe_layers(model_chunk)
    n = len(store.records)

    assert n == num_moe_layers, (
        f"Expected {num_moe_layers} records, got {n}. "
        "Hook may have missed some layers."
    )

    # Spot-check first record.
    rec0 = store.records[0]
    S, E_local = rec0.logits.shape
    _, E_global = rec0.routing_map.shape

    assert rec0.logits.dtype == torch.float64, (
        f"Expected fp64 logits (--moe-router-dtype fp64), got {rec0.logits.dtype}"
    )
    assert rec0.routing_map.dtype == torch.bool, (
        f"Expected bool routing_map, got {rec0.routing_map.dtype}"
    )

    # Derive actual top-k from routing_map (probs is dense-sparse [S, E_local], not [S, top_k]).
    true_per_row = rec0.routing_map.sum(dim=-1)
    top_k = int(true_per_row[0].item())
    assert (true_per_row == top_k).all(), (
        f"Inconsistent True-per-row in routing_map: "
        f"min={true_per_row.min().item()} max={true_per_row.max().item()}"
    )

    print_rank_0("=" * 60)
    print_rank_0("M2 validation passed.")
    print_rank_0(f"  MoE layers instrumented : {num_moe_layers}")
    print_rank_0(f"  Records collected       : {n}")
    print_rank_0(f"  Tokens (S)              : {S}")
    print_rank_0(f"  Experts local / global  : {E_local} / {E_global}")
    print_rank_0(f"  Top-k                   : {top_k}")
    print_rank_0(f"  Logit dtype             : {rec0.logits.dtype}")
    print_rank_0(f"  Routing_map dtype       : {rec0.routing_map.dtype}")
    print_rank_0(f"  Token 0, layer 0 top experts: "
                 f"{rec0.routing_map[0].nonzero(as_tuple=True)[0].tolist()}")
    print_rank_0("=" * 60)


# ---------------------------------------------------------------------------
# Dummy batch helper
# ---------------------------------------------------------------------------

def _make_dummy_batch(args):
    """Build a tiny fixed-token batch for the validation forward pass.

    Returns (input_ids, position_ids) both shaped [seq_len, batch_size].
    """
    seq_len = 4
    batch_size = 1
    vocab_size = getattr(args, 'padded_vocab_size', 32000)
    torch.manual_seed(args.seed if hasattr(args, 'seed') else 42)
    device = torch.cuda.current_device()
    # Megatron GPT model expects layout [seq_len, batch_size].
    input_ids = torch.randint(0, vocab_size, (seq_len, batch_size), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(1).expand(seq_len, batch_size)
    return input_ids, position_ids


def _build_router_study_batches(args, prompts_path: Optional[str], max_batches: int):
    """Build real prompt batches when available, otherwise a random fallback batch."""
    device = torch.cuda.current_device()
    batches = []
    if prompts_path is not None and os.path.exists(prompts_path):
        with open(prompts_path) as f:
            prompt_data = json.load(f)
        for rec in prompt_data["prompts"][:max_batches]:
            ids = torch.tensor(rec["input_ids"], dtype=torch.long, device=device).unsqueeze(1)
            pos = torch.arange(ids.shape[0], device=device).unsqueeze(1)
            batches.append((ids, pos))
        return batches, f"{len(batches)} real prompts from {os.path.basename(prompts_path)}"

    ids, pos = _make_dummy_batch(args)
    return [(ids, pos)], "random fallback batch"


def _snapshot_router_buffers(model_chunk):
    """Save router buffers that may be mutated by forwards with grad enabled."""
    saved = []
    for name, buffer in model_chunk.named_buffers():
        if any(key in name for key in ("expert_bias", "local_tokens_per_expert", "global_tokens_per_expert", "ga_steps")):
            saved.append((name, buffer, buffer.detach().clone()))
    return saved


def _restore_router_buffers(saved):
    for _name, buffer, value in saved:
        buffer.data.copy_(value)


def _zero_model_grads(model_chunk):
    args = get_args()
    main_grads_dtype = getattr(args, "main_grads_dtype", torch.float32)
    for param in model_chunk.parameters():
        param.grad = None
        main_grad = getattr(param, "main_grad", None)
        if param.requires_grad and main_grad is None:
            param.main_grad = torch.zeros_like(
                param.data, dtype=main_grads_dtype, device=param.device
            )
        elif main_grad is not None:
            main_grad.zero_()


def _get_param_grad(param):
    grad = param.grad
    if grad is None:
        grad = getattr(param, "main_grad", None)
    return grad


def _grad_group_for_name(name: str) -> List[str]:
    groups = ["all_params"]
    if "router" in name:
        groups.append("router_params")
    if "expert" in name or "experts" in name:
        groups.append("expert_params")
    if "mlp" in name or "expert" in name or "experts" in name or "router" in name:
        groups.append("moe_params")
    else:
        groups.append("non_moe_params")
    return groups


def _capture_grad_snapshot(model_chunk):
    """Move current gradients to CPU and precompute per-group norm squares."""
    grads = {}
    norm_sq = {}
    for name, param in model_chunk.named_parameters():
        grad_tensor = _get_param_grad(param)
        if grad_tensor is None:
            continue
        grad = grad_tensor.detach().float().cpu().clone()
        grads[name] = grad
        grad_norm = float((grad * grad).sum().item())
        for group in _grad_group_for_name(name):
            norm_sq[group] = norm_sq.get(group, 0.0) + grad_norm
    return grads, norm_sq


def _compare_current_grads_to_snapshot(model_chunk, ref_grads, ref_norm_sq):
    """Compute local dot/norm stats between current grads and a CPU reference snapshot."""
    stats = {}
    cur_norm_sq = {}
    for name, param in model_chunk.named_parameters():
        grad_tensor = _get_param_grad(param)
        if grad_tensor is None or name not in ref_grads:
            continue
        cur_grad = grad_tensor.detach().float().cpu()
        ref_grad = ref_grads[name]
        dot = float((ref_grad * cur_grad).sum().item())
        cur_norm = float((cur_grad * cur_grad).sum().item())
        for group in _grad_group_for_name(name):
            if group not in stats:
                stats[group] = {"dot": 0.0, "ref_norm_sq": 0.0, "cur_norm_sq": 0.0}
            stats[group]["dot"] += dot
            cur_norm_sq[group] = cur_norm_sq.get(group, 0.0) + cur_norm

    for group, ref_norm in ref_norm_sq.items():
        stats.setdefault(group, {"dot": 0.0, "ref_norm_sq": 0.0, "cur_norm_sq": 0.0})
        stats[group]["ref_norm_sq"] = ref_norm
        stats[group]["cur_norm_sq"] = cur_norm_sq.get(group, 0.0)
    return stats


def _all_reduce_grad_stats(local_stats):
    groups = sorted(local_stats.keys())
    if not groups:
        return {}
    values = []
    for group in groups:
        stat = local_stats[group]
        values.extend([stat["dot"], stat["ref_norm_sq"], stat["cur_norm_sq"]])
    tensor = torch.tensor(values, dtype=torch.float64, device=torch.cuda.current_device())
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    reduced = {}
    vals = tensor.cpu().tolist()
    for idx, group in enumerate(groups):
        dot, ref_norm_sq, cur_norm_sq = vals[3 * idx : 3 * idx + 3]
        denom = (ref_norm_sq * cur_norm_sq) ** 0.5
        cosine = dot / denom if denom > 0 else float("nan")
        rel_diff = None
        if ref_norm_sq > 0:
            rel_diff = ((ref_norm_sq + cur_norm_sq - 2.0 * dot) ** 0.5) / (ref_norm_sq ** 0.5)
        reduced[group] = {
            "cosine": cosine,
            "ref_norm": ref_norm_sq ** 0.5,
            "cur_norm": cur_norm_sq ** 0.5,
            "norm_ratio": (cur_norm_sq / ref_norm_sq) ** 0.5 if ref_norm_sq > 0 else float("nan"),
            "relative_diff": rel_diff,
        }
    return reduced


def _compute_lm_loss(model_chunk, input_ids, position_ids):
    """Run a forward pass and return next-token negative log likelihood."""
    output = model_chunk(input_ids, position_ids, attention_mask=None)
    if not isinstance(output, torch.Tensor) or output.ndim != 3:
        raise RuntimeError("Gradient cosine study requires tensor logits on this rank.")

    from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

    _args = get_args()
    padded_vocab = getattr(_args, "padded_vocab_size", 0)
    if padded_vocab > 0 and output.shape[2] < padded_vocab:
        full_logits = gather_from_tensor_model_parallel_region(output).float()
    else:
        full_logits = output.float()
    log_probs = torch.nn.functional.log_softmax(full_logits[:-1], dim=-1)
    targets = input_ids[1:, :]
    return -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1).mean()


def _natural_grad_pass(model_chunk, input_ids, position_ids, label: str):
    """Run natural routing backward pass and return routing indices + gradient snapshot."""
    saved_buffers = _snapshot_router_buffers(model_chunk)
    _restore_router_buffers(saved_buffers)
    _zero_model_grads(model_chunk)
    store = RouterActivationStore()
    handles = register_router_hooks(model_chunk, store)
    loss = _compute_lm_loss(model_chunk, input_ids, position_ids)
    loss.backward()
    remove_hooks(handles)
    stacked_maps = torch.stack([rec.routing_map for rec in store.records]).unsqueeze(0)
    routing_idx = _routing_map_to_indices(stacked_maps)[0]  # [S, L, top_k]
    grads, norm_sq = _capture_grad_snapshot(model_chunk)
    _restore_router_buffers(saved_buffers)
    print_rank_0(f"  {label}: loss={loss.detach().float().item():.6f}")
    return routing_idx, grads, norm_sq


def _forced_grad_pass(model_chunk, input_ids, position_ids, routing_idx, label: str):
    """Run backward pass while forcing router top-k to a previously captured training route."""
    saved_buffers = _snapshot_router_buffers(model_chunk)
    _restore_router_buffers(saved_buffers)
    _zero_model_grads(model_chunk)
    layer_tensors = [routing_idx[:, layer_idx, :].cuda() for layer_idx in range(routing_idx.shape[1])]
    RouterReplay.set_replay_data(layer_tensors, replay_mask=None)
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    loss = _compute_lm_loss(model_chunk, input_ids, position_ids)
    loss.backward()
    RouterReplay.clear_global_router_replay_action()
    RouterReplay.clear_global_indices()
    stats_ready = True
    _restore_router_buffers(saved_buffers)
    print_rank_0(f"  {label}: loss={loss.detach().float().item():.6f}")
    return stats_ready


def step8_gradient_cosine_routing_replay(
    model_chunk,
    args,
    results_dir: str,
    prompts_path: Optional[str] = None,
    max_batches: int = 1,
) -> None:
    """Compare gradient directions from natural routing vs forced natural top-k replay."""
    if not RouterReplay.global_router_replay_instances:
        print_rank_0(
            "Step 8 skipped: no RouterReplay instances found. Run with --moe-enable-routing-replay."
        )
        return

    batches, source_desc = _build_router_study_batches(args, prompts_path, max_batches)
    print_rank_0("=" * 60)
    print_rank_0(f"Step 8: gradient cosine natural vs forced routing ({source_desc})")

    all_results = {}
    for batch_idx, (input_ids, position_ids) in enumerate(batches):
        print_rank_0(f"Batch {batch_idx + 1}/{len(batches)}: seq_len={input_ids.shape[0]}")
        routing_a, grads_a, norm_sq_a = _natural_grad_pass(
            model_chunk, input_ids, position_ids, label="natural A"
        )

        _routing_b, _grads_b, _norm_sq_b = _natural_grad_pass(
            model_chunk, input_ids, position_ids, label="natural B"
        )
        natural_stats = _all_reduce_grad_stats(
            _compare_current_grads_to_snapshot(model_chunk, grads_a, norm_sq_a)
        )

        _forced_grad_pass(model_chunk, input_ids, position_ids, routing_a, label="forced A-route")
        forced_stats = _all_reduce_grad_stats(
            _compare_current_grads_to_snapshot(model_chunk, grads_a, norm_sq_a)
        )

        batch_results = {
            "natural_a_vs_natural_b": natural_stats,
            "natural_a_vs_forced_a_route": forced_stats,
        }
        all_results[f"batch_{batch_idx:02d}"] = batch_results

        if torch.distributed.get_rank() == 0:
            print_rank_0("  Gradient cosine summary:")
            for group in sorted(forced_stats.keys()):
                nat = natural_stats.get(group, {})
                frc = forced_stats.get(group, {})
                print_rank_0(
                    f"    {group}: "
                    f"cos(A,B)={nat.get('cosine', float('nan')):.6f}  "
                    f"cos(A,forced)={frc.get('cosine', float('nan')):.6f}  "
                    f"rel_diff_forced={frc.get('relative_diff', float('nan')):.6f}"
                )

        del grads_a, _grads_b, routing_a
        _zero_model_grads(model_chunk)

    if torch.distributed.get_rank() == 0:
        out_json = os.path.join(results_dir, "step8_gradient_cosine.json")
        with open(out_json, "w") as f:
            json.dump(
                {
                    "step": "8",
                    "description": "gradient cosine: natural routing vs forced natural top-k replay",
                    "source": source_desc,
                    "results": all_results,
                },
                f,
                indent=2,
            )
        print_rank_0(f"Saved → {out_json}")
    print_rank_0("=" * 60)


# ---------------------------------------------------------------------------
# Step 3A: within-run routing consistency (M4)
# ---------------------------------------------------------------------------

def step3a_within_run_consistency(
    model_chunk,
    args,
    results_dir: str,
    n_passes: int = 10,
    seq_len: int = 1024,
    batch_size: int = 1,
) -> None:
    """Run N identical forward passes and measure per-layer routing consistency.

    Consistency[l, s] = fraction of pass-pairs that produced the exact same
    top-k expert set for token s at MoE layer l.  Expected value is 1.0 when
    the router is fully deterministic.

    Artefacts saved to results_dir:
      step3_routing_consistency.json   — per-layer mean/min stats + summary
      step3_consistency_matrix.npy     — float32 array [num_moe_layers, S]
    """
    print_rank_0("=" * 60)
    print_rank_0(
        f"Step 3A: within-run consistency  "
        f"(N={n_passes}, seq_len={seq_len}, batch_size={batch_size})"
    )

    # Fixed, reproducible token batch.  Same seed used in M5 (across restarts)
    # so both steps see an identical token sequence.
    study_seed = getattr(args, 'seed', 6789)
    vocab_size = getattr(args, 'padded_vocab_size', 32000)
    # Respect the configured sequence length ceiling from --seq-length.
    seq_len = min(seq_len, getattr(args, 'seq_length', seq_len))
    device = torch.cuda.current_device()

    torch.manual_seed(study_seed)
    input_ids = torch.randint(0, vocab_size, (seq_len, batch_size), device=device)
    position_ids = (
        torch.arange(seq_len, device=device)
        .unsqueeze(1)
        .expand(seq_len, batch_size)
    )

    store = RouterActivationStore()
    handles = register_router_hooks(model_chunk, store)
    num_moe_layers = _count_moe_layers(model_chunk)

    # Collect one [num_moe_layers, S, num_experts] CPU bool tensor per pass.
    all_routing_maps: List[torch.Tensor] = []

    for pass_i in range(n_passes):
        store.clear()
        with torch.no_grad():
            model_chunk(input_ids, position_ids, attention_mask=None)

        # Stack per-layer records → [L, S, E] on CPU (hook already moved to CPU).
        pass_map = torch.stack([rec.routing_map for rec in store.records])
        all_routing_maps.append(pass_map)
        print_rank_0(f"  Pass {pass_i + 1}/{n_passes} done.")

    remove_hooks(handles)

    # Pairwise consistency ─────────────────────────────────────────────────────
    # For each (layer l, token s): fraction of all (i<j) pass-pairs where the
    # complete top-k expert bitmask is identical.
    stacked = torch.stack(all_routing_maps)  # [N, L, S, E] bool, on CPU

    pair_agree: List[torch.Tensor] = []
    for i in range(n_passes):
        for j in range(i + 1, n_passes):
            # [L, S]: True iff every expert bit matches between passes i and j.
            agree = (stacked[i] == stacked[j]).all(dim=-1)
            pair_agree.append(agree.float())

    # consistency[l, s] ∈ [0, 1]
    consistency = torch.stack(pair_agree).mean(dim=0)  # [L, S]

    per_layer_mean: List[float] = consistency.mean(dim=-1).tolist()
    per_layer_min:  List[float] = consistency.min(dim=-1).values.tolist()
    overall_mean = consistency.mean().item()
    overall_min  = consistency.min().item()

    # Summary ──────────────────────────────────────────────────────────────────
    print_rank_0("=" * 60)
    print_rank_0("Step 3A results:")
    print_rank_0(f"  Total tokens S            : {seq_len * batch_size}")
    print_rank_0(f"  MoE layers instrumented   : {num_moe_layers}")
    print_rank_0(f"  Pass-pairs evaluated      : {len(pair_agree)}")
    print_rank_0(f"  Overall mean consistency  : {overall_mean:.8f}")
    print_rank_0(f"  Overall min  consistency  : {overall_min:.8f}")
    print_rank_0("  Per MoE layer:")
    for l_idx, (mn, mi) in enumerate(zip(per_layer_mean, per_layer_min)):
        flag = " <<<" if mn < 1.0 else ""
        print_rank_0(f"    MoE layer {l_idx:2d}: mean={mn:.8f}  min={mi:.8f}{flag}")

    if overall_mean < 1.0:
        print_rank_0("*** NON-DETERMINISM DETECTED — consistency < 1.0 ***")
    else:
        print_rank_0("All pass-pairs fully consistent — router is deterministic within-run.")
    print_rank_0("=" * 60)

    # Persist artefacts on rank 0 ───────────────────────────────────────────────
    if torch.distributed.get_rank() == 0:
        results = {
            "step": "3A",
            "description": "within-run routing consistency",
            "n_passes": n_passes,
            "n_pass_pairs": len(pair_agree),
            "seq_len": seq_len,
            "batch_size": batch_size,
            "total_tokens_S": seq_len * batch_size,
            "num_moe_layers": num_moe_layers,
            "study_seed": study_seed,
            "overall_mean_consistency": overall_mean,
            "overall_min_consistency": overall_min,
            "per_layer_mean_consistency": per_layer_mean,
            "per_layer_min_consistency": per_layer_min,
        }
        out_json = os.path.join(results_dir, 'step3_routing_consistency.json')
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=2)
        print_rank_0(f"Saved → {out_json}")

        out_npy = os.path.join(results_dir, 'step3_consistency_matrix.npy')
        np.save(out_npy, consistency.numpy())
        print_rank_0(f"Saved → {out_npy}")


# ---------------------------------------------------------------------------
# Step 3B: across-restart routing consistency (M5)
# ---------------------------------------------------------------------------

def _collect_passes(
    model_chunk,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    n_passes: int,
    label: str,
) -> tuple:
    """Run n_passes forward passes; return (stacked_routing_maps, stacked_logprobs).

    stacked_routing_maps : torch.Tensor [N, L, S, E] bool, on CPU
    stacked_logprobs     : torch.Tensor [N, S-1] float32, on CPU — or None if
                           the rank did not receive logits (non-post-process rank).
    """
    store = RouterActivationStore()
    handles = register_router_hooks(model_chunk, store)

    all_maps: List[torch.Tensor] = []
    all_lp:   List[Optional[torch.Tensor]] = []

    for pass_i in range(n_passes):
        store.clear()
        with torch.no_grad():
            output = model_chunk(input_ids, position_ids, attention_mask=None)

        # Routing maps → [L, S, E] on CPU.
        pass_map = torch.stack([rec.routing_map for rec in store.records])
        all_maps.append(pass_map)

        # Logprobs: output is [S, B, V] on post-process ranks; None elsewhere.
        # With parallel_output=True (default), each TP rank sees only
        # [S, B, padded_vocab_size/TP].  We all-gather across TP to get the
        # full vocabulary before log_softmax so the gather() index is always
        # in-bounds regardless of TP degree.
        lp = None
        if isinstance(output, torch.Tensor) and output.ndim == 3:
            from megatron.core.tensor_parallel.mappings import (
                gather_from_tensor_model_parallel_region,
            )
            _args = get_args()
            padded_vocab = getattr(_args, 'padded_vocab_size', 0)
            vocab_out_dim = output.shape[2]
            if padded_vocab > 0 and vocab_out_dim < padded_vocab:
                # Vocab-parallel shard — all-gather across TP ranks.
                full_logits = gather_from_tensor_model_parallel_region(output).float()
            elif padded_vocab == 0 or vocab_out_dim == padded_vocab:
                full_logits = output.float()
            else:
                full_logits = None   # unexpected shape — skip
            if full_logits is not None:
                log_probs = torch.nn.functional.log_softmax(full_logits, dim=-1)
                tgt = input_ids[1:, :]                           # [S-1, B]
                per_tok = log_probs[:-1].gather(2, tgt.unsqueeze(-1)).squeeze(-1)  # [S-1, B]
                lp = per_tok.detach().cpu().mean(dim=-1)         # [S-1], avg over batch
        all_lp.append(lp)

        print_rank_0(f"  {label} pass {pass_i + 1}/{n_passes} done.")

    remove_hooks(handles)

    stacked_maps = torch.stack(all_maps)                     # [N, L, S, E]
    stacked_lp = (
        torch.stack([t for t in all_lp if t is not None])   # [N, S-1]
        if any(t is not None for t in all_lp)
        else None
    )
    return stacked_maps, stacked_lp


def step3b_across_restart_consistency(
    model_chunk,
    args,
    results_dir: str,
    n_passes: int = 10,
    seq_len: int = 1024,
    batch_size: int = 1,
) -> None:
    """Step 3B: compare routing maps and logprobs across two separate job launches.

    First launch  — run1 artefacts absent: collect N passes, save to disk, prompt
                    re-launch.
    Second launch — run1 artefacts present: collect N passes, load first-run data,
                    compare, save results.

    Artefacts written on Run 1:
      step3b_run1_routing_maps.npy  [N, L, S, E] bool
      step3b_run1_logprobs.npy      [N, S-1] float32  (absent if rank has no logits)
      step3b_run1_input_ids.npy     [S, B] int64

    Artefacts written on Run 2:
      step3b_routing_consistency.json
      step3b_cross_agreement_matrix.npy  [L, S] float32
      step3b_logprob_delta.npy           [S-1]  float32  (if logprobs available)
    """
    print_rank_0("=" * 60)
    print_rank_0(
        f"Step 3B: across-restart consistency  "
        f"(N={n_passes}, seq_len={seq_len}, batch_size={batch_size})"
    )

    # Fixed, reproducible token batch — same seed / same tokens as Step 3A.
    study_seed = getattr(args, 'seed', 6789)
    vocab_size  = getattr(args, 'padded_vocab_size', 32000)
    seq_len     = min(seq_len, getattr(args, 'seq_length', seq_len))
    device      = torch.cuda.current_device()

    torch.manual_seed(study_seed)
    input_ids = torch.randint(0, vocab_size, (seq_len, batch_size), device=device)
    position_ids = (
        torch.arange(seq_len, device=device)
        .unsqueeze(1)
        .expand(seq_len, batch_size)
    )

    # Artefact paths.
    run1_maps_path = os.path.join(results_dir, 'step3b_run1_routing_maps.npy')
    run1_lp_path   = os.path.join(results_dir, 'step3b_run1_logprobs.npy')
    run1_ids_path  = os.path.join(results_dir, 'step3b_run1_input_ids.npy')

    # All ranks agree on which phase this is (shared lustre FS, but broadcast
    # from rank 0 to avoid any stat-cache race).
    is_run1_local = [not os.path.exists(run1_maps_path)]
    torch.distributed.broadcast_object_list(is_run1_local, src=0)
    is_run1 = is_run1_local[0]

    run_label = "Run 1" if is_run1 else "Run 2"
    print_rank_0(f"  Detected: {run_label} (step3b_run1_routing_maps.npy "
                 f"{'not found' if is_run1 else 'found'})")

    # ── Collect this run's N passes ──────────────────────────────────────────
    num_moe_layers = _count_moe_layers(model_chunk)
    stacked_maps, stacked_lp = _collect_passes(
        model_chunk, input_ids, position_ids, n_passes, label=run_label
    )

    # ── Run 1: save and exit ─────────────────────────────────────────────────
    if is_run1:
        if torch.distributed.get_rank() == 0:
            np.save(run1_maps_path, stacked_maps.numpy())
            print_rank_0(f"Saved → {run1_maps_path}")
            if stacked_lp is not None:
                np.save(run1_lp_path, stacked_lp.numpy())
                print_rank_0(f"Saved → {run1_lp_path}")
            np.save(run1_ids_path, input_ids.cpu().numpy())
            print_rank_0(f"Saved → {run1_ids_path}")
        print_rank_0(
            "Step 3B Run 1 complete.\n"
            "Re-launch the study (same command) to collect Run 2 and compare."
        )
        print_rank_0("=" * 60)
        return

    # ── Run 2: load run 1 and compare ───────────────────────────────────────
    run1_maps = torch.from_numpy(np.load(run1_maps_path))          # [N, L, S, E]
    run1_lp   = (
        torch.from_numpy(np.load(run1_lp_path))
        if os.path.exists(run1_lp_path) else None
    )
    run1_ids  = np.load(run1_ids_path)

    # Sanity-check: same input tokens across restarts.
    if not np.array_equal(run1_ids, input_ids.cpu().numpy()):
        print_rank_0(
            "WARNING: input_ids differ between Run 1 and Run 2! "
            "Ensure both runs use identical --seed and --seq-length."
        )

    # Cross-run routing agreement ─────────────────────────────────────────────
    # For each (run1_pass_i, run2_pass_j): [L, S] True iff bitmask matches.
    # Average over all N² pairs to get cross_agreement[L, S].
    cross_agree_list: List[torch.Tensor] = []
    for i in range(n_passes):
        for j in range(n_passes):
            agree = (run1_maps[i] == stacked_maps[j]).all(dim=-1)  # [L, S]
            cross_agree_list.append(agree.float())
    cross_agreement = torch.stack(cross_agree_list).mean(dim=0)    # [L, S]

    # Within-Run-2 agreement for a side-by-side baseline.
    within_list: List[torch.Tensor] = []
    for i in range(n_passes):
        for j in range(i + 1, n_passes):
            agree = (stacked_maps[i] == stacked_maps[j]).all(dim=-1)
            within_list.append(agree.float())
    within_agreement = torch.stack(within_list).mean(dim=0) if within_list else cross_agreement

    per_layer_cross_mean = cross_agreement.mean(dim=-1).tolist()
    per_layer_cross_min  = cross_agreement.min(dim=-1).values.tolist()
    overall_cross_mean   = cross_agreement.mean().item()
    overall_cross_min    = cross_agreement.min().item()
    overall_within_mean  = within_agreement.mean().item()

    # Per-layer disagreement vector (1 − agreement, averaged over tokens).
    per_layer_disagree = (1.0 - cross_agreement).mean(dim=-1)  # [L]

    # Logprob delta ───────────────────────────────────────────────────────────
    lp_delta_stats: dict = {}
    lp_delta_tensor: Optional[torch.Tensor] = None
    layer_lp_corr: Optional[List[float]] = None

    if run1_lp is not None and stacked_lp is not None:
        mean_lp_run1 = run1_lp.mean(dim=0)     # [S-1]
        mean_lp_run2 = stacked_lp.mean(dim=0)  # [S-1]
        lp_delta_tensor = (mean_lp_run1 - mean_lp_run2).abs()  # [S-1]

        lp_delta_stats = {
            "mean": lp_delta_tensor.mean().item(),
            "p95":  lp_delta_tensor.quantile(0.95).item(),
            "p99":  lp_delta_tensor.quantile(0.99).item(),
            "max":  lp_delta_tensor.max().item(),
        }

        # Per-layer correlation: does more routing disagreement at layer l
        # correlate with larger logprob delta?
        # routing_disagree[l, s] vs lp_delta[s].  Use Pearson r over tokens.
        routing_disagree = (1.0 - cross_agreement)  # [L, S]
        # Align token axis: logprob delta is [S-1] (predicting token at s+1),
        # routing decisions at position s affect the output at s.  Trim last token.
        lp_trim = lp_delta_tensor                    # [S-1]
        rd_trim  = routing_disagree[:, :-1]          # [L, S-1]

        corrs: List[float] = []
        for l_idx in range(rd_trim.shape[0]):
            rd_l = rd_trim[l_idx]                    # [S-1]
            if rd_l.std() < 1e-8:
                corrs.append(float('nan'))
            else:
                # Pearson correlation.
                rd_z  = (rd_l  - rd_l.mean())  / rd_l.std()
                lp_z  = (lp_trim - lp_trim.mean()) / (lp_trim.std() + 1e-12)
                corrs.append((rd_z * lp_z).mean().item())
        layer_lp_corr = corrs

    # ── Print summary ─────────────────────────────────────────────────────────
    print_rank_0("=" * 60)
    print_rank_0("Step 3B results (across-restart comparison):")
    print_rank_0(f"  N passes per run              : {n_passes}")
    print_rank_0(f"  Cross-run pair count          : {n_passes * n_passes}")
    print_rank_0(f"  Within-Run-2 mean agreement   : {overall_within_mean:.8f}")
    print_rank_0(f"  Cross-run mean agreement      : {overall_cross_mean:.8f}")
    print_rank_0(f"  Cross-run min  agreement      : {overall_cross_min:.8f}")
    print_rank_0("  Per MoE layer (cross-run agreement):")
    for l_idx, (mn, mi) in enumerate(zip(per_layer_cross_mean, per_layer_cross_min)):
        flag = " <<<" if mn < 1.0 else ""
        corr_str = (
            f"  corr(disagree,|Δlp|)={layer_lp_corr[l_idx]:.4f}"
            if layer_lp_corr and not (isinstance(layer_lp_corr[l_idx], float) and
                                      layer_lp_corr[l_idx] != layer_lp_corr[l_idx])
            else ""
        )
        print_rank_0(
            f"    MoE layer {l_idx:2d}: mean={mn:.8f}  min={mi:.8f}{flag}{corr_str}"
        )
    if lp_delta_stats:
        print_rank_0(
            f"  |Δlogprob| (run1_avg vs run2_avg): "
            f"mean={lp_delta_stats['mean']:.6f}  "
            f"p95={lp_delta_stats['p95']:.6f}  "
            f"p99={lp_delta_stats['p99']:.6f}  "
            f"max={lp_delta_stats['max']:.6f}"
        )
    else:
        print_rank_0("  Logprob delta: not available on this rank.")

    if overall_cross_mean < 1.0:
        print_rank_0("*** CROSS-RESTART NON-DETERMINISM DETECTED ***")
    else:
        print_rank_0("Fully consistent across restarts — router is reproducible given the same seed.")
    print_rank_0("=" * 60)

    # ── Persist artefacts on rank 0 ───────────────────────────────────────────
    if torch.distributed.get_rank() == 0:
        results = {
            "step": "3B",
            "description": "across-restart routing consistency",
            "n_passes_per_run": n_passes,
            "cross_run_pair_count": n_passes * n_passes,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "total_tokens_S": seq_len * batch_size,
            "num_moe_layers": num_moe_layers,
            "study_seed": study_seed,
            "within_run2_mean_agreement": overall_within_mean,
            "cross_run_mean_agreement": overall_cross_mean,
            "cross_run_min_agreement": overall_cross_min,
            "per_layer_cross_mean_agreement": per_layer_cross_mean,
            "per_layer_cross_min_agreement": per_layer_cross_min,
            "per_layer_disagree_rate": per_layer_disagree.tolist(),
            "logprob_delta": lp_delta_stats,
            "per_layer_corr_disagree_vs_logprob_delta": layer_lp_corr,
        }
        out_json = os.path.join(results_dir, 'step3b_routing_consistency.json')
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=2)
        print_rank_0(f"Saved → {out_json}")

        out_npy = os.path.join(results_dir, 'step3b_cross_agreement_matrix.npy')
        np.save(out_npy, cross_agreement.numpy())
        print_rank_0(f"Saved → {out_npy}")

        if lp_delta_tensor is not None:
            out_lp = os.path.join(results_dir, 'step3b_logprob_delta.npy')
            np.save(out_lp, lp_delta_tensor.numpy())
            print_rank_0(f"Saved → {out_lp}")


# ---------------------------------------------------------------------------
# Step 3C: non-determinism → logprob impact (M6)
# ---------------------------------------------------------------------------

def step3c_nondeterminism_logprob_impact(
    model_chunk,
    args,
    results_dir: str,
    n_passes: int = 10,
    seq_len: int = 1024,
    batch_size: int = 1,
    prompts_path: Optional[str] = None,
) -> None:
    """Step 3C: quantify how natural GPU non-determinism translates into logprob drift.

    Runs N passes with the default (non-deterministic) GPU settings, captures
    both routing maps and per-token logprobs, then computes pairwise |Δlogprob|
    across pass pairs and correlates that with per-layer routing disagree rate.

    When prompts_path is supplied the study runs on each real prompt in that JSON
    file (same format as router_study_prompts.json) and accumulates disagree rates
    and |Δlogprob| across all prompts × tokens, giving a measurement grounded in
    the actual Calendar rollout distribution.  When prompts_path is None (default)
    a single random token batch of length seq_len is used instead.

    Skipped automatically when --deterministic-mode is active (the result would
    be 0 by construction; the interesting measurement is the non-deterministic case).

    Artefacts saved to results_dir:
      step3c_logprob_impact.json        — summary stats + per-layer correlation
      step3c_logprob_delta_matrix.npy   — [N_pairs, total_tokens] per-token |Δlogprob|
    """
    if getattr(args, 'deterministic_mode', False):
        print_rank_0(
            "Step 3C skipped: --deterministic-mode is active "
            "(routing and logprobs are identical across passes by construction)."
        )
        return

    study_seed = getattr(args, 'seed', 6789)
    device = torch.cuda.current_device()

    # ── Build the list of (input_ids, position_ids) batches to process ────────
    use_real_prompts = prompts_path is not None and os.path.exists(prompts_path)
    if use_real_prompts:
        with open(prompts_path) as f:
            prompt_data = json.load(f)
        prompt_recs = prompt_data["prompts"]
        batches = []
        for rec in prompt_recs:
            ids = torch.tensor(rec["input_ids"], dtype=torch.long, device=device).unsqueeze(1)
            pos = torch.arange(ids.shape[0], device=device).unsqueeze(1)
            batches.append((ids, pos))
        source_desc = f"{len(batches)} real Calendar rollouts from {os.path.basename(prompts_path)}"
        print_rank_0("=" * 60)
        print_rank_0(
            f"Step 3C: non-determinism → logprob impact  "
            f"(N={n_passes}, {source_desc})"
        )
    else:
        vocab_size = getattr(args, 'padded_vocab_size', 32000)
        seq_len = min(seq_len, getattr(args, 'seq_length', seq_len))
        torch.manual_seed(study_seed)
        ids = torch.randint(0, vocab_size, (seq_len, batch_size), device=device)
        pos = torch.arange(seq_len, device=device).unsqueeze(1).expand(seq_len, batch_size)
        batches = [(ids, pos)]
        source_desc = f"random batch (seq_len={seq_len}, batch_size={batch_size}, seed={study_seed})"
        print_rank_0("=" * 60)
        print_rank_0(
            f"Step 3C: non-determinism → logprob impact  "
            f"(N={n_passes}, {source_desc})"
        )

    n_pairs = n_passes * (n_passes - 1) // 2
    n_layers: int = 0

    # Per-pair accumulators (indexed [pair_idx]):
    #   disagree_sum[pair][layer] — sum of per-token disagree indicators
    #   disagree_cnt[pair][layer] — number of tokens contributing
    #   delta_lp_parts[pair]      — list of [S-1] |Δlogprob| tensors to concatenate
    disagree_sum: List[List[float]] = [[] for _ in range(n_pairs)]
    disagree_cnt: List[List[int]]   = [[] for _ in range(n_pairs)]
    delta_lp_parts: List[List[torch.Tensor]] = [[] for _ in range(n_pairs)]
    has_logprobs = False

    for b_idx, (input_ids, position_ids) in enumerate(batches):
        label = (f"3C p{b_idx+1}/{len(batches)}" if use_real_prompts else "3C")
        stacked_maps, stacked_lp = _collect_passes(
            model_chunk, input_ids, position_ids, n_passes, label=label
        )
        # stacked_maps: [N, L, S, E] bool on CPU
        # stacked_lp:   [N, S-1] float32 on CPU, or None

        if n_layers == 0:
            n_layers = stacked_maps.shape[1]
            for p in range(n_pairs):
                disagree_sum[p] = [0.0] * n_layers
                disagree_cnt[p] = [0]   * n_layers

        if stacked_lp is not None:
            has_logprobs = True
        elif b_idx == 0:
            print_rank_0(
                "WARNING: logprobs unavailable (could not gather full vocab logits). "
                "Saving routing-only results."
            )

        S = stacked_maps.shape[2]
        pair_idx = 0
        for i in range(n_passes):
            for j in range(i + 1, n_passes):
                disagree = (~(stacked_maps[i] == stacked_maps[j]).all(dim=-1)).float()  # [L, S]
                for l_idx in range(n_layers):
                    disagree_sum[pair_idx][l_idx] += disagree[l_idx].sum().item()
                    disagree_cnt[pair_idx][l_idx] += S
                if stacked_lp is not None:
                    delta_lp_parts[pair_idx].append(
                        (stacked_lp[i] - stacked_lp[j]).abs()   # [S-1]
                    )
                pair_idx += 1

    # ── Aggregate across prompts ──────────────────────────────────────────────
    # disagree_matrix[pair, layer] = disagree rate (tokens-weighted across all prompts)
    disagree_rows: List[List[float]] = []
    for pair_idx in range(n_pairs):
        row = [
            disagree_sum[pair_idx][l] / disagree_cnt[pair_idx][l]
            if disagree_cnt[pair_idx][l] > 0 else 0.0
            for l in range(n_layers)
        ]
        disagree_rows.append(row)
    disagree_matrix = torch.tensor(disagree_rows)   # [N_pairs, L]

    # delta_lp_all[pair_idx] = [total_tokens_across_prompts] concatenated
    delta_lp_list: List[torch.Tensor] = []
    if has_logprobs:
        for pair_idx in range(n_pairs):
            if delta_lp_parts[pair_idx]:
                delta_lp_list.append(torch.cat(delta_lp_parts[pair_idx]))

    # ── Summary stats and correlation ─────────────────────────────────────────
    lp_stats: dict = {}
    corrs: List[float] = [float('nan')] * n_layers
    delta_matrix: Optional[torch.Tensor] = None

    if delta_lp_list:
        delta_matrix = torch.stack(delta_lp_list)           # [N_pairs, total_tokens]
        flat = delta_matrix.flatten()
        grpo_clip = 0.2
        lp_stats = {
            "mean":            flat.mean().item(),
            "p50":             flat.quantile(0.50).item(),
            "p95":             flat.quantile(0.95).item(),
            "p99":             flat.quantile(0.99).item(),
            "max":             flat.max().item(),
            "grpo_clip_eps":   grpo_clip,
            "frac_above_clip": (flat > grpo_clip).float().mean().item(),
        }

        # Per-layer Pearson corr: disagree_rate[l] ↔ mean |Δlogprob| per pair.
        pair_mean_delta = delta_matrix.mean(dim=-1)          # [N_pairs]
        for l_idx in range(n_layers):
            x = disagree_matrix[:, l_idx]                   # [N_pairs]
            if x.std() < 1e-9 or pair_mean_delta.std() < 1e-9:
                corrs[l_idx] = float('nan')
            else:
                corrs[l_idx] = torch.corrcoef(
                    torch.stack([x, pair_mean_delta])
                )[0, 1].item()

    # ── Print ─────────────────────────────────────────────────────────────────
    print_rank_0("=" * 60)
    print_rank_0("Step 3C results:")
    print_rank_0(f"  Source                    : {source_desc}")
    print_rank_0(f"  Pass pairs evaluated      : {n_pairs}")
    if lp_stats:
        print_rank_0(f"  Total token samples       : {delta_matrix.numel()}")
        print_rank_0(f"  |Δlogprob| mean           : {lp_stats['mean']:.6f}")
        print_rank_0(f"  |Δlogprob| p50            : {lp_stats['p50']:.6f}")
        print_rank_0(f"  |Δlogprob| p95            : {lp_stats['p95']:.6f}")
        print_rank_0(f"  |Δlogprob| p99            : {lp_stats['p99']:.6f}")
        print_rank_0(f"  |Δlogprob| max            : {lp_stats['max']:.6f}")
        print_rank_0(f"  GRPO clip threshold ε     : {lp_stats['grpo_clip_eps']}")
        print_rank_0(f"  Fraction |Δlogprob| > ε   : {lp_stats['frac_above_clip']:.6f}")
        print_rank_0("  Per-layer disagree rate + corr(disagree, |Δlogprob|):")
        for l_idx in range(n_layers):
            dr = disagree_matrix[:, l_idx].mean().item()
            c  = corrs[l_idx]
            c_str = f"{c:.4f}" if c == c else "  nan"
            print_rank_0(f"    MoE layer {l_idx:2d}: disagree={dr:.4f}  corr={c_str}")
    else:
        print_rank_0("  (logprob delta not available — routing disagree rates only)")
        for l_idx in range(n_layers):
            dr = disagree_matrix[:, l_idx].mean().item()
            print_rank_0(f"    MoE layer {l_idx:2d}: disagree={dr:.4f}")
    print_rank_0("=" * 60)

    # ── Persist on rank 0 ─────────────────────────────────────────────────────
    if torch.distributed.get_rank() == 0:
        results = {
            "step": "3C",
            "description": "non-determinism logprob impact (no --deterministic-mode)",
            "source": source_desc,
            "n_passes": n_passes,
            "n_pass_pairs": n_pairs,
            "num_moe_layers": n_layers,
            "study_seed": study_seed,
            "logprob_delta": lp_stats,
            "per_layer_mean_disagree_rate": disagree_matrix.mean(dim=0).tolist(),
            "per_layer_corr_disagree_vs_logprob_delta": corrs,
        }
        out_json = os.path.join(results_dir, 'step3c_logprob_impact.json')
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=2)
        print_rank_0(f"Saved → {out_json}")

        if delta_matrix is not None:
            out_npy = os.path.join(results_dir, 'step3c_logprob_delta_matrix.npy')
            np.save(out_npy, delta_matrix.numpy())
            print_rank_0(f"Saved → {out_npy}")


# ---------------------------------------------------------------------------
# Step 3D: real Calendar prompts — probability difference study (M6.5)
# ---------------------------------------------------------------------------

def step3d_real_prompt_prob_impact(
    model_chunk,
    args,
    results_dir: str,
    prompts_path: str,
    n_passes: int = 5,
) -> None:
    """Step 3D: measure |Δprob| on real chat-formatted Calendar prompts.

    Answers: can training-side routing non-determinism produce the token
    probability differences of up to 1.0 observed in production GRPO runs?

    Runs n_passes forward passes for each prompt in prompts_path, computes
    pairwise |prob_i(token) - prob_j(token)| and reports per-confidence-tier
    statistics so high-probability tokens (where |Δprob| could reach 1.0) are
    visible separately from low-probability ones.

    Skipped automatically when --deterministic-mode is active.

    Artefacts:
      step3d_real_prob_impact.json   — summary stats by confidence tier
    """
    if getattr(args, 'deterministic_mode', False):
        print_rank_0("Step 3D skipped: --deterministic-mode is active.")
        return

    if not os.path.exists(prompts_path):
        print_rank_0(f"Step 3D skipped: prompts file not found at {prompts_path}")
        return

    with open(prompts_path) as f:
        prompt_data = json.load(f)

    prompts = prompt_data["prompts"]
    print_rank_0("=" * 60)
    print_rank_0(
        f"Step 3D: real Calendar prompts → |Δprob|  "
        f"({len(prompts)} prompts, N={n_passes} passes each)"
    )

    device = torch.cuda.current_device()

    # Collect per-token (prob_i, prob_j) for all prompt × pair combinations.
    # We store: for each token sample, (mean_prob_across_passes, delta_prob_for_pair)
    # These are kept on CPU to avoid GPU OOM across many prompts.
    all_mean_p: List[torch.Tensor]  = []   # [T] mean prob across passes, per token
    all_delta_p: List[torch.Tensor] = []   # [T] |prob_i - prob_j|, per pair per token

    for p_idx, rec in enumerate(prompts):
        ids = torch.tensor(rec["input_ids"], dtype=torch.long, device=device).unsqueeze(1)  # [S, 1]
        seq_len = ids.shape[0]
        pos = torch.arange(seq_len, device=device).unsqueeze(1)

        stacked_maps, stacked_lp = _collect_passes(
            model_chunk, ids, pos, n_passes, label=f"3D p{p_idx+1}/{len(prompts)}"
        )

        if stacked_lp is None:
            continue

        # stacked_lp: [N, S-1]  logprobs
        probs = torch.exp(stacked_lp)          # [N, S-1], in [0, 1]
        mean_p = probs.mean(dim=0)             # [S-1]

        for i in range(n_passes):
            for j in range(i + 1, n_passes):
                dp = (probs[i] - probs[j]).abs()   # [S-1]
                all_mean_p.append(mean_p)
                all_delta_p.append(dp)

        print_rank_0(f"  prompt {p_idx+1}/{len(prompts)}: seq_len={seq_len}, "
                     f"max_prob={mean_p.max():.4f}, "
                     f"max_|Δp|={(torch.stack([( torch.exp(stacked_lp[i]) - torch.exp(stacked_lp[j])).abs() for i in range(n_passes) for j in range(i+1, n_passes)]).max() if n_passes > 1 else torch.tensor(0.0)):.4f}")

    if not all_delta_p:
        print_rank_0("Step 3D: no logprob data available — aborting.")
        return

    mean_p_all  = torch.cat(all_mean_p)    # [total_token_samples]
    delta_p_all = torch.cat(all_delta_p)   # [total_token_samples]

    def tier_stats(mask_label, mask):
        n = mask.sum().item()
        if n == 0:
            return {"n": 0}
        dp = delta_p_all[mask]
        return {
            "n":    n,
            "mean": dp.mean().item(),
            "p50":  dp.quantile(0.50).item(),
            "p95":  dp.quantile(0.95).item(),
            "p99":  dp.quantile(0.99).item(),
            "max":  dp.max().item(),
        }

    tiers = {
        "all":           tier_stats("all",  torch.ones(len(delta_p_all), dtype=torch.bool)),
        "mean_p_gt_0.1": tier_stats(">0.1", mean_p_all > 0.1),
        "mean_p_gt_0.3": tier_stats(">0.3", mean_p_all > 0.3),
        "mean_p_gt_0.5": tier_stats(">0.5", mean_p_all > 0.5),
        "mean_p_gt_0.9": tier_stats(">0.9", mean_p_all > 0.9),
    }

    print_rank_0("=" * 60)
    print_rank_0("Step 3D results — |Δprob| by token confidence tier:")
    print_rank_0(f"  Total token-pair samples : {len(delta_p_all)}")
    print_rank_0(f"  {'Tier':<22}  {'n':>8}  {'mean':>8}  {'p95':>8}  {'p99':>8}  {'max':>8}")
    for label, s in tiers.items():
        if s["n"] == 0:
            print_rank_0(f"  {label:<22}  {'0':>8}  {'—':>8}  {'—':>8}  {'—':>8}  {'—':>8}")
        else:
            print_rank_0(
                f"  {label:<22}  {s['n']:>8}  {s['mean']:>8.4f}  "
                f"{s['p95']:>8.4f}  {s['p99']:>8.4f}  {s['max']:>8.4f}"
            )
    print_rank_0("=" * 60)

    if torch.distributed.get_rank() == 0:
        results = {
            "step": "3D",
            "description": "Calendar inference-engine rollouts |Δprob| by confidence tier (training-side non-determinism)",
            "n_prompts": len(prompts),
            "n_passes": n_passes,
            "tiers": tiers,
        }
        out_json = os.path.join(results_dir, 'step3d_real_prob_impact.json')
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print_rank_0(f"Saved → {out_json}")


# ---------------------------------------------------------------------------
# Step 7A: training-side rescore for inference vs training comparison (M10)
# ---------------------------------------------------------------------------

def _routing_map_to_indices(stacked_maps: torch.Tensor) -> torch.Tensor:
    """Convert bool routing bitmasks to sorted expert index tensors.

    Args:
        stacked_maps: [N, L, S, E] bool — output of _collect_passes()

    Returns:
        [N, S, L, top_k] int32 — ascending expert indices per (pass, token, layer).
        Format matches the inference engine's routing_indices field so results can
        be compared directly.
    """
    N, L, S, E = stacked_maps.shape
    top_k = int(stacked_maps[0, 0, 0].sum().item())
    # Flatten passes × layers × tokens → rows; each row has exactly top_k True entries.
    flat = stacked_maps.reshape(N * L * S, E)           # [N*L*S, E]
    nz   = flat.nonzero()                               # [N*L*S*top_k, 2] sorted
    expert_idx = nz[:, 1].reshape(N, L, S, top_k)      # [N, L, S, top_k]
    return expert_idx.permute(0, 2, 1, 3).to(torch.int32)  # [N, S, L, top_k]


def step7a_training_rescore(
    model_chunk,
    args,
    results_dir: str,
    prompts_path: str,
    n_passes: int = 5,
) -> None:
    """Step 7A: capture training-side routing indices and logprobs.

    For each prompt in prompts_path, runs n_passes forward passes and records:
    - Per-layer routing indices [S, n_layers, top_k] int32 (ascending expert order)
    - Per-token logprobs [S-1] float32

    These artefacts are later compared against inference-side captures (Step 7B)
    to decompose the inference vs training logprob gap into its determinism and
    architecture (KV-cache vs full-seq attention) components.

    The output filenames carry a determinism tag ("det" / "nodet") so both the
    --deterministic-mode and default runs can coexist in the same results_dir.

    Artefacts:
      step7a_single_{tag}_routing.npz  — keys 'p{i:02d}': [N, S, L, top_k] int32
      step7a_single_{tag}_logprobs.npz — keys 'p{i:02d}': [N, S-1] float32
      step7a_single_{tag}_stats.json   — aggregate routing disagree + |Δlogprob| stats
    """
    if not os.path.exists(prompts_path):
        print_rank_0(
            f"Step 7A skipped: prompts file not found at {prompts_path}"
        )
        return

    with open(prompts_path) as f:
        prompt_data = json.load(f)
    prompts = prompt_data["prompts"]

    det_tag   = "det" if getattr(args, 'deterministic_mode', False) else "nodet"
    study_seed = getattr(args, 'seed', 6789)
    device    = torch.cuda.current_device()

    print_rank_0("=" * 60)
    print_rank_0(
        f"Step 7A: training-side rescore  "
        f"({len(prompts)} prompts, N={n_passes} passes, mode={det_tag})"
    )

    # Per-prompt accumulators for aggregate stats.
    all_disagree_sum: List[List[float]] = []  # [pair, layer]
    all_disagree_cnt: List[List[int]]   = []
    all_delta_lp: List[torch.Tensor]    = []   # flattened |Δlogprob| across all pairs × tokens
    n_layers_global = 0

    # Per-prompt artefacts saved to disk.
    routing_arrays: dict = {}   # key → numpy array
    logprob_arrays: dict = {}

    for p_idx, rec in enumerate(prompts):
        ids = torch.tensor(rec["input_ids"], dtype=torch.long, device=device).unsqueeze(1)
        pos = torch.arange(ids.shape[0], device=device).unsqueeze(1)
        S   = ids.shape[0]

        label = f"7A-{det_tag} p{p_idx+1}/{len(prompts)}"
        stacked_maps, stacked_lp = _collect_passes(
            model_chunk, ids, pos, n_passes, label=label
        )
        # stacked_maps: [N, L, S, E] bool on CPU
        # stacked_lp:   [N, S-1] float32 on CPU, or None

        N_got, L, _S, E = stacked_maps.shape
        if n_layers_global == 0:
            n_layers_global = L
            top_k = int(stacked_maps[0, 0, 0].sum().item())

        # Convert bitmasks → sorted expert index tensors [N, S, L, top_k] int32.
        routing_idx = _routing_map_to_indices(stacked_maps)  # on CPU

        # Store per-prompt arrays.
        key = f"p{p_idx:02d}"
        routing_arrays[key] = routing_idx.numpy()
        if stacked_lp is not None:
            logprob_arrays[key] = stacked_lp.numpy()

        # ── Pairwise consistency stats for this prompt ─────────────────────────
        n_pairs = N_got * (N_got - 1) // 2
        if len(all_disagree_sum) == 0:
            all_disagree_sum = [[0.0] * L for _ in range(n_pairs)]
            all_disagree_cnt = [[0]   * L for _ in range(n_pairs)]

        pair_idx = 0
        for i in range(N_got):
            for j in range(i + 1, N_got):
                disagree = (~(stacked_maps[i] == stacked_maps[j]).all(dim=-1)).float()  # [L, S]
                for l_idx in range(L):
                    all_disagree_sum[pair_idx][l_idx] += disagree[l_idx].sum().item()
                    all_disagree_cnt[pair_idx][l_idx] += _S
                if stacked_lp is not None:
                    all_delta_lp.append((stacked_lp[i] - stacked_lp[j]).abs())
                pair_idx += 1

        print_rank_0(
            f"  prompt {p_idx+1}/{len(prompts)}: seq_len={S}, "
            f"top_k={top_k if n_layers_global > 0 else '?'}"
        )

    # ── Aggregate stats ────────────────────────────────────────────────────────
    n_pairs = N_got * (N_got - 1) // 2
    disagree_rows: List[List[float]] = []
    for pair_idx in range(n_pairs):
        row = [
            all_disagree_sum[pair_idx][l] / all_disagree_cnt[pair_idx][l]
            if all_disagree_cnt[pair_idx][l] > 0 else 0.0
            for l in range(n_layers_global)
        ]
        disagree_rows.append(row)
    disagree_matrix = torch.tensor(disagree_rows)   # [n_pairs, L]

    lp_stats: dict = {}
    if all_delta_lp:
        flat_delta = torch.cat(all_delta_lp)
        grpo_clip  = 0.2
        lp_stats = {
            "mean":            flat_delta.mean().item(),
            "p50":             flat_delta.quantile(0.50).item(),
            "p95":             flat_delta.quantile(0.95).item(),
            "p99":             flat_delta.quantile(0.99).item(),
            "max":             flat_delta.max().item(),
            "grpo_clip_eps":   grpo_clip,
            "frac_above_clip": (flat_delta > grpo_clip).float().mean().item(),
            "n_samples":       flat_delta.numel(),
        }

    per_layer_mean_disagree = disagree_matrix.mean(dim=0).tolist()

    # ── Print summary ─────────────────────────────────────────────────────────
    print_rank_0("=" * 60)
    print_rank_0(f"Step 7A results (mode={det_tag}):")
    print_rank_0(f"  Prompts processed    : {len(prompts)}")
    print_rank_0(f"  Passes per prompt    : {n_passes}")
    print_rank_0(f"  MoE layers           : {n_layers_global}")
    if lp_stats:
        print_rank_0(f"  Total token samples  : {lp_stats['n_samples']}")
        print_rank_0(f"  |Δlogprob| mean      : {lp_stats['mean']:.6f}")
        print_rank_0(f"  |Δlogprob| p50       : {lp_stats['p50']:.6f}")
        print_rank_0(f"  |Δlogprob| p95       : {lp_stats['p95']:.6f}")
        print_rank_0(f"  |Δlogprob| p99       : {lp_stats['p99']:.6f}")
        print_rank_0(f"  |Δlogprob| max       : {lp_stats['max']:.6f}")
        print_rank_0(f"  Fraction > ε=0.2     : {lp_stats['frac_above_clip']:.6f}")
    print_rank_0("  Per-layer routing disagree rate:")
    for l_idx, dr in enumerate(per_layer_mean_disagree):
        flag = " <<<" if dr > 0.05 else ""
        print_rank_0(f"    Layer {l_idx:2d}: {dr:.4f}{flag}")
    print_rank_0("=" * 60)

    # ── Persist artefacts on rank 0 ───────────────────────────────────────────
    if torch.distributed.get_rank() == 0:
        routing_path = os.path.join(results_dir, f'step7a_single_{det_tag}_routing.npz')
        np.savez(routing_path, **routing_arrays)
        print_rank_0(f"Saved → {routing_path}")

        if logprob_arrays:
            lp_path = os.path.join(results_dir, f'step7a_single_{det_tag}_logprobs.npz')
            np.savez(lp_path, **logprob_arrays)
            print_rank_0(f"Saved → {lp_path}")

        stats = {
            "step":             "7A",
            "description":      f"training-side rescore ({det_tag})",
            "deterministic":    getattr(args, 'deterministic_mode', False),
            "det_tag":          det_tag,
            "n_prompts":        len(prompts),
            "n_passes":         n_passes,
            "n_pass_pairs":     n_pairs,
            "n_moe_layers":     n_layers_global,
            "study_seed":       study_seed,
            "per_layer_mean_disagree_rate": per_layer_mean_disagree,
            "logprob_delta":    lp_stats,
        }
        stats_path = os.path.join(results_dir, f'step7a_single_{det_tag}_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print_rank_0(f"Saved → {stats_path}")


# ---------------------------------------------------------------------------
# Step 7C: inference vs training comparison (M16)
# ---------------------------------------------------------------------------

def step7c_compare(
    model_chunk,
    args,
    results_dir: str,
    rollout_dump_dir: str,
    n_passes: int = 1,
) -> None:
    """Step 7C: compare inference-side logprobs/probs/routing to training-side.

    Loads rollout files produced by the ROUTER_STUDY_DUMP_DIR hook (prompt_tokens,
    generated_tokens, generated_log_probs, routing_indices) and rescores each full
    sequence through the training model with full-sequence attention.

    Routing alignment
    -----------------
    inference routing_indices[k]   = routing when g_k was the input token (decode step k+1).
    training  routing at position P+k = same computation under full-seq attention.
    Both represent "routing for token g_k"; they may disagree because KV-cache
    vs full-sequence attention produces different hidden states entering the router.

    The logprob of g_j is set by routing at position P-1+j (= routing_indices[j-1]
    for j>=1, or the final prefill step for j=0).  Routing agreement at position P+k
    therefore relates to the logprob of g_{k+1}, not g_k.

    Outputs (per det/nodet tag, saved to results_dir):
      step7c_{tag}_comparison.json       — aggregate stats + decomposition table
      step7c_{tag}_logprob_delta.npy     — [R, max_G] float32  Δlp = train_lp - inf_lp
      step7c_{tag}_prob_delta.npy        — [R, max_G] float32  |Δp| = |exp(train) - exp(inf)|
      step7c_{tag}_routing_agree.npy     — [R, max_G, L] float32  per-token per-layer agreement
    """
    import glob as _glob

    det_tag = "det" if getattr(args, "deterministic_mode", False) else "nodet"
    device = torch.cuda.current_device()

    rollout_paths = sorted(_glob.glob(os.path.join(rollout_dump_dir, "rollout_*.npz")))
    if not rollout_paths:
        print_rank_0(
            f"Step 7C skipped: no rollout_*.npz files found in {rollout_dump_dir}"
        )
        return

    num_moe_layers = _count_moe_layers(model_chunk)

    from megatron.core import mpu as _mpu

    tp_size = _mpu.get_tensor_model_parallel_world_size()
    sp_enabled = getattr(args, "sequence_parallel", False)
    print_rank_0("=" * 60)
    print_rank_0(
        f"Step 7C: inference vs training comparison  "
        f"({len(rollout_paths)} rollouts, N={n_passes} passes, mode={det_tag})"
    )
    print_rank_0(f"  Rollout dir : {rollout_dump_dir}")
    print_rank_0(f"  TP={tp_size}  sequence_parallel={sp_enabled}")

    per_rollout_delta_lp: List[torch.Tensor] = []    # [G] each
    per_rollout_delta_p:  List[torch.Tensor] = []    # [G] each
    per_rollout_agree:    List[torch.Tensor] = []    # [G, L] each
    has_routing = False

    for r_idx, npz_path in enumerate(rollout_paths):
        data = np.load(npz_path)
        prompt_t = torch.tensor(data["prompt_tokens"], dtype=torch.long, device=device)
        gen_t    = torch.tensor(data["generated_tokens"], dtype=torch.long, device=device)
        inf_lp   = torch.tensor(data["generated_log_probs"], dtype=torch.float32)  # [G] CPU

        inf_routing: Optional[torch.Tensor] = None
        if "routing_indices" in data:
            inf_routing = torch.tensor(data["routing_indices"], dtype=torch.int32)  # [G, L, top_k] CPU
            has_routing = True

        P = prompt_t.shape[0]
        G = gen_t.shape[0]

        # Full sequence for the training-side forward pass.
        full_ids = torch.cat([prompt_t, gen_t]).unsqueeze(1)   # [P+G, 1]
        pos_ids  = torch.arange(P + G, device=device).unsqueeze(1)  # [P+G, 1]

        # The inference model receives expert_bias=0 (refit only copies named_parameters,
        # not registered buffers).  Zero out expert_bias on the training model so that
        # both sides use identical routing inputs.  We save and restore after each rollout.
        saved_biases: List[torch.Tensor] = []
        for layer in _unwrap(model_chunk).decoder.layers:
            mlp = getattr(layer, 'mlp', None)
            if mlp is None:
                continue
            router = getattr(mlp, 'router', None)
            if router is None or not hasattr(router, 'expert_bias') or router.expert_bias is None:
                continue
            saved_biases.append(router.expert_bias.data.clone())
            router.expert_bias.data.zero_()

        store   = RouterActivationStore()
        handles = register_router_hooks(model_chunk, store)

        pass_lp_list:    List[torch.Tensor] = []
        pass_agree_list: List[torch.Tensor] = []

        for _ in range(n_passes):
            store.clear()
            with torch.no_grad():
                output = model_chunk(full_ids, pos_ids, attention_mask=None)

            # ── Training logprobs for generated positions ──────────────────
            train_lp_gen: Optional[torch.Tensor] = None
            if isinstance(output, torch.Tensor) and output.ndim == 3:
                from megatron.core.tensor_parallel.mappings import (
                    gather_from_tensor_model_parallel_region,
                )
                _a = get_args()
                padded_vocab  = getattr(_a, "padded_vocab_size", 0)
                vocab_out_dim = output.shape[2]
                if padded_vocab > 0 and vocab_out_dim < padded_vocab:
                    full_logits = gather_from_tensor_model_parallel_region(output).float()
                elif padded_vocab == 0 or vocab_out_dim == padded_vocab:
                    full_logits = output.float()
                else:
                    full_logits = None

                if full_logits is not None:
                    # full_logits: [S_local, 1, V] (S_local=S/TP with SP, else S=P+G)
                    if r_idx == 0 and _ == 0:
                        print_rank_0(
                            f"[7C diag] full_logits shape={list(full_logits.shape)}"
                            f"  vocab_out_dim={vocab_out_dim}  padded_vocab={padded_vocab}"
                        )
                    log_probs = torch.nn.functional.log_softmax(full_logits, dim=-1)
                    # logprob(g_j) = log_probs[P-1+j, 0, g_j_id]
                    lp_at_gen = log_probs[P - 1: P - 1 + G, 0, :]      # [G, V] (empty if SP splits here)
                    train_lp_gen = lp_at_gen.gather(
                        1, gen_t[:lp_at_gen.shape[0]].unsqueeze(1)
                    ).squeeze(1).detach().cpu()                          # [G] or shorter if SP
                    if r_idx == 0 and _ == 0:
                        print_rank_0(
                            f"[7C diag] lp_at_gen shape={list(lp_at_gen.shape)}"
                            f"  train_lp_gen shape={list(train_lp_gen.shape)}"
                            f"  inf_lp[:5]={inf_lp[:5].tolist()}"
                        )
                        if train_lp_gen.numel() > 0:
                            print_rank_0(
                                f"[7C diag] train_lp_gen[:5]={train_lp_gen[:5].tolist()}"
                            )

            pass_lp_list.append(train_lp_gen)

            # ── Routing agreement at generated positions ───────────────────
            # Compare inference routing_indices[k] (routing for g_k)
            # vs training routing at position P+k (same computation, full-seq context).
            if inf_routing is not None and store.records:
                # Each rec.routing_map: [P+G, E_global] bool CPU
                layer_maps = torch.stack(
                    [rec.routing_map for rec in store.records]
                )                                                        # [L, S_local, E]
                gen_maps = layer_maps[:, P: P + G, :]                   # [L, G, E] bool

                # Diagnostics on first rollout, first pass only
                if r_idx == 0 and _ == 0:
                    print_rank_0(
                        f"[7C diag] r0 pass0: layer_maps shape={list(layer_maps.shape)}"
                        f"  P={P} G={G}  gen_maps shape={list(gen_maps.shape)}"
                    )
                    print_rank_0(
                        f"[7C diag] output type={type(output).__name__}"
                        f"  ndim={output.ndim if isinstance(output, torch.Tensor) else 'n/a'}"
                        f"  shape={list(output.shape) if isinstance(output, torch.Tensor) else 'n/a'}"
                    )
                    # Print router logits at gen position 0, layer 0: top-10 logits and their ranks
                    # This shows if hidden states entering the router are plausible.
                    rec0 = store.records[0]   # first MoE layer
                    logits_pos_P = rec0.logits[P]  # [E_local], fp64, for the first gen position
                    topv, topi = logits_pos_P.topk(10)
                    print_rank_0(
                        f"[7C diag] router logits at pos P(={P}), layer 0 (local E):"
                        f"  top-10 vals={topv.tolist()}  idx={topi.tolist()}"
                    )
                    # Bias at layer 0 (if set)
                    first_router = None
                    for layer in _unwrap(model_chunk).decoder.layers:
                        mlp = getattr(layer, 'mlp', None)
                        if mlp is not None and getattr(mlp, 'router', None) is not None:
                            first_router = mlp.router
                            break
                    if first_router is not None and first_router.expert_bias is not None:
                        bias = first_router.expert_bias.detach().cpu()
                        print_rank_0(
                            f"[7C diag] expert_bias at layer 0: min={bias.min().item():.4f}"
                            f"  max={bias.max().item():.4f}  mean={bias.mean().item():.4f}"
                        )

                L_cnt, G_cnt, E_cnt = gen_maps.shape
                flat = gen_maps.permute(1, 0, 2).reshape(G_cnt * L_cnt, E_cnt)
                nz   = flat.nonzero(as_tuple=False)                      # [G*L*top_k, 2]
                if nz.shape[0] > 0:
                    top_k_cnt  = nz.shape[0] // (G_cnt * L_cnt)
                    train_idx  = nz[:, 1].reshape(G_cnt, L_cnt, top_k_cnt).to(torch.int32)
                    # inf_routing: [G, L, top_k] CPU — truncate to G_cnt in case shorter
                    inf_idx    = inf_routing[:G_cnt]                     # [G, L, top_k]
                    # Sort both ascending before comparing (inf_routing is in score order, not sorted)
                    train_idx_s = train_idx.sort(dim=-1).values
                    inf_idx_s   = inf_idx.sort(dim=-1).values
                    agree      = (train_idx_s == inf_idx_s).all(dim=-1)  # [G, L] bool
                    # Diagnostics on first rollout, first pass only
                    if r_idx == 0 and _ == 0:
                        print_rank_0(
                            f"[7C diag] train_sorted[0,0]={train_idx_s[0,0].tolist()}"
                            f"  inf_sorted[0,0]={inf_idx_s[0,0].tolist()}"
                            f"  agree[0,0]={agree[0,0].item()}"
                        )
                        print_rank_0(
                            f"[7C diag] raw train_idx[0,0]={train_idx[0,0].tolist()}"
                            f"  raw inf_idx[0,0]={inf_idx[0,0].tolist()}"
                        )
                    pass_agree_list.append(agree.float())

        remove_hooks(handles)

        # Restore the expert biases that were zeroed for this rollout.
        bias_idx = 0
        for layer in _unwrap(model_chunk).decoder.layers:
            mlp = getattr(layer, 'mlp', None)
            if mlp is None:
                continue
            router = getattr(mlp, 'router', None)
            if router is None or not hasattr(router, 'expert_bias') or router.expert_bias is None:
                continue
            router.expert_bias.data.copy_(saved_biases[bias_idx])
            bias_idx += 1

        # Average across passes
        valid_lp = [lp for lp in pass_lp_list if lp is not None]
        if valid_lp:
            mean_train_lp = torch.stack(valid_lp).mean(dim=0)           # [G]
            delta_lp = mean_train_lp - inf_lp                           # signed: train − inf
            delta_p  = (torch.exp(mean_train_lp) - torch.exp(inf_lp)).abs()
            per_rollout_delta_lp.append(delta_lp)
            per_rollout_delta_p.append(delta_p)
            mean_abs = delta_lp.abs().mean().item()
        else:
            mean_abs = float("nan")

        mean_agree_summary = ""
        if pass_agree_list:
            mean_agree = torch.stack(pass_agree_list).mean(dim=0)       # [G, L]
            per_rollout_agree.append(mean_agree)
            mean_agree_summary = f"  routing_agree={mean_agree.mean().item():.3f}"

        print_rank_0(
            f"  rollout {r_idx:02d}: prompt={P} gen={G}"
            f"  |Δlp|_mean={mean_abs:.4f}{mean_agree_summary}"
        )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    if not per_rollout_delta_lp:
        print_rank_0("Step 7C: no logprob data available — aborting.")
        return

    flat_dlp  = torch.cat(per_rollout_delta_lp)     # [N_total_gen_tokens]
    flat_dp   = torch.cat(per_rollout_delta_p)
    abs_dlp   = flat_dlp.abs()
    grpo_clip = 0.2

    lp_stats = {
        "mean_signed_delta": flat_dlp.mean().item(),
        "mean":              abs_dlp.mean().item(),
        "p50":               abs_dlp.quantile(0.50).item(),
        "p95":               abs_dlp.quantile(0.95).item(),
        "p99":               abs_dlp.quantile(0.99).item(),
        "max":               abs_dlp.max().item(),
        "grpo_clip_eps":     grpo_clip,
        "frac_above_clip":   (abs_dlp > grpo_clip).float().mean().item(),
        "n_tokens":          int(flat_dlp.numel()),
    }
    p_stats = {
        "mean": flat_dp.mean().item(),
        "p50":  flat_dp.quantile(0.50).item(),
        "p95":  flat_dp.quantile(0.95).item(),
        "p99":  flat_dp.quantile(0.99).item(),
        "max":  flat_dp.max().item(),
    }

    routing_out: dict = {}
    per_layer_agree: List[float] = []
    if per_rollout_agree:
        all_agree = torch.cat(per_rollout_agree, dim=0)  # [N_total_gen, L]
        per_layer_agree = all_agree.mean(dim=0).tolist()
        routing_out = {
            "per_layer_agreement":    per_layer_agree,
            "overall_mean_agreement": all_agree.mean().item(),
        }

    # ── Print summary ─────────────────────────────────────────────────────────
    print_rank_0("=" * 60)
    print_rank_0(f"Step 7C results (mode={det_tag}):")
    print_rank_0(f"  Rollouts processed      : {len(per_rollout_delta_lp)}")
    print_rank_0(f"  Total generated tokens  : {lp_stats['n_tokens']}")
    print_rank_0(f"  Mean signed Δlogprob    : {lp_stats['mean_signed_delta']:+.6f}"
                 "  (+= train > inf, −= train < inf)")
    print_rank_0(f"  |Δlogprob| mean         : {lp_stats['mean']:.6f}")
    print_rank_0(f"  |Δlogprob| p50          : {lp_stats['p50']:.6f}")
    print_rank_0(f"  |Δlogprob| p95          : {lp_stats['p95']:.6f}")
    print_rank_0(f"  |Δlogprob| p99          : {lp_stats['p99']:.6f}")
    print_rank_0(f"  |Δlogprob| max          : {lp_stats['max']:.6f}")
    print_rank_0(f"  Frac |Δlogprob| > ε=0.2: {lp_stats['frac_above_clip']:.4f}")
    print_rank_0(f"  |Δprob| mean            : {p_stats['mean']:.6f}")
    print_rank_0(f"  |Δprob| p50             : {p_stats['p50']:.6f}")
    print_rank_0(f"  |Δprob| p95             : {p_stats['p95']:.6f}")
    print_rank_0(f"  |Δprob| max             : {p_stats['max']:.6f}")
    if per_layer_agree:
        print_rank_0(
            f"  Routing agreement inf vs train (routing_indices[k] vs training at P+k):"
        )
        for l_idx, ag in enumerate(per_layer_agree):
            flag = " <<<" if ag < 0.95 else ""
            print_rank_0(f"    MoE layer {l_idx:2d}: {ag:.4f}{flag}")
    print_rank_0("=" * 60)

    # ── Save artefacts on rank 0 ───────────────────────────────────────────────
    if torch.distributed.get_rank() == 0:
        max_G = max(t.shape[0] for t in per_rollout_delta_lp)

        def _pad(t: torch.Tensor, length: int) -> torch.Tensor:
            if t.shape[0] == length:
                return t
            return torch.cat([t, t.new_full((length - t.shape[0],), float("nan"))])

        lp_mat = torch.stack([_pad(t, max_G) for t in per_rollout_delta_lp])  # [R, max_G]
        p_mat  = torch.stack([_pad(t, max_G) for t in per_rollout_delta_p])

        out_json = os.path.join(results_dir, f"step7c_{det_tag}_comparison.json")
        with open(out_json, "w") as f:
            json.dump({
                "step":             "7C",
                "mode":             det_tag,
                "n_rollouts":       len(per_rollout_delta_lp),
                "n_passes":         n_passes,
                "rollout_dump_dir": rollout_dump_dir,
                "logprob_delta":    lp_stats,
                "prob_delta":       p_stats,
                "routing":          routing_out,
            }, f, indent=2)
        print_rank_0(f"Saved → {out_json}")

        lp_path = os.path.join(results_dir, f"step7c_{det_tag}_logprob_delta.npy")
        np.save(lp_path, lp_mat.numpy())
        print_rank_0(f"Saved → {lp_path}")

        p_path = os.path.join(results_dir, f"step7c_{det_tag}_prob_delta.npy")
        np.save(p_path, p_mat.numpy())
        print_rank_0(f"Saved → {p_path}")

        if per_rollout_agree:
            max_G2 = max(t.shape[0] for t in per_rollout_agree)

            def _pad2d(t: torch.Tensor, length: int) -> torch.Tensor:
                if t.shape[0] == length:
                    return t
                pad = t.new_full((length - t.shape[0], t.shape[1]), float("nan"))
                return torch.cat([t, pad], dim=0)

            agree_mat = torch.stack([_pad2d(t, max_G2) for t in per_rollout_agree])
            agree_path = os.path.join(results_dir, f"step7c_{det_tag}_routing_agree.npy")
            np.save(agree_path, agree_mat.numpy())
            print_rank_0(f"Saved → {agree_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_router_study(model, args):
    """Main entry point for the router study.

    Args:
        model: list of model chunks (typically length 1 for PP=1).
        args:  parsed Megatron argument namespace.  Relevant fields:
               args.router_study_results_dir   – output directory
               args.router_study_rollout_dir   – rollout dump dir for step 7C (optional)
               args.router_study_7c_only       – skip 3A/3B/3C/3D/7A, only run 7C
               args.seed                       – random seed
    """
    results_dir  = getattr(args, "router_study_results_dir", "results/router_study")
    rollout_dir  = getattr(args, "router_study_rollout_dir", None)
    only_7c      = getattr(args, "router_study_7c_only", False)
    grad_cosine_only = getattr(args, "router_study_grad_cosine_only", False)

    if torch.distributed.get_rank() == 0:
        os.makedirs(results_dir, exist_ok=True)

    print_rank_0("=" * 60)
    print_rank_0("Router study mode — model loaded successfully.")
    print_rank_0(f"  World size  : {torch.distributed.get_world_size()}")
    print_rank_0(f"  Results dir : {results_dir}")
    if rollout_dir:
        print_rank_0(f"  Rollout dir : {rollout_dir}")
    if only_7c:
        print_rank_0("  Mode        : 7C-only (skipping 3A/3B/3C/3D/7A)")
    if grad_cosine_only:
        print_rank_0("  Mode        : gradient-cosine-only")
    print_rank_0("=" * 60)

    # Set eval mode so MoE layers don't fire the training-only TP+SP guard.
    model_chunk = model[0]
    model_chunk.eval()

    import pathlib
    prompts_path = str(
        pathlib.Path(__file__).parents[3] / "experiments" / "router_study_prompts.json"
    )

    if grad_cosine_only:
        step8_gradient_cosine_routing_replay(
            model_chunk,
            args,
            results_dir,
            prompts_path=prompts_path,
            max_batches=getattr(args, "router_study_grad_cosine_max_batches", 1),
        )
        print_rank_0("Router study complete — exiting cleanly.")
        return

    if not only_7c:
        # ------------------------------------------------------------------
        # M2: instrument routers and run one validation forward pass
        # ------------------------------------------------------------------
        store   = RouterActivationStore()
        handles = register_router_hooks(model_chunk, store)

        input_ids, position_ids = _make_dummy_batch(args)

        print_rank_0("Running M2 validation forward pass …")
        with torch.no_grad():
            model_chunk(input_ids, position_ids, attention_mask=None)

        remove_hooks(handles)
        validate_store(store, model_chunk, args)

        # ------------------------------------------------------------------
        # M4: Step 3A — within-run routing consistency
        # ------------------------------------------------------------------
        step3a_within_run_consistency(model_chunk, args, results_dir)

        # ------------------------------------------------------------------
        # M5: Step 3B — across-restart routing consistency
        # ------------------------------------------------------------------
        step3b_across_restart_consistency(model_chunk, args, results_dir)

        # ------------------------------------------------------------------
        # M6: Step 3C — non-determinism → logprob impact (real Calendar rollouts)
        # ------------------------------------------------------------------
        step3c_nondeterminism_logprob_impact(model_chunk, args, results_dir,
                                             prompts_path=prompts_path)

        # ------------------------------------------------------------------
        # M6.5: Step 3D — real Calendar prompts → |Δprob| by confidence tier
        # ------------------------------------------------------------------
        step3d_real_prompt_prob_impact(model_chunk, args, results_dir, prompts_path)

        # ------------------------------------------------------------------
        # M10: Step 7A — training-side rescore for inference comparison
        # ------------------------------------------------------------------
        step7a_training_rescore(model_chunk, args, results_dir, prompts_path)

    # ------------------------------------------------------------------
    # M16: Step 7C — inference vs training comparison
    # ------------------------------------------------------------------
    if rollout_dir:
        step7c_compare(model_chunk, args, results_dir, rollout_dir)
    else:
        print_rank_0(
            "Step 7C skipped — set --router-study-rollout-dir to the rollout dump directory."
        )

    print_rank_0("Router study complete — exiting cleanly.")
