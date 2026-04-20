# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext

from megatron.core.transformer.moe.router_replay import RouterReplay


class RoutingMetadata:
    """Manages routing indices metadata for MoE layers during inference.

    This class provides static buffers for CUDA graph compatibility when
    recording routing decisions. It holds a reference to the inference context
    to automatically determine whether to use static buffers based on CUDA graph state.

    When both a training model and an inference model coexist in the same process
    (as in RL training), RouterReplay.global_router_replay_instances contains instances
    from both.  Call RouterReplay.mark_inference_boundary() before the inference model
    is initialised so that RoutingMetadata can restrict itself to inference instances
    via RouterReplay.get_inference_instances().

    Args:
        context (DynamicInferenceContext): The inference context.
        moe_router_topk (int): Number of experts selected per token.
    """

    def __init__(self, context: 'DynamicInferenceContext', moe_router_topk: int):
        self.context = context
        self.max_tokens = context.max_tokens
        self.moe_router_topk = moe_router_topk
        self.device = torch.cuda.current_device()

        # Static buffer allocated lazily in _ensure_buffer_allocated().
        # We defer allocation because RouterReplay instances don't exist yet at init time.
        self.routing_indices_buffer: Optional[torch.Tensor] = None
        self.num_moe_layers: Optional[int] = None

    def _get_inference_instances(self) -> List[RouterReplay]:
        return RouterReplay.get_inference_instances()

    def _ensure_buffer_allocated(self) -> None:
        """Allocate the static buffer if not already allocated.

        Uses only inference-model RouterReplay instances (see mark_inference_boundary()).
        """
        if self.routing_indices_buffer is not None:
            return

        instances = self._get_inference_instances()
        self.num_moe_layers = len(instances)

        if self.num_moe_layers == 0:
            return

        # Static buffer for CUDA graph compatibility.
        # Shape: [max_tokens, num_moe_layers, moe_router_topk]
        self.routing_indices_buffer = torch.empty(
            (self.max_tokens, self.num_moe_layers, self.moe_router_topk),
            dtype=torch.int32,
            device=self.device,
        )

    def get_routing_indices(self) -> Optional[torch.Tensor]:
        """Get the recorded routing indices.

        Automatically uses the static buffer when CUDA graphs are active,
        otherwise retrieves from inference-model RouterReplay instances only.

        Returns:
            Tensor of shape [num_tokens, num_moe_layers, topk] or None if not available.
        """
        if self.context.using_cuda_graph_this_step():
            # Return view of static buffer up to current token count.
            if self.routing_indices_buffer is None:
                return None
            # Only return up to active token count, to skip entries
            # for padding tokens.
            return self.routing_indices_buffer[: self.context.active_token_count]
        else:
            # Get from inference-only RouterReplay instances and stack into
            # [num_tokens, num_layers, topk].  Using get_inference_instances()
            # avoids contamination from training-model instances that have
            # recorded_topk_idx=None (they don't run during inference prefill)
            # or from a prior training forward with a different token count.
            instances = self._get_inference_instances()
            if not instances:
                return None
            recorded_data = [inst.get_recorded_indices() for inst in instances]
            if not recorded_data or recorded_data[0] is None:
                return None
            # Verify all entries share the same token count before stacking.
            if len({d.shape[0] for d in recorded_data if d is not None}) > 1:
                return None
            valid_data = [d for d in recorded_data if d is not None]
            if not valid_data:
                return None
            # Stack: list of [num_tokens, topk] -> [num_tokens, num_layers, topk]
            return torch.stack(valid_data, dim=1)

    def enable_static_buffer_recording(self) -> None:
        """Enable recording into the static buffer for CUDA graph compatibility.

        This sets up inference-model RouterReplay instances to copy routing indices
        into our pre-allocated static buffer instead of creating new tensors.
        Allocates the buffer lazily on first call.
        """
        self._ensure_buffer_allocated()
        if self.routing_indices_buffer is None:
            return
        instances = self._get_inference_instances()
        num_layers = len(instances)
        assert self.routing_indices_buffer.shape[1] == num_layers, (
            f"Buffer has {self.routing_indices_buffer.shape[1]} layers but there are "
            f"{num_layers} inference RouterReplay instances."
        )
        for layer_idx, router_instance in enumerate(instances):
            router_instance.set_static_buffer(self.routing_indices_buffer[:, layer_idx, :])

    def disable_static_buffer_recording(self) -> None:
        """Disable static buffer recording on inference instances only."""
        for inst in self._get_inference_instances():
            inst.clear_static_buffer()
