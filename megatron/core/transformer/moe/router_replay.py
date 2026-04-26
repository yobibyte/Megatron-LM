# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from enum import Enum
from typing import Callable, List, Optional, Tuple

import torch


class RouterReplayAction(Enum):
    """
    A Enum to define the actions for router replay.
    """

    RECORD = "record"  # Record the topk indices for replay
    REPLAY_FORWARD = "replay_forward"  # Replay the recorded topk indices for forward pass
    REPLAY_BACKWARD = "replay_backward"  # Replay topk indices for re-compute during backward pass


class RouterReplay:
    """
    A class to manage the recording and replaying of MoE routing decisions.
    It holds all router instances and provides static methods to globally
    control recording and replaying.
    """

    # Static variable to hold all router instances, one per MoE layer.
    global_router_replay_instances: List['RouterReplay'] = []

    # Index marking the boundary between training-model and inference-model instances.
    # Set by mark_inference_boundary() before the inference model is initialised.
    # -1 means the boundary has not been set (treat all instances as inference).
    _inference_start_idx: int = -1

    @staticmethod
    def set_replay_data(all_layers_topk_indices: List[torch.Tensor], replay_mask: Optional[torch.Tensor] = None):
        """
        Distributes the topk indices for all layers to their respective RouterReplay instances.
        :param all_layers_topk_indices: A list of tensors, where each tensor contains the
                                        topk indices for a specific layer. The order
                                        must match the instantiation order of the routers.
        :param replay_mask: Optional [S] bool tensor. When set, only positions where mask is
                            True are replayed; remaining positions compute routing normally.
                            Used to exclude padding positions from replay.
        """
        if len(all_layers_topk_indices) != len(RouterReplay.global_router_replay_instances):
            raise ValueError(
                f"The number of replay tensors ({len(all_layers_topk_indices)}) "
                f"does not match instances ({len(RouterReplay.global_router_replay_instances)})."
            )
        for i, router_instance in enumerate(RouterReplay.global_router_replay_instances):
            router_instance.set_target_indices(all_layers_topk_indices[i], replay_mask)

    @staticmethod
    def get_recorded_data() -> List[torch.Tensor]:
        """
        Collects the recorded topk indices from all RouterReplay instances.
        :return: A list of tensors, each containing the recorded topk indices for a layer.
        """
        return [
            router.get_recorded_indices() for router in RouterReplay.global_router_replay_instances
        ]

    @staticmethod
    def clear_global_indices():
        """Clears the recorded and target topk indices in all instances."""
        for router in RouterReplay.global_router_replay_instances:
            router.clear_indices()

    @staticmethod
    def set_global_router_replay_action(router_replay_action: RouterReplayAction):
        """Sets the router replay action for all router instances."""
        for router in RouterReplay.global_router_replay_instances:
            router.set_router_replay_action(router_replay_action)

    @staticmethod
    def clear_global_router_replay_action():
        """Clears the router replay action for all router instances."""
        for router in RouterReplay.global_router_replay_instances:
            router.clear_router_replay_action()

    @staticmethod
    def clear_global_router_replay_instances():
        """Clear the global list of router replay instances to prevent memory leaks."""
        RouterReplay.global_router_replay_instances.clear()
        RouterReplay._inference_start_idx = -1

    @staticmethod
    def mark_inference_boundary():
        """Record the boundary between training-model and inference-model instances.

        Call this immediately before the inference model is initialised.  Any
        RouterReplay instances created afterwards belong to the inference model
        and are returned by get_inference_instances().
        """
        RouterReplay._inference_start_idx = len(RouterReplay.global_router_replay_instances)

    @staticmethod
    def get_inference_instances() -> List['RouterReplay']:
        """Return only the inference-model RouterReplay instances.

        If mark_inference_boundary() was never called (e.g. standalone inference
        scripts with a single model), returns all instances.
        """
        if RouterReplay._inference_start_idx < 0:
            return RouterReplay.global_router_replay_instances
        return RouterReplay.global_router_replay_instances[RouterReplay._inference_start_idx:]

    @staticmethod
    def set_global_static_buffers(static_buffer: torch.Tensor):
        """Sets static buffers for all router instances from a combined buffer.

        Args:
            static_buffer: Tensor of shape [max_tokens, num_layers, topk].
                          Each layer's RouterReplay gets a slice [:, layer_idx, :].
        """
        num_layers = len(RouterReplay.global_router_replay_instances)
        assert static_buffer.shape[1] == num_layers, (
            f"Buffer has {static_buffer.shape[1]} layers but there are "
            f"{num_layers} RouterReplay instances."
        )
        for layer_idx, router_instance in enumerate(RouterReplay.global_router_replay_instances):
            # Each layer gets a view of shape [max_tokens, topk]
            router_instance.set_static_buffer(static_buffer[:, layer_idx, :])

    @staticmethod
    def clear_global_static_buffers():
        """Clears static buffers from all router instances."""
        for router in RouterReplay.global_router_replay_instances:
            router.clear_static_buffer()

    def __init__(self):
        """Initializes a RouterReplay instance for a specific layer."""
        self.target_topk_idx: Optional[torch.Tensor] = None  # Target topk indices for replay
        self.replay_mask: Optional[torch.Tensor] = None  # [S] bool — positions to replay (excludes padding)
        self.recorded_topk_idx: Optional[torch.Tensor] = None  # Recorded topk indices for replay
        self.router_replay_action: Optional[RouterReplayAction] = (
            None  # Router replay action for this layer
        )
        self.replay_backward_list: List[torch.Tensor] = (
            []
        )  # List of tensors for backward pass replay
        self.static_buffer: Optional[torch.Tensor] = None  # Static buffer for CUDA graph
        RouterReplay.global_router_replay_instances.append(self)

    def set_target_indices(self, topk_indices: torch.Tensor, replay_mask: Optional[torch.Tensor] = None):
        """Sets the target topk indices for replay."""
        self.target_topk_idx = topk_indices
        self.replay_mask = replay_mask
        self.replay_backward_list.append(topk_indices)

    def get_recorded_indices(self) -> Optional[torch.Tensor]:
        """Returns the recorded topk indices."""
        return self.recorded_topk_idx

    def clear_indices(self):
        """Clears the recorded and target topk indices."""
        self.recorded_topk_idx = None
        self.target_topk_idx = None
        self.replay_mask = None
        self.replay_backward_list = []

    def set_router_replay_action(self, router_replay_action: RouterReplayAction):
        """Sets the router replay action for this layer."""
        self.router_replay_action = router_replay_action

    def clear_router_replay_action(self):
        """Clears the router replay action for this layer."""
        self.router_replay_action = None

    def get_replay_topk(
        self,
        scores: torch.Tensor,
        topk: int,
        num_groups: Optional[int] = None,
        group_topk: Optional[int] = None,
        default_compute_topk: Callable[
            [torch.Tensor, int, Optional[int], Optional[int]], Tuple[torch.Tensor, torch.Tensor]
        ] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A wrapper for top-k computation that handles different replay actions.

        Args:
            scores (torch.Tensor): The scores to compute top-k on.
            topk (int): The number of top elements to select.
            num_groups (Optional[int]): Number of expert groups for group-limited routing.
            group_topk (Optional[int]): Number of groups to select for each token.
            default_compute_topk (Callable): The default top-k computation function, which
                                             should return a tuple of (values, indices).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the top-k values and indices.
        """
        if self.router_replay_action == RouterReplayAction.RECORD:
            probs, top_indices = default_compute_topk(
                scores, topk, num_groups=num_groups, group_topk=group_topk
            )
            self.record_indices(top_indices)
            return probs, top_indices
        elif self.router_replay_action == RouterReplayAction.REPLAY_FORWARD:
            if self.replay_mask is not None:
                # Partial replay: compute routing for all positions, then overwrite
                # non-padding positions with recorded inference indices.
                probs, top_indices = default_compute_topk(
                    scores, topk, num_groups=num_groups, group_topk=group_topk
                )
                mask = self.replay_mask.to(scores.device)
                top_indices = top_indices.clone()
                local_S = top_indices.shape[0]
                if mask.shape[0] != local_S:
                    # Sequence parallelism splits the sequence across TP ranks.
                    # replay_mask and target_topk_idx cover the full sequence; slice
                    # them to this rank's local SP shard before applying the replay.
                    from megatron.core.parallel_state import (
                        get_tensor_model_parallel_rank,
                        get_tensor_model_parallel_world_size,
                    )
                    tp_rank = get_tensor_model_parallel_rank()
                    tp_size = get_tensor_model_parallel_world_size()
                    assert mask.shape[0] % tp_size == 0, (
                        f"Replay mask length {mask.shape[0]} must be divisible by TP size {tp_size}"
                    )
                    shard_S = mask.shape[0] // tp_size
                    assert local_S == shard_S, (
                        f"Local router token count {local_S} does not match replay mask shard "
                        f"length {shard_S} for TP rank {tp_rank}/{tp_size}"
                    )
                    start = tp_rank * shard_S
                    end = start + shard_S
                    mask_local = mask[start:end]
                    target_start = int(mask[:start].sum().item())
                    target_end = target_start + int(mask_local.sum().item())
                    target_topk_idx = self.target_topk_idx[target_start:target_end].to(
                        top_indices.device
                    ).long()
                    top_indices[mask_local] = target_topk_idx
                else:
                    target_topk_idx = self.target_topk_idx.to(top_indices.device).long()
                    top_indices[mask] = target_topk_idx
                probs = scores.gather(1, top_indices)
                return probs, top_indices
            else:
                top_indices = self.target_topk_idx
                # Ensure indices are on the correct device
                top_indices = top_indices.to(scores.device)
                # Gather the scores for the replayed indices to get the probabilities
                probs = scores.gather(1, top_indices)
                return probs, top_indices
        elif self.router_replay_action == RouterReplayAction.REPLAY_BACKWARD:
            top_indices = self.replay_backward_list.pop(0)
            # Ensure indices are on the correct device
            top_indices = top_indices.to(scores.device)
            # Gather the scores for the replayed indices to get the probabilities
            probs = scores.gather(1, top_indices)
            return probs, top_indices
        else:
            return default_compute_topk(scores, topk, num_groups, group_topk)

    def set_static_buffer(self, buffer: torch.Tensor):
        """Sets a static buffer for CUDA graph compatible recording.

        Args:
            buffer: Tensor of shape [max_tokens, topk] to copy routing indices into.
        """
        self.static_buffer = buffer

    def clear_static_buffer(self):
        """Clears the static buffer."""
        self.static_buffer = None

    def record_indices(self, topk_indices: torch.Tensor):
        """Records the topk indices.

        If a static buffer is set (for CUDA graph compatibility), copies into it.
        Otherwise, just stores the tensor reference.
        """
        if self.static_buffer is not None:
            # Copy into static buffer for CUDA graph compatibility.
            num_tokens = topk_indices.shape[0]
            self.static_buffer[:num_tokens].copy_(topk_indices)
            self.recorded_topk_idx = self.static_buffer[:num_tokens]
        else:
            self.recorded_topk_idx = topk_indices
