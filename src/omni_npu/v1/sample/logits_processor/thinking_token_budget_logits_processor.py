# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)


class ThinkingTokenBudgetLogitsProcessor(LogitsProcessor):
    """Limits the number of tokens allowed inside a 'thinking' section."""

    def __init__(
            self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ):
        reasoning_config = vllm_config.reasoning_config
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs

        # Check if thinking is enabled
        self.is_enabled = (
                reasoning_config is not None and reasoning_config.is_thinking_enabled
        )
        if self.is_enabled:
            logger.info("ThinkingTokenBudgetLogitsProcessor enable.")

        self.think_start_token_ids = getattr(
            reasoning_config, "think_start_token_ids", []
        )
        self.think_end_token_ids = getattr(reasoning_config, "think_end_token_ids", [])
        self.thinking_token_budget = getattr(reasoning_config, "thinking_token_budget", 0)

        self.pin_memory = is_pin_memory
        self.device = device
        # Per-request state tracking for thinking token management
        # Key: request_index, Value: state dict containing:
        # "in_think": bool - currently in thinking mode
        # "in_end": bool - currently forcing end tokens output
        # "check_count_down": int - steps remaining until next think
        #                            start/end token parsing
        # "think_count": int - number of thinking tokens generated
        # "end_count": int - number of end tokens forced so far
        # "thinking_token_budget": int - max allowed thinking tokens
        # "output_tok_ids": list[int] - generated output tokens
        # "prev_output_length": int - previous output length for
        #                               incremental processing
        self._state: dict[int, dict[str, Any]] = {}

        # Preallocate reusable tensors
        self.mask = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)
        self.force_token_ids = torch.full(
            (max_num_reqs,), -1, dtype=torch.long, device=device
        )

    @staticmethod
    def _find_last_sequence_index(target_list: list[int], token_ids: list[int]) -> int:
        """
        Returns the index of the last occurrence of token_ids in target_list.
        Args:
          target_list (list[int]): The list of token IDs.
          token_ids (list[int]): The sequence of token IDs to find.
        """
        if not token_ids:
            return -1
        for i in range(len(target_list) - len(token_ids), -1, -1):
            if target_list[i: i + len(token_ids)] == token_ids:
                return i
        return -1

    def _init_state_entry(
            self, prompt_tok_ids: list[int] | None, thinking_token_budget: int
    ) -> dict[str, Any]:
        """Initializes the tracking state for a given sequence index."""
        if prompt_tok_ids is None:
            in_think = False
            think_count = 0
        else:
            last_start = self._find_last_sequence_index(
                prompt_tok_ids, self.think_start_token_ids
            )
            last_end = self._find_last_sequence_index(
                prompt_tok_ids, self.think_end_token_ids
            )
            in_think = last_start > last_end
            if in_think:
                think_count = len(prompt_tok_ids) - (
                        last_start + len(self.think_start_token_ids)
                )
            else:
                think_count = 0

        return {
            "in_think": in_think,  # Currently in thinking mode
            "in_end": in_think and thinking_token_budget == 0,
            "check_count_down": thinking_token_budget,
            "think_count": think_count,  # Number of tokens in thinking section
            "end_count": 0,  # Number of end tokens forced so far
            "prompt_tok_ids": prompt_tok_ids,
            "output_tok_ids": [],
            "thinking_token_budget": thinking_token_budget,
            "prev_output_length": 0,
            # Track previous output length for incremental updates
        }

    def _update_think_state(self, state: dict[str, Any]):
        """Updates the state based on newly generated output tokens."""
        if not state.get("in_end", False) and state.get("check_count_down", 0) > 0:
            state["check_count_down"] -= 1

        output = state.get("output_tok_ids", [])
        if not output:
            return

        # Track previous output length for incremental processing
        prev_length = state.get("prev_output_length", 0)
        current_length = len(output)

        if current_length <= prev_length:
            return

        # Process only newly added tokens
        new_tokens = output[prev_length:]
        state["prev_output_length"] = current_length

        # Check if new tokens contain think start or end sequences
        start_len = len(self.think_start_token_ids)
        end_len = len(self.think_end_token_ids)

        # Look for think sequences in recent tokens (including boundary)
        # Check overlapping regions where sequences might span boundaries
        check_start_idx = max(0, prev_length - max(start_len, end_len) + 1)
        recent_tokens = output[check_start_idx:]

        # Find any think start/end sequences in recent tokens
        recent_start_pos = self._find_last_sequence_index(
            recent_tokens, self.think_start_token_ids
        )
        recent_end_pos = self._find_last_sequence_index(
            recent_tokens, self.think_end_token_ids
        )

        # Update state based on recent sequences
        if not state["in_end"]:
            if recent_start_pos >= 0 and recent_end_pos >= 0:
                if recent_start_pos > recent_end_pos:
                    # Case: ...<end>...<start>... - entering think mode
                    absolute_start_pos = check_start_idx + recent_start_pos
                    new_think_count = current_length - (absolute_start_pos + start_len)
                    state["in_think"] = True
                    state["think_count"] = new_think_count
                else:
                    # Case: ...<start>...<end>... - exiting think mode
                    state["in_think"] = False
                    state["think_count"] = 0
            elif recent_start_pos >= 0:
                # Found think start - entering think mode
                absolute_start_pos = check_start_idx + recent_start_pos
                new_think_count = current_length - (absolute_start_pos + start_len)
                state["in_think"] = True
                state["think_count"] = new_think_count
            elif recent_end_pos >= 0:
                # Found think end - exiting think mode
                state["in_think"] = False
                state["think_count"] = 0
            elif state["in_think"]:
                # Continue thinking mode, increment count by new tokens
                state["think_count"] += len(new_tokens)

            # Set countdown based on current state
            if state["in_think"]:
                remaining_budget = max(
                    0, state["thinking_token_budget"] - state["think_count"]
                )
                state["check_count_down"] = remaining_budget
            else:
                state["check_count_down"] = state["thinking_token_budget"]

            # Check if need to transition to end mode
            if (
                    state["in_think"]
                    and state["think_count"] >= state["thinking_token_budget"]
            ):
                state["in_think"] = False
                state["in_end"] = True
                state["end_count"] = 0
                state["check_count_down"] = state["thinking_token_budget"]
        else:
            # In end mode
            state["end_count"] += 1
            if state["end_count"] >= len(self.think_end_token_ids):
                state.update(
                    {
                        "in_end": False,
                        "end_count": 0,
                        "check_count_down": state["thinking_token_budget"],
                    }
                )

    def is_argmax_invariant(self) -> bool:
        """This logits processor can change the outcome of
        greedy sampling by forcing that the thinking section
        ends after a certain number of tokens."""
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        if not self.is_enabled:
            return
        if batch_update:
            for index, sample_params, prompt_tok_ids, output_tok_ids in batch_update.added:
                thinking_token_budget = None
                if hasattr(sample_params, 'extra_args') and sample_params.extra_args:
                    thinking_token_budget = sample_params.extra_args.get("thinking_token_budget")

                if thinking_token_budget is not None:
                    self._state[index] = self._init_state_entry(
                        prompt_tok_ids, thinking_token_budget
                    )
                    self._state[index]["output_tok_ids"] = output_tok_ids
                else:
                    # Remove state if no thinking budget
                    self._state.pop(index, None)

            for index in batch_update.removed:
                self._state.pop(index, {})

            for i1, i2, direction in batch_update.moved:
                if direction == MoveDirectionality.SWAP:
                    state1 = self._state.get(i1, {})
                    state2 = self._state.get(i2, {})
                    if state1 or state2:
                        self._state[i1] = state2
                        self._state[i2] = state1
                else:
                    self._state[i2] = self._state.pop(i1, {})

        for state in self._state.values():
            self._update_think_state(state)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.is_enabled or not self._state:
            return logits

        batch_size = logits.size(0)
        self.mask[:batch_size] = False

        for i in range(batch_size):
            state = self._state.get(i)
            if state and state["in_end"]:
                self.mask[i] = True
                self.force_token_ids[i] = self.think_end_token_ids[state["end_count"]]

        # Check in CPU first not to sync with GPU
        has_active_thinking = any(
            state.get("in_end", False) for state in self._state.values()
        )

        if has_active_thinking:
            current_mask = self.mask[:batch_size]
            active_indices = current_mask.nonzero(as_tuple=False).view(-1)
            if len(active_indices) > 0:
                force_tokens = self.force_token_ids[active_indices]
                # Apply a large value for the end thinking token id index
                logits[active_indices, force_tokens] = 1e9

        return logits
