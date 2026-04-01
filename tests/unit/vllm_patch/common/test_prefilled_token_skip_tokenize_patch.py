import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.entrypoints.openai.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)

from omni_npu.vllm_patches.patches.common import patch_prefilled_token_skip_tokenize
from omni_npu.vllm_patches.patches.common import patch_routed_experts


class _FakeServing:
    _get_routed_experts_serialization_mode = (
        patch_routed_experts.RoutedExpertsSerializationPatch._get_routed_experts_serialization_mode
    )
    convert_ndarray_to_str = (
        patch_routed_experts.RoutedExpertsSerializationPatch.convert_ndarray_to_str
    )
    restore_str_to_ndarray = (
        patch_routed_experts.RoutedExpertsSerializationPatch.restore_str_to_ndarray
    )
    concatenate_dict_and_ndarray = (
        patch_routed_experts.RoutedExpertsSerializationPatch.concatenate_dict_and_ndarray
    )
    add_ndarray_info_to_dict = (
        patch_routed_experts.RoutedExpertsSerializationPatch.add_ndarray_info_to_dict
    )

    def __init__(
        self,
        kv_transfer_config=None,
        serialization_mode="zip_base64",
    ):
        self.engine_client = SimpleNamespace(
            vllm_config=SimpleNamespace(
                kv_transfer_config=kv_transfer_config,
                routed_experts_serialization_mode=serialization_mode,
            )
        )


def _make_kv_transfer_config(role: str | None):
    return SimpleNamespace(
        is_kv_transfer_instance=role is not None,
        kv_role=role,
        is_kv_producer=role == "kv_producer",
        is_kv_consumer=role == "kv_consumer",
    )


def _restore_payload(serving: _FakeServing, payload: dict[str, object]) -> np.ndarray:
    return serving.restore_str_to_ndarray(
        payload["routed_experts_str"],
        tuple(payload["routed_experts_shape"]),
        payload["routed_experts_dtype"],
    )


@pytest.mark.unit
def test_prefilled_chat_final_stashes_kv_payload_and_attaches_choice_routed_experts(
    monkeypatch,
):
    serving = _FakeServing(_make_kv_transfer_config("kv_producer"))
    prefill = np.arange(4, dtype=np.int32).reshape(1, 2, 2)
    decode = np.arange(8, 12, dtype=np.int32).reshape(1, 2, 2)
    request_kv_transfer_params = {}
    serving.add_ndarray_info_to_dict(prefill, request_kv_transfer_params)
    request = SimpleNamespace(kv_transfer_params=request_kv_transfer_params)
    final_res = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                index=0,
                routed_experts=decode,
                token_ids=[10],
                text="hello",
                stop_reason=None,
            )
        ],
        kv_transfer_params={},
        prompt_token_ids=[1, 2],
    )

    async def _fake_original_chat_completion_full_generator(
        _self,
        _request,
        result_generator,
        request_id,
        model_name,
        _conversation,
        _tokenizer,
        _request_metadata,
    ):
        consumed = []
        async for item in result_generator:
            consumed.append(item)
        assert consumed[0].kv_transfer_params is final_res.kv_transfer_params
        return ChatCompletionResponse(
            id=request_id,
            created=0,
            model=model_name,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="hello"),
                )
            ],
            usage=UsageInfo(
                prompt_tokens=2,
                completion_tokens=1,
                total_tokens=3,
            ),
        )

    monkeypatch.setattr(
        patch_prefilled_token_skip_tokenize,
        "_original_chat_completion_full_generator",
        _fake_original_chat_completion_full_generator,
    )

    response = asyncio.run(
        patch_prefilled_token_skip_tokenize.OpenAIServingChatPatch.chat_completion_full_generator(
            serving,
            request,
            patch_prefilled_token_skip_tokenize.to_async_iterator(final_res),
            "req-1",
            "dummy-model",
            [],
            None,
            SimpleNamespace(),
        )
    )

    stashed = _restore_payload(serving, final_res.kv_transfer_params)
    restored_choice = _restore_payload(
        serving,
        response.choices[0].model_dump(exclude_unset=False)["routed_experts"],
    )

    assert stashed.shape == (1, 2, 2)
    assert np.array_equal(stashed[0], decode[0])
    assert restored_choice.shape == (2, 2, 2)
    assert np.array_equal(restored_choice[0], prefill[0])
    assert np.array_equal(restored_choice[1], decode[0])
