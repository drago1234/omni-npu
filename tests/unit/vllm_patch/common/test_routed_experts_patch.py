import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.entrypoints.openai.protocol import (
    CompletionResponseChoice,
)
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.request import RequestStatus

from omni_npu.vllm_patches.patches.common import patch_routed_experts

patch_request_output = patch_routed_experts
patch_scheduler = patch_routed_experts
patch_serving_engine = patch_routed_experts
patch_serving_expert_id = patch_routed_experts


class _FakeServing:
    _get_routed_experts_serialization_mode = (
        patch_serving_engine.RoutedExpertsSerializationPatch._get_routed_experts_serialization_mode
    )
    convert_ndarray_to_str = (
        patch_serving_engine.RoutedExpertsSerializationPatch.convert_ndarray_to_str
    )
    restore_str_to_ndarray = (
        patch_serving_engine.RoutedExpertsSerializationPatch.restore_str_to_ndarray
    )
    concatenate_dict_and_ndarray = (
        patch_serving_engine.RoutedExpertsSerializationPatch.concatenate_dict_and_ndarray
    )
    add_ndarray_info_to_dict = (
        patch_serving_engine.RoutedExpertsSerializationPatch.add_ndarray_info_to_dict
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


async def _collect_async_items(async_iter):
    items = []
    async for item in async_iter:
        items.append(item)
    return items


def _restore_payload(serving: _FakeServing, payload: dict[str, object]) -> np.ndarray:
    return serving.restore_str_to_ndarray(
        payload["routed_experts_str"],
        tuple(payload["routed_experts_shape"]),
        payload["routed_experts_dtype"],
    )


def _collect_proxy_payloads(proxy) -> list[dict[str, object]]:
    payloads = []
    for output in proxy.outputs:
        choice = CompletionResponseChoice(index=output.index, text="hello")
        payloads.append(choice.model_dump(exclude_unset=False)["routed_experts"])
    return payloads


def _make_scheduler_request(
    request_id: str,
    *,
    status=RequestStatus.RUNNING,
    num_tokens: int,
    num_prompt_tokens: int,
    num_cached_tokens: int,
    output_kind=RequestOutputKind.FINAL_ONLY,
    client_index: int = 0,
    stop_reason: str | None = None,
    num_nans_in_logits: int = 0,
):
    request = SimpleNamespace(
        request_id=request_id,
        sampling_params=SimpleNamespace(output_kind=output_kind, logprobs=None),
        num_tokens=num_tokens,
        num_prompt_tokens=num_prompt_tokens,
        num_cached_tokens=num_cached_tokens,
        client_index=client_index,
        status=status,
        stop_reason=stop_reason,
        trace_headers={"x-request-id": request_id},
        num_nans_in_logits=num_nans_in_logits,
        pooling_params=None,
        _events=[f"event-{request_id}"],
    )

    def _take_events():
        events = request._events
        request._events = []
        return events

    request.take_events = _take_events
    request.get_finished_reason = lambda: (
        "stop" if RequestStatus.is_finished(request.status) else None
    )
    return request


@pytest.mark.unit
def test_set_pd_flags_prefers_kv_transfer_config():
    reader = SimpleNamespace()
    vllm_config = SimpleNamespace(
        kv_transfer_config=_make_kv_transfer_config("kv_producer")
    )

    patch_routed_experts._set_pd_flags(reader, vllm_config)

    assert reader.is_pd_disaggregation is True
    assert reader.is_pd_prefill is True


@pytest.mark.unit
def test_set_pd_flags_without_kv_transfer_config_is_not_pd():
    reader = SimpleNamespace()

    patch_routed_experts._set_pd_flags(reader)

    assert reader.is_pd_disaggregation is False
    assert reader.is_pd_prefill is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("role", "expected_is_prefill"),
    [
        ("prefill", True),
        ("decode", False),
    ],
)
def test_set_pd_flags_falls_back_to_role(monkeypatch, role, expected_is_prefill):
    monkeypatch.setenv("ROLE", role)
    reader = SimpleNamespace()

    patch_routed_experts._set_pd_flags(reader)

    assert reader.is_pd_disaggregation is True
    assert reader.is_pd_prefill is expected_is_prefill


@pytest.mark.unit
def test_is_prefill_node_without_kv_transfer_config_is_false():
    serving = _FakeServing()

    assert patch_serving_expert_id._is_prefill_node(serving) is False


@pytest.mark.unit
def test_is_prefill_node_prefers_kv_transfer_config():
    serving = _FakeServing(_make_kv_transfer_config("kv_producer"))

    assert patch_serving_expert_id._is_prefill_node(serving) is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("prefill", True),
        ("decode", False),
    ],
)
def test_is_prefill_node_falls_back_to_role(monkeypatch, role, expected):
    monkeypatch.setenv("ROLE", role)
    serving = _FakeServing()

    assert patch_serving_expert_id._is_prefill_node(serving) is expected


@pytest.mark.unit
def test_init_buffer_uses_npu_and_int16_dtype(
    monkeypatch,
):
    captured = {}
    num_experts_per_tok = 8

    class _FakeSharedMemory:
        def __init__(self, size: int):
            self.buf = bytearray(size)

    monkeypatch.setattr(
        patch_routed_experts,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )

    def _fake_zeros(shape, dtype, device):
        captured["zeros_shape"] = shape
        captured["zeros_dtype"] = dtype
        captured["zeros_device"] = device
        return torch.empty((0,), dtype=dtype)

    def _fake_create_or_attach_shared_memory(name, size, lock_file):
        captured["shm_name"] = name
        captured["shm_size"] = size
        captured["lock_file"] = lock_file
        return _FakeSharedMemory(size)

    monkeypatch.setattr(patch_routed_experts.torch, "zeros", _fake_zeros)
    monkeypatch.setattr(
        patch_routed_experts,
        "_create_or_attach_shared_memory",
        _fake_create_or_attach_shared_memory,
    )

    capturer = SimpleNamespace(
        _device_buffer=None,
        _shm=None,
        _host_buffer_view=None,
        _lock_file=None,
        _shm_name=None,
    )
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            num_hidden_layers=3,
            num_experts_per_tok=num_experts_per_tok,
        )
    )

    patch_routed_experts.RoutedExpertsCapturerTPAggregatePatch.init_buffer(
        capturer,
        max_num_batched_tokens=8,
        max_num_kv_tokens=16,
        model_config=model_config,
        instance_id="test",
    )

    assert captured["zeros_shape"] == (8, 3, num_experts_per_tok)
    assert captured["zeros_dtype"] is torch.int16
    assert captured["zeros_device"] == "npu"
    assert captured["shm_size"] == (
        16 * 3 * num_experts_per_tok * np.dtype(np.int16).itemsize
    )
    assert capturer._host_buffer_view.dtype == np.dtype(np.int16)
    assert capturer._host_buffer_view.shape == (16, 3, num_experts_per_tok)


@pytest.mark.unit
def test_attach_buffer_uses_int16_dtype(
    monkeypatch,
):
    class _FakeSharedMemory:
        def __init__(self, size: int):
            self.buf = bytearray(size)

    num_experts_per_tok = 8
    monkeypatch.setattr(patch_routed_experts, "_file_lock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(
        patch_routed_experts.shared_memory,
        "SharedMemory",
        lambda name: _FakeSharedMemory(
            16 * 3 * num_experts_per_tok * np.dtype(np.int16).itemsize
        ),
    )

    reader = SimpleNamespace(_shm=None, _host_buffer_view=None, _lock_file=None)
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            num_hidden_layers=3,
            num_experts_per_tok=num_experts_per_tok,
        )
    )
    vllm_config = SimpleNamespace(
        kv_transfer_config=_make_kv_transfer_config("kv_producer")
    )

    patch_routed_experts.RoutedExpertsReaderPDSupportPatch.attach_buffer(
        reader,
        max_num_kv_tokens=16,
        model_config=model_config,
        instance_id="reader",
        vllm_config=vllm_config,
    )

    assert reader._host_buffer_view.dtype == np.dtype(np.int16)
    assert reader._host_buffer_view.shape == (16, 3, num_experts_per_tok)
    assert reader.is_pd_disaggregation is True
    assert reader.is_pd_prefill is True


@pytest.mark.unit
def test_save_captured_experts_gathers_tp_group_partitions(monkeypatch):
    captured = {}

    fake_group = SimpleNamespace(world_size=2, first_rank=0, device_group=object())
    monkeypatch.setattr(
        patch_routed_experts,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        patch_routed_experts,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(patch_routed_experts, "get_tp_group", lambda: fake_group)
    monkeypatch.setattr(
        patch_routed_experts.dist,
        "all_reduce",
        lambda tensor, op, group: tensor.fill_(2),
    )

    rank1_data = torch.tensor(
        [
            [[9, 9], [8, 8]],
            [[255, 255], [255, 255]],
        ],
        dtype=torch.uint8,
    )
    rank1_mapping = torch.tensor([2, patch_routed_experts._INVALID_INDEX])

    def _fake_gather(tensor, gather_list=None, dst=None, group=None):
        if gather_list is None:
            return
        gather_list[0].copy_(tensor)
        if tensor.dim() == 1:
            gather_list[1].copy_(rank1_mapping.to(tensor.device, dtype=tensor.dtype))
        else:
            gather_list[1].copy_(rank1_data.to(tensor.device, dtype=tensor.dtype))

    monkeypatch.setattr(patch_routed_experts.dist, "gather", _fake_gather)
    monkeypatch.setattr(
        patch_routed_experts,
        "_write_to_shared_memory",
        lambda _capturer, indices, data: captured.update(
            {"indices": indices.copy(), "data": data.copy()}
        ),
    )

    capturer = SimpleNamespace(
        _device_buffer=torch.tensor(
            [
                [[1, 1], [2, 2]],
                [[3, 3], [4, 4]],
                [[5, 5], [6, 6]],
            ],
            dtype=torch.uint8,
        )
    )

    patch_routed_experts.RoutedExpertsCapturerTPAggregatePatch.save_captured_experts(
        capturer,
        np.array([0, 1, 2], dtype=np.int64),
    )

    assert np.array_equal(captured["indices"], np.array([0, 1, 2]))
    assert captured["data"].shape == (3, 2, 2)
    assert np.array_equal(captured["data"][0], np.array([[1, 1], [2, 2]], dtype=np.uint8))
    assert np.array_equal(captured["data"][1], np.array([[3, 3], [4, 4]], dtype=np.uint8))
    assert np.array_equal(captured["data"][2], np.array([[9, 9], [8, 8]], dtype=np.uint8))


@pytest.mark.unit
def test_slot_mapping_helper_returns_last_decode_tokens_for_streaming():
    observed = {}
    reader = SimpleNamespace(
        is_pd_disaggregation=False,
        is_pd_prefill=False,
        get_routed_experts=lambda indices: observed.setdefault(
            "indices", indices.copy()
        ),
    )
    request = SimpleNamespace(
        num_tokens=7,
        num_prompt_tokens=4,
        num_cached_tokens=3,
        sampling_params=SimpleNamespace(output_kind=RequestOutputKind.DELTA),
    )
    slot_mapping = np.array([40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int32)

    result = patch_scheduler._get_routed_experts_from_slot_mapping(
        reader,
        request,
        slot_mapping,
        stopped=False,
        num_new_tokens=2,
    )

    assert np.array_equal(observed["indices"], np.array([44, 45], dtype=np.int32))
    assert np.array_equal(result, np.array([44, 45], dtype=np.int32))


@pytest.mark.unit
def test_slot_mapping_helper_returns_cumulative_cached_and_fresh_tokens():
    observed = {}
    reader = SimpleNamespace(
        is_pd_disaggregation=False,
        is_pd_prefill=False,
        get_routed_experts=lambda indices: observed.setdefault(
            "indices", indices.copy()
        ),
    )
    request = SimpleNamespace(
        num_tokens=8,
        num_prompt_tokens=5,
        num_cached_tokens=3,
        sampling_params=SimpleNamespace(output_kind=RequestOutputKind.CUMULATIVE),
    )
    slot_mapping = np.array([40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int32)

    result = patch_scheduler._get_routed_experts_from_slot_mapping(
        reader,
        request,
        slot_mapping,
        stopped=False,
        num_new_tokens=2,
    )

    assert np.array_equal(
        observed["indices"], np.array([40, 41, 42, 43, 44, 45, 46], dtype=np.int32)
    )
    assert np.array_equal(
        result, np.array([40, 41, 42, 43, 44, 45, 46], dtype=np.int32)
    )


@pytest.mark.unit
def test_slot_mapping_helper_skips_prompt_tokens_for_pd_decode():
    observed = {}
    reader = SimpleNamespace(
        is_pd_disaggregation=True,
        is_pd_prefill=False,
        get_routed_experts=lambda indices: observed.setdefault(
            "indices", indices.copy()
        ),
    )
    request = SimpleNamespace(
        num_tokens=7,
        num_prompt_tokens=4,
        num_cached_tokens=0,
        sampling_params=SimpleNamespace(output_kind=RequestOutputKind.FINAL_ONLY),
    )
    slot_mapping = np.array([40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int32)

    result = patch_scheduler._get_routed_experts_from_slot_mapping(
        reader,
        request,
        slot_mapping,
        stopped=True,
        num_new_tokens=0,
    )

    assert np.array_equal(observed["indices"], np.array([44, 45], dtype=np.int32))
    assert np.array_equal(result, np.array([44, 45], dtype=np.int32))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("enable_return_routed_experts", "reader"),
    [
        (False, SimpleNamespace(get_routed_experts=lambda indices: indices)),
        (True, None),
    ],
)
def test_scheduler_update_from_output_passthrough_without_routed_experts_support(
    monkeypatch,
    enable_return_routed_experts,
    reader,
):
    captured = {}
    expected = {0: EngineCoreOutputs(outputs=[])}

    def _fake_backport_update_from_output(
        self,
        scheduler_output,
        model_runner_output,
    ):
        captured["called"] = True
        captured["scheduler_output"] = scheduler_output
        captured["model_runner_output"] = model_runner_output
        return expected

    monkeypatch.setattr(
        patch_scheduler.SchedulerRoutedExpertsPatch,
        "_omni_npu_update_from_output_backport",
        _fake_backport_update_from_output,
    )

    scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                enable_return_routed_experts=enable_return_routed_experts
            )
        ),
        routed_experts_reader=reader,
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={})
    model_runner_output = SimpleNamespace()

    result = patch_scheduler.SchedulerRoutedExpertsPatch.update_from_output(
        scheduler,
        scheduler_output,
        model_runner_output,
    )

    assert captured["called"] is True
    assert captured["scheduler_output"] is scheduler_output
    assert captured["model_runner_output"] is model_runner_output
    assert result is expected

@pytest.mark.unit
def test_scheduler_backport_update_from_output_preserves_spec_decode_empty_generation_bugfix(
    monkeypatch,
):
    request = _make_scheduler_request(
        "req-empty",
        num_tokens=5,
        num_prompt_tokens=4,
        num_cached_tokens=0,
        output_kind=RequestOutputKind.CUMULATIVE,
    )
    request.num_computed_tokens = 5
    request.num_output_placeholders = 3

    scheduler = SimpleNamespace(
        perf_metrics=None,
        connector=None,
        kv_cache_manager=SimpleNamespace(take_events=lambda: None),
        finished_req_ids_dict={},
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enable_return_routed_experts=False)
        ),
        requests={"req-empty": request},
        structured_output_manager=SimpleNamespace(
            should_advance=lambda _request: False
        ),
        make_stats=lambda *args, **kwargs: None,
        make_spec_decoding_stats=lambda *args, **kwargs: (
            (_ for _ in ()).throw(
                AssertionError("spec decode stats should not be updated")
            )
        ),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req-empty": 1},
        scheduled_spec_decode_tokens={
            "req-empty": [100, 101],
        },
        num_invalid_spec_tokens=0,
    )
    model_runner_output = SimpleNamespace(
        sampled_token_ids=[[]],
        logprobs=None,
        prompt_logprobs_dict={},
        req_id_to_index={"req-empty": 0},
        pooler_output=None,
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
    )

    result = patch_scheduler.SchedulerRoutedExpertsPatch._omni_npu_update_from_output_backport(
        scheduler,
        scheduler_output,
        model_runner_output,
    )

    assert request.num_computed_tokens == 5
    assert request.num_output_placeholders == 3
    assert scheduler_output.scheduled_spec_decode_tokens == {
        "req-empty": [100, 101],
    }
    assert result == {}


@pytest.mark.unit
def test_scheduler_update_from_output_overwrites_existing_output_routed_experts(
    monkeypatch,
):
    observed = {}
    request = _make_scheduler_request(
        "req-stream",
        num_tokens=5,
        num_prompt_tokens=3,
        num_cached_tokens=2,
        output_kind=RequestOutputKind.DELTA,
    )

    def _fake_original_update_from_output(self, scheduler_output, model_runner_output):
        observed["called"] = True
        request.num_tokens = 7
        return {
            0: EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id="req-stream",
                        new_token_ids=[30, 31],
                    )
                ]
            )
        }

    monkeypatch.setattr(
        patch_scheduler.SchedulerRoutedExpertsPatch,
        "_omni_npu_update_from_output_backport",
        _fake_original_update_from_output,
    )

    scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enable_return_routed_experts=True)
        ),
        routed_experts_reader=SimpleNamespace(
            is_pd_disaggregation=False,
            is_pd_prefill=False,
            get_routed_experts=lambda indices: observed.setdefault(
                "indices", indices.copy()
            ),
        ),
        kv_cache_manager=SimpleNamespace(
            get_blocks=lambda _request_id: SimpleNamespace(
                get_block_ids=lambda: [[10, 11]]
            )
        ),
        block_size=4,
        requests={"req-stream": request},
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"req-stream": 2})
    model_runner_output = SimpleNamespace(
        req_id_to_index={"req-stream": 0},
        prompt_logprobs_dict={},
        pooler_output=None,
    )

    result = patch_scheduler.SchedulerRoutedExpertsPatch.update_from_output(
        scheduler,
        scheduler_output,
        model_runner_output,
    )

    assert observed["called"] is True
    assert np.array_equal(observed["indices"], np.array([44, 45], dtype=np.int32))
    assert np.array_equal(
        result[0].outputs[0].routed_experts,
        np.array([44, 45], dtype=np.int32),
    )


@pytest.mark.unit
def test_scheduler_update_from_output_overwrites_stopped_output_after_free(
    monkeypatch,
):
    observed = {}
    request = _make_scheduler_request(
        "req-stop",
        num_tokens=5,
        num_prompt_tokens=4,
        num_cached_tokens=0,
        output_kind=RequestOutputKind.FINAL_ONLY,
        stop_reason="eos",
    )

    def _fake_original_update_from_output(self, scheduler_output, model_runner_output):
        observed["called"] = True
        request.num_tokens = 7
        request.status = RequestStatus.FINISHED_STOPPED
        del self.requests["req-stop"]
        return {
            0: EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id="req-stop",
                        new_token_ids=[],
                        finish_reason="stop",
                        stop_reason="eos",
                        routed_experts=np.zeros((1, 2, 2), dtype=np.int32),
                    )
                ]
            )
        }

    monkeypatch.setattr(
        patch_scheduler.SchedulerRoutedExpertsPatch,
        "_omni_npu_update_from_output_backport",
        _fake_original_update_from_output,
    )

    scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enable_return_routed_experts=True)
        ),
        routed_experts_reader=SimpleNamespace(
            is_pd_disaggregation=True,
            is_pd_prefill=False,
            get_routed_experts=lambda indices: observed.setdefault(
                "indices", indices.copy()
            ),
        ),
        kv_cache_manager=SimpleNamespace(
            get_blocks=lambda _request_id: SimpleNamespace(
                get_block_ids=lambda: [[10, 11]]
            )
        ),
        block_size=4,
        requests={"req-stop": request},
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"req-stop": 1})
    model_runner_output = SimpleNamespace(
        req_id_to_index={"req-stop": 0},
        prompt_logprobs_dict={},
        pooler_output=None,
    )

    result = patch_scheduler.SchedulerRoutedExpertsPatch.update_from_output(
        scheduler,
        scheduler_output,
        model_runner_output,
    )

    assert observed["called"] is True
    assert "req-stop" not in scheduler.requests
    assert np.array_equal(observed["indices"], np.array([44, 45], dtype=np.int32))
    assert np.array_equal(
        result[0].outputs[0].routed_experts,
        np.array([44, 45], dtype=np.int32),
    )


@pytest.mark.unit
def test_scheduler_update_from_output_appends_output_when_only_routed_experts_exist(
    monkeypatch,
):
    observed = {}
    request = _make_scheduler_request(
        "req-final-only",
        num_tokens=5,
        num_prompt_tokens=4,
        num_cached_tokens=1,
        output_kind=RequestOutputKind.CUMULATIVE,
        num_nans_in_logits=7,
    )
    prompt_logprobs = object()
    pooler_output = object()

    def _fake_original_update_from_output(self, scheduler_output, model_runner_output):
        observed["called"] = True
        request.num_tokens = 7
        request.status = RequestStatus.FINISHED_STOPPED
        del self.requests["req-final-only"]
        return {}

    monkeypatch.setattr(
        patch_scheduler.SchedulerRoutedExpertsPatch,
        "_omni_npu_update_from_output_backport",
        _fake_original_update_from_output,
    )

    scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enable_return_routed_experts=True)
        ),
        routed_experts_reader=SimpleNamespace(
            is_pd_disaggregation=False,
            is_pd_prefill=False,
            get_routed_experts=lambda indices: observed.setdefault(
                "indices", indices.copy()
            ),
        ),
        kv_cache_manager=SimpleNamespace(
            get_blocks=lambda _request_id: SimpleNamespace(
                get_block_ids=lambda: [[10, 11]]
            )
        ),
        block_size=4,
        requests={"req-final-only": request},
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"req-final-only": 1})
    model_runner_output = SimpleNamespace(
        req_id_to_index={"req-final-only": 0},
        prompt_logprobs_dict={"req-final-only": prompt_logprobs},
        pooler_output=[pooler_output],
    )

    result = patch_scheduler.SchedulerRoutedExpertsPatch.update_from_output(
        scheduler,
        scheduler_output,
        model_runner_output,
    )

    assert observed["called"] is True
    assert np.array_equal(
        observed["indices"], np.array([40, 41, 42, 43, 44, 45], dtype=np.int32)
    )
    assert 0 in result
    assert len(result[0].outputs) == 1
    output = result[0].outputs[0]
    assert output.request_id == "req-final-only"
    assert output.new_token_ids == []
    assert output.finish_reason == "stop"
    assert output.new_prompt_logprobs_tensors is prompt_logprobs
    assert output.pooling_output is pooler_output
    assert output.events == ["event-req-final-only"]
    assert output.num_nans_in_logits == 7
    assert np.array_equal(
        output.routed_experts,
        np.array([40, 41, 42, 43, 44, 45], dtype=np.int32),
    )

@pytest.mark.unit
def test_serialization_helpers_round_trip_zip_base64():
    serving = _FakeServing()
    data = np.arange(12, dtype=np.int32).reshape(2, 2, 3)
    payload = {}

    serving.add_ndarray_info_to_dict(data, payload)
    restored = serving.restore_str_to_ndarray(
        payload["routed_experts_str"],
        tuple(payload["routed_experts_shape"]),
        payload["routed_experts_dtype"],
    )

    assert np.array_equal(restored, data)
    assert payload["routed_experts_str_len"] == len(payload["routed_experts_str"])


@pytest.mark.unit
def test_serialization_helpers_round_trip_base64():
    serving = _FakeServing(serialization_mode="base64")
    data = np.arange(12, dtype=np.int32).reshape(2, 2, 3)
    payload = {}

    serving.add_ndarray_info_to_dict(data, payload)
    restored = serving.restore_str_to_ndarray(
        payload["routed_experts_str"],
        tuple(payload["routed_experts_shape"]),
        payload["routed_experts_dtype"],
    )

    assert np.array_equal(restored, data)
    assert payload["routed_experts_str_len"] == len(payload["routed_experts_str"])


@pytest.mark.unit
def test_restore_str_to_ndarray_raises_on_serialization_mode_mismatch():
    zipped_serving = _FakeServing(serialization_mode="zip_base64")
    plain_serving = _FakeServing(serialization_mode="base64")
    data = np.arange(12, dtype=np.int32).reshape(2, 2, 3)
    payload = {}

    zipped_serving.add_ndarray_info_to_dict(data, payload)

    with pytest.raises(ValueError):
        plain_serving.restore_str_to_ndarray(
            payload["routed_experts_str"],
            tuple(payload["routed_experts_shape"]),
            payload["routed_experts_dtype"],
        )


@pytest.mark.unit
def test_concatenate_dict_and_ndarray_raises_on_shape_mismatch():
    serving = _FakeServing()
    prefill = np.arange(4, dtype=np.int32).reshape(1, 2, 2)
    decode = np.arange(6, dtype=np.int32).reshape(1, 3, 2)
    payload = {}
    serving.add_ndarray_info_to_dict(prefill, payload)

    with pytest.raises(ValueError):
        serving.concatenate_dict_and_ndarray(payload, decode)


@pytest.mark.unit
def test_build_routed_experts_payload_merges_prefill_and_decode_for_non_stream():
    serving = _FakeServing()
    prefill = np.arange(4, dtype=np.int32).reshape(1, 2, 2)
    decode = np.arange(8, 16, dtype=np.int32).reshape(2, 2, 2)
    kv_transfer_params = {}
    serving.add_ndarray_info_to_dict(prefill, kv_transfer_params)
    output = SimpleNamespace(routed_experts=decode, token_ids=[10, 11])

    payload = patch_serving_expert_id._build_routed_experts_payload(
        serving,
        output,
        kv_transfer_params,
    )
    restored = _restore_payload(serving, payload)

    assert restored.shape == (3, 2, 2)
    assert np.array_equal(restored[0], prefill[0])
    assert np.array_equal(restored[1:], decode)


@pytest.mark.unit
def test_wrap_request_output_stream_attaches_prefill_once_per_choice():
    patch_serving_expert_id._ensure_choice_patches_applied()
    serving = _FakeServing()
    prefill = np.arange(4, dtype=np.int32).reshape(1, 2, 2)
    first_decode_0 = np.arange(8, 12, dtype=np.int32).reshape(1, 2, 2)
    first_decode_1 = np.arange(12, 16, dtype=np.int32).reshape(1, 2, 2)
    second_decode_0 = np.arange(16, 20, dtype=np.int32).reshape(1, 2, 2)
    second_decode_1 = np.arange(20, 24, dtype=np.int32).reshape(1, 2, 2)
    kv_transfer_params = {}
    serving.add_ndarray_info_to_dict(prefill, kv_transfer_params)
    request = SimpleNamespace(kv_transfer_params=kv_transfer_params)

    async def _result_generator():
        yield SimpleNamespace(
            outputs=[
                SimpleNamespace(index=0, routed_experts=first_decode_0, token_ids=[10]),
                SimpleNamespace(index=1, routed_experts=first_decode_1, token_ids=[20]),
            ],
            kv_transfer_params=None,
        )
        yield SimpleNamespace(
            outputs=[
                SimpleNamespace(index=0, routed_experts=second_decode_0, token_ids=[11]),
                SimpleNamespace(index=1, routed_experts=second_decode_1, token_ids=[21]),
            ],
            kv_transfer_params=None,
        )

    proxies = asyncio.run(
        _collect_async_items(
            patch_serving_expert_id._wrap_request_output_stream(
                serving,
                request,
                _result_generator(),
            )
        )
    )
    first_payloads = _collect_proxy_payloads(proxies[0])
    second_payloads = _collect_proxy_payloads(proxies[1])

    first_restored_0 = _restore_payload(serving, first_payloads[0])
    first_restored_1 = _restore_payload(serving, first_payloads[1])
    second_restored_0 = _restore_payload(serving, second_payloads[0])
    second_restored_1 = _restore_payload(serving, second_payloads[1])

    assert first_restored_0.shape == (2, 2, 2)
    assert np.array_equal(first_restored_0[0], prefill[0])
    assert np.array_equal(first_restored_0[1], first_decode_0[0])
    assert first_restored_1.shape == (2, 2, 2)
    assert np.array_equal(first_restored_1[0], prefill[0])
    assert np.array_equal(first_restored_1[1], first_decode_1[0])

    assert second_restored_0.shape == (1, 2, 2)
    assert np.array_equal(second_restored_0[0], second_decode_0[0])
    assert second_restored_1.shape == (1, 2, 2)
    assert np.array_equal(second_restored_1[0], second_decode_1[0])
    assert "routed_experts_str" in request.kv_transfer_params


@pytest.mark.unit
def test_wrap_request_output_stream_prefilled_chunk_attaches_prefill_once():
    patch_serving_expert_id._ensure_choice_patches_applied()
    serving = _FakeServing()
    prefill = np.arange(4, dtype=np.int32).reshape(1, 2, 2)
    decode = np.arange(8, 12, dtype=np.int32).reshape(1, 2, 2)
    kv_transfer_params = {}
    serving.add_ndarray_info_to_dict(prefill, kv_transfer_params)
    request = SimpleNamespace(kv_transfer_params=kv_transfer_params)

    async def _result_generator():
        yield SimpleNamespace(
            outputs=[SimpleNamespace(index=0, routed_experts=None, token_ids=[10])],
            kv_transfer_params=None,
        )
        yield SimpleNamespace(
            outputs=[SimpleNamespace(index=0, routed_experts=decode, token_ids=[11])],
            kv_transfer_params=None,
        )

    proxies = asyncio.run(
        _collect_async_items(
            patch_serving_expert_id._wrap_request_output_stream(
                serving,
                request,
                _result_generator(),
            )
        )
    )
    first_payload = _collect_proxy_payloads(proxies[0])[0]
    second_payload = _collect_proxy_payloads(proxies[1])[0]

    first_restored = _restore_payload(serving, first_payload)
    second_restored = _restore_payload(serving, second_payload)

    assert first_restored.shape == (1, 2, 2)
    assert np.array_equal(first_restored[0], prefill[0])
    assert second_restored.shape == (1, 2, 2)
    assert np.array_equal(second_restored[0], decode[0])
    assert "routed_experts_str" in request.kv_transfer_params


@pytest.mark.unit
def test_wrap_indexed_request_output_stream_attaches_prefill_once_per_prompt():
    patch_serving_expert_id._ensure_choice_patches_applied()
    serving = _FakeServing()
    prefill = np.arange(4, dtype=np.int32).reshape(1, 2, 2)
    decode_prompt_0_first = np.arange(8, 12, dtype=np.int32).reshape(1, 2, 2)
    decode_prompt_1_first = np.arange(12, 16, dtype=np.int32).reshape(1, 2, 2)
    decode_prompt_0_second = np.arange(16, 20, dtype=np.int32).reshape(1, 2, 2)
    kv_transfer_params = {}
    serving.add_ndarray_info_to_dict(prefill, kv_transfer_params)
    request = SimpleNamespace(kv_transfer_params=kv_transfer_params)

    async def _result_generator():
        yield 0, SimpleNamespace(
            outputs=[SimpleNamespace(index=0, routed_experts=decode_prompt_0_first, token_ids=[10])],
            kv_transfer_params=None,
        )
        yield 1, SimpleNamespace(
            outputs=[SimpleNamespace(index=0, routed_experts=decode_prompt_1_first, token_ids=[20])],
            kv_transfer_params=None,
        )
        yield 0, SimpleNamespace(
            outputs=[SimpleNamespace(index=0, routed_experts=decode_prompt_0_second, token_ids=[11])],
            kv_transfer_params=None,
        )

    wrapped = asyncio.run(
        _collect_async_items(
            patch_serving_expert_id._wrap_indexed_request_output_stream(
                serving,
                request,
                _result_generator(),
            )
        )
    )
    first_prompt_payload = _collect_proxy_payloads(wrapped[0][1])[0]
    second_prompt_payload = _collect_proxy_payloads(wrapped[1][1])[0]
    third_prompt_payload = _collect_proxy_payloads(wrapped[2][1])[0]

    first_prompt_restored = _restore_payload(serving, first_prompt_payload)
    second_prompt_restored = _restore_payload(serving, second_prompt_payload)
    third_prompt_restored = _restore_payload(serving, third_prompt_payload)

    assert first_prompt_restored.shape == (2, 2, 2)
    assert np.array_equal(first_prompt_restored[0], prefill[0])
    assert np.array_equal(first_prompt_restored[1], decode_prompt_0_first[0])
    assert second_prompt_restored.shape == (2, 2, 2)
    assert np.array_equal(second_prompt_restored[0], prefill[0])
    assert np.array_equal(second_prompt_restored[1], decode_prompt_1_first[0])
    assert third_prompt_restored.shape == (1, 2, 2)
    assert np.array_equal(third_prompt_restored[0], decode_prompt_0_second[0])
    assert "routed_experts_str" in request.kv_transfer_params

@pytest.mark.unit
def test_stash_prefill_routed_experts_writes_kv_transfer_params():
    serving = _FakeServing(_make_kv_transfer_config("kv_producer"))
    experts = np.arange(4, dtype=np.int32).reshape(1, 2, 2)
    request_output = SimpleNamespace(
        outputs=[SimpleNamespace(routed_experts=experts)],
        kv_transfer_params=None,
    )

    patch_serving_expert_id._stash_prefill_routed_experts(serving, request_output)

    assert request_output.kv_transfer_params is not None
    assert request_output.kv_transfer_params["routed_experts_dtype"] == "int32"


@pytest.mark.unit
def test_request_output_proxy_resets_routed_experts_context_between_choices():
    patch_serving_expert_id._ensure_choice_patches_applied()
    payload = {
        "routed_experts_shape": (1, 1, 1),
        "routed_experts_dtype": "int32",
        "routed_experts_str_len": 4,
        "routed_experts_str": "abcd",
    }
    proxy = patch_serving_expert_id._RequestOutputProxy(
        SimpleNamespace(outputs=[SimpleNamespace(routed_experts=np.zeros((1, 1, 1)))]),
        lambda _output: payload,
    )

    for _output in proxy.outputs:
        choice = CompletionResponseChoice(index=0, text="hello")

    dumped = choice.model_dump(exclude_unset=False)
    assert dumped["routed_experts"] == payload

    other_choice = CompletionResponseChoice(index=1, text="world")
    assert "routed_experts" not in other_choice.model_dump(exclude_unset=False)


@pytest.mark.unit
def test_request_output_add_aggregates_routed_experts(monkeypatch):
    observed = {}
    original_add = RequestOutput.add

    def _observed_original_add(self, next_output, aggregate):
        observed["called"] = True
        return original_add(self, next_output, aggregate)

    monkeypatch.setattr(
        patch_request_output,
        "_original_add",
        _observed_original_add,
    )

    first = RequestOutput(
        request_id="req-agg",
        prompt="hello",
        prompt_token_ids=[1, 2],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="a",
                token_ids=[10],
                cumulative_logprob=0.1,
                logprobs=None,
                routed_experts=np.arange(4, dtype=np.int32).reshape(1, 2, 2),
            )
        ],
        finished=False,
    )
    second = RequestOutput(
        request_id="req-agg",
        prompt="hello",
        prompt_token_ids=[1, 2],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="bc",
                token_ids=[11, 12],
                cumulative_logprob=0.3,
                logprobs=None,
                routed_experts=np.arange(8, 16, dtype=np.int32).reshape(2, 2, 2),
                finish_reason="stop",
            )
        ],
        finished=True,
    )

    patch_request_output.RequestOutputRoutedExpertsAggregationPatch.add(
        first,
        second,
        aggregate=True,
    )

    assert observed["called"] is True
    assert first.finished is True
    assert first.outputs[0].text == "abc"
    assert list(first.outputs[0].token_ids) == [10, 11, 12]
    assert np.array_equal(
        first.outputs[0].routed_experts,
        np.array(
            [
                [[0, 1], [2, 3]],
                [[8, 9], [10, 11]],
                [[12, 13], [14, 15]],
            ],
            dtype=np.int32,
        ),
    )


@pytest.mark.unit
@pytest.mark.parametrize("scenario", ["replace", "append"])
def test_request_output_add_non_aggregating_paths_keep_original_behavior(
    monkeypatch,
    scenario,
):
    observed = {}
    original_add = RequestOutput.add

    def _observed_original_add(self, next_output, aggregate):
        observed["called"] = True
        return original_add(self, next_output, aggregate)

    monkeypatch.setattr(
        patch_request_output,
        "_original_add",
        _observed_original_add,
    )

    next_completion = CompletionOutput(
        index=0 if scenario == "replace" else 1,
        text="replaced" if scenario == "replace" else "b",
        token_ids=[20] if scenario == "replace" else [11],
        cumulative_logprob=0.2,
        logprobs=None,
        routed_experts=(
            np.arange(4, 8, dtype=np.int32).reshape(1, 2, 2)
            if scenario == "replace"
            else np.arange(8, 12, dtype=np.int32).reshape(1, 2, 2)
        ),
    )
    first = RequestOutput(
        request_id=f"req-{scenario}",
        prompt="hello",
        prompt_token_ids=[1, 2],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="a",
                token_ids=[10],
                cumulative_logprob=0.1,
                logprobs=None,
                routed_experts=np.arange(4, dtype=np.int32).reshape(1, 2, 2),
            )
        ],
        finished=False,
    )
    second = RequestOutput(
        request_id=f"req-{scenario}",
        prompt="hello",
        prompt_token_ids=[1, 2],
        prompt_logprobs=None,
        outputs=[next_completion],
        finished=True,
    )

    patch_request_output.RequestOutputRoutedExpertsAggregationPatch.add(
        first,
        second,
        aggregate=scenario == "append",
    )

    assert observed["called"] is True
    if scenario == "replace":
        assert first.outputs[0] is next_completion
        assert np.array_equal(
            first.outputs[0].routed_experts,
            next_completion.routed_experts,
        )
    else:
        assert len(first.outputs) == 2
        assert first.outputs[1] is next_completion
        assert np.array_equal(
            first.outputs[1].routed_experts,
            next_completion.routed_experts,
        )
