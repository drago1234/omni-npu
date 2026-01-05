import os
import socket
import struct
import threading
import time
from unittest import mock
from unittest.mock import patch, MagicMock

import pytest
import torch
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_world_group

from omni_npu.connector.llmdatadist_manager_v1 import (
    LLMDataDistManager,
    LLMDataDistConfig,
    TORCH_DTYPE_TO_NPU_DTYPE,
    SCHEDULER_LINK_BATCH_SIZE,
    SCHEDULER_LINK_INTERVAL,
    KV_CACHE_RETRY_TIMES,
    KV_CACHE_RETRY_WAIT_SECOND,
    SYNC_KV_TIMEOUT,
    LINK_TIMEOUT,
    RETRYABLE_CODES,
    NUM_DIE_PER_MACH,
    ip_port_to_int,
    unzip_kv_cache_dict,
    unzip_kv_cache_list,
    maybe_merge_kv_caches,
    maybe_split_kv_caches_for_spec_layers,
    LLMStatusCode,
    LLMRole
)

# Define path constants
VLLM_KV_TRANSFER_MANAGER_PATH = 'omni_npu.connector.llmdatadist_manager_v1'
LLM_DATADIST_PATH = 'omni_npu.connector.llmdatadist_manager_v1'


class TestLLMDataDistManager:
    @pytest.fixture
    def mock_llm_datadist(self):
        with patch(f'{LLM_DATADIST_PATH}.LLMDataDist') as mock_datadist:
            mock_instance = MagicMock()
            mock_datadist.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_world_group(self):
        with patch(f'{VLLM_KV_TRANSFER_MANAGER_PATH}.get_world_group') as mock_get_world_group:
            mock_world_group = MagicMock()
            mock_world_group.rank_in_group = 0
            mock_world_group.local_rank = 0
            mock_get_world_group.return_value = mock_world_group
            yield mock_world_group

    @pytest.fixture
    def mock_vllm_config(self):
        config = MagicMock(spec=VllmConfig)
        config.kv_transfer_config = MagicMock()
        config.kv_transfer_config.kv_role = 'kv_producer'
        config.kv_transfer_config.kv_parallel_size = 2
        config.kv_transfer_config.kv_connector_extra_config = {'kv_producer_dp_size': 1}
        config.parallel_config = MagicMock()
        config.parallel_config.data_parallel_rank = 0
        config.parallel_config.tensor_parallel_size = 1
        config.parallel_config.data_parallel_size = 1
        yield config

    @pytest.fixture
    def mock_block_cache_key(self):
        with patch(f"{VLLM_KV_TRANSFER_MANAGER_PATH}.BlocksCacheKey") as mock_obj:
            yield mock_obj

    @pytest.fixture
    def mock_kv_cache_retry_times(self):
        from omni_npu.connector import llmdatadist_manager_v1
        ori = llmdatadist_manager_v1.KV_CACHE_RETRY_TIMES
        llmdatadist_manager_v1.KV_CACHE_RETRY_TIMES = 3
        yield llmdatadist_manager_v1.KV_CACHE_RETRY_TIMES
        llmdatadist_manager_v1.KV_CACHE_RETRY_TIMES = ori

    def test_init_llm_data_dist_manager(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        assert manager.rank == 0
        assert manager.local_rank == 0
        assert manager.tp_size == 1
        assert manager.dp_size == 1
        assert manager.dp_rank == 0
        assert manager.prefill_dp_size == 1
        assert manager.data_dist_engine == mock_llm_datadist

    @pytest.mark.parametrize(
        "remote_cluster_id,remote_dp_rank,expected_result",
        [
            ((12345, 67890), 0, [(12345, 67890)]),
            ([12345, 67890], 0, [(12345, 67890)]),
        ]
    )
    def test_get_real_remote_cluster_ids_found(self, mock_vllm_config, mock_llm_datadist, mock_world_group, 
                                               remote_cluster_id, remote_dp_rank, expected_result):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        meta = MagicMock()
        meta.remote_cluster_id = remote_cluster_id
        meta.remote_dp_rank = remote_dp_rank
        
        key = (tuple(remote_cluster_id) if isinstance(remote_cluster_id, list) else remote_cluster_id, 
               remote_dp_rank, 0)
        manager.registered_link_infos[key] = expected_result
        
        result = manager.get_real_remote_cluster_ids(meta)
        
        assert result == expected_result

    def test_get_real_remote_cluster_ids_not_found_register_link(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        meta = MagicMock()
        meta.remote_cluster_id = (12345, 67890)
        meta.remote_dp_rank = 0

        manager.registered_link_infos[(54321, 67890), 0, 0] = None

        # Mock register_link to avoid side effects
        with patch.object(manager, 'register_link') as mock_register_link, \
             patch.object(manager, 'close_link') as mock_close_link:
            result = manager.get_real_remote_cluster_ids(meta)
            
            mock_register_link.assert_called_once_with((12345, 67890), 0, 0, 0)
            assert result is None  # Since we don't set it in the dict
            mock_close_link.assert_called()

    def test_register_link_success(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        mock_llm_datadist.link_clusters.return_value = LLMStatusCode.LLM_SUCCESS, None

        with patch.object(manager, '_get_cluster_id_list', return_value=[12345]):
            with patch.object(manager, 'cluster_id_to_ip_port', return_value=("127.0.0.1:8000", 1, 0)):
                with patch.object(manager, '_get_local_ip', return_value="127.0.0.1"):
                    manager.register_link((12345,), 0, 0)
        
        mock_llm_datadist.link_clusters.assert_called_once()

    def test_register_link_failure(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)

        mock_llm_datadist.link_clusters.return_value = (1, "error")
        
        with patch.object(manager, '_get_cluster_id_list', return_value=[12345]):
            with patch.object(manager, 'cluster_id_to_ip_port', return_value=("127.0.0.1:8000", 1, 0)):
                with patch.object(manager, '_get_local_ip', return_value="127.0.0.1"):
                    with pytest.raises(Exception, match="link failed"):
                        manager.register_link((12345,), 0, 0)

    def test_close_link_success(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        manager.data_dist_config.is_prefill = False

        mock_llm_datadist.unlink_clusters.return_value = LLMStatusCode.LLM_SUCCESS, None
        
        with patch.object(manager, '_get_cluster_id_list', return_value=[12345]):
            with patch.object(manager, 'cluster_id_to_ip_port', return_value=("127.0.0.1:8000", 1, 0)):
                with patch.object(manager, '_get_local_ip', return_value="127.0.0.1"):
                    manager.close_link((12345, ), 0, 0, 0)
        
        mock_llm_datadist.unlink_clusters.assert_called_once()

    def test_close_link_failure(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        mock_llm_datadist.unlink_clusters.return_value = (1, "error")  # Non-success status
        
        with patch.object(manager, '_get_cluster_id_list', return_value=[12345]):
            with patch.object(manager, 'cluster_id_to_ip_port', return_value=("127.0.0.1:8000", 1, 0)):
                with patch.object(manager, '_get_local_ip', return_value="127.0.0.1"):
                    with pytest.raises(Exception, match="unlink failed"):
                        manager.close_link(12345, 0, 0, 0)

    def test_force_unlink(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        manager.force_unlink(12345)
        
        mock_llm_datadist.unlink_clusters.assert_called_once()

    @pytest.mark.parametrize(
        "exception_type, status_code, expected_result",
        [
            (None, None, True),  # Success case
            ("LLMException", 0, False),  # Non-retryable exception
            ("LLMException", 1, False),  # Retryable exception, max retries reached
        ]
    )
    def test_pull_blocks(self, mock_vllm_config, mock_llm_datadist, mock_world_group, 
                         exception_type, status_code, expected_result):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        if exception_type == "LLMException":
            from llm_datadist import LLMException, LLMStatusCode
            if status_code in RETRYABLE_CODES:
                mock_llm_datadist.cache_manager.pull_blocks.side_effect = [
                    LLMException("test", status_code),
                    LLMException("test", status_code)
                ]
            else:
                mock_llm_datadist.cache_manager.pull_blocks.side_effect = [
                    LLMException("test", status_code)
                ]
        else:
            # Success case
            pass

        src_cache_key = MagicMock()
        dst_cache = MagicMock()
        src_blocks = [0]
        dst_blocks = [0]
        
        result = manager._pull_blocks(src_cache_key, dst_cache, src_blocks, dst_blocks)
        
        assert result == expected_result

    def test_pull_blocks_retryable_code_success_after_retry(self, mock_vllm_config, mock_llm_datadist, mock_world_group, mock_kv_cache_retry_times):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        from llm_datadist import LLMException, LLMStatusCode
        # First call fails with retryable code, second succeeds
        mock_llm_datadist.cache_manager.pull_blocks.side_effect = [
            LLMException("test", status_code=LLMStatusCode.LLM_TIMEOUT),
            None  # Success on second try
        ]
        
        src_cache_key = MagicMock()
        dst_cache = MagicMock()
        src_blocks = [0]
        dst_blocks = [0]
        
        result = manager._pull_blocks(src_cache_key, dst_cache, src_blocks, dst_blocks)
        
        assert result == True
        assert mock_llm_datadist.cache_manager.pull_blocks.call_count == 2

    def test_pull_blocks_failure_after_retry(self, mock_vllm_config, mock_llm_datadist, mock_world_group, mock_kv_cache_retry_times):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)

        from llm_datadist import LLMException, LLMStatusCode
        # First call fails with retryable code, second succeeds
        mock_llm_datadist.cache_manager.pull_blocks.side_effect = LLMException("test", status_code=LLMStatusCode.LLM_TIMEOUT)

        mock_cache = MagicMock()
        manager.registered_kv_caches = [mock_cache]

        with patch.object(manager, "_refresh_link"):
            with pytest.raises(RuntimeError):
                manager.pull_kv([0], [0], 12345, 0)

        mock_llm_datadist.cache_manager.pull_blocks.side_effect = ValueError

        with patch.object(manager, "_refresh_link"):
            with pytest.raises(RuntimeError):
                manager.pull_kv([0], [0], 12345, 0)

        from omni_npu.connector import llmdatadist_manager_v1
        llmdatadist_manager_v1.KV_CACHE_RETRY_TIMES = 0

        with patch.object(manager, "_refresh_link"):
            with pytest.raises(RuntimeError):
                manager.pull_kv([0], [0], 12345, 0)

    def test_pull_kv_success(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # Mock registered_kv_caches
        mock_cache = MagicMock()
        manager.registered_kv_caches = [mock_cache]
        
        # Mock _pull_blocks to return True
        with patch.object(manager, '_pull_blocks', return_value=True):
            manager.pull_kv([0], [0], 12345, 0)
        
            # Verify _pull_blocks was called
            manager._pull_blocks.assert_called_once()

    def test_pull_kv_failure_then_success_with_refresh(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # Mock registered_kv_caches
        mock_cache = MagicMock()
        manager.registered_kv_caches = [mock_cache]
        
        # Mock _pull_blocks to fail first, then succeed
        with patch.object(manager, '_pull_blocks', side_effect=[False, True]):
            with patch.object(manager, '_refresh_link'):
                manager.pull_kv([0], [0], 12345, 0)
        
            # _pull_blocks should be called twice (first fail, then success after refresh)
            assert manager._pull_blocks.call_count == 2

    def test_pull_kv_failure_even_after_refresh(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # Mock registered_kv_caches
        mock_cache = MagicMock()
        manager.registered_kv_caches = [mock_cache]
        
        # Mock _pull_blocks to always fail
        with patch.object(manager, '_pull_blocks', return_value=False):
            with patch.object(manager, '_refresh_link'):
                with pytest.raises(RuntimeError, match="Failed to pull kv even if rebuild the kv link!"):
                    manager.pull_kv([0], [0], 12345, 0)

    def test_refresh_link_success(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # Mock _get_host_cluster_id to return a valid host_cluster_id
        with patch.object(manager, '_get_host_cluster_id', return_value=((12345,), 0, 0)):
            with patch.object(manager, 'close_link'):
                with patch.object(manager, 'register_link'):
                    manager._refresh_link(12345, 0, 0)

                    manager.register_link.assert_called_once()
                manager.close_link.assert_called_once()

    def test_refresh_link_no_host_cluster_id(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # Mock _get_host_cluster_id to return None
        with patch.object(manager, '_get_host_cluster_id', return_value=(None, None, None)):
            with pytest.raises(RuntimeError, match="Unregistered host cluster id!!!"):
                manager._refresh_link(12345, 0, 0)

    def test_get_host_cluster_id_found(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # Add a matching entry to registered_link_infos
        manager.registered_link_infos[((12345,), 0, 0)] = [12345]
        
        result = manager._get_host_cluster_id(12345, 0, 0)
        
        assert result == ((12345,), 0, 0)

    def test_get_host_cluster_id_not_found(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # No matching entry
        result = manager._get_host_cluster_id(12345, 0, 0)
        
        assert result is None

    def test_get_cluster_id_list(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # Mock cluster_id_to_ip_port
        with patch.object(manager, 'cluster_id_to_ip_port', return_value=("127.0.0.1:8000", 1, 0)):
            result = manager._get_cluster_id_list([12345], 0, 0, 0)
        
        assert len(result) == 1
        assert isinstance(result[0], int)

        with patch.object(manager, 'cluster_id_to_ip_port', return_value=("127.0.0.1:8000", 1, 0)):
            result = manager._get_cluster_id_list(12345, 0, 0, 0)

        assert len(result) == 1
        assert isinstance(result[0], int)

    def test_register_memory_dense_model(self, mock_vllm_config, mock_llm_datadist, mock_world_group, mock_block_cache_key):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # Create mock KV caches
        kv_cache = {
            'layer.0': torch.randn(2, 4, 8, 16, dtype=torch.float16)
        }
        
        mock_cache = MagicMock()
        mock_llm_datadist.cache_manager.register_blocks_cache.return_value = mock_cache
        
        manager.register_memory(kv_cache)
        
        # Verify the cache was registered
        assert len(manager.registered_kv_caches) == 1
        assert manager.registered_kv_caches[0] == mock_cache

    def test_register_memory_dense_model_tuple(self, mock_vllm_config, mock_llm_datadist, mock_world_group, mock_block_cache_key):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # Create mock KV caches with tuple
        kv_cache = {
            'layer.0': (torch.randn(4, 8, 16, dtype=torch.float16), torch.randn(4, 8, 16, dtype=torch.float16))
        }
        
        mock_cache = MagicMock()
        mock_llm_datadist.cache_manager.register_blocks_cache.return_value = mock_cache
        
        manager.register_memory(kv_cache)
        
        # Verify the cache was registered
        assert len(manager.registered_kv_caches) == 2

    def test_register_memory_duplicate_call(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # Pre-populate registered_kv_caches
        manager.registered_kv_caches = [MagicMock()]
        
        kv_cache = {
            'layer.0': torch.randn(2, 4, 8, 16, dtype=torch.float16)
        }
        
        with pytest.raises(ValueError, match="Attr `registered_kv_caches` must be empty before register kv_caches."):
            manager.register_memory(kv_cache)

    def test_cluster_id_to_ip_port(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # Create a test cluster_id
        cluster_id = ip_port_to_int("127.0.0.1:8000", 2)
        
        ip_port, tp_size, tp_rank = manager.cluster_id_to_ip_port(cluster_id)
        
        assert ip_port == "127.0.0.1:8000"
        assert tp_size == 2
        assert tp_rank == 0

    def test_cluster_id_to_ip_port_invalid_type(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        with pytest.raises(TypeError, match="cluster_id must be int type"):
            manager.cluster_id_to_ip_port("not_an_int")


class TestHelper:
    @pytest.mark.parametrize(
        "kv_caches,expect_len",
        [
            ({"layer.0": torch.zeros(1)}, 1),
            ({"layer.0": (torch.zeros(1), torch.ones(1))}, 2),
        ],
    )
    def test_unzip_kv_cache_dict_basic(self, kv_caches, expect_len):
        out = unzip_kv_cache_dict(kv_caches)
        assert len(out) == expect_len
        assert out[0][0] in kv_caches["layer.0"] if isinstance(kv_caches["layer.0"], tuple) else out[0][0] == kv_caches["layer.0"]


    def test_unzip_kv_cache_dict_not_implemented(self):
        kv_caches = {
            "layer.0": torch.zeros(1),
            "layer.dup.0": torch.ones(1),
        }
        with pytest.raises(NotImplementedError):
            unzip_kv_cache_dict(kv_caches)


    @pytest.mark.parametrize(
        "kv_list,expect_len",
        [
            ([torch.zeros(1)], 1),
            ([(torch.zeros(1), torch.ones(1))], 2),
        ],
    )
    def test_unzip_kv_cache_list(self, kv_list, expect_len):
        out = unzip_kv_cache_list(kv_list)
        assert len(out) == expect_len


    @pytest.mark.skip
    def test_maybe_merge_kv_caches(self):
        t = torch.zeros(2, 1, 1, 1, 1)
        flatten = [[t]]
        out = maybe_merge_kv_caches(flatten)
        assert len(out) == 2
        assert out[0][0].shape == (1, 1, 1, 1)


    def test_maybe_merge_kv_caches_no_merge(self):
        t = torch.zeros(1, 1)
        flatten = [[t]]
        assert maybe_merge_kv_caches(flatten) == flatten


    def test_maybe_split_kv_caches(self):
        t1 = torch.zeros(1, 2)
        t2 = torch.zeros(2, 2)
        flatten = [[t1, t2]]
        out = maybe_split_kv_caches_for_spec_layers(flatten)
        assert len(out) == 2
        assert t1 in out[0] or t1 in out[1]


    def test_maybe_split_kv_caches_no_split(self):
        t = torch.zeros(1, 2)
        flatten = [[t, t.clone()]]
        assert maybe_split_kv_caches_for_spec_layers(flatten) == flatten


    @pytest.mark.parametrize(
        "ip_port,tp_size",
        [
            ("127.0.0.1:8000", 1),
            ("0.0.0.0:0", 65535),
        ],
    )
    def test_ip_port_to_int(self, ip_port, tp_size):
        val = ip_port_to_int(ip_port, tp_size)
        assert isinstance(val, int)


    def test_ip_port_to_int_invalid_port(self):
        with pytest.raises(ValueError):
            ip_port_to_int("127.0.0.1:70000", 1)


class TestLLMDataDistConfig:
    @pytest.fixture
    def vllm_config(self):
        """Mock VllmConfig object"""
        mock_vllm_config = MagicMock()
        mock_vllm_config.kv_transfer_config.kv_role = "kv_producer"
        mock_vllm_config.parallel_config.data_parallel_rank = 0
        mock_vllm_config.parallel_config.tensor_parallel_size = 2
        mock_vllm_config.parallel_config.data_parallel_size = 2
        mock_vllm_config.kv_transfer_config.kv_parallel_size = 2
        mock_vllm_config.kv_transfer_config.kv_connector_extra_config = {"kv_producer_dp_size": 1}
        return mock_vllm_config

    @pytest.fixture
    def mock_ray(self):
        import sys
        pre_ray = sys.modules.get("ray", None)
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = False
        sys.modules["ray"] = mock_ray

        yield mock_ray

        if pre_ray is not None:
            sys.modules["ray"] = pre_ray
        else:
            del sys.modules["ray"]

    @pytest.fixture
    def block_ray_import(self):
        import sys

        class BlockModuleImporter:
            def __init__(self, blocked_name: str):
                self.blocked_name = blocked_name

            def find_spec(self, fullname, path, target=None):
                # 精确匹配 or 阻断子模块
                if fullname == self.blocked_name or fullname.startswith(self.blocked_name + "."):
                    raise ImportError(f"Module '{fullname}' is blocked")
                return None  # 交给下一个 importer

        pre_ray = sys.modules.get("ray", None)
        if pre_ray is not None:
            del sys.modules["ray"]

        module_name = "ray"
        importer = BlockModuleImporter(module_name)
        sys.meta_path.insert(0, importer)
        try:
            yield
        finally:
            if importer in sys.meta_path:
                sys.meta_path.remove(importer)
            if pre_ray:
                sys.modules["ray"] = pre_ray

    @pytest.fixture
    def mock_get_world_group(self):
        with mock.patch("omni_npu.connector.llmdatadist_manager_v1.get_world_group") as mock_world_group:
            yield mock_world_group

    @pytest.fixture
    def mock_ip_port_to_int(self):
        """Mock ip_port_to_int"""
        with mock.patch('omni_npu.connector.llmdatadist_manager_v1.ip_port_to_int') as mock_ip:
            yield mock_ip


    @pytest.mark.parametrize(
        "ignore_load_rank, expected_rank, expected_local_rank, expected_cluster_id, expected_ip_list",
        [
            (True, -1, -1, -1, ["127.0.0.1"]),
            (False, 0, 0, 123456, ["127.0.0.1"]),
        ]
    )
    def test_init(self, ignore_load_rank, expected_rank, expected_local_rank, expected_cluster_id, expected_ip_list, vllm_config,
                  mock_ip_port_to_int, mock_get_world_group, block_ray_import):
        """Test initialization with different ignore_load_rank values."""
        mock_ip_port_to_int.return_value = 123456 if not ignore_load_rank else -1
        if ignore_load_rank:
            mock_ip_port_to_int.return_value = expected_cluster_id
        else:
            mock_ip_port_to_int.return_value = expected_cluster_id
            mock_get_world_group.return_value.rank_in_group = expected_rank
            mock_get_world_group.return_value.local_rank = expected_local_rank

        config = LLMDataDistConfig(vllm_config, "127.0.0.1", 8080, ignore_load_rank=ignore_load_rank)

        assert config.rank == expected_rank
        assert config.local_rank == expected_local_rank
        assert config.cluster_id == expected_cluster_id
        assert config.host_ip_list == expected_ip_list


    @pytest.mark.parametrize(
        "ray_nodes, expected_ips",
        [
            ([], ["127.0.0.1"]),  # No nodes
            ([{"Alive": True, "NodeManagerAddress": "192.168.1.1", "GcsAddress": "192.168.1.1:12345"},
              {"Alive": True, "NodeManagerAddress": "192.168.1.2", "GcsAddress": "192.168.1.2:12345"}],
             ["192.168.1.2", "192.168.1.1"]),
            ([{"Alive": False, "NodeManagerAddress": "192.168.1.1"},
              {"Alive": True, "NodeManagerAddress": "192.168.1.2", "GcsAddress": "192.168.1.2:12345"}],
             ["192.168.1.2"]),
            ([{"Alive": False, "NodeManagerAddress": "192.168.1.1"},
              {"Alive": True, "NodeManagerAddress": "192.168.1.2"}],
             ["192.168.1.2"]),
        ]
    )
    def test_get_worker_ips(self, ray_nodes, expected_ips, vllm_config, mock_ray):
        """Test _get_worker_ips with different Ray cluster states."""
        mock_ray.nodes.return_value = ray_nodes
        config = LLMDataDistConfig(vllm_config, "127.0.0.1", 8080, ignore_load_rank=True)

        assert config._get_worker_ips() == expected_ips

        mock_ray.init.side_effect = [RuntimeError]
        config = LLMDataDistConfig(vllm_config, "127.0.0.1", 8080, ignore_load_rank=True)
        assert config._get_worker_ips() == ["127.0.0.1"]

    @pytest.mark.parametrize(
        "role_str, expected_role",
        [
            ("kv_producer", LLMRole.PROMPT),  # Simulating 'PROMPT' role
        ]
    )
    def test_role_property(self, role_str, expected_role, vllm_config):
        """Test role property."""
        vllm_config.kv_transfer_config.kv_role = role_str
        config = LLMDataDistConfig(vllm_config, "127.0.0.1", 8080, ignore_load_rank=True)

        assert config.role == expected_role


    @pytest.mark.parametrize(
        "role_str, expected_prefill",
        [
            ("kv_producer", True),  # 'PROMPT' role is expected to be prefill
        ]
    )
    def test_is_prefill_property(self, role_str, expected_prefill, vllm_config):
        """Test is_prefill property."""
        vllm_config.kv_transfer_config.kv_role = role_str
        config = LLMDataDistConfig(vllm_config, "127.0.0.1", 8080, ignore_load_rank=True)

        assert config.is_prefill == expected_prefill
