from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request
from omni_npu.vllm_patches.core import VLLMPatch, register_patch


logger = init_logger(__name__)


@register_patch("WaitingRemoteKvPatch", Scheduler)
class WaitingRemoteKvPatch(VLLMPatch):

    _attr_names_to_apply = ['_update_waiting_for_remote_kv']

    def _update_waiting_for_remote_kv(self: Scheduler, request: Request) -> bool:
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            num_computed_tokens = request.num_tokens - 1
            self.kv_cache_manager.cache_blocks(request, num_computed_tokens)
            request.num_computed_tokens = num_computed_tokens

        # Return that we are ready.
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True
