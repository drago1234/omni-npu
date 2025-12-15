# init for omni_npu.distributed
from .communicator import NPUCommunicator

__all__ = ["NPUCommunicator"]

import sys
from omni_npu.distributed import eplb_state
from vllm.distributed.eplb.eplb_state import EplbState
sys.modules["vllm.distributed.eplb.eplb_state"] = eplb_state
