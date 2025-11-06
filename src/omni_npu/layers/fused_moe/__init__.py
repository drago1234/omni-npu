from omni_npu.layers.fused_moe.layer import AscendFusedMoE, AscendNonOverlapSharedFusedMoE
import vllm.model_executor.models.deepseek_v2 as deepseek_v2


def update_moe_layers():
    deepseek_v2.FusedMoE = AscendFusedMoE
    deepseek_v2.SharedFusedMoE = AscendNonOverlapSharedFusedMoE


update_moe_layers()
