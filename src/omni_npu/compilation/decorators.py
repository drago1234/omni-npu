import functools
import vllm.compilation.decorators as _dec_mododule
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger


logger = init_logger("vllm.omni_npu.compilation.decorators")


def _bypass_prefill(self, *args, **kwargs):
    # patch vllm's _support_torch_compile's __call__
    batch_descriptor = get_forward_context().batch_descriptor
    if batch_descriptor is None or not batch_descriptor.uniform_decode:
        logger.debug(f"<<< use original forward")
        return True, self.forward(*args, **kwargs)
    return False, None

def _wrap_call(original_call):
    @functools.wraps(original_call)
    def _new_call(self, *args, **kwargs):
        hit, retval = _bypass_prefill(self, *args, **kwargs)
        logger.debug(f"<<< {hit=}, {retval=}")
        if hit:
            return retval
        logger.debug(f"<<< {hit=}, {retval=}, use original_call")
        return original_call(self, *args, **kwargs)
    return _new_call

def patch_compile_decorators():
    _original_decorator = _dec_mododule._support_torch_compile

    def _patched_support_torch_compile(cls, dynamic_arg_dims, enable_if=None):
        cls = _original_decorator(cls, dynamic_arg_dims, enable_if)

        cls.__call__ = _wrap_call(cls.__call__)
        logger.debug("<<< cls.__call__ wrapped!")
        return cls

    _dec_mododule._support_torch_compile = _patched_support_torch_compile
    logger.debug("<<< _patched_support_torch_compile applied!")
