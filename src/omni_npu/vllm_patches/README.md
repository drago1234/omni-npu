# 添加自定义vLLM补丁到omni-npu中

本模块参考自：https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html

本指南将帮助你把自定义的vLLM补丁添加到omni-npu中。

## 1. 准备补丁文件
将你的补丁文件放置在`src/omni_npu/vllm_patches/patches`目录。

补丁文件应定义至少一个继承自`VLLMPatch`类的补丁类，并添加`@register_patch(name, target)`装饰器实现注册，同时需要定义好补丁类自己的`_attr_names_to_apply`以表示目标类/公共函数中需要被新增或修改的部分。

`@register_patch(name, target)`装饰器中的`name`表示在`PatchManager`中的注册名，`target`表示vLLM中被打补丁的类/模块（类内修改->补丁类；公共函数修改->对应模块）。

可以参考`src/omni_npu/vllm_patches/patches/examples/llm_engine_hello_world.py`中的`LLMEngineHelloWorldPatch`补丁类，如下所示：

类函数拓展：
```python
from vllm.v1.engine.llm_engine import LLMEngine

@register_patch("LLMEngineHelloWorld", LLMEngine)
class LLMEngineHelloWorldPatch(VLLMPatch):
    """
    Makes LLMEngines print 'Hello World' when get supported tasks.
    """

    _attr_names_to_apply = ['print_hello_world', 'get_supported_tasks']

    @staticmethod
    def print_hello_world():
        print("Hello World")

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        self.print_hello_world()
        return self.engine_core.get_supported_tasks()
```
在上述例子中，`LLMEngineHelloWorldPatch`这个补丁类在`PatchManager`中注册了名为`"LLMEngineHelloWorld"`的补丁，vLLM中目标类为`vllm.v1.engine.llm_engine.LLMEngine`。
`LLMEngineHelloWorldPatch`通过`_attr_names_to_apply`指明了需要新增或替换`vllm.v1.engine.llm_engine.LLMEngine`中的`print_hello_world`及`get_supported_tasks`，实现调用`get_supported_tasks`时打印出Hello World。

模块级函数拓展：
```python
import vllm.engine.arg_utils as arg_utils

@register_patch("GetKwargsHelloWorld", arg_utils)
class GetKwargsHelloWorldPatch(VLLMPatch):
    _attr_names_to_apply = ["get_kwargs"]

    def get_kwargs(cls):
        logger.info(">>> Hello World: get_kwargs is called for %s", cls)
        return copy.deepcopy(_compute_kwargs(cls))
```
在上述例子中，`GetKwargsHelloWorldPatch`这个补丁类在`PatchManager`中注册了名为`"GetKwargsHelloWorld"`的补丁类，vLLM中目标模块为`vllm.engine.arg_utils`。
`GetKwargsHelloWorldPatch`通过`_attr_names_to_apply`指明了需要新增或替换`vllm.engine.arg_utils`模块中的`get_kwargs`方法，实现原函数拓展，打印出Hello World。

## 2. 运行时指定执行补丁

补丁文件需要被注册以及执行两个环节：
    通过补丁类中打上`@register_patch`以及对应文件可以被导入实现注册;
    通过配置`OMNI_NPU_VLLM_PATCHES`参数实现代码被应用

## 2.1 补丁文件注册

`/patches` 文件夹中 `/patches/common` 用于存放公共的补丁文件；`/patches/modes/xxxmodel(qwen、deepseek、pangu)` 用于存放对应模型的补丁文件，目前并未显示的区分具体模型下具体版本的patch文件;
服务启动时通过指定`omni_npu_patches` 会主动将`/patches/common` 以及`/patches/modes/xxxmodel` 下的补丁文件注册到`registered_patches`;其中`/patches/common`为默认注册，`/patches/modes/xxxmodel`通过`model_type`匹配文件夹名称进行注册。

`patches/models/xxxmodel`注册逻辑：

```python
    try:
        MODEL_PATH = Path(sys.argv[2])
    except IndexError:
        raise EnvironmentError("The model path must be passed as the second parameter through the command line.")
    
    model_type = get_model_type_from_config(MODEL_PATH)
    models_root = patches_root / "models"
    model_dir = find_patch_dir_for_model(model_type, models_root)
```


`model="/data/models/DeepSeek-V3.2-INT8"`
上述例子中通过启动参数 `VLLM_PLUGINS="omni-npu,omni_npu_patches,omni_custom_models" vllm serve "$model" `， 获取model的配置路径，读取路径下`config.json`中 `model_type`，与`/patches/models/xxxmodel` 的文件夹名称匹配。


`model_type` 与具体模型文件名称的匹配逻辑支持如下三种方式，按照 `- Mapping table - Prefix matching - containment match`顺序优先匹配返回:

    - Mapping table     `src/omni_npu/vllm_patches/__init__.py` `MODEL_PATCH_MAP = {"deepseek_v3": "deepseek"}` 手动维护MAP，配置model_type 与`patches/models/`  下文件名称`xxxmodel`映射
    - Prefix matching    支持`patches/models/`中 `xxxmodel` 是`model_type`的前缀匹配    
    - containment match  支持`patches/models/`中 `xxxmodel` 是`model_type`的子串匹配

# 2.2 补丁文件执行

通过环境变量`OMNI_NPU_VLLM_PATCHES`指定具体被执行的补丁。
当环境变量`OMNI_NPU_VLLM_PATCHES`为`"ALL"`时，将自动执行`src/omni_npu/vllm_patches/patches`目录中`/common` ;`/models/xxxmodel` 对应patch文件;

否则，根据环境变量`OMNI_NPU_VLLM_PATCHES`执行指定补丁，补丁之间以`,`隔开，格式如下：

```
OMNI_NPU_VLLM_PATCHES="PatchA,PatchB"
```
其中，`PatchA`及`PatchB`为补丁在`PatchManager`类中的注册名，即`@register_patch(name, target)`装饰器中指定的`name`。

## 3. 补丁生效范围
补丁通过 vllm.plugins.load_general_plugins 函数加载替换，只有在patch执行后import的函数，才会被替换成补丁之后的版本