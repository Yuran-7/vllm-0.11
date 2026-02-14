# vLLM 注意力后端配置指南

## 1. 为什么 Flash-Attn 是默认注意力后端？

### vLLM 内置的 Flash Attention 实现

vLLM **并不需要用户手动安装** `flash-attn` 包，因为它在构建时已经将 Flash Attention 集成到了代码库中。

#### 实现机制

1. **vLLM 维护了自己的 Flash Attention fork**
   - 仓库地址：https://github.com/vllm-project/flash-attention
   - Git commit: `ee4d25bd84e0cbc7e0b9b9685085fd5db2dcb62a`
   - 参考：`cmake/external_projects/vllm_flash_attn.cmake`

2. **编译时自动构建**
   ```cmake
   FetchContent_Declare(
       vllm-flash-attn
       GIT_REPOSITORY https://github.com/vllm-project/flash-attention.git
       GIT_TAG ee4d25bd84e0cbc7e0b9b9685085fd5db2dcb62a
   )
   ```
   - 在编译 vLLM 时，CMake 会自动下载并编译 Flash Attention
   - 编译后的库安装到 `vllm/vllm_flash_attn/` 目录

3. **导入方式**
   ```python
   # 不是从外部包导入
   # from flash_attn import flash_attn_varlen_func  ❌
   
   # 而是从 vLLM 内部模块导入
   from vllm.vllm_flash_attn import flash_attn_varlen_func  ✅
   ```

#### 为什么要这样做？

| 优势 | 说明 |
|------|------|
| ✅ 版本控制精确 | 锁定特定的 commit，避免版本不兼容 |
| ✅ 无需手动安装 | 用户不需要单独安装 flash-attn 包 |
| ✅ 可定制化 | vLLM 可以应用自己的补丁和优化 |
| ✅ 避免冲突 | 不与用户环境中的 flash-attn 冲突 |
| ✅ 统一构建 | 确保所有用户使用相同版本 |

### V1 引擎中的默认后端选择逻辑

在 V1 引擎（`VLLM_USE_V1=1`，默认启用）中，后端选择遵循以下优先级：

```python
# 参考：vllm/platforms/cuda.py:get_attn_backend_cls()

# 1. Blackwell GPU (SM 10.0，如 H200)
if compute_capability == 10.0:
    默认后端 = FlashInfer (需要安装)
    备选后端 = Flash Attention

# 2. Ampere+ GPU (SM 8.0+，如 A100, RTX 4090, H100)
elif compute_capability >= 8.0:
    if has_sink or use_fp8_kv_cache:
        默认后端 = Triton Attention
    else:
        默认后端 = Flash Attention  # ← 4090 走这里

# 3. 更老的 GPU (SM < 8.0)
else:
    默认后端 = FlexAttention
```

#### 为什么 Flash Attention 是 Ampere+ GPU 的默认选择？

1. **性能优秀**：Flash Attention 在 Ampere 及以后的架构上有出色的性能
2. **内置支持**：已编译集成，无需额外依赖
3. **广泛验证**：经过大量测试，稳定可靠
4. **功能完整**：支持 FA2 和 FA3，覆盖多种场景

---

## 2. 如何在 RTX 4090 上使用 FlashInfer 后端？

### RTX 4090 硬件信息

- **架构**：Ada Lovelace (Ampere 后继)
- **Compute Capability**：8.9 (SM 8.9)
- **默认后端**：Flash Attention

### 环境变量工作原理

在了解具体配置方法之前，先理解为什么设置 `VLLM_ATTENTION_BACKEND=FLASHINFER` 就能生效。

#### 代码执行流程

```
用户设置环境变量
    ↓
vllm/attention/selector.py::_cached_get_attn_backend()
    ↓
读取 envs.VLLM_ATTENTION_BACKEND (值为 "FLASHINFER")
    ↓
backend_name_to_enum("FLASHINFER") 
    → 将字符串转换为 _Backend.FLASHINFER 枚举
    ↓
调用 current_platform.get_attn_backend_cls(selected_backend=_Backend.FLASHINFER, ...)
    ↓
vllm/platforms/cuda.py::get_attn_backend_cls()
    → 检查 if selected_backend == _Backend.FLASHINFER: ✅成立
    ↓
返回 FlashInfer 后端实现类
```

#### 关键代码解析

**1. 环境变量读取**（`vllm/attention/selector.py:190`）
```python
# 从环境变量读取后端名称
backend_by_env_var: Optional[str] = envs.VLLM_ATTENTION_BACKEND
# 此时 backend_by_env_var = "FLASHINFER"
```

**2. 字符串转枚举**（`vllm/attention/selector.py:200`）
```python
# 将字符串 "FLASHINFER" 转换为枚举 _Backend.FLASHINFER
selected_backend = backend_name_to_enum(backend_by_env_var)

# backend_name_to_enum 函数实现（selector.py:21-31）：
def backend_name_to_enum(backend_name: str) -> Optional[_Backend]:
    # 使用 Python 枚举的字符串访问机制
    # _Backend["FLASHINFER"] → _Backend.FLASHINFER
    return _Backend[backend_name] if backend_name in _Backend.__members__ else None
```

**3. 后端选择判断**（`vllm/platforms/cuda.py:311`）
```python
if selected_backend == _Backend.FLASHINFER:
    logger.info_once("Using FlashInfer backend on V1 engine.")
    if cls.has_device_capability(100):  # Blackwell GPU
        from vllm.v1.attention.backends.utils import set_kv_cache_layout
        set_kv_cache_layout("HND")
    return FLASHINFER_V1  # 返回 FlashInfer 实现类路径
```

#### 有效的环境变量值

环境变量的值必须是 `_Backend` 枚举的**成员名称**（定义在 `vllm/platforms/interface.py:41-61`）：

```python
class _Backend(enum.Enum):
    FLASH_ATTN = enum.auto()      # ✅ 有效
    FLASHINFER = enum.auto()      # ✅ 有效
    FLEX_ATTENTION = enum.auto()  # ✅ 有效
    TRITON_ATTN = enum.auto()     # ✅ 有效
    XFORMERS = enum.auto()        # ✅ 有效
    TREE_ATTN = enum.auto()       # ✅ 有效
    # ... 其他后端
```

**示例：**
- ✅ `export VLLM_ATTENTION_BACKEND=FLASHINFER` → 正确，匹配枚举名
- ✅ `export VLLM_ATTENTION_BACKEND=FLASH_ATTN` → 正确
- ❌ `export VLLM_ATTENTION_BACKEND=flashinfer` → 错误，大小写不匹配
- ❌ `export VLLM_ATTENTION_BACKEND=FlashInfer` → 错误，必须全大写

#### 优先级顺序

后端选择有明确的优先级（从高到低）：

1. **代码强制指定**（`global_force_attn_backend()`）- 最高优先级
2. **环境变量** `VLLM_ATTENTION_BACKEND` - 次之
3. **自动选择**（基于 GPU 类型和配置）- 默认行为

参考代码（`vllm/attention/selector.py:178-205`）：
```python
# 1. 检查是否有代码强制指定
backend_by_global_setting = get_global_forced_attn_backend()
if backend_by_global_setting is not None:
    selected_backend = backend_by_global_setting  # 最高优先级
else:
    # 2. 检查环境变量
    backend_by_env_var = envs.VLLM_ATTENTION_BACKEND
    if backend_by_env_var is not None:
        selected_backend = backend_name_to_enum(backend_by_env_var)
    # 3. 如果都没有，selected_backend = None，走自动选择逻辑
```

---

### 方法一：设置环境变量（推荐）

在启动 vLLM 之前设置环境变量：

#### Linux/macOS:
```bash
export VLLM_ATTENTION_BACKEND=FLASHINFER
python your_script.py
```

或者一次性运行：
```bash
VLLM_ATTENTION_BACKEND=FLASHINFER python your_script.py
```


### 方法二：在 Python 代码中设置

```python
import os
os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

from vllm import LLM, SamplingParams

# 初始化模型（会使用 FlashInfer 后端）
llm = LLM(model="meta-llama/Llama-2-7b-hf")
```

### 方法三：在启动服务时指定

如果使用 vLLM 的 OpenAI 兼容服务器：

```bash
VLLM_ATTENTION_BACKEND=FLASHINFER \
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --port 8000
```

### 前提条件：安装 FlashInfer

FlashInfer 是一个**独立的第三方库**，需要单独安装：

```bash
# 安装 FlashInfer
pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/

# 或者从源码安装
pip install flashinfer
```

**注意**：
- FlashInfer 与 Flash Attention 不同，是独立的库
- 需要根据你的 CUDA 版本和 PyTorch 版本选择对应的 wheel
- 如果不安装 FlashInfer，设置环境变量后会报错

### 验证后端是否生效

启动 vLLM 后，查看日志输出：

```
INFO 12:00:00 cuda.py:313] Using FlashInfer backend on V1 engine.
```

或者在代码中检查：

```python
import logging
logging.basicConfig(level=logging.INFO)

from vllm import LLM

llm = LLM(model="meta-llama/Llama-2-7b-hf")
# 查看日志中的 "Using ... backend on V1 engine"
```

---

## 3. 所有可用的注意力后端

### V1 引擎支持的后端列表

| 后端名称 | 环境变量值 | 适用场景 | 是否需要安装 |
|---------|-----------|---------|-------------|
| **Flash Attention** | `FLASH_ATTN` | SM 8.0+ GPU，通用推理 | ❌ 内置 |
| **FlashInfer** | `FLASHINFER` | SM 10.0 (Blackwell) 优先，高性能 | ✅ 需要安装 |
| **FlexAttention** | `FLEX_ATTENTION` | 旧 GPU (< SM 8.0)，兼容性好 | ❌ 内置 |
| **Triton Attention** | `TRITON_ATTN` | FP8 KV cache，特殊场景 | ❌ 内置 |
| **XFormers** | `XFORMERS` | 兼容性备选 | ✅ 需要安装 |
| **Tree Attention** | `TREE_ATTN` | 树形结构推理 | ❌ 内置 |

### MLA (Multi-head Latent Attention) 专用后端

如果使用支持 MLA 的模型（如 DeepSeek-V2）：

| 后端名称 | 环境变量值 | 适用场景 |
|---------|-----------|---------|
| **FlashMLA** | `FLASHMLA` | SM 9.0 GPU，block_size=64 |
| **CutlassMLA** | `CUTLASS_MLA` | SM 10.0 (Blackwell)，block_size=128 |
| **FlashInferMLA** | `FLASHINFER_MLA` | SM 10.0，block_size=32/64 |
| **FlashAttnMLA** | `FLASH_ATTN_MLA` | 通用 MLA 实现 |
| **TritonMLA** | `TRITON_MLA` | 备选 MLA 实现 |

---



## 4. 相关环境变量

除了 `VLLM_ATTENTION_BACKEND`，还有其他相关环境变量：

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `VLLM_USE_V1` | 是否使用 V1 引擎 | `1` (启用) |
| `VLLM_FLASH_ATTN_VERSION` | 指定 Flash Attn 版本 (2 或 3) | 自动选择 |
| `VLLM_ATTENTION_BACKEND` | 指定注意力后端 | 自动选择 |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | V1 多进程模式 | `1` (启用) |

---


**文档日期**：2026年2月14日  
**vLLM 版本**：基于最新主分支  
**维护者**：vLLM Community
