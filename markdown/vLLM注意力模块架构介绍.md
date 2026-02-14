# vLLM 注意力模块架构介绍

本文档介绍 vLLM 中两个核心注意力模块：`vllm/attention` 和 `vllm/v1/attention` 的架构设计和职责分工。

---

## 目录
- [概述](#概述)
- [vllm/attention - 统一注意力框架](#vllmattention---统一注意力框架)
- [vllm/v1/attention - V1 引擎专用后端](#vllmv1attention---v1-引擎专用后端)
- [两者关系与协作](#两者关系与协作)
- [使用建议](#使用建议)

---

## 概述

vLLM 0.11.0 采用**双层架构**设计注意力机制：

| 模块 | 定位 | 职责 |
|------|------|------|
| **vllm/attention** | 统一注意力框架层 | 提供抽象接口、后端选择、层封装 |
| **vllm/v1/attention** | V1 引擎执行层 | 实现具体的注意力计算内核 |

```
┌───────────────────────────────────────────────────────┐
│              vllm/attention                           │
│  ┌─────────────────────────────────────────────────┐ │
│  │  selector.py - 后端选择器                        │ │
│  │  layer.py - Attention 类（torch.nn.Module）      │ │
│  │  backends/abstract.py - 抽象接口                 │ │
│  └─────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
                          │
                          │ 调用
                          ▼
┌───────────────────────────────────────────────────────┐
│           vllm/v1/attention/backends                  │
│  ┌─────────────────────────────────────────────────┐ │
│  │  flash_attn.py - Flash Attention 实现            │ │
│  │  flashinfer.py - FlashInfer 实现                 │ │
│  │  flex_attention.py - FlexAttention 实现          │ │
│  │  triton_attn.py - Triton Attention 实现          │ │
│  │  mla/ - Multi-head Latent Attention 实现         │ │
│  └─────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
```

---

## vllm/attention - 统一注意力框架

### 📁 目录结构

```
vllm/attention/
├── __init__.py               # 导出核心接口
├── selector.py               # 后端选择逻辑
├── layer.py                  # Attention 层实现
├── backends/                 # 后端抽象层
│   ├── abstract.py           # AttentionBackend 抽象接口
│   └── utils.py              # 后端工具函数
├── layers/                   # 特殊注意力层
│   ├── chunked_local_attention.py    # 分块局部注意力
│   ├── cross_attention.py            # 交叉注意力
│   └── encoder_only_attention.py     # 编码器专用注意力
├── ops/                      # 注意力操作实现
│   ├── paged_attn.py         # 分页注意力
│   ├── prefix_prefill.py     # 前缀预填充
│   ├── flashmla.py           # Flash MLA 操作
│   ├── merge_attn_states.py  # 注意力状态合并
│   └── ...                   # 其他操作
└── utils/                    # 工具函数
    └── fa_utils.py           # Flash Attention 工具
```

### 核心职责

#### 1. **后端选择器 (selector.py)**

负责根据环境变量、硬件平台、模型配置选择最合适的注意力后端。

**关键函数**：
- `backend_name_to_enum(backend_name: str)` - 字符串转枚举
- `get_env_variable_attn_backend()` - 读取环境变量
- `get_attn_backend()` - 返回选中的后端类

**选择优先级**：
```python
1. 全局强制指定 (force_attn_backend_ctx_manager)
2. 环境变量指定 (VLLM_ATTENTION_BACKEND)
3. 平台自动选择 (current_platform.get_attn_backend_cls)
```

#### 2. **注意力层 (layer.py)**

提供 `Attention` 类作为 `torch.nn.Module`，封装注意力计算。

**核心特性**：
```python
class Attention(nn.Module, AttentionLayerBase):
    """
    注意力层实现：
    1. 存储 key/value 到 KV cache
    2. 执行多头/多查询/分组查询注意力
    3. 返回注意力输出张量
    """
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        cache_config: Optional[CacheConfig] = None,
        use_mla: bool = False,  # Multi-head Latent Attention
        attn_type: str = AttentionType.DECODER,
        ...
    )
```

**支持的注意力类型**：
- `DECODER` - 解码器自注意力
- `ENCODER` - 编码器自注意力（编码器-解码器）
- `ENCODER_ONLY` - 纯编码器自注意力
- `ENCODER_DECODER` - 编码器-解码器交叉注意力

#### 3. **抽象接口 (backends/abstract.py)**

定义所有注意力后端必须实现的接口。

**核心抽象类**：
```python
class AttentionBackend(ABC):
    """所有注意力后端的基类"""
    
    accept_output_buffer: bool = False
    supports_quant_query_input: bool = False
    
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """返回后端名称（如 'FLASH_ATTN'）"""
    
    @staticmethod
    @abstractmethod
    def get_impl_cls() -> Type["AttentionImpl"]:
        """返回具体实现类"""
    
    @staticmethod
    @abstractmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        """返回元数据类"""
    
    @staticmethod
    @abstractmethod
    def get_kv_cache_shape(...) -> Tuple[int, ...]:
        """返回 KV cache 形状"""
```

#### 4. **高级注意力层 (layers/)**

提供特殊场景的注意力实现：

- **chunked_local_attention.py** - 分块局部注意力，适用于长序列
- **cross_attention.py** - 交叉注意力，用于编码器-解码器架构
- **encoder_only_attention.py** - 纯编码器注意力（BERT 类模型）

#### 5. **注意力操作 (ops/)**

底层注意力操作的集合：

- **paged_attn.py** - PagedAttention，支持不连续内存块
- **prefix_prefill.py** - 前缀缓存预填充
- **flashmla.py** - Flash MLA 操作（DeepSeek-V2 等）
- **merge_attn_states.py** - 合并多个注意力状态
- **triton_flash_attention.py** - Triton 实现的 Flash Attention

**关键操作示例**：
```python
# PagedAttention 函数签名
def paged_attention_v1(
    query: torch.Tensor,          # [num_seqs, num_heads, head_size]
    key_cache: torch.Tensor,      # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache: torch.Tensor,    # [num_blocks, block_size, num_kv_heads, head_size]
    block_tables: torch.Tensor,   # [num_seqs, max_num_blocks_per_seq]
    context_lens: torch.Tensor,   # [num_seqs]
    ...
) -> torch.Tensor:
```

---

## vllm/v1/attention - V1 引擎专用后端

### 📁 目录结构

```
vllm/v1/attention/
├── __init__.py               # 空文件（后端由 vllm/attention 导入使用）
└── backends/                 # V1 引擎专用实现
    ├── flash_attn.py         # Flash Attention V1 实现
    ├── flashinfer.py         # FlashInfer V1 实现
    ├── flex_attention.py     # FlexAttention V1 实现
    ├── triton_attn.py        # Triton Attention V1 实现
    ├── xformers.py           # XFormers V1 实现
    ├── tree_attn.py          # TreeAttention V1 实现
    ├── cpu_attn.py           # CPU 注意力实现
    ├── rocm_attn.py          # ROCm/AMD GPU 实现
    ├── pallas.py             # Google Pallas/TPU 实现
    ├── gdn_attn.py           # GDN Attention 实现
    ├── linear_attn.py        # Linear Attention 实现
    ├── short_conv_attn.py    # 短卷积注意力
    ├── mamba*.py             # Mamba/SSM 状态空间模型
    ├── mla/                  # Multi-head Latent Attention 目录
    │   ├── common.py         # MLA 公共代码
    │   ├── flashmla.py       # Flash MLA 实现
    │   ├── cutlass_mla.py    # CUTLASS MLA 实现
    │   ├── flashinfer_mla.py # FlashInfer MLA 实现
    │   ├── flashattn_mla.py  # Flash Attention MLA 实现
    │   ├── triton_mla.py     # Triton MLA 实现
    │   └── indexer.py        # MLA 索引器
    └── utils.py              # V1 后端工具函数
```

### 核心职责

#### 1. **Flash Attention V1 后端 (flash_attn.py)**

基于 vLLM 内置的 Flash Attention 实现（从 vLLM fork 编译）。

**关键特性**：
```python
class FlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supports_quant_query_input: bool = True
    
    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]
    
    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]
    
    @staticmethod
    def get_kv_cache_shape(...) -> tuple[int, ...]:
        # 返回 (2, num_blocks, block_size, num_kv_heads, head_size)
        return (2, num_blocks, block_size, num_kv_heads, head_size)
```

**适用场景**：
- SM 8.0+ NVIDIA GPU（A100, A6000, RTX 3090, RTX 4090 等）
- FP16/BF16 精度推理
- Head size ≤ 256

#### 2. **FlashInfer V1 后端 (flashinfer.py)**

使用独立的 FlashInfer 库（需额外安装）。

**关键特性**：
```python
class FlashInferBackend(AttentionBackend):
    """
    使用 FlashInfer 实现：
    - BatchDecodeWithPagedKVCacheWrapper（解码阶段）
    - BatchPrefillWithPagedKVCacheWrapper（预填充阶段）
    - MultiLevelCascadeAttentionWrapper（级联注意力）
    """
    
    # 支持 TRT-LLM 加速路径
    supports_trtllm_attention: bool = True
    
    # 支持 FP8 KV cache
    supports_fp8_kv_cache: bool = True
```

**适用场景**：
- SM 10.0 (Blackwell B200/B100) 优先
- SM 8.0-9.0 也支持
- 需要高性能分页注意力
- FP8 KV cache 量化

**安装方式**：
```bash
pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/
```

#### 3. **FlexAttention V1 后端 (flex_attention.py)**

使用 PyTorch 内置的 FlexAttention API（torch.nn.attention.flex_attention）。

**适用场景**：
- SM < 8.0 的旧 GPU（P100, V100 等）
- 不支持 Flash Attention 的硬件
- 任意 head size
- 兼容性后备方案

#### 4. **MLA 专用实现 (mla/ 目录)**

Multi-head Latent Attention（多头潜在注意力）的专门优化，用于 DeepSeek-V2、DeepSeek-V3 等模型。

**支持的 MLA 后端**：

| 后端 | 文件 | 适用 GPU | Block Size |
|------|------|----------|------------|
| **FlashMLA** | `flashmla.py` | SM 9.0 (H100, H200) | 64 |
| **CUTLASS MLA** | `cutlass_mla.py` | SM 10.0 (B200, B100) | 128 |
| **FlashInfer MLA** | `flashinfer_mla.py` | SM 10.0 | 32/64 |
| **FlashAttn MLA** | `flashattn_mla.py` | SM 8.0+ | 通用 |
| **Triton MLA** | `triton_mla.py` | 所有 GPU | 通用 |

**MLA 索引器 (indexer.py)**：
```python
class MLAIndexer:
    """
    管理 MLA 的压缩 KV 存储：
    - kv_lora_rank: 压缩维度（如 512）
    - qk_rope_head_dim: RoPE 旋转位置编码维度
    - v_head_dim: Value 投影维度
    """
```

#### 5. **特殊架构支持**

- **Mamba/SSM** (`mamba*.py`) - 状态空间模型（非 Transformer 注意力）
- **Linear Attention** (`linear_attn.py`) - 线性复杂度注意力
- **Short Conv Attention** (`short_conv_attn.py`) - 短卷积注意力
- **GDN Attention** (`gdn_attn.py`) - Gated Depthwise Network 注意力

#### 6. **平台特定实现**

- **CPU** (`cpu_attn.py`) - CPU 优化实现（用于调试或纯 CPU 推理）
- **ROCm** (`rocm_attn.py`, `rocm_aiter_fa.py`) - AMD GPU 实现
- **TPU** (`pallas.py`) - Google TPU 使用 Pallas 编译器

---

## 两者关系与协作

### 调用流程

```
用户代码
   │
   ├─ 导入: from vllm import LLM
   │
   ├─ 初始化: llm = LLM(model="...")
   │
   └─> vllm/attention/layer.py
           │
           ├─ Attention.__init__()
           │   └─> vllm/attention/selector.py
           │       └─ get_attn_backend()
           │           ├─ 1. 读取 VLLM_ATTENTION_BACKEND 环境变量
           │           ├─ 2. 调用 backend_name_to_enum("FLASHINFER")
           │           └─ 3. 返回 FlashInferBackend 类
           │
           └─ Attention.forward()
               └─> vllm/v1/attention/backends/flashinfer.py
                   └─ FlashInferImpl.forward()
                       └─ flashinfer.BatchPrefillWithPagedKVCacheWrapper.run()
```

### 关键接口契约

`vllm/attention` 定义接口，`vllm/v1/attention` 提供实现：

```python
# vllm/attention/backends/abstract.py
class AttentionBackend(ABC):
    @staticmethod
    @abstractmethod
    def get_impl_cls() -> Type["AttentionImpl"]:
        """返回具体实现类"""

# vllm/v1/attention/backends/flash_attn.py
class FlashAttentionBackend(AttentionBackend):
    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl  # 具体实现
```

### 数据流

```
输入张量
   │
   ▼
vllm/attention/layer.py (Attention.forward)
   │
   ├─ Q/K/V 投影
   ├─ 量化（如果启用）
   │
   ▼
vllm/v1/attention/backends/*.py
   │
   ├─ KV cache 写入
   ├─ 注意力计算
   │   ├─ Prefill: flash_attn_varlen_func()
   │   └─ Decode: paged_decode_kernel()
   │
   ▼
输出张量
```

### 共享组件

| 组件 | 位置 | 作用 |
|------|------|------|
| **AttentionMetadata** | `vllm/attention/backends/abstract.py` | 元数据基类 |
| **AttentionType** | `vllm/attention/backends/abstract.py` | 注意力类型枚举 |
| **merge_attn_states** | `vllm/attention/ops/merge_attn_states.py` | 合并注意力状态 |
| **reshape_and_cache_flash** | `vllm/attention/utils/fa_utils.py` | Flash Attention KV cache Reshape |

---

## 使用建议

### 1. 选择合适的后端

| 硬件平台 | 推荐后端 | 配置方法 |
|---------|---------|---------|
| **RTX 4090 / SM 8.9** | FlashInfer | `export VLLM_ATTENTION_BACKEND=FLASHINFER` |
| **A100 / SM 8.0** | Flash Attention | 默认（无需配置） |
| **H100 / SM 9.0** | FlashInfer | `export VLLM_ATTENTION_BACKEND=FLASHINFER` |
| **B200 / SM 10.0** | FlashInfer | 默认（自动选择） |
| **V100 / SM 7.0** | FlexAttention | `export VLLM_ATTENTION_BACKEND=FLEX_ATTENTION` |
| **AMD ROCm** | ROCm Attention | 自动检测 |
| **Google TPU** | Pallas | 自动检测 |
| **CPU** | CPU Attention | 自动检测 |

### 2. MLA 模型推荐配置

如果使用 DeepSeek-V2/V3 等 MLA 架构模型：

```bash
# H100 GPU（SM 9.0）
export VLLM_ATTENTION_BACKEND=FLASHMLA

# B200 GPU（SM 10.0）
export VLLM_ATTENTION_BACKEND=CUTLASS_MLA

# 通用配置（自动选择）
export VLLM_ATTENTION_BACKEND=FLASH_ATTN_MLA
```

### 3. 调试建议

**查看实际使用的后端**：
```python
import logging
logging.basicConfig(level=logging.INFO)

from vllm import LLM
llm = LLM(model="meta-llama/Llama-2-7b-hf")
# 日志输出: INFO cuda.py:313] Using FlashInfer backend on V1 engine.
```

**验证后端功能**：
```python
from vllm.attention.selector import get_attn_backend
from vllm.platforms import _Backend

backend = get_attn_backend(head_size=128, dtype=torch.float16, kv_cache_dtype="auto")
print(f"Selected backend: {backend.get_name()}")
print(f"Supports FP8 KV cache: {hasattr(backend, 'supports_fp8_kv_cache')}")
```

### 4. 性能优化建议

- **启用 FP8 KV cache**（FlashInfer 后端）：减少显存占用
  ```bash
  export VLLM_ATTENTION_BACKEND=FLASHINFER
  # KV cache 量化在启动时自动启用（如果硬件支持）
  ```

- **使用 CUDA Graph**：减少 kernel 启动开销
  ```python
  llm = LLM(
      model="meta-llama/Llama-2-7b-hf",
      enforce_eager=False,  # 启用 CUDA Graph（默认）
  )
  ```

- **调整 block_size**：
  ```python
  llm = LLM(
      model="meta-llama/Llama-2-7b-hf",
      block_size=16,  # 默认值，可调整为 32/64（取决于后端）
  )
  ```

---

## 参考代码位置

| 功能 | 文件路径 |
|------|---------|
| 后端选择逻辑 | [vllm/attention/selector.py](../vllm/attention/selector.py) |
| Attention 层 | [vllm/attention/layer.py](../vllm/attention/layer.py) |
| 抽象接口 | [vllm/attention/backends/abstract.py](../vllm/attention/backends/abstract.py) |
| Flash Attention V1 | [vllm/v1/attention/backends/flash_attn.py](../vllm/v1/attention/backends/flash_attn.py) |
| FlashInfer V1 | [vllm/v1/attention/backends/flashinfer.py](../vllm/v1/attention/backends/flashinfer.py) |
| MLA 实现 | [vllm/v1/attention/backends/mla/](../vllm/v1/attention/backends/mla/) |
| PagedAttention 操作 | [vllm/attention/ops/paged_attn.py](../vllm/attention/ops/paged_attn.py) |

---

**文档版本**：v1.0  
**更新日期**：2026 年 2 月 14 日  
**适用 vLLM 版本**：0.11.0+  
**维护者**：vLLM Community
