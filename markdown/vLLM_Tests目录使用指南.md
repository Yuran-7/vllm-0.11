# vLLM Tests 目录使用指南

本文档介绍 vLLM 测试套件的组织结构、运行方式以及各个测试模块的功能说明。

---

## 目录
- [如何运行测试](#如何运行测试)
- [测试文件的两种形式](#测试文件的两种形式)
- [Tests 目录结构](#tests-目录结构)
- [测试分类与标记](#测试分类与标记)
- [常用测试命令](#常用测试命令)

---

## 如何运行测试

### 方式一：使用 pytest（推荐）

vLLM 使用 pytest 作为测试框架，所有测试文件都可以通过 pytest 运行。

#### 运行所有测试
```bash
pytest tests/
```

#### 运行特定测试文件
```bash
pytest tests/cuda/test_cuda_context.py
```

#### 运行特定测试类或函数
```bash
# 运行测试类
pytest tests/cuda/test_cuda_context.py::TestSetCudaContext

# 运行特定测试函数
pytest tests/cuda/test_cuda_context.py::TestSetCudaContext::test_set_cuda_context_parametrized

# 运行匹配模式的测试
pytest tests/ -k "cuda"
```

#### 显示详细输出
```bash
# 显示所有输出
pytest tests/cuda/test_cuda_context.py -v -s

# 显示测试覆盖率
pytest tests/ --cov=vllm --cov-report=html
```

---

### 方式二：直接运行（仅限包含 `__main__` 的测试文件）

有些测试文件包含 `if __name__ == "__main__":` 块，可以直接作为 Python 脚本运行。

```bash
python tests/cuda/test_cuda_context.py
```

**示例代码**：
```python
# tests/cuda/test_cuda_context.py

class TestSetCudaContext:
    def test_set_cuda_context_parametrized(self, device_input, expected_device_id):
        # 测试代码
        pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # 直接运行会调用 pytest
```

---

## 测试文件的两种形式

### 形式一：纯 pytest 测试文件（无 `__main__`）

**特点**：
- 只能通过 `pytest` 命令运行
- 大部分测试文件属于此类
- 遵循 pytest 标准结构

**示例**：
```python
# tests/test_config.py
import pytest
from vllm.config import VllmConfig

def test_compile_config_repr_succeeds():
    config = VllmConfig()
    val = repr(config)
    assert 'VllmConfig' in val

class TestModelConfig:
    def test_model_config_creation(self):
        # 测试代码
        pass
```

**运行方式**：
```bash
pytest tests/test_config.py
```

---

### 形式二：包含 `__main__` 的测试文件

**特点**：
- 可以直接作为 Python 脚本运行
- 也可以通过 `pytest` 运行
- 通常用于调试或独立运行的测试

**示例**：
```python
# tests/cuda/test_cuda_context.py
import pytest

class TestSetCudaContext:
    @pytest.mark.skipif(not current_platform.is_cuda(),
                        reason="CUDA not available")
    def test_set_cuda_context_parametrized(self, device_input, expected_device_id):
        # 测试代码
        pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**运行方式**（两种方式等效）：
```bash
# 方式 1: 使用 pytest（推荐）
pytest tests/cuda/test_cuda_context.py -v

# 方式 2: 直接运行（仅限包含 __main__ 的文件）
python tests/cuda/test_cuda_context.py
```

**包含 `__main__` 的测试文件（部分列表）**：
```
tests/cuda/test_cuda_context.py
tests/v1/e2e/test_min_tokens.py
tests/v1/kv_connector/nixl_integration/test_disagg_accuracy.py
tests/model_executor/test_weight_utils.py
tests/quantization/test_torchao.py
tests/kernels/test_flex_attention.py
tests/kernels/moe/test_flashinfer_moe.py
tests/kv_transfer/test_send_recv.py
tests/distributed/test_shm_buffer.py
tests/evals/gsm8k/gsm8k_eval.py
tests/compile/piecewise/test_toy_llama.py
```

**对比**：

| 特性 | 无 `__main__` | 有 `__main__` |
|------|--------------|--------------|
| **数量** | 大多数测试文件 | 少数测试文件 |
| **运行方式** | 只能用 pytest | pytest 或直接运行 |
| **用途** | 标准单元测试 | 调试/独立测试/性能测试 |
| **示例** | `test_config.py` | `test_cuda_context.py` |

---

## Tests 目录结构

### 📁 测试目录树

```
tests/
├── conftest.py                    # pytest 全局配置和 fixtures
├── utils.py                       # 测试工具函数
├── ci_envs.py                     # CI 环境配置
│
├── basic_correctness/             # 基础正确性测试
│   └── test_basic_correctness.py  # vLLM vs HuggingFace 输出对比
│
├── models/                        # 模型测试
│   ├── language/                  # 语言模型
│   ├── multimodal/                # 多模态模型（视觉、音频）
│   ├── quantization/              # 量化模型
│   ├── test_registry.py           # 模型注册表测试
│   └── test_initialization.py     # 模型初始化测试
│
├── kernels/                       # CUDA/Triton Kernel 测试
│   ├── attention/                 # 注意力内核
│   ├── moe/                       # MoE 内核
│   ├── mamba/                     # Mamba/SSM 内核
│   ├── quantization/              # 量化内核
│   ├── test_flex_attention.py     # FlexAttention 测试
│   └── test_triton_flash_attention.py  # Triton Flash Attention
│
├── v1/                            # V1 引擎测试
│   ├── engine/                    # 引擎核心
│   ├── core/                      # 调度器、KV Cache
│   ├── e2e/                       # 端到端测试
│   ├── entrypoints/               # API 入口测试
│   ├── kv_connector/              # KV 传输测试
│   └── executor/                  # 执行器测试
│
├── distributed/                   # 分布式测试
│   ├── test_shm_buffer.py         # 共享内存缓冲区
│   ├── test_same_node.py          # 同节点通信
│   └── test_eplb_algo.py          # 负载均衡算法
│
├── quantization/                  # 量化测试
│   ├── test_torchao.py            # TorchAO 量化
│   ├── test_fp8.py                # FP8 量化
│   └── test_compressed_tensors.py # 压缩张量
│
├── cuda/                          # CUDA 功能测试
│   └── test_cuda_context.py       # CUDA 上下文管理
│
├── entrypoints/                   # API 入口点测试
│   ├── openai/                    # OpenAI 兼容 API
│   └── test_cli.py                # 命令行接口
│
├── compile/                       # torch.compile 测试
│   └── piecewise/                 # Piecewise CUDA Graph
│
├── lora/                          # LoRA 适配器测试
├── tokenization/                  # Tokenization 测试
├── multimodal/                    # 多模态输入测试
├── samplers/                      # 采样器测试
├── speculative_decoding/          # 推测解码测试
├── kv_transfer/                   # KV Cache 传输测试
├── engine/                        # 引擎测试
├── config/                        # 配置测试
├── detokenizer/                   # 反 Tokenization 测试
├── evals/                         # 评估测试
│   └── gsm8k/                     # GSM8K 数学评估
├── benchmarks/                    # 基准测试
├── standalone_tests/              # 独立测试
├── plugins_tests/                 # 插件测试
├── runai_model_streamer_test/     # RunAI 模型流测试
├── reasoning/                     # 推理测试
├── tool_use/                      # 工具调用测试
└── tpu/                           # TPU 测试
```

---

## 详细模块说明

### 1. **basic_correctness/** - 基础正确性测试

**目的**：验证 vLLM 输出与 HuggingFace Transformers 的一致性

**关键测试文件**：
- `test_basic_correctness.py`: 对比 vLLM 和 HF 的生成输出

**示例测试**：
```python
def test_vllm_gc_ed():
    """验证 vLLM 实例被正确回收"""
    llm = LLM("distilbert/distilgpt2")
    weak_llm = weakref.ref(llm)
    del llm
    assert weak_llm() is None
```

**运行**：
```bash
pytest tests/basic_correctness/ -v
```

---

### 2. **models/** - 模型测试

**目的**：测试各种模型架构的加载、初始化和推理

**子目录**：
- **language/**: 纯语言模型（Llama, GPT, Qwen 等）
- **multimodal/**: 多模态模型（LLaVA, Qwen2-VL, InternVL 等）
- **quantization/**: 量化模型（AWQ, GPTQ, FP8 等）

**关键测试**：
- `test_registry.py`: 模型注册表测试
- `test_initialization.py`: 模型初始化流程测试
- `test_transformers.py`: Transformers 集成测试

**示例**：
```bash
# 测试 Llama 模型
pytest tests/models/language/ -k "llama"

# 测试多模态模型
pytest tests/models/multimodal/ -v
```

---

### 3. **kernels/** - CUDA/Triton Kernel 测试

**目的**：测试底层计算内核的正确性和性能

**子目录**：
- **attention/**: 注意力机制内核（Flash Attention, FlashInfer 等）
- **moe/**: MoE（Mixture of Experts）内核
- **mamba/**: Mamba/State Space Models 内核
- **quantization/**: 量化内核（FP8, INT8 等）

**关键测试文件**：
- `test_flex_attention.py`: FlexAttention 后端测试
- `test_triton_flash_attention.py`: Triton 实现的 Flash Attention
- `moe/test_flashinfer_moe.py`: FlashInfer MoE 内核

**示例**：
```bash
# 测试 Flash Attention
pytest tests/kernels/test_flex_attention.py -v

# 测试 MoE 内核
pytest tests/kernels/moe/ -v
```

**注意**: 这些测试需要特定的 GPU 硬件（SM 8.0+）

---

### 4. **v1/** - V1 引擎测试

**目的**：测试 vLLM V1 引擎的各个组件

**子目录**：
- **engine/**: 引擎核心（AsyncLLM, EngineCore）
- **core/**: 调度器、KV Cache、内存管理
- **e2e/**: 端到端测试
- **entrypoints/**: API 入口点测试
- **kv_connector/**: KV Cache 传输测试（disagg 架构）
- **executor/**: 模型执行器测试

**关键测试**：
- `test_async_llm_dp.py`: Data Parallel 测试
- `test_kv_sharing.py`: KV Cache 共享测试
- `e2e/test_min_tokens.py`: 最小 token 生成测试

**示例**：
```bash
# 测试 V1 引擎
pytest tests/v1/engine/ -v

# 测试端到端场景
pytest tests/v1/e2e/ -v
```

---

### 5. **distributed/** - 分布式测试

**目的**：测试多 GPU 通信、张量并行、流水线并行

**关键测试文件**：
- `test_shm_buffer.py`: 共享内存缓冲区测试
- `test_shm_storage.py`: 共享内存存储测试
- `test_same_node.py`: 同节点多 GPU 通信
- `test_node_count.py`: 节点计数检测
- `test_eplb_algo.py`: Elastic Parallel Load Balancing 算法

**示例**：
```bash
# 需要多 GPU 环境
pytest tests/distributed/ -v

# 测试共享内存
python tests/distributed/test_shm_buffer.py
```

---

### 6. **quantization/** - 量化测试

**目的**：测试各种量化方案的正确性

**支持的量化方法**：
- AWQ (Activation-aware Weight Quantization)
- GPTQ (Generative Pre-trained Transformer Quantization)
- FP8 (8-bit Floating Point)
- Compressed Tensors
- TorchAO

**关键测试文件**：
- `test_torchao.py`: TorchAO 量化框架
- `test_fp8.py`: FP8 量化
- `test_compressed_tensors.py`: 压缩张量格式

**示例**：
```bash
# 测试 FP8 量化
pytest tests/quantization/test_fp8.py -v

# 测试所有量化方法
pytest tests/quantization/ -v
```

---

### 7. **cuda/** - CUDA 功能测试

**目的**：测试 CUDA 相关的底层功能

**关键测试**：
- `test_cuda_context.py`: CUDA 上下文管理和多线程隔离

**示例**：
```python
class TestSetCudaContext:
    @pytest.mark.skipif(not current_platform.is_cuda(),
                        reason="CUDA not available")
    def test_set_cuda_context_parametrized(self, device_input, expected_device_id):
        # 测试在隔离的线程中设置 CUDA 上下文
        pass
```

**运行**：
```bash
pytest tests/cuda/ -v
# 或
python tests/cuda/test_cuda_context.py
```

---

### 8. **entrypoints/** - API 入口点测试

**目的**：测试各种 API 入口点的功能

**子目录**：
- **openai/**: OpenAI 兼容 API 测试
- **offline_mode/**: 离线模式测试
- **chat/**: 聊天接口测试

**示例**：
```bash
# 测试 OpenAI API
pytest tests/entrypoints/openai/ -v
```

---

### 9. **compile/** - torch.compile 测试

**目的**：测试 PyTorch 2.x torch.compile 功能

**子目录**：
- **piecewise/**: Piecewise CUDA Graph 测试

**关键测试**：
- `test_toy_llama.py`: 简化 Llama 模型的编译测试

**示例**：
```bash
pytest tests/compile/ -v
```

---

### 10. **其他重要目录**

| 目录 | 功能 |
|------|------|
| **lora/** | LoRA 适配器加载、切换、多 LoRA 推理 |
| **tokenization/** | Tokenizer 正确性、特殊 token 处理 |
| **multimodal/** | 多模态输入处理（图片、音频、视频） |
| **samplers/** | 采样算法测试（top-p, top-k, beam search） |
| **speculative_decoding/** | 推测解码（Medusa, EAGLE 等） |
| **kv_transfer/** | KV Cache 传输和共享 |
| **engine/** | 引擎核心组件（V0 引擎） |
| **config/** | 配置系统测试 |
| **detokenizer/** | 反 Tokenization 测试 |
| **evals/** | 模型评估（GSM8K, MMLU 等） |
| **benchmarks/** | 性能基准测试 |
| **standalone_tests/** | 独立运行的测试 |
| **plugins_tests/** | 插件系统测试 |
| **tpu/** | TPU 平台测试 |

---

## 测试分类与标记

vLLM 使用 pytest markers 来分类和过滤测试。

### pytest 标记定义

在 `pyproject.toml` 中定义：

```toml
[tool.pytest.ini_options]
markers = [
    "slow_test",                    # 慢速测试（需要长时间运行）
    "skip_global_cleanup",          # 跳过全局清理
    "core_model",                   # 核心模型（每个 PR 都测试）
    "hybrid_model",                 # 混合模型（包含 Mamba 层）
    "cpu_model",                    # CPU 测试
    "split",                        # 分割运行的测试
    "distributed",                  # 分布式 GPU 测试
    "skip_v1",                      # 不在 V1 引擎上运行
    "optional",                     # 可选测试（默认跳过）
]
```

### 使用标记过滤测试

```bash
# 只运行快速测试（排除 slow_test）
pytest tests/ -m "not slow_test"

# 只运行核心模型测试
pytest tests/models/ -m "core_model"

# 只运行分布式测试
pytest tests/ -m "distributed"

# 运行可选测试
pytest tests/ --optional

# 排除 V1 引擎测试
pytest tests/ -m "not skip_v1"
```

---

## 常用测试命令

### 基础命令

```bash
# 运行所有测试
pytest tests/

# 运行特定目录
pytest tests/models/

# 运行单个文件
pytest tests/cuda/test_cuda_context.py

# 显示详细输出
pytest tests/cuda/test_cuda_context.py -v -s

# 只显示失败的测试
pytest tests/ -v --tb=short

# 停在第一个失败的测试
pytest tests/ -x

# 重新运行失败的测试
pytest tests/ --lf  # last-failed
```

---

### 过滤和选择测试

```bash
# 按名称模式匹配
pytest tests/ -k "cuda"
pytest tests/ -k "test_set_cuda_context"

# 按标记过滤
pytest tests/ -m "not slow_test"
pytest tests/ -m "core_model or cpu_model"

# 运行特定测试函数
pytest tests/cuda/test_cuda_context.py::TestSetCudaContext::test_set_cuda_context_parametrized
```

---

### 并行运行测试

```bash
# 安装 pytest-xdist
pip install pytest-xdist

# 使用多核并行运行
pytest tests/ -n auto  # 自动检测 CPU 核心数
pytest tests/ -n 4     # 使用 4 个进程
```

---

### 测试覆盖率

```bash
# 安装 pytest-cov
pip install pytest-cov

# 生成覆盖率报告
pytest tests/ --cov=vllm --cov-report=html

# 在终端显示覆盖率
pytest tests/ --cov=vllm --cov-report=term-missing
```

---

### 调试测试

```bash
# 显示完整的堆栈跟踪
pytest tests/cuda/test_cuda_context.py -vv --tb=long

# 进入失败测试的 pdb 调试器
pytest tests/cuda/test_cuda_context.py --pdb

# 显示所有 print 输出
pytest tests/cuda/test_cuda_context.py -s

# 显示 fixture 使用情况
pytest tests/ --fixtures
```

---

### GPU 测试

```bash
# 指定 GPU
CUDA_VISIBLE_DEVICES=0 pytest tests/cuda/

# 多 GPU 测试
CUDA_VISIBLE_DEVICES=0,1 pytest tests/distributed/ -m "distributed"

# 跳过需要多 GPU 的测试
pytest tests/ -m "not distributed"
```

---

### CI 环境测试

```bash
# 设置目标测试套件
TARGET_TEST_SUITE=L4 pytest tests/

# 跳过慢速测试
pytest tests/ -m "not slow_test"

# 只运行核心模型测试
pytest tests/models/ -m "core_model"
```

---

## 常见问题

### 1. 测试失败：CUDA out of memory

**解决方案**：
```bash
# 减少并行度
pytest tests/ -n 1

# 运行单个测试
pytest tests/cuda/test_cuda_context.py

# 清理 GPU 缓存
python -c "import torch; torch.cuda.empty_cache()"
```

---

### 2. 测试跳过：缺少依赖

**解决方案**：
```bash
# 安装开发依赖
pip install -e ".[dev]"

# 安装测试依赖
pip install pytest pytest-asyncio pytest-xdist pytest-cov
```

---

### 3. 运行慢速测试

```bash
# 只运行慢速测试
pytest tests/ -m "slow_test" -v

# 设置超时时间
pytest tests/ --timeout=300  # 5 分钟超时
```

---

### 4. 查看测试收集信息

```bash
# 显示所有测试但不运行
pytest tests/ --collect-only

# 显示测试统计
pytest tests/ --fixtures
```

---

## conftest.py 文件

`tests/conftest.py` 包含全局 pytest 配置和 fixtures。

**关键 fixtures**：
- `vllm_runner`: 创建 vLLM 实例
- `hf_runner`: 创建 HuggingFace 模型实例
- `example_prompts`: 测试提示词
- `image_assets`: 图片测试资源
- `audio_assets`: 音频测试资源

**使用示例**：
```python
def test_basic_generation(vllm_runner):
    """使用 vllm_runner fixture"""
    with vllm_runner("meta-llama/Llama-2-7b-hf") as llm:
        outputs = llm.generate(["Hello"], sampling_params)
        assert len(outputs) == 1
```

---

## 最佳实践

### 1. 编写新测试

```python
import pytest
from vllm import LLM

class TestMyFeature:
    """测试类应该用 Test 前缀"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(),
                        reason="CUDA not available")
    def test_basic_functionality(self):
        """测试函数用 test_ 前缀"""
        llm = LLM("facebook/opt-125m")
        outputs = llm.generate(["Hello"])
        assert len(outputs) == 1
    
    @pytest.mark.parametrize("model", [
        "facebook/opt-125m",
        "facebook/opt-350m",
    ])
    def test_multiple_models(self, model):
        """使用参数化测试多个场景"""
        llm = LLM(model)
        assert llm.llm_engine is not None
```

---

### 2. 使用 fixtures

```python
@pytest.fixture
def sample_llm():
    """创建可重用的 fixture"""
    llm = LLM("facebook/opt-125m")
    yield llm
    del llm  # 清理资源

def test_with_fixture(sample_llm):
    outputs = sample_llm.generate(["Hello"])
    assert len(outputs) == 1
```

---

### 3. 添加测试标记

```python
@pytest.mark.slow_test
def test_large_model():
    """标记慢速测试"""
    pass

@pytest.mark.distributed
def test_multi_gpu():
    """标记分布式测试"""
    pass
```

---

## 参考资料

| 资源 | 说明 |
|------|------|
| **pytest 文档** | https://docs.pytest.org/ |
| **vLLM 官方文档** | https://docs.vllm.ai/ |
| **CI 配置** | `.github/workflows/` 目录 |
| **测试配置** | `pyproject.toml` `[tool.pytest.ini_options]` |

---

**文档版本**: v1.0  
**更新日期**: 2026 年 2 月 14 日  
**适用 vLLM 版本**: 0.11.0+  
**维护者**: vLLM Community
