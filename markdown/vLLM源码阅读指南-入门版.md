# vLLM 源码阅读指南 (入门版)

> **目标读者**：刚开始阅读 vLLM 源码的小白  
> **关注重点**：Transformers 架构、Prefill、Attention 计算、离线/在线接口、V1 引擎  
> **忽略内容**：非 decode-only 模型、量化、分布式、LoRA 等高级特性

---

## 📚 前言：如何阅读 vLLM 源码？

vLLM 是一个高性能的大模型推理引擎，代码量庞大。作为初学者，**最佳的学习路径是"顺着请求的处理流程走"**：

```
用户请求 → 引擎调度 → KV Cache 管理 → 模型执行 → Attention 计算 → 结果返回
```

本指南将按照这个流程，逐层剖析 vLLM 的核心模块。

---

## 🎯 第一部分：入口层 - 你如何与 vLLM 交互？

### 📂 目录：`vllm/entrypoints/`

这是你代码阅读的**起点**，决定了请求如何进入引擎。

#### 1. **离线接口**：`llm.py`

**作用**：提供最简单的离线推理 API，是初学者的最佳入口。

**核心类**：
- `LLM`: 用户直接调用的类，封装了模型加载、推理、生成等功能
  - `__init__()`: 初始化引擎，加载模型
  - `generate()`: 执行批量推理，返回生成结果
  - `chat()`: 对话式接口，内部调用 `generate()`
  - `encode()`: Embedding 模型的编码接口

**使用示例**：
```python
from vllm import LLM, SamplingParams

# 创建 LLM 实例
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# 生成文本
outputs = llm.generate(prompts=["Hello, world!"], sampling_params=SamplingParams())
```

**推荐阅读**：
- `generate()` 方法的完整流程
- 它如何将请求传递给 `LLMEngine`

---

#### 2. **在线服务**：`entrypoints/openai/api_server.py`

**作用**：启动 OpenAI 兼容的 HTTP 服务器，处理并发请求。

**核心组件**：
- `api_server.py`: FastAPI 应用入口，定义路由
- `serving_chat.py`: `/v1/chat/completions` 接口实现
- `serving_completion.py`: `/v1/completions` 接口实现
- `serving_engine.py`: 引擎管理器，封装 `AsyncLLMEngine`（有疑问）

**关键流程**：
```
HTTP 请求 → FastAPI 路由 → serving_chat.py → AsyncLLMEngine → LLMEngine
```

**推荐阅读**：
1. `api_server.py` 中的路由定义
2. `serving_chat.py` 中的 `create_chat_completion()` 方法
3. 如何使用 `AsyncLLMEngine.generate()` 处理流式/非流式请求

---

#### 3. **重要配置文件**：`entrypoints/llm.py` 和 `engine/arg_utils.py`

**作用**：
- `arg_utils.py`: 定义所有引擎参数（`EngineArgs`），是理解 vLLM 配置的关键
- 包含 `max_model_len`、`block_size`、`gpu_memory_utilization` 等核心参数

**推荐阅读**：
- `EngineArgs` 类的定义，了解每个参数的含义

---

## ⚙️ 第二部分：引擎层 - 调度的大脑

### 📂 目录：`vllm/engine/`

这是 vLLM 的**核心枢纽**，负责接收请求、调度执行、返回结果。

#### 1. **核心引擎**：`llm_engine.py`

**作用**：整个系统的"指挥官"，协调所有组件完成推理。

**核心类**：
- `LLMEngine`: 主引擎类
  - `__init__()`: 初始化模型、调度器、执行器
  - `add_request()`: 添加新请求到队列
  - `step()`: **【核心方法】** 执行一次调度+推理循环
  - `_process_model_outputs()`: 处理模型输出，更新请求状态

**LLMEngine.step() 的流程**：
```python
1. 调用 scheduler.schedule() → 决定哪些请求要执行
2. 调用 executor.execute_model() → 在 GPU 上执行模型
3. 处理输出 → 更新 KV Cache、采样结果
4. 返回已完成的请求
```

**推荐阅读**：
- `step()` 方法的完整实现
- 如何与 `Scheduler` 和 `Executor` 交互

---

#### 2. **异步引擎**：`async_llm_engine.py`

**作用**：为在线服务提供异步接口，支持高并发。

**核心类**：
- `AsyncLLMEngine`: 异步版本的 `LLMEngine`
  - `generate()`: 异步生成接口，返回 `AsyncGenerator`
  - `_run_engine_loop()`: 后台线程持续调用 `LLMEngine.step()`

**与 LLMEngine 的关系**：
```
AsyncLLMEngine (异步接口) → LLMEngine (同步核心)
```

**推荐阅读**：
- `generate()` 方法如何使用 `asyncio` 生成流式输出

---

## 🧠 第三部分：调度层 - V1 引擎的核心

### 📂 目录：`vllm/v1/`

这是 **vLLM 的下一代架构（V1 引擎）**，代码更清晰、性能更高。

#### 1. **V1 引擎**：`v1/engine/llm_engine.py`

**作用**：V1 版本的核心引擎，架构更模块化。

**核心改进**：
- 更清晰的调度逻辑
- 更高效的 KV Cache 管理
- 更好的 Prefill/Decode 分离

**推荐阅读**：
- 与传统 `engine/llm_engine.py` 的对比
- 新的调度策略实现

---

#### 2. **调度器**：`v1/core/sched/`

**作用**：**【必看】** 实现 vLLM 最著名的 **Continuous Batching** 逻辑。

**核心文件**：
- `scheduler.py`: 主调度器
  - `schedule()`: 决定哪些请求处于 Prefill/Decode 阶段
  - `_get_num_new_tokens()`: 计算每个请求需要生成多少 token
  - `_allocate_and_set_running()`: 分配 KV Cache 并标记请求为运行状态

**关键概念**：
- **Continuous Batching**: 动态地将处于不同阶段的请求打包成一个 Batch
- **Prefill**: 计算 prompt 的 KV Cache（计算密集型）
- **Decode**: 逐 token 生成（内存密集型）

**推荐阅读**：
- `schedule()` 方法的调度逻辑
- 如何平衡 Prefill 和 Decode 请求

---

#### 3. **KV Cache 管理器**：`v1/core/kv_cache_manager.py`

**作用**：**【核心概念：PagedAttention】** 将物理显存映射为逻辑块。

**核心概念**：
- **Block**: 固定大小的 KV Cache 单元（例如 16 个 token）
- **PagedAttention**: 像操作系统的虚拟内存一样管理 GPU 显存
- **Copy-on-Write**: 多个请求共享相同的 Prompt KV Cache

**核心文件**：
- `kv_cache_manager.py`: 管理 Block 的分配和释放
- `block_pool.py`: Block 池管理

**推荐阅读**：
- `allocate()` 方法：如何分配 Block
- `free()` 方法：如何回收 Block
- Prefix Caching 的实现

---

## 🔧 第四部分：执行层 - 模型如何运行？

### 📂 目录：`vllm/executor/` 和 `vllm/worker/`

这里负责**真正的计算**，将调度器的决策转化为 GPU 操作。

#### 1. **执行器**：`executor/uniproc_executor.py`

**作用**：协调 Worker 执行模型推理。

**核心方法**：
- `execute_model()`: 调用 Worker 执行模型

**简化版流程**：
```python
executor.execute_model(scheduler_output)
  → worker.execute_model()
    → model_runner.execute_model()
      → 调用 CUDA kernels
```

**推荐阅读**：
- `execute_model()` 的完整调用链

---

#### 2. **Worker**：`worker/worker.py` 和 `v1/worker/gpu_worker.py`

**作用**：管理 GPU 资源、加载模型权重、执行推理。

**核心职责**：
- 初始化 GPU 设备
- 加载模型权重
- 管理 KV Cache 显存
- 调用 `ModelRunner` 执行推理

**推荐阅读**：
- `__init__()`: 如何初始化 GPU 和模型
- `execute_model()`: 如何调用 ModelRunner

---

## 🏗️ 第五部分：模型层 - Transformers 架构

### 📂 目录：`vllm/model_executor/`

这里定义了各种模型的实现，是理解 **Transformers 架构在 vLLM 中如何运行**的关键。

#### 1. **模型实现**：`model_executor/models/llama.py`

**作用**：**【重点推荐】** Llama 模型的实现，代表性的 Decoder-only 架构。

**核心类**：
- `LlamaForCausalLM`: 完整的 Llama 模型
  - `forward()`: 前向传播
- `LlamaModel`: Transformer 主体
  - 包含多个 `LlamaDecoderLayer`
- `LlamaDecoderLayer`: 单层 Transformer
  - `LlamaAttention`: 自注意力层（**关键：调用 vLLM 的自定义 Attention**）
  - `LlamaMLP`: 前馈神经网络

**与 HuggingFace 的区别**：
- Attention 层被替换为 vLLM 的 `PagedAttention`
- 增加了 KV Cache 管理逻辑
- 优化了内存布局

**推荐阅读**：
1. `LlamaForCausalLM.forward()`: 模型前向传播的完整流程
2. `LlamaAttention`: 如何调用 vLLM 的 Attention 算子
3. 对比 HuggingFace 的 `LlamaForCausalLM`，理解 vLLM 的优化

---

#### 2. **其他重要模型**：
- `gpt2.py`: GPT-2 架构（适合理解最基础的 Transformer）
- `gemma.py`: Google Gemma 模型
- `qwen2.py`: Qwen 系列模型

**推荐**：如果 Llama 太复杂，先从 `gpt2.py` 开始。

---

#### 3. **模型层定义**：`model_executor/layers/`

**作用**：定义各种神经网络层的实现。

**核心文件**：
- `attention.py`: Attention 层的抽象接口
- `sampler.py`: **【重要】** 采样器，负责从 logits 生成下一个 token
- `linear.py`: 优化过的线性层
- `rotary_embedding.py`: RoPE 位置编码

**推荐阅读**：
- `sampler.py` 中的 `Sampler.forward()`: 如何实现 top-p、top-k、temperature 采样
- `attention.py`: Attention 接口的定义

---

## ⚡ 第六部分：Attention 计算 - 性能的核心

### 📂 目录：`vllm/attention/`

这是 vLLM **性能优势的源泉**，实现了高效的 Attention 计算。

#### 1. **核心概念：PagedAttention**

**什么是 PagedAttention？**
- 将 KV Cache 划分为固定大小的 Block
- 使用类似操作系统虚拟内存的方式管理显存
- 支持多个请求共享相同的 Prompt KV Cache（Prefix Caching）

**优势**：
- 减少显存碎片
- 提高显存利用率
- 支持更大的 Batch Size

---

#### 2. **Attention 后端**：`attention/backends/`

**作用**：实现不同硬件平台的 Attention 算子。

**核心文件**：
- `abstract.py`: Attention 后端的抽象接口
- `utils.py`: 后端选择逻辑

**后端类型**（V1 引擎）：
- **FlashAttention**: 用于 Prefill 阶段（计算密集型）
- **PagedAttention**: 用于 Decode 阶段（内存密集型）

**关键流程**：
```python
if is_prefill:
    使用 FlashAttention (csrc/attention/flash_attn.cu)
else:
    使用 PagedAttention (csrc/attention/paged_attn_v2.cu)
```

**推荐阅读**：
- `abstract.py`: Attention 后端的接口定义
- 如何根据 Prefill/Decode 阶段选择不同的算子

---

#### 3. **Attention 层**：`attention/layer.py`

**作用**：封装 Attention 的高层逻辑。

**核心类**：
- `Attention`: 供模型调用的 Attention 层
  - `forward()`: 根据当前阶段调用对应的后端

**推荐阅读**：
- `forward()` 方法：如何判断 Prefill/Decode 并调用不同算子

---

#### 4. **底层 CUDA 实现**：`csrc/attention/`

**作用**：CUDA/C++ 实现的高性能算子（属于 C++ 扩展）。

**核心文件**：
- `csrc/attention/flash_attn.cu`: FlashAttention 的 CUDA 实现
- `csrc/attention/paged_attn_v2.cu`: PagedAttention 的 CUDA 实现

**推荐**：如果你熟悉 CUDA，这里是深入理解性能优化的关键。

---

## 📦 第七部分：数据流 - 请求如何表示？

### 📂 核心文件

#### 1. **请求定义**：`sequence.py` 和 `v1/request.py`

**作用**：定义请求的数据结构。

**核心类**：
- `Sequence`: 表示一个生成序列
  - 包含 token IDs、KV Cache 的 Block 映射
- `SequenceGroup`: 一组相关的序列（例如 Beam Search）
- V1 版本的 `Request`: 更简洁的请求表示

**推荐阅读**：
- `Sequence` 的状态管理
- 如何追踪生成的 token

---

#### 2. **输出定义**：`outputs.py` 和 `v1/outputs.py`

**作用**：定义推理结果的数据结构。

**核心类**：
- `RequestOutput`: 单个请求的输出
- `CompletionOutput`: 生成的文本结果
- `SampleLogprobs`: Log 概率信息

---

#### 3. **采样参数**：`sampling_params.py`

**作用**：定义生成参数（temperature、top_p、top_k 等）。

**核心类**：
- `SamplingParams`: 采样参数
  - `temperature`: 控制随机性
  - `top_p`: Nucleus Sampling
  - `top_k`: Top-K Sampling

---



## 🎓 总结

vLLM 的核心创新在于：
1. **Continuous Batching**: 动态批处理，提高吞吐量
2. **PagedAttention**: 高效显存管理，减少碎片
3. **Prefill/Decode 分离**: 针对不同阶段的优化策略

**核心代码路径**：
```
用户请求 (entrypoints/)
    ↓
引擎调度 (engine/ & v1/engine/)
    ↓
调度器决策 (v1/core/sched/)
    ↓
KV Cache 分配 (v1/core/kv_cache_manager.py)
    ↓
Worker 执行 (worker/ & v1/worker/)
    ↓
模型推理 (model_executor/models/)
    ↓
Attention 计算 (attention/)
    ↓
采样生成 (model_executor/layers/sampler.py)
    ↓
结果返回 (outputs.py)
```

希望这份指南能帮你快速入门 vLLM 源码！如果在阅读过程中遇到具体问题，欢迎随时提问。

---

**作者**：GitHub Copilot  
**日期**：2026-02-15  
**版本**：v1.0
