# KV Cache 内存计算流程详解

从 `EngineCore` 出发，经过 `Executor`，到 `GPUWorker`，完整追踪 **"Available KV cache memory"** 的计算过程。

---

## 目录

- [完整调用链总览](#完整调用链总览)
- [第一步：EngineCore._initialize_kv_caches()](#第一步enginecore_initialize_kv_caches)
- [第二步：Executor.determine_available_memory()](#第二步executordetermine_available_memory)
- [第三步：GPUWorker.init_device()——内存快照基准](#第三步gpuworkerinit_device内存快照基准)
- [第四步：GPUWorker.determine_available_memory()——核心计算](#第四步gpuworkerdetermine_available_memory核心计算)
- [第五步：memory_profiling()——Profile 运行测峰值](#第五步memory_profiling----profile-运行测峰值)
- [第六步：get_kv_cache_configs()——换算成 Block 数量](#第六步get_kv_cache_configs换算成-block-数量)

---

## 完整调用链总览

(EngineCore_DP0 pid=2031378) INFO 02-21 23:15:07 [gpu_worker.py:302] Available KV cache memory: 4.93 GiB

(EngineCore_DP0 pid=2031378) INFO 02-21 23:15:07 [kv_cache_utils.py:1087] GPU KV cache size: 40,416 tokens

```
EngineCore.__init__()                           [v1/engine/core.py]
  └─ _initialize_kv_caches()
       ├─ model_executor.get_kv_cache_specs()
       ├─ model_executor.determine_available_memory()
       │    └─ collective_rpc("determine_available_memory")
       │         └─ GPUWorker.determine_available_memory()  [v1/worker/gpu_worker.py]
       │              ├─ model_runner.profile_run()          [虚拟前向，测峰值]
       │              ├─ memory_profiling(init_snapshot, weights_memory)
       │              │    └─ MemorySnapshot.measure()       [采集 torch / CUDA 内存]
       │              └─ available_kv = requested_memory - non_kv_cache_memory
       │                                                     → "Available KV cache memory: X GiB"
       └─ get_kv_cache_configs(available_gpu_memory)        [v1/core/kv_cache_utils.py]
            ├─ get_num_blocks(available_memory, page_size)
            │    └─ num_blocks = available_memory // page_size // num_layers
            └─ "GPU KV cache size: X tokens"
```

---

## 第一步：EngineCore._initialize_kv_caches()

**文件**：[`vllm/v1/engine/core.py`](../vllm/v1/engine/core.py)，约第 171 行。

```python
def _initialize_kv_caches(self, vllm_config) -> tuple[int, int, KVCacheConfig]:
    # 1. 查询模型每层的 KV Cache 规格（num_kv_heads、head_size、dtype 等）
    kv_cache_specs = self.model_executor.get_kv_cache_specs()

    # 2. 通过 profile run 测量可用内存（核心步骤）
    available_gpu_memory = self.model_executor.determine_available_memory()

    # 3. 根据可用内存 + KV 规格 → 计算 block 数量
    kv_cache_configs = get_kv_cache_configs(vllm_config, kv_cache_specs,
                                            available_gpu_memory)

    # 4. 初始化 KV Cache tensors + warmup
    self.model_executor.initialize_from_config(kv_cache_configs)
```

`determine_available_memory()` 的返回值是一个 `list[int]`（字节），对应每个 KV cache group 的可用内存量。该值直接决定最终分配多少 KV Cache block。

---

## 第二步：Executor.determine_available_memory()

**文件**：[`vllm/v1/executor/abstract.py`](../vllm/v1/executor/abstract.py)，约第 84 行。

```python
def determine_available_memory(self) -> list[int]:  # in bytes
    return self.collective_rpc("determine_available_memory")
```

`collective_rpc` 会将方法调用分发到所有 Worker 进程（通过 RPC），并汇总返回值。单卡场景（`UniProcExecutor`）直接在当前进程调用；多卡 TP 场景（`ExecutorWithExternalLauncher`）会对所有 rank 做 `all_reduce MIN`，取最小可用内存保证一致性。

---

## 第三步：GPUWorker.init_device()——内存快照基准

**文件**：[`vllm/v1/worker/gpu_worker.py`](../vllm/v1/worker/gpu_worker.py)，约第 160 行。

在 Worker 初始化时（调用 `init_device()`），vLLM 会**在 NCCL 初始化完成之后**立即拍一张内存快照作为基准线：

```python
# NCCL 初始化后，清理并拍快照
gc.collect()
torch.cuda.empty_cache()

self.init_snapshot = MemorySnapshot()           # 基准快照 (B0)
self.requested_memory = (
    self.init_snapshot.total_memory             # GPU 总显存
    * self.cache_config.gpu_memory_utilization  # 例如 0.9
)
```

| 变量 | 含义 |
|------|------|
| `init_snapshot.total_memory` | GPU 物理总显存（如 24 GiB） |
| `init_snapshot.free_memory` | 此时空闲显存（已扣除 NCCL 等开销） |
| `requested_memory` | `total_memory × gpu_memory_utilization`，vLLM 被允许"使用"的上限 |

> **关键点**：快照在 NCCL 初始化之后拍，因此 NCCL 自身的内存占用已被包含在 `cuda_memory` 里，后续不会被误算为可用内存。

---

## 第四步：GPUWorker.determine_available_memory()——核心计算

**文件**：[`vllm/v1/worker/gpu_worker.py`](../vllm/v1/worker/gpu_worker.py)，约第 223 行。

```python
def determine_available_memory(self) -> int:
    # 1. 清理并重置 peak 统计
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 2. profile run：用最大 batch 跑一次虚拟前向，测出峰值 activation 内存
    with memory_profiling(
        self.init_snapshot,                              # B0 基准快照
        weights_memory=int(self.model_runner.model_memory_usage),
    ) as profile_result:
        self.model_runner.profile_run()                  # 虚拟前向

    # 3. 计算可用 KV Cache 内存
    self.available_kv_cache_memory_bytes = (
        self.requested_memory                            # util × total
        - profile_result.non_kv_cache_memory            # 非 KV 占用
    )

    logger.info("Available KV cache memory: %.2f GiB",
                self.available_kv_cache_memory_bytes / GiB_bytes)

    return int(self.available_kv_cache_memory_bytes)
```

**核心公式**：

$$
\text{available\_kv\_cache} = \underbrace{(\text{total} \times \text{util})}_{\text{requested\_memory}} - \underbrace{(\text{weights} + \text{peak\_activation} + \text{non\_torch})}_{\text{non\_kv\_cache\_memory}}
$$

---

## 第五步：memory_profiling()——Profile 运行测峰值

**文件**：[`vllm/utils/__init__.py`](../vllm/utils/__init__.py)，约第 2783 行。

`memory_profiling` 是一个上下文管理器，在 `profile_run()` 前后分别拍快照，测量三类内存增量：

```
GPU 显存可分为三类：
  类别 1：其他进程使用（非 vLLM）
  类别 2：本 vLLM 进程通过 PyTorch 使用（torch.cuda.memory_reserved）
  类别 3：本 vLLM 进程使用，但不经过 PyTorch（如 NCCL buffer、某些 attention 后端）
```

| 快照时刻 | 符号 | 含义 |
|---------|------|------|
| `init_device()` 调用时 | `B0 (before_create)` | NCCL 已初始化，模型尚未加载 |
| `profile_run()` 执行前 | `B1 (before_profile)` | 模型已加载，权重占满显存 |
| `profile_run()` 执行后 | `A  (after_profile)` | 虚拟前向完成，临时 tensor 已释放 |

```python
# non_kv_cache_memory 的三个组成部分：
result.weights_memory      = model_memory_usage          # (a) 模型权重
result.torch_peak_increase = A.torch_peak - B1.torch_peak  # (b) 峰值 activation
result.non_torch_increase  = A.non_torch  - B0.non_torch   # (c) NCCL / 其他非 torch 占用

result.non_kv_cache_memory = (a) + (b) + (c)
```

**具体含义**：

| 分项 | 如何测量 | 典型值（Llama 3.1-8B, BF16） |
|------|---------|---------------------------|
| **(a) 模型权重** | `model_memory_usage`（加载时记录） | 14.99 GiB |
| **(b) 峰值 activation** | `torch.cuda.memory_stats()["allocated_bytes.all.peak"]` 的增量 | 1.191GiB |
| **(c) 非 Torch 内存增量** | `(cuda_memory - torch.cuda.memory_reserved())` 的增量（B0→A） | 36MB |

---

## 第六步：get_kv_cache_configs()——换算成 Block 数量

**文件**：[`vllm/v1/core/kv_cache_utils.py`](../vllm/v1/core/kv_cache_utils.py)，约第 813 行。

拿到 `available_gpu_memory`（字节）之后，按页大小换算成 block 数：

```python
def get_num_blocks(vllm_config, num_layers, available_memory, page_size):
    num_blocks = int(available_memory // page_size // num_layers)
    return num_blocks
```

其中 `page_size` 即每个 block（`block_size` 个 token）所有注意力层共享的字节数：

```python
# FullAttentionSpec.page_size_bytes（每层每个 block 的字节数）
page_size_per_layer = 2 * block_size * num_kv_heads * head_size * dtype_size
#                    ↑K+V  ↑token数      ↑头数          ↑头维度     ↑fp16=2字节
```

最终：

```python
# 总 token 数
total_tokens = num_blocks * block_size

logger.info("GPU KV cache size: %s tokens", total_tokens)
logger.info("Maximum concurrency for %s tokens per request: %.2fx",
            max_model_len, total_tokens / max_model_len)
```

