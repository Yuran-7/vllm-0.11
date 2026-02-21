# vLLM 离线推理（LLM）核心类层级梳理

> 适用于 vLLM V1，离线（offline）场景，即使用 `LLM(...)` 发起推理。

---

## 整体层级概览

```
用户层
└── LLM                          (vllm/entrypoints/llm.py)

引擎调度层（主进程）
└── LLMEngine                    (vllm/v1/engine/llm_engine.py)
    └── EngineCoreClient [抽象]  (vllm/v1/engine/core_client.py)
        ├── InprocClient         ← 非 MP 模式（单进程调试用）
        |—— AsyncMPClient        ← 在线服务（vllm serve）走的是 AsyncLLM
        |   └── DP>1 + 外部负载均衡         ← DPAsyncMPClient
        |   └── DP>1 + 内部负载均衡         ← DPLBAsyncMPClient      
        └── SyncMPClient         ← 离线 LLM 默认使用
            └── MPClient [基类]

引擎核心层
├── EngineCore                   (vllm/v1/engine/core.py)  ← 非 MP 模式：InprocClient 在主进程内直接持有
└── EngineCoreProc(EngineCore)                             ← MP 模式：以独立子进程运行（DP=1 时）
    └── DPEngineCoreProc(EngineCoreProc)                   ← MP 模式：以独立子进程运行（DP>1 时）

执行器层（子进程1 内）
└── ExecutorBase                 (vllm/executor/executor_base.py)
    └── Executor [V1抽象]        (vllm/v1/executor/abstract.py)
        ├── UniProcExecutor      ← TP=1 默认（Worker 直接运行在子进程1内，无额外子进程）
        └── MultiprocExecutor    ← TP>1 默认（为每个 GPU 各启动一个 WorkerProc 子进程）

Worker 层（两种路径都经过 WorkerWrapperBase）
├── [TP=1]  UniProcExecutor.driver_worker ──► WorkerWrapperBase    (vllm/worker/worker_base.py)
└── [TP>1]  WorkerProc.worker ──────────────► WorkerWrapperBase    (vllm/worker/worker_base.py)
└── WorkerWrapperBase → Worker (vllm/v1/worker/gpu_worker.py) → GPUModelRunner (vllm/v1/worker/gpu_model_runner.py)
```

---

## 各类详细说明

### 1. `LLM`
- **文件**：`vllm/entrypoints/llm.py`
- **职责**：用户直接使用的入口 API（`LLM("meta-llama/...")`）
- **关键行为**：
  - 内部创建 `LLMEngine`，并设置 `multiprocess_mode=True`（默认启用多进程）
  - 提供 `generate()`、`encode()` 等同步接口
- **不涉及 GPU**，纯粹的 Python 接口层

---

### 2. `LLMEngine`
- **文件**：`vllm/v1/engine/llm_engine.py`
- **继承**：无（独立类，"Legacy" 名称仅为向后兼容）
- **职责**：离线引擎的主进程核心，负责：
  - 请求的 tokenize / multimodal 特征提取（`InputPreprocessor`）
  - 通过 `EngineCoreClient` 与后台 EngineCore 进程通信
  - 汇总输出并返回给调用方
- **关键属性**：
  - `self.engine_core`：一个 `EngineCoreClient` 实例（通常是 `SyncMPClient`）
  - `self.processor`：处理原始输入，转为 `EngineCoreRequest`

---

### 3. `EngineCoreClient`（抽象基类）
- **文件**：`vllm/v1/engine/core_client.py`
- **职责**：定义与 EngineCore（内层循环）通信的统一接口，包含 `add_request()`、`get_output()`、`abort_requests()` 等
- **派生类**（离线 LLM 相关）：

| 子类 | 使用场景 | 通信方式 |
|---|---|---|
| `InprocClient` | 非 MP 模式（单进程） | 直接调用 `EngineCore.step()` |
| `SyncMPClient` | 离线 LLM（MP 模式，**默认**） | ZMQ socket + 后台线程 |

---

### 4. `MPClient`（`SyncMPClient` 的父类）
- **文件**：`vllm/v1/engine/core_client.py`
- **职责**：多进程客户端的公共基类，负责：
  - 调用 `launch_core_engines()` 启动 **EngineCore 子进程**
  - 创建 ZMQ input/output socket，用于与 EngineCore 进程双向通信
  - 启动 `MPClientEngineMonitor` 守护线程监控子进程存活

---

### 5. `SyncMPClient`
- **文件**：`vllm/v1/engine/core_client.py`
- **继承**：`MPClient -> EngineCoreClient`
- **职责**：同步（阻塞式）多进程客户端，供 `LLM`（非 async）使用：
  - 启动 `EngineCoreOutputQueueThread` 守护线程，持续从 ZMQ socket 读取输出放入队列
  - `get_output()` 从队列中阻塞获取 `EngineCoreOutputs`
  - `add_request()` 序列化请求并通过 ZMQ 发送到 EngineCore 进程

---

### 6. `EngineCore`
- **文件**：`vllm/v1/engine/core.py`
- **职责**：vLLM 的"内层循环"（Inner Loop），调度与执行的核心：
  - 持有 `Scheduler`（调度器）和 `Executor`（执行器）
  - `step()` 方法：调用 `scheduler.schedule()` → `executor.execute_model()` → `scheduler.update_from_output()`
  - 管理 KV Cache 初始化
- **不直接使用 GPU**，通过 `Executor` 向 Worker 下发任务
- **使用场景**：非 MP 模式（`VLLM_ENABLE_V1_MULTIPROCESSING=0`）下，`InprocClient` 在**主进程内**直接实例化并持有 `EngineCore`，通过 `engine_core.step_fn()` 同步调用，无子进程、无 ZMQ

---

### 7. `EngineCoreProc`
- **文件**：`vllm/v1/engine/core.py`
- **继承**：`EngineCoreProc(EngineCore)`
- **职责**：为 EngineCore 套上 ZMQ 通信外壳，使其能在**独立子进程**中运行（**MP 模式、DP=1 时使用**）：
  - 启动 `process_input_sockets` 线程（从 ZMQ 读请求 → 放入 `input_queue`）
  - 启动 `process_output_sockets` 线程（从 `output_queue` 读结果 → 写到 ZMQ）
  - `run_busy_loop()`：主循环，不断消费 `input_queue` 并调用 `_process_engine_step()`
- **静态方法** `run_engine_core()`：子进程入口函数，由 `CoreEngineProcManager` 通过 `multiprocessing.Process` 调用；内部根据 `data_parallel_size > 1` 决定实例化 `EngineCoreProc` 还是 `DPEngineCoreProc`

---

### 7b. `DPEngineCoreProc`
- **文件**：`vllm/v1/engine/core.py`
- **继承**：`DPEngineCoreProc(EngineCoreProc)`
- **使用场景**：**MP 模式、DP>1（Data Parallel 多引擎）时使用**，每个 DP rank 对应一个独立子进程
- **在 `EngineCoreProc` 基础上新增**：
  - `_init_data_parallel()`：初始化跨 DP 引擎的 stateless process group
  - `run_busy_loop()` 扩展：每步后执行 `_has_global_unfinished_reqs()`（all-reduce 判断全局请求），协调所有 DP 引擎同步进入/退出 idle
  - `_maybe_publish_request_counts()`：向 Coordinator 发布本引擎的请求计数，供客户端负载均衡（`DPLBAsyncMPClient`）
  - `reinitialize_distributed()`：支持 Elastic EP 动态扩缩容

---

### 8. `ExecutorBase`
- **文件**：`vllm/executor/executor_base.py`
- **职责**：V0/V1 共用的执行器抽象接口，定义 `collective_rpc()`、`initialize_cache()` 等

---

### 9. `Executor`（V1 抽象类）
- **文件**：`vllm/v1/executor/abstract.py`
- **继承**：`Executor(ExecutorBase)`
- **职责**：V1 专用扩展，新增：
  - `execute_model(scheduler_output)` → 调用 `collective_rpc("execute_model", ...)`
  - `get_kv_cache_specs()` 获取各 Worker 的 KV cache 规格
  - `initialize_from_config()` 初始化 KV cache 并 warm up 模型
- **派生类**：

| 子类 | 使用场景 | 对应 `distributed_executor_backend` |
|---|---|---|
| `UniProcExecutor` | TP=1 时默认（Worker 直接在 EngineCoreProc 内运行） | `"uni"` |
| `MultiprocExecutor` | TP>1 时默认（为每个 GPU 启动 WorkerProc 子进程） | `"mp"` |
| `RayDistributedExecutor` | 多节点分布式场景 | `"ray"` |
| `ExecutorWithExternalLauncher` | 外部启动器场景（如 torchrun） | `"external_launcher"` |

---

### 10. `MultiprocExecutor`
- **文件**：`vllm/v1/executor/multiproc_executor.py`
- **继承**：`MultiprocExecutor(Executor)`
- **职责**：管理多个 Worker 子进程：
  - 为每个 rank（0 到 `world_size-1`）创建一个 `WorkerProc` 子进程
  - 通过**共享内存 `MessageQueue`** 广播 `SchedulerOutput` 给所有 Worker（零拷贝）
  - `collective_rpc(method, ...)` 向所有 Worker 下发指令（如 `execute_model`、`initialize_cache`）
  - 启动 `WorkerMonitor` 守护线程监控 Worker 存活

---

---

### 10b. `UniProcExecutor`
- **文件**：`vllm/v1/executor/abstract.py`
- **继承**：`UniProcExecutor(UniProcExecutorV0, Executor)`
- **职责**：TP=1 时的轻量执行器，**不启动任何 WorkerProc 子进程**：
  - UniProcExecutorV0的`_init_executor()` 中：`self.driver_worker = WorkerWrapperBase(...)` — 以 `driver_worker` 属性持有
  - `collective_rpc()` 直接调用 `run_method(self.driver_worker, method, ...)` 而非跨进程通信
  - 节省子进程创建开销，适合单 GPU 场景
- **与 `MultiprocExecutor` 的核心区别**：无共享内存 `MessageQueue`，无 `WorkerMonitor` 线程，无跨进程 IPC

---

### 11. `WorkerProc`
- **文件**：`vllm/v1/executor/multiproc_executor.py`
- **职责**：在独立子进程中运行的 Worker 包装器（每个 GPU 对应一个）：
  - `__init__()` 中：`wrapper = WorkerWrapperBase(...); self.worker = wrapper` — 以 `worker` 属性持有
  - 从共享内存 `MessageQueue` 接收 `SchedulerOutput`
  - 调用 `worker.execute_model(scheduler_output)` 触发 GPU 推理
  - 将 `ModelRunnerOutput` 写入另一个 `MessageQueue` 返回给 `MultiprocExecutor`
- **不是** `Worker` 的父类，而是"包含" Worker 的进程容器

---

### 12. `WorkerBase`
- **文件**：`vllm/worker/worker_base.py`
- **职责**：Worker 的通用抽象基类，定义 `init_device()`、`load_model()`、`execute_model()` 等接口

---

### 13. `Worker`
- **文件**：`vllm/v1/worker/gpu_worker.py`
- **继承**：`Worker(WorkerBase)`
- **职责**：单个 GPU 上的工作单元，负责：
  - `init_device()`：初始化 CUDA 设备、建立分布式通信（NCCL/torch.distributed）、实例化 `GPUModelRunner`
  - `load_model()`：调用 `model_runner.load_model()`，将模型权重加载到 GPU 显存
  - `determine_available_memory()`：profile run 测量可用显存
  - `initialize_from_config()`：初始化 KV cache tensor
- **关键属性**：`self.model_runner: GPUModelRunner`

---

### 14. `GPUModelRunner`
- **文件**：`vllm/v1/worker/gpu_model_runner.py`
- **继承**：`GPUModelRunner(LoRAModelRunnerMixin, KVConnectorModelRunnerMixin)`
- **职责**：真正执行 GPU 推理的核心：
  - **持有模型权重**：`self.model: nn.Module`（PyTorch 模型，权重存在 GPU 显存）
  - **持有 KV Cache**：`self.kv_caches: list[torch.Tensor]`
  - `_prepare_inputs(scheduler_output)`：将调度结果转为 GPU 输入张量（`input_ids`、`positions`、`attn_metadata` 等）
  - `execute_model(scheduler_output)`：完整推理流程（prepare → forward → sample）
  - 管理 CUDA Graph 的 capture 和 replay
  - 维护 `self.requests`：每个请求的 `CachedRequestState`（token ids、block ids 等）

---

## 调用链路总图

### 非 MP 模式（`VLLM_ENABLE_V1_MULTIPROCESSING=0`，共 1 个进程）

```
[主进程（唯一进程）]
LLM.generate()
  └── LLMEngine.step()
        ├── processor.preprocess()
        ├── engine_core.add_request()   # InprocClient → 直接调用 EngineCore.add_request()
        └── engine_core.get_output()    # InprocClient → 直接调用 EngineCore.step_fn()
              └── EngineCore.step()
                    ├── scheduler.schedule()
                    ├── executor.execute_model()  # UniProcExecutor（同进程）
                    │     └── model_runner.execute_model()  → GPU 推理
                    └── scheduler.update_from_output()
```

### TP=1（默认，共 2 个进程）

```
[主进程]
LLM.generate()
  └── LLMEngine.step()
        ├── processor.preprocess()         # tokenize，主进程
        ├── engine_core.add_request()      # SyncMPClient → ZMQ → EngineCore进程
        └── engine_core.get_output()       # 阻塞等待，从队列取结果

[子进程 1: EngineCoreProc]                 # 唯一的子进程，也负责 GPU 推理
run_busy_loop()
  └── _process_engine_step()
        └── EngineCore.step()
              ├── scheduler.schedule()     # 生成 SchedulerOutput
              ├── executor.execute_model() # UniProcExecutor → 直接调用（同进程）
              │     └── worker.execute_model(scheduler_output)
              │           └── model_runner.execute_model(scheduler_output)
              │                 ├── _prepare_inputs()    # 准备 GPU 输入张量
              │                 └── model.forward(...)   # nn.Module 前向推理（GPU）
              └── scheduler.update_from_output()
```

### TP=4（共 6 个进程）

```
[主进程]
LLM.generate()
  └── LLMEngine.step()
        ├── processor.preprocess()         # tokenize，主进程
        ├── engine_core.add_request()      # SyncMPClient → ZMQ → EngineCore进程
        └── engine_core.get_output()       # 阻塞等待，从队列取结果

[子进程 1: EngineCoreProc]
run_busy_loop()
  └── _process_engine_step()
        └── EngineCore.step()
              ├── scheduler.schedule()     # 生成 SchedulerOutput
              ├── executor.execute_model() # MultiprocExecutor → 共享内存 → Workers
              └── scheduler.update_from_output()

[子进程 2-5: WorkerProc × 4（每个持有1张GPU）]
WorkerProc 收到 SchedulerOutput
  └── worker.execute_model(scheduler_output)
        └── model_runner.execute_model(scheduler_output)
              ├── _update_states()         # 更新请求状态
              ├── _prepare_inputs()        # 准备 GPU 输入张量
              └── model.forward(...)       # nn.Module 前向推理（GPU）
                    └── sampler.forward()  # 采样，生成 token id
```

### DP=2 × TP=1（共 5 个进程，在线服务场景）

```
[前端进程（API server）]
AsyncLLM → DPLBAsyncMPClient → ZMQ（负载均衡路由请求到两个引擎）

[DPEngineCoreProc 子进程 × 2（每个 DP rank 一个）]
run_busy_loop()                            # 扩展版：含 all-reduce 同步
  ├── _process_engine_step()               # 同 EngineCoreProc
  ├── _maybe_publish_request_counts()      # 向 Coordinator 上报请求数
  └── _has_global_unfinished_reqs()        # all-reduce 判断全局是否有未完成请求
        # 两个 DP 引擎互相同步，全部空闲才退出 busy loop

[各自的 Worker（UniProcExecutor，TP=1 时在同一子进程内）]
  └── model_runner.execute_model() → GPU 推理
```

---

## 进程/组件关系速查

### TP=1（`distributed_executor_backend="uni"`，共 2 个进程）

| 组件 | 所在进程 | 是否使用 GPU |
|---|---|---|
| `LLM` | 主进程 | ✗ |
| `LLMEngine` | 主进程 | ✗ |
| `SyncMPClient` | 主进程 | ✗ |
| `EngineCoreProc` | 子进程 1 | ✗（容器） |
| `EngineCore` | 子进程 1 | ✗ |
| `UniProcExecutor` | 子进程 1（内） | ✗ |
| `Worker` | **子进程 1（内）** | ✓ |
| `GPUModelRunner` | **子进程 1（内）** | ✓（模型权重+KV Cache 在此） |

### TP>1（`distributed_executor_backend="mp"`，共 TP+2 个进程）

| 组件 | 所在进程 | 是否使用 GPU |
|---|---|---|
| `LLM` | 主进程 | ✗ |
| `LLMEngine` | 主进程 | ✗ |
| `SyncMPClient` | 主进程 | ✗ |
| `EngineCoreProc` | 子进程 1 | ✗ |
| `EngineCore` | 子进程 1 | ✗ |
| `MultiprocExecutor` | 子进程 1（内） | ✗ |
| `WorkerProc` | 子进程 2~(TP+1) | ✓（通过 Worker） |
| `Worker` | 子进程 2~(TP+1) | ✓ |
| `GPUModelRunner` | 子进程 2~(TP+1) | ✓（模型权重+KV Cache 在此） |

---

## `distributed_executor_backend` 配置说明

### 如何指定

```python
# 方式一：通过 LLM 构造参数（最常用）
llm = LLM(
    model="meta-llama/Llama-3-8B",
    tensor_parallel_size=2,
    distributed_executor_backend="mp",  # 显式指定
)

# 方式二：通过 EngineArgs（服务端场景）
from vllm.engine.arg_utils import EngineArgs
args = EngineArgs(
    model="meta-llama/Llama-3-8B",
    tensor_parallel_size=2,
    distributed_executor_backend="ray",
)

# 方式三：传入自定义 Executor 类（高级用法）
from vllm.executor.executor_base import ExecutorBase
class MyExecutor(ExecutorBase): ...
llm = LLM(model="...", distributed_executor_backend=MyExecutor)
```

### 可选值

| 值 | 对应 Executor 类 | 说明 |
|---|---|---|
| `"uni"` | `UniProcExecutor` | 不启动 WorkerProc 子进程，Worker 直接运行在 EngineCoreProc 内。**TP=1 默认自动选择** |
| `"mp"` | `MultiprocExecutor` | 为每个 GPU rank 启动一个 WorkerProc 子进程。**TP>1 默认自动选择** |
| `"ray"` | `RayDistributedExecutor` | 使用 Ray 管理分布式 Worker，适合多节点场景 |
| `"external_launcher"` | `ExecutorWithExternalLauncher` | 由外部工具（如 `torchrun`）管理进程启动，vLLM 不自行 spawn 子进程 |
| 自定义类（`Type[ExecutorBase]`） | 用户自定义 | 高级扩展，需继承 `ExecutorBase` |
| 自定义类的全限定名（字符串） | 用户自定义 | 如 `"mypackage.MyExecutor"`，通过 `resolve_obj_by_qualname` 动态加载 |

### 自动选择逻辑（`ParallelConfig.__post_init__`，位于 `vllm/config/parallel.py`）

```
用户未设置 distributed_executor_backend（即为 None）时：
  ├── world_size == 1  →  "uni"
  └── world_size > 1
        ├── TPU + SPMD 模式           →  "uni"
        ├── CUDA 且 GPU 数 < world_size →  "ray"（需要多节点）
        ├── data_parallel_backend=ray  →  "ray"
        ├── Ray 已初始化且有 placement_group → "ray"
        └── 其他                       →  "mp"
```
