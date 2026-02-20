# vLLM Step 函数调用链分析

## 概述

vLLM 的 `step` 函数确实像 nano-vllm 一样重要，是整个推理引擎的**核心循环（Inner Loop）**，负责协调调度、模型执行和输出处理。

## 完整调用链

### 1. LLMEngine.step() 
**文件**: [vllm/v1/engine/llm_engine.py](vllm/v1/engine/llm_engine.py#L220-L243)

```python
def step(self) -> Union[list[RequestOutput], list[PoolingRequestOutput]]:
    # 1) 从 EngineCore 获取输出
    outputs = self.engine_core.get_output()
    
    # 2) 处理输出
    processed_outputs = self.output_processor.process_outputs(
        outputs.outputs,
        engine_core_timestamp=outputs.timestamp,
        iteration_stats=iteration_stats)
    
    # 3) 中止已完成的请求
    self.engine_core.abort_requests(processed_outputs.reqs_to_abort)
    
    # 4) 记录统计信息
    if self.logger_manager is not None:
        self.logger_manager.record(...)
    
    return processed_outputs.request_outputs
```

**关键操作**: `self.engine_core.get_output()` - 触发核心调度和推理

---

### 2. EngineCoreClient.get_output()
**文件**: [vllm/v1/engine/core_client.py](vllm/v1/engine/core_client.py)

根据不同的运行模式，有多种实现：

| 模式 | 适用场景 | 进程模型 | 通信方式 |
|------|---------|---------|---------|
| **InprocClient** | V0 兼容、调试 | 单进程 | 直接调用 |
| **SyncMPClient** ⭐ | LLM (默认) | 多进程 | ZMQ + Queue |
| **AsyncMPClient** | AsyncLLM | 多进程 | ZMQ + asyncio |

#### 2.1 InprocClient (进程内模式)
**用于**: V0 兼容模式，单进程运行

```python
def get_output(self) -> EngineCoreOutputs:
    outputs, _ = self.engine_core.step_fn()  # 直接调用 EngineCore.step()
    return outputs and outputs.get(0) or EngineCoreOutputs()
```

#### 2.2 SyncMPClient (多进程模式) ⭐ 详解
**用于**: 多进程环境，通过 ZMQ 通信
**文件**: [vllm/v1/engine/core_client.py](vllm/v1/engine/core_client.py#L597-L730)

这是 vLLM **默认使用的模式**，EngineCore 在独立后台进程中运行 busy loop。

```python
def get_output(self) -> EngineCoreOutputs:
    # 从 Python Queue 中获取输出（阻塞）
    outputs = self.outputs_queue.get()
    if isinstance(outputs, Exception):
        raise self._format_exception(outputs) from None
    if outputs.wave_complete is not None:
        self.engines_running = False
    return outputs
```

**完整通信链路** (跨进程 + 跨线程):

```
┌─────────────────────────────────────────────────────────────────────┐
│ 前端进程 (LLMEngine)                                                  │
│                                                                       │
│  LLMEngine.step()                                                    │
│    └→ SyncMPClient.get_output()                                     │
│         └→ outputs_queue.get()  ← ⬅️ 从 Python Queue 读取（阻塞）     │
│                ↑                                                     │
│                │ (2. 写入 Queue)                                      │
│                │                                                     │
│         process_outputs_socket() 线程                                │
│         └→ out_socket.recv_multipart()  ← ⬅️ 从 ZMQ socket 接收      │
│                ↑                                                     │
└────────────────┼─────────────────────────────────────────────────────┘
                 │ ZMQ (进程间通信)
                 │
┌────────────────┼─────────────────────────────────────────────────────┐
│ 后台进程 (EngineCoreProc)          ↓ (1. 发送到 ZMQ)                 │
│                                                                       │
│         process_output_sockets() 线程                                │
│         └→ sockets[i].send_multipart(buffers)  ➡️ 发送到 ZMQ         │
│                ↑                                                     │
│                │ (4. 从 Queue 读取)                                   │
│                │                                                     │
│         self.output_queue.put_nowait(output)  ← (3. 写入)            │
│                ↑                                                     │
│         _process_engine_step()                                       │
│         └→ outputs, _ = self.step_fn()  ⬅️ 调用核心 step            │
│                ↑                                                     │
│         run_busy_loop()  ♻️ 主循环                                   │
│         └→ while True:                                               │
│              _process_input_queue()   # 处理输入请求                  │
│              _process_engine_step()   # 执行 step                    │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

**关键数据结构**:
- `outputs_queue`: Python `queue.Queue`，线程安全队列
- `output_queue`: Python `queue.Queue`，在 EngineCoreProc 中
- `out_socket`: ZMQ PULL socket (前端进程)
- `sockets`: ZMQ PUSH sockets (后台进程)

**工作流程详解**:

1. **后台进程启动** (在 `__init__` 时)
   ```python
   # 启动独立 Python 进程运行 EngineCore
   EngineCore.run_engine_core(...)
   ```

2. **后台进程 Busy Loop** (在 `run_busy_loop()` 中)
   ```python
   while True:
       # 从输入 socket 获取新请求
       _process_input_queue()
       
       # 执行 step，生成输出
       _process_engine_step():
           outputs, model_executed = self.step_fn()  # ← 核心！
           # 将输出写入 output_queue
           for output in outputs.items():
               self.output_queue.put_nowait(output)
   ```

3. **输出线程发送** (`process_output_sockets()` 线程)
   ```python
   while True:
       output = self.output_queue.get()  # 从队列读取
       # 通过 ZMQ 发送给前端
       sockets[client_index].send_multipart(buffers, copy=False)
   ```

4. **前端线程接收** (`process_outputs_socket()` 线程)
   ```python
   while True:
       frames = out_socket.recv_multipart(copy=False)  # ZMQ 接收
       outputs = decoder.decode(frames)
       outputs_queue.put_nowait(outputs)  # 写入 Python Queue
   ```

5. **前端阻塞读取**
   ```python
   # LLMEngine.step() → SyncMPClient.get_output()
   outputs = self.outputs_queue.get()  # 阻塞直到有输出
   ```

**为什么需要这么复杂的架构？**
- ✅ **进程隔离**: GPU 操作在独立进程，崩溃不影响主进程
- ✅ **并发**: busy loop 持续运行，无需每次 step 启动
- ✅ **零拷贝**: ZMQ `copy=False` 高效传输大张量
- ✅ **异步**: 输入输出在独立线程，不阻塞核心循环

#### 2.3 AsyncMPClient (异步多进程模式)
**用于**: AsyncLLM，异步 API

```python
async def get_output_async(self) -> EngineCoreOutputs:
    # 异步从 ZMQ 队列获取输出
```

---

### 3. EngineCore.step() ⭐ 核心！
**文件**: [vllm/v1/engine/core.py](vllm/v1/engine/core.py#L274-L297)

这是整个推理引擎的**核心函数**，类似于 nano-vllm 的 `step()`。

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    """调度、执行、生成输出"""
    
    # 检查是否有待处理请求
    if not self.scheduler.has_requests():
        return {}, False
    
    # ========== 核心三步骤 ==========
    
    # 1️⃣ 调度：决定哪些请求执行、分配多少 token
    scheduler_output = self.scheduler.schedule()
    
    # 2️⃣ 推理：执行模型前向计算
    model_output = self.execute_model_with_error_logging(
        self.model_executor.execute_model,
        scheduler_output)
    
    # 3️⃣ 更新：处理模型输出，更新调度器状态
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output)
    
    return (engine_core_outputs, 
            scheduler_output.total_num_scheduled_tokens > 0)
```

**在多进程模式下的调用上下文**:

在 `EngineCoreProc` 后台进程中，`step()` 在 busy loop 中被持续调用：

```python
# 文件: vllm/v1/engine/core.py, EngineCoreProc 类
def run_busy_loop(self):
    """EngineCore 的核心 busy loop"""
    while True:
        # 1) 处理输入队列（添加/中止请求等）
        self._process_input_queue()
        
        # 2) 执行 step 并返回输出
        self._process_engine_step()

def _process_engine_step(self) -> bool:
    """执行一次 step"""
    # 调用 step_fn (即 self.step 或 self.step_with_batch_queue)
    outputs, model_executed = self.step_fn()
    
    # 将输出放入 output_queue（由输出线程消费）
    for output in (outputs.items() if outputs else ()):
        self.output_queue.put_nowait(output)
    
    # 后处理（如更新推测解码的 draft tokens）
    self.post_step(model_executed)
    
    return model_executed
```

**关键点**:
- `self.step_fn` 在 `__init__` 时设置：
  - 无 Pipeline 并行 → `self.step`
  - 有 Pipeline 并行 → `self.step_with_batch_queue`
- Busy loop **持续运行**，无需外部触发
- 输出通过 `output_queue` 异步传递给前端

#### 3.1 变体：step_with_batch_queue() (Pipeline 并行)
**文件**: [vllm/v1/engine/core.py](vllm/v1/engine/core.py#L302-L360)

用于 Pipeline 并行场景，支持异步批次队列，消除流水线气泡。

---

### 4. Scheduler.schedule() 
**文件**: [vllm/v1/core/sched/scheduler.py](vllm/v1/core/sched/scheduler.py#L179)

```python
def schedule(self) -> SchedulerOutput:
    """核心调度算法"""
    
    # 调度策略：让每个请求的 num_computed_tokens 追赶 num_tokens_with_spec
    # 支持：分块预填充、前缀缓存、推测解码
    
    # 1. 调度 RUNNING 请求（优先级最高）
    while req_index < len(self.running) and token_budget > 0:
        # 分配 token 预算、encoder 输入、推测解码 token
        ...
    
    # 2. 调度 WAITING 请求（新请求或恢复的请求）
    # 3. 可能抢占低优先级请求
    # 4. 分配/释放 KV Cache blocks
    
    return SchedulerOutput(
        scheduled_new_reqs=...,
        scheduled_resumed_reqs=...,
        scheduled_running_reqs=...,
        preempted_reqs=...,
        ...
    )
```

**关键功能**:
- Token 预算分配 (类似 nano-vllm 的 seqs 调度)
- KV Cache Block 管理
- 请求优先级和抢占

---

### 5. Executor.execute_model()
**文件**: [vllm/v1/executor/abstract.py](vllm/v1/executor/abstract.py#L98)

```python
def execute_model(
    self,
    scheduler_output: SchedulerOutput,
    non_block: bool = False
) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
    """执行模型前向推理"""
    
    # 1. 准备模型输入 (从 scheduler_output)
    # 2. 调用 GPU worker 执行推理
    # 3. 返回 token_ids 和其他输出 (logprobs, hidden_states 等)
```

**实现类**:
- `MultiprocessExecutor` - 多进程 GPU
- `RayDistributedExecutor` - Ray 分布式
- 等

---

### 6. Scheduler.update_from_output()
**文件**: [vllm/v1/core/sched/scheduler.py](vllm/v1/core/sched/scheduler.py)

```python
def update_from_output(
    self,
    scheduler_output: SchedulerOutput,
    model_output: ModelRunnerOutput
) -> dict[int, EngineCoreOutputs]:
    """根据模型输出更新调度器状态"""
    
    # 1. 将生成的 token 添加到请求
    # 2. 检测 finish 条件 (EOS, max_tokens, stop_strings)
    # 3. 释放完成请求的 KV Cache blocks (类似 nano-vllm 的 postprocess)
    # 4. 生成 EngineCoreOutputs 返回给上层
    
    return engine_core_outputs
```

---

## 与 nano-vllm 的对应关系

| nano-vllm | vLLM v1 | 说明 |
|-----------|---------|------|
| `scheduler.schedule()` | `Scheduler.schedule()` | 调度请求，分配资源 |
| `model_runner.call("run", seqs, is_prefill)` | `Executor.execute_model(scheduler_output)` | 模型推理 |
| `scheduler.postprocess(seqs, token_ids)` | `Scheduler.update_from_output(...)` | 处理输出，释放 KV Cache |
| `seqs` (List[Sequence]) | `SchedulerOutput` | 调度结果 |
| `token_ids` (List[int]) | `ModelRunnerOutput` | 模型输出 |

---

## 关键差异

1. **架构更复杂**: vLLM 支持多进程、分布式、异步等多种模式
2. **抽象更完善**: 清晰的分层架构 (LLMEngine → EngineCoreClient → EngineCore)
3. **功能更丰富**: 支持 Pipeline 并行、LoRA、结构化输出等高级特性
4. **调度更灵活**: 无严格的 "prefill/decode" 阶段划分，统一的 token 分配算法

---

## 总结

**vLLM 的 `step` 函数确实是核心中的核心**，调用链清晰：

```
LLMEngine.step()
  → EngineCoreClient.get_output()
    → EngineCore.step()  ⭐ 核心循环
      → Scheduler.schedule()        # 调度
      → Executor.execute_model()    # 推理
      → Scheduler.update_from_output()  # 更新
```

这与 nano-vllm 的设计理念一致，但在工程实现上更为复杂和完善。

---

## 附录：SyncMPClient 完整数据流图

以 **SyncMPClient (默认多进程模式)** 为例，从用户调用到返回结果的完整流程：

```
┌──────────────────────────────────────────────────────────────────────┐
│ 用户代码                                                               │
│   for output in llm.generate(...):                                   │
│       └→ LLMEngine.step()                                            │
└───────────────────────┬──────────────────────────────────────────────┘
                        │
                        ↓
┌──────────────────────────────────────────────────────────────────────┐
│ 前端进程 (LLMEngine 所在进程)                                          │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────┐         │
│  │ LLMEngine.step()                                         │         │
│  │   outputs = self.engine_core.get_output()  ①            │         │
│  │   processed = self.output_processor.process_outputs(...) │         │
│  │   return processed.request_outputs                       │         │
│  └───────────────┬─────────────────────────────────────────┘         │
│                  │                                                     │
│                  ↓                                                     │
│  ┌─────────────────────────────────────────────────────────┐         │
│  │ SyncMPClient.get_output()                                │         │
│  │   outputs = self.outputs_queue.get()  ② 阻塞读取         │         │
│  └───────────────┬─────────────────────────────────────────┘         │
│                  ↑                                                     │
│                  │ (从 Python Queue 读取)                              │
│                  │                                                     │
│  ┌───────────────┴─────────────────────────────────────────┐         │
│  │ process_outputs_socket() 线程 (持续运行)                 │         │
│  │   while True:                                            │         │
│  │     frames = out_socket.recv_multipart()  ③ ZMQ 接收    │         │
│  │     outputs = decoder.decode(frames)                     │         │
│  │     outputs_queue.put_nowait(outputs)  写入 Queue        │         │
│  └───────────────┬─────────────────────────────────────────┘         │
│                  ↑                                                     │
└──────────────────┼─────────────────────────────────────────────────────┘
                   │ ZMQ (PUSH/PULL 进程间通信)
                   │
┌──────────────────┼─────────────────────────────────────────────────────┐
│ 后台进程 (EngineCore 独立进程)         ↓                                │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────┐          │
│  │ process_output_sockets() 线程 (持续运行)                 │          │
│  │   while True:                                            │          │
│  │     output = self.output_queue.get()  ④ 从队列读取      │          │
│  │     buffers = encoder.encode_into(output)                │          │
│  │     socket.send_multipart(buffers)  发送到 ZMQ           │          │
│  └───────────────┬─────────────────────────────────────────┘          │
│                  ↑                                                      │
│                  │ (写入 output_queue)                                  │
│                  │                                                      │
│  ┌───────────────┴─────────────────────────────────────────┐          │
│  │ run_busy_loop() 主循环 (持续运行)                        │          │
│  │   while True:                                            │          │
│  │     _process_input_queue()       # 处理新请求            │          │
│  │     _process_engine_step()  ⑤                           │          │
│  │       outputs, _ = self.step_fn()  ⬅️ 核心！           │          │
│  │       for output in outputs.items():                     │          │
│  │         self.output_queue.put_nowait(output)             │          │
│  └───────────────┬─────────────────────────────────────────┘          │
│                  │                                                      │
│                  ↓                                                      │
│  ┌─────────────────────────────────────────────────────────┐          │
│  │ EngineCore.step()  ⭐⭐⭐ 核心！                         │          │
│  │   scheduler_output = self.scheduler.schedule()  ⑥       │          │
│  │   model_output = self.model_executor.execute_model(...)⑦│          │
│  │   outputs = self.scheduler.update_from_output(...)  ⑧   │          │
│  │   return outputs                                         │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

关键步骤：
  ① LLMEngine 调用 get_output 获取结果
  ② 从线程安全队列阻塞读取（等待数据）
  ③ 接收线程从 ZMQ 接收数据并解码
  ④ 发送线程从队列读取并通过 ZMQ 发送
  ⑤ Busy loop 执行 step_fn
  ⑥ 调度：决定执行哪些请求
  ⑦ 推理：GPU 执行模型前向传播
  ⑧ 更新：处理输出、释放 KV Cache、生成结果
```

**性能优化点**:
- ⚡ **Busy Loop**: 后台进程持续运行，无启动开销
- ⚡ **零拷贝**: ZMQ `copy=False` + msgpack 直接序列化张量
- ⚡ **双缓冲**: 队列解耦生产者/消费者，提高吞吐
- ⚡ **异步线程**: 输入输出 IO 不阻塞核心计算

---

## 快速索引：关键文件和函数

| 功能 | 文件 | 关键函数/类 |
|------|------|-----------|
| **入口** | `vllm/v1/engine/llm_engine.py` | `LLMEngine.step()` |
| **客户端** | `vllm/v1/engine/core_client.py` | `SyncMPClient.get_output()` |
| **核心循环** | `vllm/v1/engine/core.py` | `EngineCore.step()` <br> `EngineCoreProc.run_busy_loop()` |
| **调度器** | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule()` <br> `Scheduler.update_from_output()` |
| **执行器** | `vllm/v1/executor/multiproc_executor.py` | `MultiprocessExecutor.execute_model()` |
| **Worker** | `vllm/v1/worker/worker_base.py` | `WorkerBase.execute_model()` |
| **输出处理** | `vllm/v1/engine/output_processor.py` | `OutputProcessor.process_outputs()` |

**调试断点建议**:
- `EngineCore.step()` - 查看调度和推理的核心逻辑
- `Scheduler.schedule()` - 理解如何分配 token 预算
- `_process_engine_step()` - 观察输出如何进入队列
- `SyncMPClient.get_output()` - 追踪前端如何获取结果
