# vLLM 架构调用链

## 调用层次结构

```
LLMEngine (包装层, vllm/v1/engine/llm_engine.py)
  └── EngineCoreClient (客户端)
        └── EngineCore (核心引擎, vllm/v1/engine/core.py)
              ├── Scheduler (调度器)
              └── Executor (模型执行器)
                    └── Worker (工作进程)
                          └── GPUModelRunner (GPU模型运行器)
```

```
MultiprocExecutor (Executor子类)
  └── WorkerProc (在独立的并行进程中运行)
        └── WorkerWrapperBase (懒加载包装器)
              └── Worker (GPUWorker, 在 vllm/v1/worker/gpu_worker.py)
                    └── GPUModelRunner (在 gpu_worker.py 的 init_device 中创建)
```
## 核心组件详解

### 1. Scheduler (调度器)

**职责**: 管理请求队列和KV缓存分配

**核心功能**:
- 维护 `waiting`、`running`、`finished` 三个请求队列
- 根据KV缓存可用性决定哪些请求可以执行
- 生成 `SchedulerOutput`：
  - `scheduled_new_reqs`: 新调度的请求
  - `scheduled_cached_reqs`: 已缓存的请求
  - `num_scheduled_tokens`: 每个请求调度的token数
  - `scheduled_encoder_inputs`: 编码器输入（多模态）
  - `scheduled_spec_decode_tokens`: 推测解码token

**输出**: `SchedulerOutput` 对象传递给 ModelExecutor

### 2. ModelExecutor (模型执行器)

**职责**: 管理分布式模型执行

**关键方法**:
```python
def execute_model(
    self,
    scheduler_output: SchedulerOutput,
) -> ModelRunnerOutput:
    # 调用 Worker 执行模型推理
    return self.worker.execute_model(scheduler_output)
```

**支持的并行策略**:
- Tensor Parallelism (TP)
- Pipeline Parallelism (PP)
- Data Parallelism (DP)
- Expert Parallelism (EP)

### 3. GPUModelRunner (GPU模型运行器)

**核心流程**:

#### 3.1 模型加载 (`load_model()`)

位置: `vllm/v1/worker/gpu_model_runner.py:2597`

```python
def load_model(self, eep_scale_up: bool = False) -> None:
    # 1. 获取模型加载器
    model_loader = get_model_loader(self.load_config)
    
    # 2. 加载模型（例如 LLaMA、Qwen等）
    self.model = model_loader.load_model(
        vllm_config=self.vllm_config,
        model_config=self.model_config
    )
    
    # 3. 可选：加载LoRA适配器
    if self.lora_config:
        self.model = self.load_lora_model(...)
    
    # 4. 可选：包装为CUDA Graph（性能优化）
    if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
        self.model = CUDAGraphWrapper(self.model, ...)
```

**模型加载详细步骤** (`BaseModelLoader.load_model`):

```python
def load_model(self, vllm_config, model_config) -> nn.Module:
    # Step 1: 初始化模型架构（如 LlamaForCausalLM）
    model = initialize_model(
        vllm_config=vllm_config,
        model_config=model_config
    )
    
    # Step 2: 加载权重文件
    self.load_weights(model, model_config)
    
    # Step 3: 后处理（量化、设备迁移等）
    process_weights_after_loading(model, model_config, target_device)
    
    return model.eval()
```

**支持的加载格式**:
- `auto`: 自动检测格式
- `safetensors`: 推荐格式（更安全）
- `pt`: PyTorch checkpoint
- `gguf`: GGUF格式
- `tensorizer`: 优化的序列化格式

#### 3.2 模型执行 (`execute_model()`)

位置: `vllm/v1/worker/gpu_model_runner.py:2230`

```python
def execute_model(
    self,
    scheduler_output: SchedulerOutput,
    intermediate_tensors: Optional[IntermediateTensors] = None,
) -> ModelRunnerOutput:
    # 1. 更新状态
    self._update_states(scheduler_output)
    
    # 2. 准备输入
    (attn_metadata, logits_indices, spec_decode_metadata,
     num_scheduled_tokens, ...) = self._prepare_inputs(scheduler_output)
    
    # 3. 前处理（多模态、嵌入等）
    input_ids, inputs_embeds, positions = self._preprocess(...)
    
    # 4. 模型前向传播
    with set_forward_context(...):
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=self.kv_caches,
            attn_metadata=attn_metadata,
            ...
        )
    
    # 5. 采样
    logits = self.model.compute_logits(hidden_states)
    sampler_output = self.sampler(logits, sampling_metadata)
    
    # 6. 返回结果
    return ModelRunnerOutput(
        req_ids=req_ids,
        sampled_token_ids=sampled_token_ids,
        logprobs=logprobs,
        ...
    )
```

## 完整执行流程示例

```
用户请求 "你好，世界"
    ↓
1. LLMEngine.generate()
    ↓
2. EngineCoreClient.add_request()
    ↓
3. EngineCore.step()
    ├── Scheduler.schedule()
    │     ├── 分配KV缓存块
    │     ├── 决定处理哪些请求
    │     └── 生成 SchedulerOutput
    │
    └── ModelExecutor.execute_model(scheduler_output)
          └── Worker.execute_model(scheduler_output)
                └── GPUModelRunner.execute_model(scheduler_output)
                      ├── _update_states()        # 更新请求状态
                      ├── _prepare_inputs()        # 准备输入张量
                      ├── model.forward()          # 模型推理
                      ├── sampler()                # 采样下一个token
                      └── 返回 ModelRunnerOutput
    ↓
4. 返回生成的token给用户
```

## 关键数据结构

### SchedulerOutput
```python
@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestsData
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    finished_req_ids: list[str]
```

### ModelRunnerOutput
```python
@dataclass
class ModelRunnerOutput:
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]]
    logprobs: Optional[list[dict]]
```

## 与 nano-vllm 的对比

| 组件 | nano-vllm | vLLM |
|------|-----------|------|
| 模型加载 | `__init__` 中直接加载 | 独立 `load_model()` 方法 |
| 模型位置 | `ModelRunner.model` | `GPUModelRunner.model` |
| 调度器 | 简化版 | 完整的 `Scheduler` 类 |
| 分布式 | 手动 `dist.init_process_group` | 抽象的 `ModelExecutor` |
| 架构 | 扁平化 | 分层设计（Engine → Core → Executor → Runner） |

## 总结

**vLLM的设计优势**:
1. **模块化**: 每层职责清晰（调度、执行、推理分离）
2. **可扩展**: 支持多种并行策略和硬件后端
3. **性能优化**: CUDA Graph、PagedAttention、Continuous Batching
4. **分布式友好**: 抽象的Executor层管理多进程/多机通信

**关键执行路径**: `Scheduler` 决定"做什么" → `ModelExecutor` 分配"谁来做" → `GPUModelRunner` 执行"怎么做"
