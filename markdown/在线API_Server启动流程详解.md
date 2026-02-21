# vLLM API Server 启动流程详解

本文档详细分析 `python -m vllm.entrypoints.openai.api_server` 命令的完整执行流程。

---

## 目录
- [命令示例](#命令示例)
- [日志输出分析](#日志输出分析)
- [完整执行流程](#完整执行流程)
- [核心组件初始化](#核心组件初始化)
- [时间线分析](#时间线分析)
- [关键代码路径](#关键代码路径)

---

## 命令示例

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model /data/ysh/models/Llama-3.1-8B-Instruct/ \
    --served-model-name meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 6578 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 10000
```

---

## 日志输出分析

根据日志输出，启动过程可以分为以下几个阶段：

| 阶段 | 时间戳 | 关键日志 | 耗时 |
|------|--------|---------|------|
| **1. 服务器初始化** | 16:10:22 | `vLLM API server version 0.1.dev9966` | - |
| **2. 参数解析** | 16:10:22 | `non-default args: {...}` | < 1s |
| **3. 模型配置加载** | 16:10:22 | `Resolved architecture: LlamaForCausalLM` | < 1s |
| **4. 平台检测** | 16:10:25 | `Automatically detected platform cuda` | 3s |
| **5. 引擎核心初始化** | 16:10:27 | `Initializing a V1 LLM engine` | 2s |
| **6. 模型加载** | 16:10:27-33 | `Loading safetensors checkpoint shards` | 6s |
| **7. torch.compile** | 16:10:38-11:04 | `Compiling a graph for dynamic shape` | 26s |
| **8. KV cache 分配** | 16:11:05 | `GPU KV cache size: 40,528 tokens` | 1s |
| **9. CUDA Graph 捕获** | 16:11:05-11 | `Capturing CUDA graphs` | 6s |
| **10. 服务启动** | 16:11:12 | `Starting vLLM API server 0 on http://0.0.0.0:6578` | 1s |

**总耗时**：约 50 秒（从启动到服务就绪）

---

## 完整执行流程

### 流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    命令行入口                                    │
│  python -m vllm.entrypoints.openai.api_server [args]           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. __main__ 入口 (api_server.py:1942)                         │
│     - cli_env_setup()           # CLI 环境设置                  │
│     - make_arg_parser(parser)   # 创建参数解析器                │
│     - parser.parse_args()       # 解析命令行参数                │
│     - validate_parsed_serve_args(args)  # 验证参数              │
│     - uvloop.run(run_server(args))      # 启动异步服务器        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. run_server() (api_server.py:1877)                          │
│     - decorate_logs("APIServer")  # 添加日志前缀                │
│     - setup_server(args)          # 设置监听地址和 socket      │
│     - run_server_worker(...)      # 启动工作进程                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. run_server_worker() (api_server.py:1887)                   │
│     ┌───────────────────────────────────────────────────────┐  │
│     │ async with build_async_engine_client(args):          │  │
│     │   # 引擎客户端生命周期管理                             │  │
│     └───────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. build_async_engine_client() (api_server.py:152)            │
│     - AsyncEngineArgs.from_cli_args(args)  # 转换为引擎参数    │
│     - build_async_engine_client_from_engine_args(...)          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. build_async_engine_client_from_engine_args()               │
│     (api_server.py:190)                                         │
│                                                                 │
│     ┌─────────────────────────────────────────────────┐       │
│     │ # 创建 VllmConfig                                │       │
│     │ vllm_config = engine_args.create_engine_config() │       │
│     └─────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│     ┌─────────────────────────────────────────────────┐       │
│     │ # 创建 V1 AsyncLLM 引擎，它和LLMEngine是平级关系  │       │
│     │ async_llm = AsyncLLM.from_vllm_config(          │       │
│     │     vllm_config=vllm_config,                    │       │
│     │     usage_context=...,                          │       │
│     │     enable_log_requests=...,                    │       │
│     │     disable_log_stats=...,                      │       │
│     │     client_count=...,                           │       │
│     │     client_index=...                            │       │
│     │ )                                               │       │
│     └─────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│     ┌─────────────────────────────────────────────────┐       │
│     │ # 清理多模态缓存                                 │       │
│     │ await async_llm.reset_mm_cache()                │       │
│     └─────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│     ┌─────────────────────────────────────────────────┐       │
│     │ yield async_llm  # 返回引擎客户端               │       │
│     └─────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. AsyncLLM.__init__() (v1/engine/async_llm.py:54)            │
│                                                                 │
│     ┌─────────────────────────────────────────────────┐       │
│     │ # 初始化 Tokenizer                              │       │
│     │ self.tokenizer = init_tokenizer_from_configs(   │       │
│     │     model_config=vllm_config.model_config       │       │
│     │ )                                               │       │
│     └─────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│     ┌─────────────────────────────────────────────────┐       │
│     │ # 创建输入处理器                                 │       │
│     │ self.processor = Processor(                     │       │
│     │     vllm_config=vllm_config,                    │       │
│     │     tokenizer=self.tokenizer,                   │       │
│     │     mm_registry=mm_registry                     │       │
│     │ )                                               │       │
│     └─────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│     ┌─────────────────────────────────────────────────┐       │
│     │ # 创建输出处理器                                 │       │
│     │ self.output_processor = OutputProcessor(        │       │
│     │     self.tokenizer,                             │       │
│     │     log_stats=self.log_stats                    │       │
│     │ )                                               │       │
│     └─────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│     ┌─────────────────────────────────────────────────┐       │
│     │ # 创建引擎核心客户端（多进程）                    │       │
│     │ self.engine_core = EngineCoreClient.           │       │
│     │     make_async_mp_client(                       │       │
│     │         vllm_config=vllm_config,                │       │
│     │         executor_class=executor_class,          │       │
│     │         log_stats=self.log_stats,               │       │
│     │         client_count=client_count,              │       │
│     │         client_index=client_index               │       │
│     │     )                                           │       │
│     └─────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│     ┌─────────────────────────────────────────────────┐       │
│     │ # 创建统计日志管理器                             │       │
│     │ self.logger_manager = StatLoggerManager(...)    │       │
│     └─────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│     ┌─────────────────────────────────────────────────┐       │
│     │ # 启动输出处理协程                               │       │
│     │ self._run_output_handler()                      │       │
│     └─────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. EngineCoreClient.make_async_mp_client()                     │
│     (v1/engine/core_client.py:85)                               │
│                                                                 │
│     创建多进程引擎核心：                                         │
│     - 启动独立的 EngineCore 进程（EngineCore_DP0）              │
│     - 通过 ZMQ socket 进行 IPC 通信                             │
│     - 发送初始化消息到引擎核心                                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  8. EngineCore.__init__() (v1/engine/core.py:62)               │
│     [在独立进程 EngineCore_DP0 中运行]                          │
│                                                                 │
│     ┌─────────────────────────────────────────────────┐       │
│     │ # 加载插件                                       │       │
│     │ load_general_plugins()                          │       │
│     └─────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│     ┌─────────────────────────────────────────────────┐       │
│     │ # 创建模型执行器                                 │       │
│     │ self.model_executor = executor_class(           │       │
│     │     vllm_config                                 │       │
│     │ )                                               │       │
│     └─────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│     ┌─────────────────────────────────────────────────┐       │
│     │ # 初始化 KV Cache（内存分配 + 性能分析）         │       │
│     │ num_gpu_blocks, num_cpu_blocks, kv_cache_config │       │
│     │     = self._initialize_kv_caches(vllm_config)   │       │
│     │                                                 │       │
│     │ 包括：                                          │       │
│     │ - 模型加载到 GPU                                │       │
│     │ - GPU 内存分析                                  │       │
│     │ - 分配 KV cache blocks                          │       │
│     └─────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│     ┌─────────────────────────────────────────────────┐       │
│     │ # 创建调度器                                     │       │
│     │ self.scheduler = V1Scheduler(...)               │       │
│     └─────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│     ┌─────────────────────────────────────────────────┐       │
│     │ # 创建结构化输出管理器                           │       │
│     │ self.structured_output_manager = ...            │       │
│     └─────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  9. _initialize_kv_caches() (v1/engine/core.py:内部方法)        │
│     [在 EngineCore_DP0 进程中]                                  │
│                                                                 │
│     ┌─────────────────────────────────────────────────┐       │
│     │ Step 1: 模型加载                                │       │
│     │ ────────────────────────────────────────       │       │
│     │ INFO: Starting to load model                   │       │
│     │ INFO: Loading model from scratch...            │       │
│     │ INFO: Using Flash Attention backend on V1      │       │
│     │                                                 │       │
│     │ - 加载 safetensors 权重文件                     │       │
│     │ - 4 个 shard 文件逐个加载                       │       │
│     │ - 模型权重占用：14.9889 GiB                     │       │
│     │ - 耗时：3.565280 秒                             │       │
│     └─────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│     ┌─────────────────────────────────────────────────┐       │
│     │ Step 2: torch.compile 编译                      │       │
│     │ ────────────────────────────────────────       │       │
│     │ INFO: Using cache directory for torch.compile  │       │
│     │ INFO: Dynamo bytecode transform time: 4.43 s   │       │
│     │ INFO: Compiling a graph for dynamic shape      │       │
│     │       takes 18.28 s                             │       │
│     │ INFO: torch.compile takes 22.70 s in total     │       │
│     │                                                 │       │
│     │ - 编译 level: 3（默认）                         │       │
│     │ - 后端: inductor                                │       │
│     │ - 缓存目录: ~/.cache/vllm/torch_compile_cache/  │       │
│     └─────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│     ┌─────────────────────────────────────────────────┐       │
│     │ Step 3: 性能分析与 KV Cache 分配                │       │
│     │ ────────────────────────────────────────       │       │
│     │ INFO: Available KV cache memory: 4.95 GiB      │       │
│     │ INFO: GPU KV cache size: 40,528 tokens         │       │
│     │ INFO: Maximum concurrency: 4.05x               │       │
│     │                                                 │       │
│     │ 计算逻辑：                                      │       │
│     │ - 总显存：24 GB (RTX 3090/4090)                │       │
│     │ - 模型占用：14.99 GiB                           │       │
│     │ - 编译缓存：0.69 GiB                            │       │
│     │ - gpu_memory_utilization: 0.90                 │       │
│     │ - 可用 KV cache: 24*0.9 - 14.99 - 0.69 ≈ 4.95 │       │
│     │ - num_gpu_blocks: 2533 (block_size=16)         │       │
│     │ - 总 tokens: 2533 * 16 = 40,528               │       │
│     └─────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│     ┌─────────────────────────────────────────────────┐       │
│     │ Step 4: CUDA Graph 捕获                         │       │
│     │ ────────────────────────────────────────       │       │
│     │ Capturing CUDA graphs (mixed prefill-decode,   │       │
│     │                        PIECEWISE): 67/67        │       │
│     │ Capturing CUDA graphs (decode, FULL): 35/35    │       │
│     │                                                 │       │
│     │ INFO: Graph capturing finished in 6 secs       │       │
│     │ INFO: Graph memory: 0.69 GiB                   │       │
│     │                                                 │       │
│     │ CUDA Graph 用途：                               │       │
│     │ - 减少 kernel 启动开销                          │       │
│     │ - 捕获 67 个 prefill-decode 混合场景           │       │
│     │ - 捕获 35 个纯 decode 场景                      │       │
│     │ - 支持的 batch sizes: 1-512                    │       │
│     └─────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│     ┌─────────────────────────────────────────────────┐       │
│     │ INFO: init engine (profile, create kv cache,   │       │
│     │       warmup model) took 37.12 seconds          │       │
│     └─────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  10. run_server_worker() 继续 (api_server.py:1907)             │
│      [回到 APIServer 主进程]                                    │
│                                                                 │
│      ┌─────────────────────────────────────────────────┐      │
│      │ # 构建 FastAPI 应用                             │      │
│      │ app = build_app(args)                           │      │
│      └─────────────────────────────────────────────────┘      │
│                          │                                      │
│                          ▼                                      │
│      ┌─────────────────────────────────────────────────┐      │
│      │ # 初始化应用状态                                 │      │
│      │ vllm_config = await engine_client.              │      │
│      │     get_vllm_config()                           │      │
│      │ await init_app_state(                           │      │
│      │     engine_client, vllm_config, app.state, args │      │
│      │ )                                               │      │
│      └─────────────────────────────────────────────────┘      │
│                          │                                      │
│                          ▼                                      │
│      ┌─────────────────────────────────────────────────┐      │
│      │ # 启动 HTTP 服务器                              │      │
│      │ await serve_http(app, sock=sock, ...)          │      │
│      │                                                 │      │
│      │ INFO: Starting vLLM API server 0 on            │      │
│      │       http://0.0.0.0:6578                       │      │
│      └─────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  11. build_app() (api_server.py:1511)                          │
│                                                                 │
│      ┌─────────────────────────────────────────────────┐      │
│      │ # 创建 FastAPI 应用                             │      │
│      │ app = FastAPI(lifespan=lifespan)                │      │
│      └─────────────────────────────────────────────────┘      │
│                          │                                      │
│                          ▼                                      │
│      ┌─────────────────────────────────────────────────┐      │
│      │ # 注册路由                                       │      │
│      │ app.include_router(router)                      │      │
│      │                                                 │      │
│      │ 支持的路由：                                     │      │
│      │ - /v1/chat/completions                          │      │
│      │ - /v1/completions                               │      │
│      │ - /v1/embeddings                                │      │
│      │ - /v1/models                                    │      │
│      │ - /health, /ping                                │      │
│      │ - /tokenize, /detokenize                        │      │
│      │ - /v1/score, /rerank                            │      │
│      │ - 等 30+ 个端点                                 │      │
│      └─────────────────────────────────────────────────┘      │
│                          │                                      │
│                          ▼                                      │
│      ┌─────────────────────────────────────────────────┐      │
│      │ # 添加中间件                                     │      │
│      │ - CORSMiddleware (跨域)                         │      │
│      │ - AuthenticationMiddleware (API Key)            │      │
│      │ - XRequestIdMiddleware (请求 ID)                │      │
│      │ - ScalingMiddleware (扩缩容检查)                │      │
│      └─────────────────────────────────────────────────┘      │
│                          │                                      │
│                          ▼                                      │
│      ┌─────────────────────────────────────────────────┐      │
│      │ # 挂载监控                                       │      │
│      │ mount_metrics(app)  # Prometheus /metrics       │      │
│      └─────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  12. init_app_state() (api_server.py:1606)                     │
│                                                                 │
│      ┌─────────────────────────────────────────────────┐      │
│      │ # 获取支持的任务类型                             │      │
│      │ supported_tasks = await engine_client.          │      │
│      │     get_supported_tasks()                       │      │
│      │ INFO: Supported_tasks: ['generate']             │      │
│      └─────────────────────────────────────────────────┘      │
│                          │                                      │
│                          ▼                                      │
│      ┌─────────────────────────────────────────────────┐      │
│      │ # 加载 Chat Template                            │      │
│      │ resolved_chat_template = load_chat_template(    │      │
│      │     args.chat_template                          │      │
│      │ )                                               │      │
│      └─────────────────────────────────────────────────┘      │
│                          │                                      │
│                          ▼                                      │
│      ┌─────────────────────────────────────────────────┐      │
│      │ # 初始化各个 Serving 组件                        │      │
│      │                                                 │      │
│      │ state.openai_serving_responses = ...            │      │
│      │ state.openai_serving_chat = OpenAIServingChat(  │      │
│      │     engine_client, model_config, ...            │      │
│      │ )                                               │      │
│      │ state.openai_serving_completion = ...           │      │
│      │ state.openai_serving_pooling = ...              │      │
│      │ state.openai_serving_embedding = ...            │      │
│      │ state.openai_serving_score = ...                │      │
│      │ state.openai_serving_tokenization = ...         │      │
│      │                                                 │      │
│      │ INFO: Using default chat sampling params:      │      │
│      │       {'temperature': 0.6, 'top_p': 0.9}        │      │
│      └─────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  13. serve_http() (entrypoints/launcher.py)                    │
│                                                                 │
│      ┌─────────────────────────────────────────────────┐      │
│      │ # 启动 Uvicorn ASGI 服务器                      │      │
│      │ config = uvicorn.Config(                        │      │
│      │     app=app,                                    │      │
│      │     host=args.host,                             │      │
│      │     port=args.port,                             │      │
│      │     log_level=args.uvicorn_log_level,           │      │
│      │     timeout_keep_alive=60,                      │      │
│      │     access_log=not args.disable_access_log      │      │
│      │ )                                               │      │
│      │ server = uvicorn.Server(config)                 │      │
│      │ await server.serve()                            │      │
│      └─────────────────────────────────────────────────┘      │
│                          │                                      │
│                          ▼                                      │
│      ┌─────────────────────────────────────────────────┐      │
│      │ INFO: Started server process [1883203]          │      │
│      │ INFO: Waiting for application startup.          │      │
│      │ INFO: Application startup complete.             │      │
│      │                                                 │      │
│      │ 🎉 服务已就绪，可以接收请求                      │      │
│      └─────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心组件初始化

### 1. AsyncLLM (V1 引擎)

**位置**: `vllm/v1/engine/async_llm.py`

**职责**：
- 异步 LLM 引擎的主入口
- 管理请求生命周期
- 协调输入处理、引擎核心、输出处理

**初始化流程**：
```python
AsyncLLM(
    vllm_config=vllm_config,
    executor_class=Executor.get_class(vllm_config),
    log_stats=not disable_log_stats,
    usage_context=UsageContext.OPENAI_API_SERVER,
)
```

**核心组件**：
- **Processor**: 输入预处理（Tokenization、多模态处理）
- **OutputProcessor**: 输出后处理（Detokenization、采样参数应用）
- **EngineCoreClient**: 引擎核心客户端（多进程 IPC）
- **StatLoggerManager**: 统计日志管理

---

### 2. EngineCore (V1 引擎核心)

**位置**: `vllm/v1/engine/core.py`

**进程**: 独立进程 `EngineCore_DP0` (Data Parallel rank 0)

**职责**：
- 模型加载与执行
- KV Cache 管理
- 请求调度
- CUDA Graph 优化

**初始化流程**：
```python
EngineCore(
    vllm_config=vllm_config,
    executor_class=executor_class,
    log_stats=log_stats,
)
```

**核心组件**：
- **ModelExecutor**: 模型执行器（GPU 推理）
- **V1Scheduler**: 请求调度器（Chunked Prefill + Continuous Batching）
- **KV Cache**: 分页键值缓存（PagedAttention）
- **StructuredOutputManager**: 结构化输出管理（JSON Schema 约束）

---

### 3. ModelExecutor (模型执行)

**位置**: `vllm/v1/executor/`

**职责**：
- 加载模型权重到 GPU
- 执行前向推理
- 管理分布式并行（TP/PP/DP）

**关键操作**：

#### 3.1 模型加载
```python
# 日志输出示例
INFO: Starting to load model /data/ysh/models/Llama-3.1-8B-Instruct/...
INFO: Loading model from scratch...

Loading safetensors checkpoint shards: 100% | 4/4 [00:03<00:00, 1.31it/s]
INFO: Loading weights took 3.33 seconds
INFO: Model loading took 14.9889 GiB and 3.565280 seconds
```

**加载过程**：
1. 读取 `config.json` 解析模型架构
2. 根据架构创建模型实例（`LlamaForCausalLM`）
3. 从 safetensors 文件加载权重（4 个 shard）
4. 将权重传输到 GPU（FP16/BF16）

#### 3.2 torch.compile 编译
```python
# 日志输出示例
INFO: Using cache directory: ~/.cache/vllm/torch_compile_cache/...
INFO: Dynamo bytecode transform time: 4.43 s
INFO: Compiling a graph for dynamic shape takes 18.28 s
INFO: torch.compile takes 22.70 s in total
```

**编译配置**：
```python
CompilationConfig(
    level=3,              # 最高优化级别
    backend="inductor",   # PyTorch Inductor 后端
    use_inductor=True,
    use_cudagraph=True,   # 启用 CUDA Graph
)
```

**优化效果**：
- 融合 kernel 操作
- 减少 Python 开销
- 提升推理吞吐量 20-40%

#### 3.3 KV Cache 分配
```python
# 日志输出示例
INFO: Available KV cache memory: 4.95 GiB
INFO: GPU KV cache size: 40,528 tokens
INFO: Maximum concurrency for 10,000 tokens per request: 4.05x
```

**计算公式**：
```python
total_memory = 24 GB  # GPU 显存（如 RTX 4090）
model_memory = 14.99 GiB  # 模型权重
compile_memory = 0.69 GiB  # torch.compile + CUDA Graph

kv_cache_memory = total_memory * gpu_memory_utilization - model_memory - compile_memory
                = 24 * 0.9 - 14.99 - 0.69
                = 4.95 GiB

# 每个 token 的 KV cache 大小（Llama-3.1-8B）
bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_size
                = 2 * 32 * 8 * 128 * 2  # BF16 = 2 bytes
                = 131,072 bytes
                = 128 KB

# Block 大小和数量
block_size = 16  # tokens per block
block_memory = 128 KB * 16 = 2 MB per block

num_gpu_blocks = kv_cache_memory / block_memory
               = 4.95 GiB / 2 MB
               = 2,533 blocks

total_tokens = num_gpu_blocks * block_size
             = 2,533 * 16
             = 40,528 tokens

# 最大并发数（max_model_len = 10,000）
max_concurrency = total_tokens / max_model_len
                = 40,528 / 10,000
                = 4.05x
```

#### 3.4 CUDA Graph 捕获
```python
# 日志输出示例
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100% | 67/67 [00:03<00:00]
Capturing CUDA graphs (decode, FULL): 100% | 35/35 [00:01<00:00]

INFO: Graph capturing finished in 6 secs, took 0.69 GiB
```

**捕获策略**：
- **Mixed Prefill-Decode (PIECEWISE)**: 67 个场景
  - 捕获不同 batch size 的混合场景（prefill + decode）
  - 用于动态批处理
  
- **Decode-only (FULL)**: 35 个场景
  - 捕获纯 decode 的 batch sizes: [1, 2, 4, 8, ..., 512]
  - 用于生成阶段的批量推理

**优化效果**：
- 减少 kernel 启动开销 10-20%
- 固定内存布局，减少碎片
- 支持高吞吐量推理

---

### 4. V1Scheduler (调度器)

**位置**: `vllm/v1/core/sched/scheduler.py`

**职责**：
- 请求队列管理
- 分块预填充（Chunked Prefill）
- 连续批处理（Continuous Batching）
- 优先级调度

**关键特性**：

#### 4.1 Chunked Prefill
```python
# 日志输出示例
INFO: Chunked prefill is enabled with max_num_batched_tokens=2048.
```

**工作原理**：
- 将长 prefill 请求分成多个 chunk（每个 2048 tokens）
- 与 decode 请求混合批处理
- 避免长请求阻塞短请求

**示例**：
```
请求 A: prefill 8192 tokens → 分成 4 个 chunk (2048 * 4)
请求 B: decode 128 tokens
请求 C: prefill 1024 tokens → 1 个 chunk

Batch 1: [A_chunk_1, B_decode, C_prefill]  # 2048 + 128 + 1024 = 3200 tokens
Batch 2: [A_chunk_2, B_decode]
Batch 3: [A_chunk_3, B_decode]
Batch 4: [A_chunk_4, B_decode]
```

#### 4.2 Continuous Batching
- 动态调整 batch size
- 完成的请求立即从 batch 中移除
- 新请求立即加入 batch（如有空间）

---

### 5. FastAPI 应用

**位置**: `vllm/entrypoints/openai/api_server.py`

**职责**：
- HTTP API 服务
- 请求验证与路由
- OpenAI 协议兼容

**路由列表** (共 30+ 个端点)：

| 路由 | 方法 | 功能 |
|------|------|------|
| `/v1/chat/completions` | POST | Chat 对话完成 |
| `/v1/completions` | POST | 文本完成 |
| `/v1/embeddings` | POST | 文本嵌入 |
| `/v1/models` | GET | 列出可用模型 |
| `/health` | GET | 健康检查 |
| `/ping` | GET/POST | 服务存活检查 |
| `/tokenize` | POST | Tokenization |
| `/detokenize` | POST | Detokenization |
| `/v1/score` | POST | 序列评分 |
| `/rerank` | POST | 文档重排序 |
| `/v1/audio/transcriptions` | POST | 语音转文本 |
| `/metrics` | GET | Prometheus 监控指标 |

**中间件**：
- **CORSMiddleware**: 处理跨域请求
- **AuthenticationMiddleware**: API Key 验证
- **XRequestIdMiddleware**: 请求 ID 追踪
- **ScalingMiddleware**: 扩缩容状态检查






---

## 日志关键字解析

| 日志关键字 | 含义 | 代码位置 |
|-----------|------|---------|
| `vLLM API server version` | 服务器版本信息 | `api_server.py:打印启动信息` |
| `non-default args` | 用户指定的参数 | `utils.py:233` |
| `Resolved architecture` | 检测到的模型架构 | `model.py:547` |
| `Using max model len` | 最大序列长度 | `model.py:1510` |
| `Chunked prefill is enabled` | 分块预填充开启 | `scheduler.py:205` |
| `Automatically detected platform` | 检测到的硬件平台 | `__init__.py:216` |
| `Initializing a V1 LLM engine` | V1 引擎初始化开始 | `core.py:77` |
| `Starting to load model` | 开始加载模型 | `gpu_model_runner.py:2602` |
| `Using Flash Attention backend` | 注意力后端选择 | `cuda.py:366` |
| `Loading weights took X seconds` | 权重加载耗时 | `default_loader.py:267` |
| `torch.compile takes X s in total` | 编译总耗时 | `monitor.py:34` |
| `Available KV cache memory` | 可用 KV cache 内存 | `gpu_worker.py:298` |
| `GPU KV cache size: X tokens` | KV cache 容量 | `kv_cache_utils.py:1087` |
| `Maximum concurrency` | 最大并发请求数 | `kv_cache_utils.py:1091` |
| `Capturing CUDA graphs` | CUDA Graph 捕获进度 | 进度条输出 |
| `Graph capturing finished` | 捕获完成 | `gpu_model_runner.py:3480` |
| `init engine took X seconds` | 引擎初始化总耗时 | `core.py:210` |
| `Supported_tasks` | 支持的任务类型 | `api_server.py:1634` |
| `Using default chat sampling params` | 默认采样参数 | `serving_chat.py:139` |
| `Starting vLLM API server` | 服务器启动 | `api_server.py:1912` |
| `Available routes` | 可用的 API 路由 | `launcher.py:34` |
| `Application startup complete` | 应用启动完成 | Uvicorn 输出 |

---



**文档版本**: v1.0  
**更新日期**: 2026 年 2 月 14 日  
**适用 vLLM 版本**: 0.11.0+ (V1 引擎)  
**作者**: vLLM Community
