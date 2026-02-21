available_kv = requested_memory - non_kv_cache_memory，

其中 requested_memory = total_memory * gpu_memory_utilization

non_kv_cache_memory 由三部分组成：weights + peak_activation + non_torch，见vllm/utils/__init__.py

权重weights已知约 14.98 GiB

peak_activation = diff_profile.torch_peak，torch_peak 取的是 allocated_bytes.all.peak
含义是：在 profile_run() 这段期间，PyTorch 张量分配的“峰值增量”（主要是激活、临时工作区、采样/attention 过程中的临时 tensor）

non_torch_memory = cuda_memory - torch_memory_reserved，从“创建 vLLM 前”到“profile 结束后”仍然存在的非 torch 显存增长（例如 NCCL/通信缓冲、驱动/后端侧缓冲等）