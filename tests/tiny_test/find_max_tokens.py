import json
import time
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# 配置
DATA_FILE = "/home/ysh/datasets/LongBench/data/hotpotqa_e.jsonl"
MODEL_PATH = "/data/ysh/models/Llama-3.1-8B-Instruct/"
START_TOKEN = 20000
TOKEN_INCREMENT = 1000

def read_and_concat_contexts(file_path, num_lines=10):
    """读取 jsonl 文件的前 num_lines 行的 context 并拼接"""
    contexts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                data = json.loads(line.strip())
                if 'context' in data:
                    contexts.append(data['context'])
                else:
                    print(f"Warning: Line {i} doesn't have 'context' field")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return None
    
    concatenated_text = " ".join(contexts)
    return concatenated_text

def count_tokens(text, tokenizer):
    """计算文本的 token 数量"""
    tokens = tokenizer.encode(text)
    return len(tokens)

def truncate_to_token_length(text, tokenizer, target_length):
    """截取文本到指定的 token 长度"""
    tokens = tokenizer.encode(text)
    if len(tokens) <= target_length:
        return text, len(tokens)
    
    truncated_tokens = tokens[:target_length]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    return truncated_text, target_length

def test_oom(llm, text, token_length):
    """测试指定 token 长度是否会 OOM"""
    try:
        print(f"\n{'='*60}")
        print(f"Testing with {token_length} tokens...")
        print(f"{'='*60}")
        
        # 创建 prompt
        prompt = f"Based on the following context, please provide a summary:\n\n{text}\n\nSummary:"
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=50,
        )
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行推理
        outputs = llm.generate([prompt], sampling_params)
        
        # 记录结束时间
        end_time = time.time()
        inference_time = end_time - start_time
        
        print(f"✓ Success! {token_length} tokens processed successfully")
        print(f"⏱️  Inference time: {inference_time:.2f} seconds ({inference_time/60:.2f} minutes)")
        print(f"Output: {outputs[0].outputs[0].text[:100]}...")
        
        # 清理 GPU 缓存（vLLM 会自动管理 KV cache）
        torch.cuda.empty_cache()
        
        return True, inference_time
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"✗ OOM Error at {token_length} tokens!")
        print(f"Error: {str(e)}")
        torch.cuda.empty_cache()
        return False
    except Exception as e:
        print(f"✗ Error at {token_length} tokens: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        torch.cuda.empty_cache()
        return False

def main():
    print("="*60)
    print("Long Context OOM Test for vLLM")
    print("="*60)
    
    # 步骤 1: 读取并拼接数据
    print("\n[Step 1] Reading and concatenating contexts from data file...")
    full_text = read_and_concat_contexts(DATA_FILE, num_lines=10)
    
    if full_text is None:
        print("Failed to read data file. Exiting.")
        return
    
    print(f"Successfully read and concatenated {len(full_text)} characters")
    
    # 步骤 2: 加载 tokenizer 并计算总 token 数
    print(f"\n[Step 2] Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    total_tokens = count_tokens(full_text, tokenizer)
    print(f"Total tokens in concatenated text: {total_tokens}")
    
    if total_tokens < START_TOKEN:
        print(f"\nWarning: Total tokens ({total_tokens}) is less than START_TOKEN ({START_TOKEN})")
        print("Will repeat the text to reach the required length...")
        # 重复文本以达到所需长度
        repeat_times = (START_TOKEN // total_tokens) + 2
        full_text = (full_text + " ") * repeat_times
        total_tokens = count_tokens(full_text, tokenizer)
        print(f"After repeating, total tokens: {total_tokens}")
    
    # 步骤 3: 加载模型一次（设置足够大的 max_model_len）
    print(f"\n[Step 3] Loading model (this will only be done once)...")
    try:
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=40000,  # 设置一个足够大的值，Llama-3.1 支持 128K
            enforce_eager=True,  # 避免 CUDA graph 的内存开销
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # 步骤 4: 从 START_TOKEN 开始测试
    print(f"\n[Step 4] Starting OOM test from {START_TOKEN} tokens...")
    print(f"Incrementing by {TOKEN_INCREMENT} tokens each iteration")
    
    current_token_length = START_TOKEN
    oom_found = False
    last_successful_length = 0
    
    while not oom_found:
        # 截取到当前 token 长度
        truncated_text, actual_length = truncate_to_token_length(
            full_text, tokenizer, current_token_length
        )
        
        # 测试是否 OOM (传入已加载的 llm 实例)
        success = test_oom(llm, truncated_text, actual_length)
        
        if success:
            last_successful_length = current_token_length
            current_token_length += TOKEN_INCREMENT
        else:
            oom_found = True
            print(f"\n{'='*60}")
            print(f"OOM DETECTED!")
            print(f"Last successful length: {last_successful_length} tokens")
            print(f"OOM occurred at: {current_token_length} tokens")
            print(f"{'='*60}")
            break
    
    # 清理
    del llm
    torch.cuda.empty_cache()
    print("\n[Test Complete]")

if __name__ == "__main__":
    main()
