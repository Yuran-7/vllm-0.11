import json
import os
from transformers import AutoTokenizer

# 配置
DATA_FILE = "/home/ysh/datasets/LongBench/data/hotpotqa_e.jsonl"
OUTPUT_FILE = "/home/ysh/datasets/LongBench/data/hotpotqa_e_variable_length.jsonl"
MODEL_PATH = "/data/ysh/models/Llama-3.1-8B-Instruct/"

# 目标 token 长度列表
TARGET_TOKEN_LENGTHS = [2000, 4000, 6000, 8000, 10000, 15000, 20000, 25000, 28000, 30000, 35000, 40000]
LINES_PER_GROUP = 10

def read_all_contexts(file_path):
    """读取所有行的 context"""
    contexts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if 'context' in data:
                    contexts.append(data['context'])
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return None
    
    return contexts

def concat_group_contexts(contexts, start_idx, group_size):
    """拼接一组 context"""
    group_contexts = contexts[start_idx:start_idx + group_size]
    concatenated = " ".join(group_contexts)
    return concatenated

def truncate_to_token_length(text, tokenizer, target_length):
    """截取文本到指定的 token 长度"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= target_length:
        print(f"Warning: Text has {len(tokens)} tokens, less than target {target_length}")
        return text, len(tokens)
    
    truncated_tokens = tokens[:target_length]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    # 验证实际 token 数量
    actual_tokens = tokenizer.encode(truncated_text, add_special_tokens=False)
    return truncated_text, len(actual_tokens)

def create_dataset():
    print("="*60)
    print("Variable Length Dataset Creation")
    print("="*60)
    
    # 步骤 1: 读取所有数据
    print(f"\n[Step 1] Reading all contexts from {DATA_FILE}...")
    all_contexts = read_all_contexts(DATA_FILE)
    
    if all_contexts is None:
        print("Failed to read data file. Exiting.")
        return
    
    print(f"Total contexts read: {len(all_contexts)}")
    
    required_lines = len(TARGET_TOKEN_LENGTHS) * LINES_PER_GROUP
    if len(all_contexts) < required_lines:
        print(f"Warning: Need {required_lines} lines but only have {len(all_contexts)}")
        print("Will reuse contexts if necessary...")
    
    # 步骤 2: 加载 tokenizer
    print(f"\n[Step 2] Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("✓ Tokenizer loaded successfully")
    
    # 步骤 3: 生成数据
    print(f"\n[Step 3] Generating dataset with variable token lengths...")
    print(f"Target token lengths: {TARGET_TOKEN_LENGTHS}")
    print(f"Lines per group: {LINES_PER_GROUP}")
    
    output_data = []
    current_idx = 0
    
    for group_id, target_length in enumerate(TARGET_TOKEN_LENGTHS):
        print(f"\n--- Processing Group {group_id + 1}/{len(TARGET_TOKEN_LENGTHS)} ---")
        print(f"Target token length: {target_length}")
        
        # 拼接当前组的 contexts
        start_idx = current_idx % len(all_contexts)  # 循环使用如果不够
        concatenated_text = concat_group_contexts(all_contexts, start_idx, LINES_PER_GROUP)
        
        print(f"Concatenated {LINES_PER_GROUP} contexts (lines {start_idx} to {start_idx + LINES_PER_GROUP - 1})")
        print(f"Original text length: {len(concatenated_text)} characters")
        
        # 如果拼接后的文本 token 不够，重复文本
        initial_tokens = tokenizer.encode(concatenated_text, add_special_tokens=False)
        print(f"Initial token count: {len(initial_tokens)}")
        
        if len(initial_tokens) < target_length:
            print(f"Warning: Concatenated text has {len(initial_tokens)} tokens, less than target {target_length}")
            print("Repeating text to reach target length...")
            repeat_times = (target_length // len(initial_tokens)) + 2
            concatenated_text = (concatenated_text + " ") * repeat_times
            repeated_tokens = tokenizer.encode(concatenated_text, add_special_tokens=False)
            print(f"After repeating {repeat_times} times: {len(repeated_tokens)} tokens")
        
        # 截取到目标 token 长度
        truncated_text, actual_length = truncate_to_token_length(
            concatenated_text, tokenizer, target_length
        )
        
        print(f"✓ Truncated to {actual_length} tokens")
        
        # 创建数据条目
        data_entry = {
            "id": group_id,
            "token_length": actual_length,
            "target_token_length": target_length,
            "context": truncated_text
        }
        
        output_data.append(data_entry)
        current_idx += LINES_PER_GROUP
    
    # 步骤 4: 写入输出文件
    print(f"\n[Step 4] Writing output to {OUTPUT_FILE}...")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for entry in output_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"✓ Successfully wrote {len(output_data)} entries to {OUTPUT_FILE}")
    except Exception as e:
        print(f"✗ Error writing output file: {e}")
        return
    
    # 步骤 5: 验证和摘要
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"{'ID':<5} {'Target Tokens':<15} {'Actual Tokens':<15} {'Context Preview':<30}")
    print(f"{'-'*60}")
    
    for entry in output_data:
        preview = entry['context'][:50].replace('\n', ' ')
        print(f"{entry['id']:<5} {entry['target_token_length']:<15} {entry['token_length']:<15} {preview}...")
    
    print(f"{'='*60}")
    print(f"Total entries created: {len(output_data)}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"{'='*60}")
    
    print("\n[Complete]")

if __name__ == "__main__":
    # 在执行逻辑前清空文件
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.truncate(0)
        print(f"注意：已清空旧的生成文件 {OUTPUT_FILE}")
    
    create_dataset()
  
# python tests/tiny_test/create_variable_length_dataset.py
