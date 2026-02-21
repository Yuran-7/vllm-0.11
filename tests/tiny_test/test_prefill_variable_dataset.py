import argparse
import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# 配置
DATA_FILE = "/NV1/ysh/GraphRAG/datasets/LongBench/data/hotpotqa_e_variable_length.jsonl"
MODEL_PATH = "/NV1/ykw/models/Meta-Llama-3.1-8B-Instruct/"
MAX_NEW_TOKENS = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Forward test with variable-length dataset")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "transformers"],
        default="vllm",
        help="Inference backend to use",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        choices=[0, 1],
        default=0,
        help="GPU index to use (0 or 1)",
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=None,
        help="Maximum number of batched tokens. If not set, uses vLLM default.",
    )
    return parser.parse_args()


def read_variable_length_dataset(file_path):
    """读取 variable length jsonl 数据集。"""
    entries = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if "context" not in data:
                    print(f"Warning: Line {i} has no 'context' field, skipped")
                    continue
                entries.append(data)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return None

    return entries


def make_prompt(context):
    return (
        "Based on the following context, please provide a brief summary:\n\n"
        f"{context}\n\n"
        "Summary:"
    )


def run_single_forward(llm, prompt, sampling_params):
    """执行单条前向推理，返回是否成功、耗时和结果文本。"""
    try:
        start = time.time()
        outputs = llm.generate([prompt], sampling_params)
        latency = time.time() - start
        text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        return True, latency, text, None
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return False, None, None, f"OOM: {str(e)}"
    except Exception as e:
        torch.cuda.empty_cache()
        return False, None, None, f"{type(e).__name__}: {str(e)}"


def run_single_forward_transformers(model, tokenizer, prompt, device):
    """使用 transformers 执行单条前向推理，返回是否成功、耗时和结果文本。"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        start = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        latency = time.time() - start

        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return True, latency, text, None
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return False, None, None, f"OOM: {str(e)}"
    except Exception as e:
        torch.cuda.empty_cache()
        return False, None, None, f"{type(e).__name__}: {str(e)}"


def main():
    args = parse_args()

    print("=" * 80)
    print("Forward Test with Variable-Length Dataset")
    print("=" * 80)
    print(f"Backend: {args.backend} | GPU: {args.gpu}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"\n[Step 1] Reading dataset from: {DATA_FILE}")
    entries = read_variable_length_dataset(DATA_FILE)

    print(f"Loaded {len(entries)} entries")

    print(f"\n[Step 2] Loading tokenizer from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("✓ Tokenizer loaded")

    token_lengths = []
    for entry in entries:
        if "token_length" in entry:
            token_lengths.append(int(entry["token_length"]))
        else:
            token_lengths.append(len(tokenizer.encode(entry["context"], add_special_tokens=False)))

    max_context_tokens = max(token_lengths)
    # prompt 模板 + 预留生成长度
    max_model_len = max_context_tokens + 100

    llm = None
    hf_model = None
    device = torch.device("cuda:0")

    print("\n[Step 3] Loading model once...")
    try:
        if args.backend == "vllm":
            print(f"Using vLLM with max_model_len={max_model_len}, max_num_batched_tokens={args.max_num_batched_tokens}")
            llm = LLM(
                model=MODEL_PATH,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                max_model_len=max_model_len,
                enforce_eager=True,
                max_num_batched_tokens=args.max_num_batched_tokens,
            )
        else:
            print(f"Using transformers on cuda:{args.gpu}")
            device = torch.device(f"cuda:{args.gpu}")
            hf_model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float16,
            )
            hf_model.to(device)
            hf_model.eval()
        print("✓ Model loaded")
    except torch.cuda.OutOfMemoryError as e:
        print(f"✗ OOM when loading model: {e}")
        torch.cuda.empty_cache()
        return
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        torch.cuda.empty_cache()
        return

    print("\n[Step 4] Running forward pass for each entry")
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
    )

    results = []
    for idx, entry in enumerate(entries):
        target = entry.get("target_token_length", "N/A")
        token_len = token_lengths[idx]
        prompt = make_prompt(entry["context"])

        print(f"\n--- Entry {idx + 1}/{len(entries)} | target={target}, context_tokens={token_len} ---")
        if args.backend == "vllm":
            success, latency, output_text, err = run_single_forward(llm, prompt, sampling_params)
        else:
            success, latency, output_text, err = run_single_forward_transformers(
                hf_model,
                tokenizer,
                prompt,
                device,
            )

        if success:
            print(
                f"✓ Success | backend={args.backend} | latency={latency:.2f}s "
                f"| output='{output_text[:80]}...'"
            )
            results.append({
                "id": entry.get("id", idx),
                "target_token_length": target,
                "context_token_length": token_len,
                "success": True,
                "latency_sec": latency,
                "error": "",
                "backend": args.backend,
                "gpu": args.gpu,
            })
        else:
            print(f"✗ Failed | backend={args.backend} | {err}")
            results.append({
                "id": entry.get("id", idx),
                "target_token_length": target,
                "context_token_length": token_len,
                "success": False,
                "latency_sec": None,
                "error": err,
                "backend": args.backend,
                "gpu": args.gpu,
            })

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count
    latencies = [r["latency_sec"] for r in results if r["latency_sec"] is not None]

    for r in results:
        status = "OK" if r["success"] else "FAIL"
        latency_str = f"{r['latency_sec']:.2f}s" if r["latency_sec"] is not None else "-"
        print(
            f"id={r['id']:<3} target={str(r['target_token_length']):<8} "
            f"context={r['context_token_length']:<8} status={status:<4} latency={latency_str:<8}"
        )

    if llm is not None:
        del llm
    if hf_model is not None:
        del hf_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

# python tests/tiny_test/test_prefill_variable_dataset.py --backend vllm --gpu 0
# python tests/tiny_test/test_prefill_variable_dataset.py --backend transformers --gpu 1
# python tests/tiny_test/test_prefill_variable_dataset.py --backend vllm --gpu 1
# python tests/tiny_test/test_prefill_variable_dataset.py --backend transformers --gpu 0
# python tests/tiny_test/test_prefill_variable_dataset.py --backend vllm --gpu 0 --max_num_batched_tokens 4096
