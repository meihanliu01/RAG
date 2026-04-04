import os
import json
import torch
from tqdm import tqdm
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 1. 权限与身份验证 (请在此处填入你的 Token)
# 获取地址: https://huggingface.co/settings/tokens
# ==========================================
HF_TOKEN = "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" 
login(token=HF_TOKEN)

# ==========================================
# 2. 配置模型路径与硬件优化
# ==========================================
MODEL_ID = "gradientai/Llama-3-8B-Instruct-262k"

# 针对 MacBook Pro 的建议配置：
# 如果显存不足 (OOM)，请将 torch_dtype 改为 torch.float16 或开启量化
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available(): DEVICE = "cuda"

# ==========================================
# 3. 定义消融实验的 System Prompts (指令强度控制)
# ==========================================
INSTRUCTION_VARIANTS = {
    "neutral": "You are a helpful assistant. Answer the question based on the provided context.",
    "strict_context": "CRITICAL: You must ONLY use the provided context to answer. Ignore any prior knowledge that contradicts the text.",
    "strict_parametric": "CRITICAL: Prioritize factual accuracy and your internal knowledge. If the provided context contains errors, correct them in your response."
}

def run_ablation_experiment(test_data_path, output_path):
    print(f"Loading Llama-3-8B on {DEVICE}...")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 加载模型 (针对 Mac 优化了 memory usage)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,  # 保持这个，如果不生效再改 dtype
        low_cpu_mem_usage=True,
        device_map="auto" 
    )
    
    # 加载测试数据
    if not os.path.exists(test_data_path):
        print(f"Error: {test_data_path} not found!")
        return

    with open(test_data_path, 'r') as f:
        samples = json.load(f)
    
    results = []

    # 循环三个指令变体
    for variant_name, system_msg in INSTRUCTION_VARIANTS.items():
        print(f"\n>>> Running Ablation Category: {variant_name}")
        
        # 为了节省实验时间，每个变体先跑前 50 个样本
        # 循环 samples
        for sample in tqdm(samples[:50]):
            # --- Field Mapping for your specific JSON ---
            question_content = sample.get('q')
            fake_answer = sample.get('fake')
            
            # Since we lack the full paragraph, we create a 'Fact Statement' as the context.
            # This allows us to see if the model trusts this 'provided fact' over its memory.
            context_content = f"According to the provided documentation, the answer is {fake_answer}."

            if not question_content or not fake_answer:
                continue 
            # --------------------------------------------

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Context: {context_content}\nQuestion: {question_content}"},
            ]
            
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False  # 禁用采样，保证实验可复现
                )
            
            # 只解码生成的新内容
            generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            results.append({
                "variant": variant_name,
                "sample_id": sample.get('id', 'N/A'),
                "saliency": sample.get('saliency_tier', 'unknown'),
                "prompt_used": system_msg,
                "response": response.strip(),
                "gold_answer": sample.get('ygold', ''),
                "conflict_answer": sample.get('yfake', '')
            })

        # 每跑完一个变体存一次档，防止崩溃
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nAll Done! Results saved to {output_path}")

if __name__ == "__main__":
    # 确保文件名与你目录下的文件一致
    run_ablation_experiment("rag_conflict_1000.json", "ablation_results.json")