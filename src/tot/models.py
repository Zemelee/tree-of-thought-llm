import os
import time
import torch
import backoff
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

completion_tokens = prompt_tokens = 0

model_path = "/home/zemelee/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
).eval()


# 定义重试机制
@backoff.on_exception(backoff.expo, (torch.cuda.OutOfMemoryError, RuntimeError))
def generate_with_backoff(**kwargs):
    return model.generate(**kwargs)


def gpt(prompt, temperature=0.7, max_tokens=1000, n=1, stop=None):
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
    )

# n用于重复生成n个回复
def chatgpt(messages, temperature=0.7, max_tokens=1000, n=1, stop=None):
    global completion_tokens, prompt_tokens
    system_prompt = "You are a helpful assistant."
    model_messages = [{"role": "system", "content": system_prompt}] + messages
    inputs = tokenizer.apply_chat_template(model_messages, return_tensors="pt").to(
        model.device
    )
    prompt_token_count = inputs.shape[1]
    prompt_tokens += prompt_token_count
    # 处理停止词
    stop_ids = []
    if stop:
        for s in stop:
            stop_ids.append(tokenizer.encode(s, add_special_tokens=False)[0])
    # 生成回复
    outputs = []
    for _ in range(n):
        # 生成参数配置
        generate_args = {
            "input_ids": inputs,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0.0,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if stop_ids:
            generate_args["bad_words_ids"] = [[stop_id] for stop_id in stop_ids]
        # 生成回复并统计token数
        with torch.no_grad():
            outputs_tensor = generate_with_backoff(**generate_args)
        # 计算生成的token数
        generated_tokens = outputs_tensor[0, prompt_token_count:]
        completion_token_count = generated_tokens.shape[0]
        completion_tokens += completion_token_count
        # 解码回复
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # 处理停止词（手动截断）
        if stop:
            for s in stop:
                if s in output_text:
                    output_text = output_text[: output_text.index(s)]
        outputs.append(output_text)
    return outputs


def gpt_usage(backend="Qwen2.5-3B-Instruct"):
    global completion_tokens, prompt_tokens
    # 计算近似成本（基于OpenAI价格作为参考）
    if backend.lower().startswith("gpt-4"):
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend.lower().startswith("gpt-3.5"):
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    else:  # 本地模型的近似成本
        cost = (
            (completion_tokens + prompt_tokens) / 1000 * 0.0001
        )  # 假设每1k token 0.01美分
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cost": cost,
    }
