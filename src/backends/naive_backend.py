import time
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class NaiveBackend:
    """HuggingFace Transformers 原生推理封装"""
    def __init__(self, model_path, device="cuda"):
        self.model_path = model_path
        self.device = device
        
        # 路径存在性检查
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        if not os.path.isdir(model_path):
            raise NotADirectoryError(f"Model path is not a directory: {model_path}")
        
        print(f"[Naive] Loading model from local path: {model_path}")
        
        # 强制 local_files_only=True，避免 transformers 误判为远程 repo_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )

        # ← 新增：decoder-only 模型 batch 推理必须用 left padding
        self.tokenizer.padding_side = 'left'
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        self.model.eval()
        print("[Naive] Model loaded. GPU memory:", 
              f"{torch.cuda.memory_allocated()/1024**2:.0f} MB")
    
    def generate(self, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        input_len = inputs.input_ids.shape[1]
        output_tokens = outputs[0][input_len:]
        output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        num_output_tokens = output_tokens.shape[0]
        total_time = end_time - start_time
        
        return {
            "output_text": output_text,
            "num_input_tokens": int(input_len),
            "num_output_tokens": int(num_output_tokens),
            "total_time": total_time,
            "throughput": num_output_tokens / total_time if total_time > 0 else 0,
        }
    
    def generate_batch(self, prompts, max_new_tokens=256, temperature=0.7, top_p=0.9):
        """Static batching inference. All prompts are processed in one forward pass."""
        if not prompts:
            return []

        # 1. Apply chat template to all prompts
        texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)

        # 2. Batch tokenize with left padding (decoder-only 模型必须 left padding)
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.device)

        # padding 后的统一序列长度（所有样本对齐后的长度）
        input_seq_len = inputs.input_ids.shape[1]

        # 3. Record batch-level timing
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        # 4. Split batch outputs into individual results
        total_time = end_time - start_time

        results = []
        for i, output in enumerate(outputs):
            # 从 input_seq_len 处切分，得到生成的 tokens
            generated_tokens = output[input_seq_len:]
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            num_output_tokens = generated_tokens.shape[0]
            
            # 实际输入 token 数（不含 padding）
            num_input_tokens = int(inputs.attention_mask[i].sum().item())

            results.append({
                "output_text": output_text,
                "num_input_tokens": num_input_tokens,
                "num_output_tokens": int(num_output_tokens),
                "total_time": total_time,
                "throughput": num_output_tokens / total_time if total_time > 0 else 0,
            })

        return results