import time
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class VLLMBackend:
    """vLLM 推理封装"""
    def __init__(self, model_path, tensor_parallel_size=1, gpu_memory_utilization=0.9):
        self.model_path = model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        if not os.path.isdir(model_path):
            raise NotADirectoryError(f"Model path is not a directory: {model_path}")
        
        print(f"[vLLM] Loading model from local path: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # vLLM 的 LLM() 对本地路径支持较好，但同样加上 local_files_only 保险
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype="float16",
            max_model_len=8192,  
        )
        print("[vLLM] Model loaded.")
    
    def generate(self, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        
        start_time = time.perf_counter()
        outputs = self.llm.generate([text], sampling_params)
        end_time = time.perf_counter()
        
        output = outputs[0]
        output_text = output.outputs[0].text
        num_input_tokens = len(output.prompt_token_ids)
        num_output_tokens = len(output.outputs[0].token_ids)
        total_time = end_time - start_time
        
        return {
            "output_text": output_text,
            "num_input_tokens": int(num_input_tokens),
            "num_output_tokens": int(num_output_tokens),
            "total_time": total_time,
            "throughput": num_output_tokens / total_time if total_time > 0 else 0,
        }

    def generate_batch(self, prompts, max_new_tokens=256, temperature=0.7, top_p=0.9):
        """Batch generation using vLLM internal continuous batching.
        Args:
            prompts: list of raw prompt strings.
        Returns:
            List of result dicts, one per prompt.
        """
        texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)

        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_new_tokens
        )

        start_time = time.perf_counter()
        outputs = self.llm.generate(texts, sampling_params)
        end_time = time.perf_counter()

        results = []
        total_time = end_time - start_time
        for output in outputs:
            output_text = output.outputs[0].text
            num_input_tokens = len(output.prompt_token_ids)
            num_output_tokens = len(output.outputs[0].token_ids)
            results.append({
                "output_text": output_text,
                "num_input_tokens": int(num_input_tokens),
                "num_output_tokens": int(num_output_tokens),
                "total_time": total_time,  # batch 模式下所有请求共享总时间
                "throughput": num_output_tokens / total_time if total_time > 0 else 0,
            })
        return results
