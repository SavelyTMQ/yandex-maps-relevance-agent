"""Baseline на Qwen2.5-3B."""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import BASELINE_MODEL


class QwenBaseline:
    def __init__(self):
        print(f"Загружаю {BASELINE_MODEL}...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            BASELINE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    def predict(self, row, system_prompt, user_prompt_fn, fewshot_rows=None):
        messages = [{"role": "system", "content": system_prompt}]
        
        if fewshot_rows is not None:
            for _, ex in fewshot_rows.iterrows():
                messages.append({"role": "user", "content": user_prompt_fn(ex)})
                messages.append({"role": "assistant", "content": json.dumps(
                    {"reasoning": "…", "label": ex["label"]}, ensure_ascii=False
                )})
        
        messages.append({"role": "user", "content": user_prompt_fn(row)})
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=300, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        answer = self.tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return answer
