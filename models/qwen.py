#!/usr/bin/env python3
"""
Text‑only Qwen wrapper for either HF‑Transformers or vLLM back‑ends.
"""

from typing import List, Dict, Any

import torch
from torch import amp
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams


class Qwen:
    def __init__(self, args, model_name_path: str = "Qwen/Qwen1.5-7B-Chat"):
        # ──────────────────────────────────────────────────────────
        # common generation hyper‑params
        # ──────────────────────────────────────────────────────────
        self.gen_prompt_suffix = args.gen_prompt_suffix
        self.gen_engine        = args.gen_engine.lower()
        self.max_new_tokens    = args.max_new_tokens
        self.top_k             = args.top_k
        self.top_p             = args.top_p
        self.temperature       = args.temperature
        self.n_generations     = args.n_generations
        self.do_sample         = self.temperature > 0

        # ──────────────────────────────────────────────────────────
        # Hugging Face back‑end
        # ──────────────────────────────────────────────────────────
        if self.gen_engine == "hf":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_path, trust_remote_code=True)
            self.tokenizer.padding_side = "left"

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()

            EOS = self.tokenizer.eos_token_id  # <|im_end|> → 151643 for Qwen2

            self.gen_params = dict(
                num_return_sequences=self.n_generations,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=EOS,          # ← stop as soon as <|im_end|> appears
                pad_token_id=EOS,          # avoids warning if no pad token
            )
            if self.do_sample:
                self.gen_params.update(
                    do_sample=True,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    temperature=self.temperature,
                )

        # ──────────────────────────────────────────────────────────
        # vLLM back‑end
        # ──────────────────────────────────────────────────────────
        elif self.gen_engine == "vllm":
            # still need tokenizer for prompt building / EOS id
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_path, trust_remote_code=True)

            self.model = LLM(
                model=model_name_path,
                dtype="auto",
                tensor_parallel_size=torch.cuda.device_count(),
                trust_remote_code=True,
                max_model_len=2048,
            )

            EOS = self.tokenizer.eos_token_id

            self.gen_params = SamplingParams(
                n=self.n_generations,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                stop_token_ids=[EOS],      # ← vLLM stop list
            )
        else:
            raise ValueError(f"Unsupported gen_engine: {self.gen_engine}")

    # ──────────────────────────────────────────────────────────
    # helper: build ChatML prompt with <|im_end|>
    # ──────────────────────────────────────────────────────────
    def _make_prompts(self, batch: List[Dict[str, Any]]) -> List[str]:
        convs = [
            [
                {
                    "role": "user",
                    "content": item["question_prompt"] + self.gen_prompt_suffix,
                }
            ]
            for item in batch
        ]
        return self.tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=True,  # appends <|im_start|>assistant\n
            tokenize=False,
        )

    # ──────────────────────────────────────────────────────────
    # generation back‑ends
    # ──────────────────────────────────────────────────────────
    def _gen_hf(self, batch_inputs):
        prompts = self._make_prompts(batch_inputs)
        enc     = self.tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")

        with torch.no_grad(), amp.autocast(device_type="cuda",
                                           dtype=next(self.model.parameters()).dtype):
            gen_ids = self.model.generate(**enc, **self.gen_params)

        seq_len   = enc.input_ids.shape[1]  # prompt length
        out_texts = self.tokenizer.batch_decode(
            [ids[seq_len:] for ids in gen_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return out_texts

    def _gen_vllm(self, batch_inputs):
        prompts  = self._make_prompts(batch_inputs)
        requests = [{"prompt": p} for p in prompts]
        outputs  = self.model.generate(requests, self.gen_params)

        responses = []
        for out in outputs:
            for seq in out.outputs:
                responses.append(seq.text)
        return responses

    # ──────────────────────────────────────────────────────────
    # public API
    # ──────────────────────────────────────────────────────────
    def generate_response(self, batched_input: List[Dict[str, Any]]):
        if self.gen_engine == "hf":
            return self._gen_hf(batched_input)
        elif self.gen_engine == "vllm":
            return self._gen_vllm(batched_input)

    def shutdown(self):
        if self.gen_engine == "vllm":
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel
            except ImportError:
                from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            if hasattr(self.model, "shutdown"):
                self.model.shutdown()

        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


# #!/usr/bin/env python3
# """
# Text‑only Qwen wrapper for either HF‑Transformers or vLLM back‑ends.

# Usage
# -----
# from models import qwen
# model = qwen.Qwen(args, "Qwen/Qwen1.5-7B-Chat")
# outputs = model.generate_response([{"question_prompt": "..."}])
# """

# from typing import List, Dict, Any
# import torch
# from torch import amp
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from vllm import LLM, SamplingParams


# class Qwen:
#     def __init__(self, args, model_name_path: str = "Qwen/Qwen1.5-7B-Chat"):
#         # ----- common config -----
#         self.gen_prompt_suffix = args.gen_prompt_suffix
#         self.gen_engine        = args.gen_engine.lower()
#         self.max_new_tokens    = args.max_new_tokens
#         self.top_k             = args.top_k
#         self.top_p             = args.top_p
#         self.temperature       = args.temperature
#         self.n_generations     = args.n_generations
#         self.do_sample         = self.temperature > 0

#         # ----- HF backend -----
#         if self.gen_engine == "hf":
#             self.tokenizer = AutoTokenizer.from_pretrained(model_name_path)
#             self.tokenizer.padding_side = "left"
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_name_path, torch_dtype="auto", device_map="auto"
#             )
#             self.model.eval()

#             self.gen_params = dict(
#                 num_return_sequences=self.n_generations,
#                 max_new_tokens=self.max_new_tokens,
#             )
#             if self.do_sample:
#                 self.gen_params.update(
#                     do_sample=True,
#                     top_k=self.top_k,
#                     top_p=self.top_p,
#                     temperature=self.temperature,
#                 )

#         # ----- vLLM backend -----
#         elif self.gen_engine == "vllm":
#             self.model = LLM(
#                 model=model_name_path,
#                 dtype="auto",
#                 tensor_parallel_size=torch.cuda.device_count(),
#                 trust_remote_code=True,
#                 max_model_len=2048,
#             )
#             self.gen_params = SamplingParams(
#                 n=self.n_generations,
#                 max_tokens=self.max_new_tokens,
#                 temperature=self.temperature,   # if =0 → greedy
#                 top_k=self.top_k,
#                 top_p=self.top_p,
#             )
#             # vLLM needs its own tokenizer instance for text‑only requests
#             self.tokenizer = None  # not used

#         else:
#             raise ValueError(f"Unsupported gen_engine: {self.gen_engine}")

#     # ------------------------------------------------------------------
#     # helpers
#     # ------------------------------------------------------------------
#     # def _make_prompts(self, batch: List[Dict[str, Any]]) -> List[str]:
#     #     return [
#     #         item["question_prompt"] + self.gen_prompt_suffix for item in batch
#     #     ]
#     def _make_prompts(self, batch):
#         convs = [
#             [
#                 {
#                     "role": "user",
#                     "content": item["question_prompt"] + self.gen_prompt_suffix,
#                 }
#             ]
#             for item in batch
#         ]
#         # let the tokenizer build the ChatML string
#         return self.tokenizer.apply_chat_template(
#             convs, add_generation_prompt=True, tokenize=False
#         )


#     # ------------------------------------------------------------------
#     # generation back‑ends
#     # ------------------------------------------------------------------
#     def _gen_hf(self, batch_inputs):
#         prompts = self._make_prompts(batch_inputs)
#         enc     = self.tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")

#         with torch.no_grad(), amp.autocast(device_type="cuda", dtype=next(self.model.parameters()).dtype):
#             gen_ids = self.model.generate(**enc, **self.gen_params)

#         # strip off the prompt tokens for each returned sequence
#         seq_len   = enc.input_ids.shape[1]
#         out_texts = self.tokenizer.batch_decode(
#             [ids[seq_len:] for ids in gen_ids],
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=False,
#         )
#         return out_texts

#     def _gen_vllm(self, batch_inputs):
#         prompts  = self._make_prompts(batch_inputs)
#         requests = [{"prompt": p} for p in prompts]
#         outputs  = self.model.generate(requests, self.gen_params)

#         responses = []
#         for out in outputs:
#             for seq in out.outputs:
#                 responses.append(seq.text)
#         return responses

#     # ------------------------------------------------------------------
#     # public API
#     # ------------------------------------------------------------------
#     def generate_response(self, batched_input: List[Dict[str, Any]]):
#         if self.gen_engine == "hf":
#             return self._gen_hf(batched_input)
#         elif self.gen_engine == "vllm":
#             return self._gen_vllm(batched_input)

#     def shutdown(self):
#         if self.gen_engine == "vllm":
#             try:
#                 from vllm.distributed.parallel_state import destroy_model_parallel
#             except ImportError:
#                 from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
#             destroy_model_parallel()

#             if hasattr(self.model, "shutdown"):
#                 self.model.shutdown()
