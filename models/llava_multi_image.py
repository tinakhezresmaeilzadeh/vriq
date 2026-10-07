from vllm import LLM, SamplingParams
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
import torch
from typing import List, Dict, Any
from PIL import Image
from torch import amp
#supported models: HF: llava-hf/llava-v1.6-mistral-7b-hf, llava-hf/llava-v1.6-vicuna-7b-hf
#supported models: vLLM: llava-hf/llava-v1.6-vicuna-7b-hf

class LLaVA:
    """
    Unified wrapper that lets you run LLaVA (1.5 / 1.6 / NeXT) either
    with Hugging Face generation or vLLM’s multi‑modal backend.

    The public entry‑point is `generate_response(batched_input)`,
    where `batched_input` is a list of dicts, each dict containing

        {
            "question_prompt": <str>,   # user text
            "decoded_image":   <PIL.Image.Image>  # the associated image
        }

    Parameters expected in `args` (CLI / dataclass, etc.):

    • gen_engine          : "hf"  or  "vllm"
    • gen_prompt_suffix   : text appended after each question prompt
    • max_new_tokens      : int
    • top_k, top_p        : usual sampling knobs
    • temperature         : float
    • n_generations       : number of completions to return
    """
    def __init__(
        self,
        args,
        model_name_path: str = "llava-hf/llava-v1.6-mistral-7b-hf",
    ):
        # ── shared ────────────────────────────────────────────────────────────
        self.processor = LlavaNextProcessor.from_pretrained(model_name_path)
        self.processor.tokenizer.padding_side = "left"
        self.gen_prompt_suffix = args.gen_prompt_suffix
        self.gen_engine        = args.gen_engine.lower()
        self.max_new_tokens    = args.max_new_tokens
        self.top_k             = args.top_k
        self.top_p             = args.top_p
        self.temperature       = args.temperature
        self.n_generations     = args.n_generations
        self.do_sample         = self.temperature > 0

        # ── backend selection ────────────────────────────────────────────────
        if self.gen_engine == "hf":
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name_path,
                torch_dtype="auto",
                device_map="auto",
            ).eval()

            tok = self.processor.tokenizer
            if tok.pad_token_id is None:            # many LLaVA checkpoints ship without one
                tok.pad_token = tok.eos_token       # reuse the EOS token as PAD

            self.gen_params = dict(
                num_return_sequences=self.n_generations,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=tok.pad_token_id, 
            )

            if self.do_sample:                    
                self.gen_params.update(
                    do_sample=True,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    temperature=self.temperature,
                )

        elif self.gen_engine == "vllm":
            self.model = LLM(
                model=model_name_path,
                dtype="auto",
                tensor_parallel_size=torch.cuda.device_count(),
                limit_mm_per_prompt={"image": 2},
                trust_remote_code=True,
                max_model_len=4096
            )

            self.gen_params = SamplingParams(
                n=self.n_generations,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature, #vllm greedy must have temp = 0, when temp=0, top_p and top_k are ignored
                top_p=self.top_p,
                top_k=self.top_k,
            )
        else:
            raise ValueError(f"Unknown gen_engine {self.gen_engine!r}")
        
    def _make_batched_conversations(self, batch):
        convs = []
        for item in batch:
            prompt = item["question_prompt"] + self.gen_prompt_suffix
            imgs   = item["decoded_images"]            # now a list[Image]

            content = [{"type": "image", "image": im} for im in imgs]
            content.append({"type": "text", "text": prompt})

            convs.append([{"role": "user", "content": content}])
        return convs

    # ── Hugging Face path ───────────────────────────────────────────────────
    @torch.inference_mode()
    def _gen_hf(self, batched_input: List[Dict[str, Any]]) -> List[str]:
        convs = self._make_batched_conversations(batched_input)
        text_inputs = self.processor.apply_chat_template(
            convs, tokenizer=False, add_generation_prompt=True
        )

        # processor handles image preprocessing internally
        model_inputs = self.processor(
            text=text_inputs,
            images=[c[0]["content"][0]["image"] for c in convs],
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # generate
        with torch.no_grad(), amp.autocast(device_type="cuda", dtype=next(self.model.parameters()).dtype):
            gen_ids = self.model.generate(**model_inputs, **self.gen_params)

        # strip prompt tokens for each returned sequence
        seq_len   = model_inputs.input_ids.size(1)
        trimmed   = [ids[seq_len:] for ids in gen_ids]
        responses = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return responses

    # ── vLLM path ────────────────────────────────────────────────────────────
    def _gen_vllm(self, batch):
        convs   = self._make_batched_conversations(batch)
        prompts = self.processor.apply_chat_template(
                    convs, tokenize=False, add_generation_prompt=True)

        # 1. **DON’T** run process_vision_info here – it mangles the list
        images_per_conv = [item["decoded_images"] for item in batch]

        # 2. Pass the raw PIL list straight to vLLM
        requests = [{"prompt": p,
                    "multi_modal_data": {"image": imgs}}
                    for p, imgs in zip(prompts, images_per_conv)]

        outs = self.model.generate(requests, self.gen_params)
        return [o.outputs[0].text for o in outs]

    # ── public API ───────────────────────────────────────────────────────────
    def generate_response(self, batched_input: List[Dict[str, Any]]) -> List[str]:
        if self.gen_engine == "hf":
            return self._gen_hf(batched_input)
        if self.gen_engine == "vllm":
            return self._gen_vllm(batched_input)
        raise RuntimeError("Unreachable")

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