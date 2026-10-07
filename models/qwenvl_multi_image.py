from vllm import LLM, SamplingParams
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,   # NEW
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
import torch
from typing import List, Dict, Any
from PIL import Image
from torch import amp


class QwenVL:
    def __init__(
        self,
        args,
        model_name_path: str = "Qwen/Qwen2-VL-2B-Instruct",
    ):
        # ---- Processor / tokenizer ----
        self.processor = AutoProcessor.from_pretrained(model_name_path)

        # Some models give you a Processor with .tokenizer,
        # others (older setups) might give you a bare tokenizer.
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

        # shared
        self.gen_prompt_suffix = args.gen_prompt_suffix
        self.gen_engine        = args.gen_engine.lower()
        self.max_new_tokens    = args.max_new_tokens
        self.top_k             = args.top_k
        self.top_p             = args.top_p
        self.temperature       = args.temperature
        self.n_generations     = args.n_generations
        self.do_sample         = self.temperature > 0

        if self.gen_engine == "hf":
            lower_name = model_name_path.lower()

            if "qwen2-vl" in lower_name:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name_path,
                    torch_dtype="auto",
                    device_map="auto",
                )
            elif "qwen2.5-vl" in lower_name:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name_path,
                    torch_dtype="auto",
                    device_map="auto",
                )
            elif "qwen3-vl" in lower_name:
                # NEW: Qwen3-VL HF support
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name_path,
                    torch_dtype="auto",
                    device_map="auto",
                )
            else:
                raise ValueError(f"Unsupported HF Qwen model: {model_name_path}")

            self.model.eval()

            # gen params
            self.gen_params = dict(
                num_return_sequences=self.n_generations,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )

            if self.do_sample:
                self.gen_params.update(
                    do_sample=True,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    temperature=self.temperature,
                )

        elif self.gen_engine == "vllm":
            # vLLM already supports Qwen2 / 2.5 / 3 VL as long as the model
            # name is supported in your vLLM version.
            self.model = LLM(
                model=model_name_path,
                dtype="auto",
                limit_mm_per_prompt={"image": 2},
                tensor_parallel_size=torch.cuda.device_count(),
                trust_remote_code=True,
                max_model_len=16384,
            )

            self.gen_params = SamplingParams(
                n=self.n_generations,
                #max_tokens=self.max_new_tokens,
                #temperature=self.temperature,  # temp=0 => greedy in vLLM
                max_tokens=16384,
                temperature=0.6,
                top_p=self.top_p,
                top_k=self.top_k,
            )
        else:
            raise ValueError(f"Unknown gen_engine: {self.gen_engine}")

    def _make_batched_conversations(self, batch):
        convs = []
        for item in batch:
            prompt = item["question_prompt"] + self.gen_prompt_suffix
            imgs   = item["decoded_images"]  # list[Image]

            content = [{"type": "image", "image": im} for im in imgs]
            content.append({"type": "text", "text": prompt})

            convs.append([{"role": "user", "content": content}])
        return convs

    def _gen_hugginface(self, batched_input):
        # prep data
        batched_convs = self._make_batched_conversations(batched_input)
        text_inputs = self.processor.apply_chat_template(
            batched_convs,
            tokenize=False,              # use the HF-style arg name
            add_generation_prompt=True,
        )
        img_inputs, vid_inputs = process_vision_info(batched_convs)

        hf_inputs = self.processor(
            text=text_inputs,
            images=img_inputs,
            videos=vid_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # Qwen3-VL examples usually drop token_type_ids if present
        hf_inputs.pop("token_type_ids", None)

        # generate
        with torch.no_grad(), amp.autocast(
            device_type="cuda",
            dtype=next(self.model.parameters()).dtype,
        ):
            gen_ids = self.model.generate(**hf_inputs, **self.gen_params)

        # trim off prompt tokens
        batch_size = hf_inputs.input_ids.size(0)
        seq_len    = hf_inputs.input_ids.size(1)
        trimmed = []
        for i in range(batch_size):
            for j in range(self.n_generations):
                idx = i * self.n_generations + j
                trimmed.append(gen_ids[idx][seq_len:])

        # decode
        batched_output = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return batched_output

    def _gen_vllm(self, batch):
        convs   = self._make_batched_conversations(batch)
        prompts = self.processor.apply_chat_template(
            convs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        print(f"[DEBUG] Prompt preview: {prompts[0][:500]}")

        # don't process_vision_info here; just pass PILs directly
        images_per_conv = [item["decoded_images"] for item in batch]

        requests = [
            {
                "prompt": p,
                "multi_modal_data": {"image": imgs},
            }
            for p, imgs in zip(prompts, images_per_conv)
        ]

        outs = self.model.generate(requests, self.gen_params)
        return [o.outputs[0].text for o in outs]

    def generate_response(self, batched_input: List[Dict[str, Any]]):
        if self.gen_engine == "hf":
            return self._gen_hugginface(batched_input)
        elif self.gen_engine == "vllm":
            return self._gen_vllm(batched_input)

    def shutdown(self):
        if self.gen_engine == "vllm":
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel
            except ImportError:
                from vllm.model_executor.parallel_utils.parallel_state import (
                    destroy_model_parallel,
                )

            destroy_model_parallel()

            if hasattr(self.model, "shutdown"):
                self.model.shutdown()

        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
