from vllm import LLM, SamplingParams
from transformers import (
    AutoProcessor,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
from torch import amp
from typing import List, Dict, Any
from PIL import Image


class KeyeVL:
    def __init__(
        self,
        args,
        model_name_path: str = "Kwai-Keye/Keye-VL-8B-Preview",
    ):
        # ============================================================
        # Processor / Tokenizer (Keye uses a combined AutoProcessor)
        # ============================================================
        self.processor = AutoProcessor.from_pretrained(model_name_path, trust_remote_code=True)

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

        # Shared generation args
        self.gen_prompt_suffix = args.gen_prompt_suffix
        self.gen_engine        = args.gen_engine.lower()
        self.max_new_tokens    = args.max_new_tokens
        self.top_k             = args.top_k
        self.top_p             = args.top_p
        self.temperature       = args.temperature
        self.n_generations     = args.n_generations
        self.do_sample         = self.temperature > 0

        # ============================================================
        # HuggingFace engine
        # ============================================================
        if self.gen_engine == "hf":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )

            self.model.eval()

            self.gen_params = dict(
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.n_generations,
                use_cache=True,
            )

            if self.do_sample:
                self.gen_params.update(
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                )

        # ============================================================
        # vLLM engine
        # ============================================================
        elif self.gen_engine == "vllm":
            self.model = LLM(
                model=model_name_path,
                trust_remote_code=True,
                dtype="auto",
                tensor_parallel_size=torch.cuda.device_count(),
                limit_mm_per_prompt={"image": 4},
                max_model_len=4096,
            )

            self.gen_params = SamplingParams(
                n=self.n_generations,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
        else:
            raise ValueError(f"Unknown gen_engine: {self.gen_engine}")

    # ================================================================
    # Build a batch of Keye chat-format conversations
    # ================================================================
    def _make_batched_conversations(self, batch):
        convs = []
        for item in batch:
            prompt = item["question_prompt"] + self.gen_prompt_suffix
            item["question_prompt"] = prompt
            images = item["decoded_images"]    # list[PIL.Image]

            content = []
            for im in images:
                content.append({"type": "image", "image": im})
            content.append({"type": "text", "text": prompt})

            convs.append([{"role": "user", "content": content}])
        return convs

    # ================================================================
    # HuggingFace Generation
    # ================================================================
    def _gen_hf(self, batched_input):
        conversations = self._make_batched_conversations(batched_input)

        # Keye uses apply_chat_template
        text_inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Processor handles both images and text directly
        images = [item["decoded_images"] for item in batched_input]

        hf_inputs = self.processor(
            text=text_inputs,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad(), amp.autocast(
            device_type="cuda",
            dtype=next(self.model.parameters()).dtype,
        ):
            output_ids = self.model.generate(**hf_inputs, **self.gen_params)

        # Trim prompt
        prompt_len = hf_inputs.input_ids.size(1)
        trimmed = []
        for i in range(len(batched_input)):
            for j in range(self.n_generations):
                idx = i * self.n_generations + j
                trimmed.append(output_ids[idx][prompt_len:])

        texts = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return texts

    # ================================================================
    # vLLM Generation
    # ================================================================
    def _gen_vllm(self, batch):
        conversations = self._make_batched_conversations(batch)

        prompts = self.processor.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
        )

        requests = []
        for p, item in zip(prompts, batch):
            requests.append({
                "prompt": p,
                "multi_modal_data": {"image": item["decoded_images"]},
            })

        outputs = self.model.generate(requests, self.gen_params)
        return [o.outputs[0].text for o in outputs]

    # ================================================================
    # API
    # ================================================================
    def generate_response(self, batched_input):
        if self.gen_engine == "hf":
            return self._gen_hf(batched_input)
        else:
            return self._gen_vllm(batched_input)

    # ================================================================
    # Cleanup
    # ================================================================
    def shutdown(self):
        if self.gen_engine == "vllm":
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel
            except ImportError:
                from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

            destroy_model_parallel()

            if hasattr(self.model, "shutdown"):
                self.model.shutdown()
