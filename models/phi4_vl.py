# models/phi4_vl.py

import torch
from torch import amp
from typing import List, Dict, Any

from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig


class Phi4Multimodal:
    """
    Wrapper for microsoft/Phi-4-multimodal-instruct.

    - Uses AutoProcessor + AutoModelForCausalLM with trust_remote_code=True.
    - Supports multiple images per example (list of PIL Images in `decoded_images`).
    - Matches the interface of your other wrappers (QwenVL, DeepSeekVL2, etc.).
    """

    def __init__(
        self,
        args,
        model_name_path: str = "microsoft/Phi-4-multimodal-instruct",
    ):
        # ===== shared args (same as your Qwen class) =====
        # self.gen_prompt_suffix = args.gen_prompt_suffix
        self.gen_prompt_suffix = " \n Please provide your thinking process before you answer"
        self.gen_engine        = args.gen_engine.lower()
        self.max_new_tokens    = args.max_new_tokens
        self.top_k             = args.top_k
        self.top_p             = args.top_p
        self.temperature       = args.temperature
        self.n_generations     = args.n_generations
        self.do_sample         = self.temperature > 0

        if self.gen_engine != "hf":
            # vLLM has its own Phi-4-MM integration, but wiring multimodal correctly
            # there is a separate project; for now we keep this wrapper HF-only.
            raise ValueError(
                f"Phi4Multimodal currently supports only gen_engine='hf', "
                f"but got: {self.gen_engine}"
            )

        # ===== Processor & tokenizer =====
        # This will load Phi4MMProcessor from processing_phi4mm.py
        # and the underlying tokenizer / image / audio processors.
        self.processor = AutoProcessor.from_pretrained(
            model_name_path,
            trust_remote_code=True,
        )

        # Try to grab tokenizer from the processor (Phi4MMProcessor exposes it)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is not None:
            # Left padding / truncation for batched generation
            if hasattr(self.tokenizer, "padding_side"):
                self.tokenizer.padding_side = "left"
            if hasattr(self.tokenizer, "truncation_side"):
                self.tokenizer.truncation_side = "left"

        # ===== Model (HF, remote code) =====
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            # you can pass _attn_implementation here if you want flash-attn
            # _attn_implementation="flash_attention_2",
        )
        self.model.eval()

        # Optionally load GenerationConfig from the repo (as in the official example)
        try:
            self.generation_config = GenerationConfig.from_pretrained(
                model_name_path
            )
        except Exception:
            self.generation_config = None

        # ===== Generation parameters =====
        # We mimic the example but still respect your args.
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
        else:
            self.gen_params.update(do_sample=False)

        # If there is a GenerationConfig, let it control things that you didn't
        # override explicitly.
        if self.generation_config is not None:
            self.gen_params.setdefault("generation_config", self.generation_config)

    # ======================================================================
    # Prompt builder following the official model card
    # ======================================================================
    @staticmethod
    def _build_prompt_core(
        question: str,
        num_images: int = 0,
        has_audio: bool = False,
        suffix: str = "",
    ) -> str:
        """
        Build a prompt string like:

            <|user|><|image_1|>...question...<|end|><|assistant|>

        or, for pure text:

            <|user|>question<|end|><|assistant|>
        """
        user_tok      = "<|user|>"
        assistant_tok = "<|assistant|>"
        end_tok       = "<|end|>"

        # image placeholders: <|image_1|><|image_2|>...
        image_part = "".join(f"<|image_{i+1}|>" for i in range(num_images))

        # audio placeholder if needed (you probably don't use audio in TIR)
        audio_part = "<|audio_1|>" if has_audio else ""

        body = question + suffix
        # body = "please describe this image"

        return f"{user_tok}{image_part}{audio_part}{body}{end_tok}{assistant_tok}"

    # ======================================================================
    # Generation (HF) – we process one sample at a time to keep multimodal
    # handling simple and robust.
    # ======================================================================
    def _gen_hf(self, batched_input: List[Dict[str, Any]]) -> List[str]:
        """
        Each element of `batched_input` should have:
            - "question_prompt": str
            - "decoded_images": list[PIL.Image.Image] (possibly empty / absent)

        Returns:
            A flat list of strings of length = batch_size * n_generations.
        """
        all_outputs: List[str] = []

        for item in batched_input:
            question = item["question_prompt"]
            imgs     = item.get("decoded_images", None) or []
            num_imgs = len(imgs)

            # For TIR you don't use audio, so we set has_audio=False.
            prompt = self._build_prompt_core(
                question=question,
                num_images=num_imgs,
                has_audio=False,
                suffix=self.gen_prompt_suffix,
            )

            # Build inputs as in the official example
            if num_imgs == 0:
                inputs = self.processor(
                    text=prompt,
                    return_tensors="pt",
                )
            elif num_imgs == 1:
                inputs = self.processor(
                    text=prompt,
                    images=imgs[0],
                    return_tensors="pt",
                )
            else:
                # Multiple images are supported: pass the list directly.
                inputs = self.processor(
                    text=prompt,
                    images=imgs,
                    return_tensors="pt",
                )

            inputs = inputs.to(self.model.device)

            # Generate
            with torch.no_grad(), amp.autocast(
                device_type="cuda",
                dtype=next(self.model.parameters()).dtype,
            ):
                gen_ids = self.model.generate(
                    **inputs,
                    **self.gen_params,
                )

            # Trim off the prompt tokens so we keep only new tokens
            # Shape: (num_return_sequences, total_seq_len)
            input_len = inputs["input_ids"].shape[1]
            num_seqs  = gen_ids.shape[0]

            trimmed_ids = [
                gen_ids[i][input_len:] for i in range(num_seqs)
            ]

            # Decode
            if self.tokenizer is not None:
                texts = self.tokenizer.batch_decode(
                    trimmed_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            else:
                texts = self.processor.batch_decode(
                    trimmed_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

            all_outputs.extend(texts)

        return all_outputs

    # ======================================================================
    # Public interface
    # ======================================================================
    def generate_response(self, batched_input: List[Dict[str, Any]]) -> List[str]:
        if self.gen_engine == "hf":
            return self._gen_hf(batched_input)
        else:
            # Should not happen because we guard in __init__
            raise ValueError(f"Unsupported gen_engine for Phi4Multimodal: {self.gen_engine}")

    def shutdown(self):
        # Nothing special for pure HF
        pass
