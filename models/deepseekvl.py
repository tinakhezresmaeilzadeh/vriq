# # models/deepseekvl.py

# import torch
# from torch import amp
# from typing import List, Dict, Any

# from transformers import AutoModelForCausalLM
# from deepseek_vl2.models import DeepseekVLV2Processor


# class DeepSeekVL2:
#     """
#     Wrapper for DeepSeek-VL2 models (e.g. deepseek-ai/deepseek-vl2, -small, -tiny).

#     - Uses the official DeepseekVLV2Processor + AutoModelForCausalLM (trust_remote_code=True).
#     - Supports multiple images per sample.
#     - Follows the same interface as your QwenVL wrapper.
#     - If gen_engine='vllm' is requested, we currently fall back to HF (see __init__).
#     """

#     def __init__(
#         self,
#         args,
#         model_name_path: str = "deepseek-ai/deepseek-vl2",
#     ):
#         # ===== shared settings (mirroring your Qwen class) =====
#         self.gen_prompt_suffix = args.gen_prompt_suffix
#         self.gen_engine = args.gen_engine.lower()
#         self.max_new_tokens = args.max_new_tokens
#         self.top_k = args.top_k
#         self.top_p = args.top_p
#         self.temperature = args.temperature
#         self.n_generations = args.n_generations
#         self.do_sample = self.temperature > 0
#         self.debug = getattr(args, "debug", False)

#         # DeepSeek-VL2 has vLLM support in recent vLLM versions, but this
#         # wrapper currently only uses HF. If vllm is requested, fall back.
#         if self.gen_engine == "vllm":
#             print(
#                 "\n[WARNING] DeepSeek-VL2 vLLM path is not wired in this wrapper yet. "
#                 "Falling back to HuggingFace backend (gen_engine='hf').\n"
#             )
#             self.gen_engine = "hf"
#         elif self.gen_engine != "hf":
#             raise ValueError(
#                 f"DeepSeek-VL2 wrapper only supports gen_engine 'hf' (or 'vllm' "
#                 f"which falls back to hf), but got: {self.gen_engine}"
#             )

#         # ===== Processor & tokenizer =====
#         # This processor is responsible for:
#         #   - adding pad token <｜▁pad▁｜> if missing
#         #   - adding <image>, grounding tokens, chat tokens (<|User|>, <|Assistant|>), etc.
#         #   - formatting messages into SFT format
#         self.processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(
#             model_name_path
#         )
#         self.tokenizer = self.processor.tokenizer

#         # ===== Model (HF, remote code) =====
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name_path,
#             trust_remote_code=True,
#         )
#         # DeepSeek’s example: bfloat16 + CUDA + eval
#         self.model = self.model.to(torch.bfloat16).cuda().eval()

#         # ===== Generation parameters =====
#         self.gen_params = dict(
#             max_new_tokens=self.max_new_tokens,
#             num_return_sequences=self.n_generations,
#             use_cache=True,
#         )

#         if self.do_sample:
#             self.gen_params.update(
#                 do_sample=True,
#                 top_k=self.top_k,
#                 top_p=self.top_p,
#                 temperature=self.temperature,
#             )
#         else:
#             self.gen_params.update(do_sample=False)

#     # ======================================================================
#     # Conversation builder
#     # ======================================================================
#     def _make_conversations(
#         self, batch: List[Dict[str, Any]]
#     ):
#         """
#         Build DeepSeek-VL2-style conversations and corresponding images list.

#         Each item in `batch` is expected to contain:
#             - "question_prompt": str
#             - "decoded_images": list[PIL.Image.Image]

#         Returns:
#             all_convs:  list[list[{"role": str, "content": str}]]
#             all_images: list[list[PIL.Image.Image]]
#         """
#         all_convs = []
#         all_images = []

#         for item in batch:
#             prompt = item["question_prompt"] + self.gen_prompt_suffix
#             imgs = item["decoded_images"]  # list of PIL.Image

#             if len(imgs) == 0:
#                 # text-only
#                 content = prompt
#             elif len(imgs) == 1:
#                 # single image: official pattern uses `<image>` prefix
#                 content = "<image>\n" + prompt
#             else:
#                 # multi-image: use <image_placeholder> once per image
#                 # (DeepSeek-VL2 processor uses these to align images to text)
#                 placeholders = "".join(["<image_placeholder>"] * len(imgs))
#                 content = f"{placeholders}\n{prompt}"

#             conv = [
#                 {
#                     "role": "<|User|>",
#                     "content": content,
#                 },
#                 {
#                     "role": "<|Assistant|>",
#                     "content": "",
#                 },
#             ]

#             all_convs.append(conv)
#             all_images.append(imgs)

#         return all_convs, all_images

#     # ======================================================================
#     # HF generation (process one example at a time, matching official doc)
#     # ======================================================================
#     def _gen_hf(self, batched_input: List[Dict[str, Any]]):
#         """
#         batched_input: list of items with keys:
#             - 'question_prompt'
#             - 'decoded_images' (list[PIL.Image.Image])

#         Returns:
#             list of strings of length batch_size * n_generations
#         """
#         all_outputs: List[str] = []

#         for idx, item in enumerate(batched_input):
#             # 1) Build a single conversation + images
#             conversations, images = self._make_conversations([item])
#             conversation = conversations[0]   # list[{"role","content"}]
#             imgs = images[0]                  # list[PIL.Image]

#             # 2) Processor call (matches official DeepSeek-VL2 example)
#             prepare_inputs = self.processor(
#                 conversations=conversation,
#                 images=imgs,
#                 force_batchify=True,
#                 system_prompt="",    # customize if needed
#             ).to(self.model.device)

#             # 3) Multimodal encoding to get input embeddings
#             inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

#             # Some DeepSeek versions expose `.language_model`, others `.language`;
#             # fall back to whole model if neither is present.
#             lm = getattr(self.model, "language_model", None)
#             if lm is None:
#                 lm = getattr(self.model, "language", None)
#             if lm is None:
#                 lm = self.model

#             # 4) Generate (NO manual trimming here — we decode full seq)
#             with torch.no_grad(), amp.autocast(
#                 device_type="cuda",
#                 dtype=next(self.model.parameters()).dtype,
#             ):
#                 gen_ids = lm.generate(
#                     inputs_embeds=inputs_embeds,
#                     attention_mask=prepare_inputs.attention_mask,
#                     pad_token_id=self.tokenizer.eos_token_id,
#                     bos_token_id=self.tokenizer.bos_token_id,
#                     eos_token_id=self.tokenizer.eos_token_id,
#                     **self.gen_params,
#                 )

#             # 5) Decode the full sequences, just like the official demo
#             texts = self.tokenizer.batch_decode(
#                 gen_ids,
#                 skip_special_tokens=True,
#                 clean_up_tokenization_spaces=False,
#             )

#             # Optional debug: show the formatted prompt + first decoded answer
#             if self.debug and idx == 0:
#                 try:
#                     sft = prepare_inputs["sft_format"][0]
#                     print("\n[DeepSeek-VL2 DEBUG] SFT format:\n", sft)
#                 except Exception:
#                     pass
#                 print("[DeepSeek-VL2 DEBUG] Decoded outputs:")
#                 for t_i, t in enumerate(texts):
#                     print(f"  sample {t_i}: {repr(t[:400])}")  # truncate view

#             # Flatten into a single list across num_return_sequences
#             all_outputs.extend(texts)

#         return all_outputs

#     # ======================================================================
#     # Public interface
#     # ======================================================================
#     def generate_response(self, batched_input: List[Dict[str, Any]]):
#         """
#         Main entry point, matching your QwenVL class.
#         """
#         if self.gen_engine == "hf":
#             return self._gen_hf(batched_input)
#         else:
#             # Should not happen because we fall back to hf in __init__,
#             # but keep this guard in case.
#             raise ValueError(
#                 f"DeepSeek-VL2: unsupported gen_engine={self.gen_engine}"
#             )

#     def shutdown(self):
#         """
#         No special shutdown required for pure HF.
#         """
#         pass

# models/deepseekvl.py

import torch
from torch import amp
from typing import List, Dict, Any

from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor


class DeepSeekVL2:
    """
    Wrapper for DeepSeek-VL2 models (e.g. deepseek-ai/deepseek-vl2, -small, -tiny).

    - Uses the official DeepseekVLV2Processor + AutoModelForCausalLM (trust_remote_code=True).
    - Supports multiple images per sample.
    - Follows the same interface as your QwenVL wrapper.
    - Currently only supports HuggingFace backend (gen_engine='hf').
      If 'vllm' is requested, we fall back to HF with a warning.
    """

    def __init__(
        self,
        args,
        model_name_path: str = "deepseek-ai/deepseek-vl2",
    ):
        # ===== shared settings (mirroring your Qwen class) =====
        self.gen_prompt_suffix = args.gen_prompt_suffix
        self.gen_engine = args.gen_engine.lower()
        self.max_new_tokens = args.max_new_tokens
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.temperature = args.temperature
        self.n_generations = args.n_generations
        self.do_sample = self.temperature > 0
        self.debug = bool(getattr(args, "debug", False))

        # vLLM not wired yet -> fall back to HF
        if self.gen_engine == "vllm":
            print(
                "\n[WARNING] DeepSeek-VL2 is not wired for vLLM in this wrapper. "
                "Falling back to HuggingFace backend (gen_engine='hf').\n"
            )
            self.gen_engine = "hf"
        elif self.gen_engine != "hf":
            raise ValueError(
                f"DeepSeek-VL2 wrapper only supports gen_engine 'hf' or 'vllm' (fallback to hf), "
                f"but got: {self.gen_engine}"
            )

        # ===== Processor & tokenizer =====
        # Processor takes care of:
        #   - adding pad/image/grounding/chat tokens if missing
        #   - formatting messages into SFT format
        self.processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(
            model_name_path
        )
        self.tokenizer = self.processor.tokenizer

        # ===== Model (HF, remote code) =====
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_path,
            trust_remote_code=True,
        )
        # DeepSeek’s example: bfloat16 + CUDA + eval
        self.model = self.model.to(torch.bfloat16).cuda().eval()

        # ===== Generation parameters (base) =====
        # We'll add max_length / max_new_tokens per-example after we know prompt length.
        self.base_gen_params = dict(
            num_return_sequences=self.n_generations,
            use_cache=True,
        )

        if self.do_sample:
            self.base_gen_params.update(
                do_sample=True,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
            )
        else:
            self.base_gen_params.update(do_sample=False)

    # ======================================================================
    # Conversation builder
    # ======================================================================
    def _make_conversations(
        self, batch: List[Dict[str, Any]]
    ):
        """
        Build DeepSeek-VL2-style conversations and corresponding images list.

        Each item in `batch` is expected to contain:
            - "question_prompt": str
            - "decoded_images": list[PIL.Image.Image]

        Returns:
            all_convs:  list[list[{"role": str, "content": str}]]
            all_images: list[list[PIL.Image.Image]]
        """
        all_convs = []
        all_images = []

        for item in batch:
            prompt = item["question_prompt"] + self.gen_prompt_suffix
            imgs = item["decoded_images"]  # list of PIL.Image

            if len(imgs) == 0:
                # text-only
                content = prompt
            elif len(imgs) == 1:
                # single image: official pattern uses `<image>` prefix
                content = "<image>\n" + prompt
            else:
                # multi-image: use <image_placeholder> once per image
                placeholders = "".join(["<image_placeholder>"] * len(imgs))
                content = f"{placeholders}\n{prompt}"

            conv = [
                {
                    "role": "<|User|>",
                    "content": content,
                },
                {
                    "role": "<|Assistant|>",
                    "content": "",
                },
            ]

            all_convs.append(conv)
            all_images.append(imgs)

        return all_convs, all_images

    # ======================================================================
    # HF generation (process one example at a time, matching official doc)
    # ======================================================================
    def _gen_hf(self, batched_input: List[Dict[str, Any]]):
        """
        batched_input: list of items with keys:
            - 'question_prompt'
            - 'decoded_images' (list[PIL.Image.Image])

        Returns:
            list of strings of length batch_size * n_generations
        """
        all_outputs: List[str] = []

        for idx, item in enumerate(batched_input):
            # 1) Build a single conversation + images
            conversations, images = self._make_conversations([item])
            conversation = conversations[0]   # list[{"role","content"}]
            imgs = images[0]                  # list[PIL.Image]

            # 2) Processor call -> prepare inputs (includes input_ids, attention_mask, etc.)
            prepare_inputs = self.processor(
                conversations=conversation,
                images=imgs,
                force_batchify=True,
                system_prompt="",    # customize if needed
            ).to(self.model.device)

            # SFT format for debugging
            if self.debug:
                print("\n[DeepSeek-VL2 DEBUG] SFT format:\n", prepare_inputs["sft_format"][0], "\n")

            # 3) Multimodal encoding to get input embeddings
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

            # Base LM: some versions have `.language_model`, others `.language`.
            lm = getattr(self.model, "language_model", None)
            if lm is None:
                lm = getattr(self.model, "language", None)
            if lm is None:
                lm = self.model

            # 4) Per-example generation params
            prompt_len = prepare_inputs.input_ids.shape[1]
            gen_params = dict(self.base_gen_params)
            # Explicitly set both max_new_tokens and max_length to avoid default 20-token cap.
            gen_params["max_new_tokens"] = self.max_new_tokens
            gen_params["max_length"] = prompt_len + self.max_new_tokens

            # 5) Generate
            with torch.no_grad(), amp.autocast(
                device_type="cuda",
                dtype=next(self.model.parameters()).dtype,
            ):
                gen_ids = lm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **gen_params,
                )

            # We *do not* trim off the prompt here. DeepSeek’s own demo decodes
            # the full sequence. If gen_ids already exclude the prompt,
            # this keeps everything; if they include it, extraction still works.
            decoded_texts = self.tokenizer.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            if self.debug:
                for j, t in enumerate(decoded_texts):
                    print(
                        f"[DeepSeek-VL2 DEBUG] sample {idx}, seq {j}: "
                        f"len(tokens)={gen_ids[j].shape[0]}, text[:400]={repr(t[:400])}"
                    )

            all_outputs.extend(decoded_texts)

        return all_outputs

    # ======================================================================
    # Public interface
    # ======================================================================
    def generate_response(self, batched_input: List[Dict[str, Any]]):
        """
        Main entry point, matching your QwenVL class.
        """
        if self.gen_engine == "hf":
            return self._gen_hf(batched_input)
        else:
            # Should not happen because we fall back to hf in __init__,
            # but keep this guard in case.
            raise ValueError(
                f"DeepSeek-VL2: unsupported gen_engine={self.gen_engine}"
            )

    def shutdown(self):
        """
        No special shutdown required for pure HF.
        """
        pass
