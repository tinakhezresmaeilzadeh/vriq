from vllm import LLM, SamplingParams
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoProcessor
import torch
from typing import List, Dict, Any

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


class InternVL:
    """
    Wrapper for InternVL2/3 family with:
      - HF path using model.chat() (InternVL2 official)
      - vLLM path for models that vLLM supports
      - InternVL2 official tiling preprocessing
      - Transformers>=4.50 compatibility patches

    args:
      - gen_engine: "hf" | "vllm"
      - gen_prompt_suffix: str
      - max_new_tokens: int
      - top_k: int
      - top_p: float
      - temperature: float (0 => greedy)
      - n_generations: int
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, args, model_name_path: str = "OpenGVLab/InternVL2-26B"):
        self.model_name_path = model_name_path
        self.image_tok = "<image>"

        # --- args ---
        self.gen_prompt_suffix = getattr(args, "gen_prompt_suffix", "")
        self.gen_engine = getattr(args, "gen_engine", "hf").lower()
        self.max_new_tokens = int(getattr(args, "max_new_tokens", 256))
        self.top_k = int(getattr(args, "top_k", 0))
        self.top_p = float(getattr(args, "top_p", 1.0))
        self.temperature = float(getattr(args, "temperature", 0.0))
        self.n_generations = int(getattr(args, "n_generations", 1))
        self.do_sample = self.temperature > 0

        # ---- Tokenizer: explicit for InternVL2 ----
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_path,
            trust_remote_code=True,
            use_fast=False,
        )
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"
        if getattr(self.tokenizer, "pad_token_id", None) is None and hasattr(self.tokenizer, "eos_token"):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- Optional processor (only for vLLM path) ----
        try:
            self.processor = AutoProcessor.from_pretrained(model_name_path, trust_remote_code=True)
        except Exception:
            self.processor = None

        # --------------------
        # Engine: HF
        # --------------------
        if self.gen_engine == "hf":
            _ = AutoConfig.from_pretrained(model_name_path, trust_remote_code=True)

            self.model = AutoModel.from_pretrained(
                model_name_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            ).eval()

            # Patch InternVL2/InternLM2 generation issues for transformers>=4.50
            self._patch_language_model_generate_and_config()

            # IMPORTANT:
            # 1) DO NOT put `use_cache` in this dict. InternVL2 remote code already forwards
            #    `use_cache` to language_model.generate(), and adding it here causes:
            #    "got multiple values for keyword argument 'use_cache'".
            # 2) We disable cache globally on the language model (done in patch function).
            self.generation_config = dict(
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                top_p=self.top_p,
                top_k=self.top_k,
            )
            if self.do_sample:
                self.generation_config["temperature"] = self.temperature

            # InternVL2 official preprocessing defaults
            self.input_size = 448
            self.max_num_tiles = 12
            self.use_thumbnail = True
            self._transform = self._build_transform(self.input_size)

        # --------------------
        # Engine: vLLM
        # --------------------
        elif self.gen_engine == "vllm":
            if self.processor is None or not hasattr(self.processor, "apply_chat_template"):
                raise RuntimeError(
                    "vLLM path requires AutoProcessor with apply_chat_template. "
                    "Your repo didn't provide one. Use HF path for InternVL2-26B."
                )

            self.model = LLM(
                model=model_name_path,
                dtype="auto",
                limit_mm_per_prompt={"image": 1},
                tensor_parallel_size=max(1, torch.cuda.device_count()),
                trust_remote_code=True,
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

    # ---------------------------------------------------------------------
    # Transformers>=4.50 compatibility:
    # - ensure InternVL2 remote code can call self.language_model.generate(...)
    # - ensure generation_config exists (not None)
    # - disable KV cache globally to avoid past_key_values None issues
    # ---------------------------------------------------------------------
    def _patch_language_model_generate_and_config(self):
        lm = getattr(self.model, "language_model", None)
        if lm is None:
            return

        # 1) Ensure .generate exists (GenerationMixin)
        if not hasattr(lm, "generate"):
            from transformers.generation.utils import GenerationMixin
            lm.__class__ = type(lm.__class__.__name__, (lm.__class__, GenerationMixin), {})

        # 2) Ensure generation_config exists
        if getattr(lm, "generation_config", None) is None:
            from transformers import GenerationConfig
            lm.generation_config = GenerationConfig.from_model_config(lm.config)

        # 3) Some remote code expects model.generation_config too
        if getattr(self.model, "generation_config", None) is None:
            try:
                self.model.generation_config = lm.generation_config
            except Exception:
                pass

        # 4) Disable cache globally (DO NOT pass use_cache via chat() dict)
        try:
            lm.config.use_cache = False
        except Exception:
            pass
        try:
            if getattr(lm, "generation_config", None) is not None:
                lm.generation_config.use_cache = False
        except Exception:
            pass

    # -------------------------
    # helpers
    # -------------------------
    def _get_model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _get_model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    # ---- InternVL2 official preprocessing (ported to PIL input) ----
    def _build_transform(self, input_size: int):
        mean, std = self.IMAGENET_MEAN, self.IMAGENET_STD
        return T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width = int(image_size * target_aspect_ratio[0])
        target_height = int(image_size * target_aspect_ratio[1])
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        tiles_per_row = target_width // image_size

        for i in range(blocks):
            box = (
                (i % tiles_per_row) * image_size,
                (i // tiles_per_row) * image_size,
                ((i % tiles_per_row) + 1) * image_size,
                ((i // tiles_per_row) + 1) * image_size,
            )
            processed_images.append(resized_img.crop(box))

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)

        return processed_images

    def _pil_to_pixel_values(self, pil_image) -> torch.Tensor:
        device = self._get_model_device()
        dtype = self._get_model_dtype()

        tiles = self._dynamic_preprocess(
            pil_image,
            image_size=self.input_size,
            use_thumbnail=self.use_thumbnail,
            max_num=self.max_num_tiles,
        )
        pixel_values = [self._transform(t) for t in tiles]
        pixel_values = torch.stack(pixel_values, dim=0)  # [N,3,H,W]
        return pixel_values.to(device=device, dtype=dtype)

    def _make_convs_text_only(self, batch: List[Dict[str, Any]]):
        convs = []
        for item in batch:
            prompt = item["question_prompt"] + self.gen_prompt_suffix
            convs.append([{"role": "user", "content": f"{self.image_tok}\n{prompt}"}])
        return convs

    # -------------------------
    # HF generation (InternVL2 via chat())
    # -------------------------
    def _gen_hf(self, batch: List[Dict[str, Any]]):
        if not hasattr(self.model, "chat"):
            raise RuntimeError("Loaded InternVL model does not expose model.chat().")

        # If sampling and want multiple generations, rerun chat() multiple times.
        runs = self.n_generations if (self.do_sample and self.n_generations > 1) else 1

        outputs: List[str] = []
        for _ in range(runs):
            for item in batch:
                prompt = item["question_prompt"] + self.gen_prompt_suffix
                pil = item["decoded_image"]  # PIL.Image

                question = f"{self.image_tok}\n{prompt}"
                pixel_values = self._pil_to_pixel_values(pil)

                with torch.no_grad():
                    # Official: model.chat(tokenizer, pixel_values, question, generation_config, ...)
                    resp = self.model.chat(
                        self.tokenizer,
                        pixel_values,
                        question,
                        self.generation_config,
                    )

                outputs.append(resp if isinstance(resp, str) else str(resp))

        # If greedy + n_generations>1, repeat deterministically (pointless but matches expected length)
        if (not self.do_sample) and self.n_generations > 1:
            base = outputs[: len(batch)]
            return base * self.n_generations

        return outputs

    # -------------------------
    # vLLM generation
    # -------------------------
    def _gen_vllm(self, batch: List[Dict[str, Any]]):
        convs = self._make_convs_text_only(batch)
        text_prompts = self.processor.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
        images = [item["decoded_image"] for item in batch]

        requests = [{"prompt": p, "multi_modal_data": {"image": img}} for p, img in zip(text_prompts, images)]
        outputs = self.model.generate(requests, self.gen_params)
        return [seq.text for out in outputs for seq in out.outputs]

    # -------------------------
    # public API
    # -------------------------
    def generate_response(self, batch: List[Dict[str, Any]]):
        if self.gen_engine == "hf":
            return self._gen_hf(batch)
        if self.gen_engine == "vllm":
            return self._gen_vllm(batch)
        raise ValueError(f"Unknown gen_engine: {self.gen_engine}")

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

