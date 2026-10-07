"""
LLaVA-OneVision-1.5 model wrapper using HuggingFace Transformers.
Uses AutoModelForCausalLM + AutoProcessor with trust_remote_code=True.
"""

import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from typing import List, Dict, Any


class LLaVAOneVision15:
    def __init__(self, args, model_name_path="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"):
        self.args = args
        self.model_name = model_name_path

        print(f"[LLaVA-OV-1.5] Loading {model_name_path} via HuggingFace Transformers...")

        self.processor = AutoProcessor.from_pretrained(
            model_name_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

        print(f"[LLaVA-OV-1.5] Model loaded successfully on {self.model.device}")

    def generate_response(self, batched_input: List[Dict[str, Any]]) -> List[str]:
        from qwen_vl_utils import process_vision_info

        results = []
        suffix = ""
        if self.args and hasattr(self.args, 'gen_prompt_suffix'):
            suffix = self.args.gen_prompt_suffix

        max_new_tokens = 1024
        if self.args and hasattr(self.args, 'max_new_tokens'):
            max_new_tokens = self.args.max_new_tokens

        for item in batched_input:
            prompt = item.get("question_prompt", "") + suffix
            img = item.get("decoded_image", None)

            # Build message in the format expected by the processor
            content = []
            if img is not None:
                # Save PIL image to a temp path for qwen_vl_utils, or use inline
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content}]

            try:
                # Apply chat template
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Process vision info
                image_inputs, video_inputs = process_vision_info(messages)

                # Prepare inputs
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)

                # Generate
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )

                # Decode only the generated part (skip input tokens)
                input_len = inputs["input_ids"].shape[1]
                generated_ids = output_ids[0][input_len:]
                response = self.processor.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                ).strip()

                results.append(response)
                print(f"[LLaVA-OV-1.5] Response: {response[:120]}...")

            except Exception as e:
                print(f"[LLaVA-OV-1.5] Error: {e}")
                import traceback
                traceback.print_exc()
                results.append(f"[ERROR] {e}")

        return results

    def shutdown(self):
        """Free GPU memory."""
        del self.model
        del self.processor
        torch.cuda.empty_cache()
        print(f"[LLaVA-OV-1.5] Shut down {self.model_name}.")
