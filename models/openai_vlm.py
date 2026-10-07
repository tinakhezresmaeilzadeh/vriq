import os
import base64
import traceback
from io import BytesIO
from typing import List, Dict, Any, Optional

from openai import OpenAI


def _require_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY before calling the OpenAI API.")
    return key

class GPT:
    def __init__(self,
                 model_name_path: str = "gpt-4o-mini",
                 args=None,
                 temperature: float = 1,
                 max_tokens: int = 1024,
                 n_generations: int = 1,
                 patience: int = 2):
        self.model = model_name_path
        self.temperature = None
        if args is not None and getattr(args, "max_new_tokens", None):
            max_tokens = args.max_new_tokens
        self.max_tokens = max_tokens
        if args is not None and getattr(args, "n_generations", None):
            n_generations = args.n_generations
        self.n_generations = n_generations
        self.max_patience = patience
        self.client = OpenAI(api_key=_require_openai_key())
        self.args = args

        # Tool control for o3/o4. GPT-5 models use reasoning but no code_interpreter by default.
        self.no_tools = getattr(args, "no_o3_tools", False) if args else False

        # Reasoning effort for Responses API reasoning-capable models.
        # Use: none, low, medium, high.
        self.reasoning_effort = getattr(args, "reasoning_effort", "none") if args else "none"

    # ----------------------------
    # API selection helpers
    # ----------------------------
    def _uses_responses_api(self) -> bool:
        name = self.model.lower()
        return (
            name.startswith("gpt-5")
            or "gpt-5" in name
            or "o3" in name
            or "o4" in name
        )

    def _is_o_model(self) -> bool:
        name = self.model.lower()
        return ("o3" in name) or ("o4" in name)

    def _reasoning_kwargs(self) -> Dict[str, Any]:
        effort = getattr(self, "reasoning_effort", "none")
        if not effort or effort == "none":
            return {}
        return {"reasoning": {"effort": effort}}

    @staticmethod
    def _encode_image(img) -> str:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    @staticmethod
    def _response_text_from_responses_api(response) -> str:
        text = getattr(response, "output_text", None)
        if text:
            return text.strip()

        # Fallback for SDK/object variants where output_text is not populated.
        try:
            chunks = []
            for item in getattr(response, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    c_text = getattr(c, "text", None)
                    if c_text:
                        chunks.append(c_text)
            return "\n".join(chunks).strip()
        except Exception:
            return ""

    # ----------------------------
    # Answer extraction
    # ----------------------------
    def extract_answer_from_raw_response(self, prompt: str) -> str:
        """
        Lightweight extractor. Defaults to gpt-4o-mini unless you instantiate GPT with another model.
        For GPT-5/o3/o4, uses Responses API; otherwise uses Chat Completions.
        """
        extraction = "None"
        patience = self.max_patience

        while patience > 0:
            patience -= 1
            try:
                if self._uses_responses_api():
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": prompt}],
                        }
                    ]

                    kwargs = {
                        "model": self.model,
                        "input": messages,
                        "max_output_tokens": self.max_tokens,
                        "timeout": 6000,
                    }
                    kwargs.update(self._reasoning_kwargs())

                    response = self.client.responses.create(**kwargs)
                    pred = self._response_text_from_responses_api(response).strip()
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}],
                        }
                    ]

                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_tokens,
                        n=1,
                    )
                    pred = response.choices[0].message.content.strip()

                if pred:
                    extraction = pred
                    break
                else:
                    print("Empty response from GPT extractor.")

            except Exception as e:
                print(f"Exception during GPT extraction call: {e.__class__.__name__}: {e}")
                traceback.print_exc()

        return extraction

    # ----------------------------
    # Main multimodal generation
    # ----------------------------
    def generate_response(self, batched_input: List[Dict[str, Any]]) -> List[str]:
        results = []

        for item in batched_input:
            prompt = item["question_prompt"]

            if self.args and hasattr(self.args, "gen_prompt_suffix"):
                prompt += self.args.gen_prompt_suffix

            img = item.get("decoded_image", None)

            use_responses = self._uses_responses_api()
            is_o_model = self._is_o_model()

            if use_responses:
                content = []

                if img is not None:
                    try:
                        b64_img = self._encode_image(img)
                        content.append({
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{b64_img}",
                        })
                    except Exception as e:
                        print(f"Failed to encode image: {e}")

                content.append({"type": "input_text", "text": prompt})
                messages = [{"role": "user", "content": content}]
            else:
                content = []

                if img is not None:
                    try:
                        b64_img = self._encode_image(img)
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_img}",
                            },
                        })
                    except Exception as e:
                        print(f"Failed to encode image: {e}")

                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]

            retries = self.max_patience

            while retries > 0:
                retries -= 1

                try:
                    if use_responses:
                        instructions = (
                            "You are a helpful assistant that analyzes the image "
                            "and answers the question clearly."
                        )

                        kwargs = {
                            "model": self.model,
                            "instructions": instructions,
                            "input": messages,
                            "max_output_tokens": self.max_tokens,
                            "timeout": 6000,
                        }
                        kwargs.update(self._reasoning_kwargs())

                        # Preserve old behavior: o3/o4 with tools unless --no_o3_tools is set.
                        # GPT-5.x uses reasoning without code_interpreter by default.
                        if is_o_model and not self.no_tools:
                            kwargs["tools"] = [
                                {
                                    "type": "code_interpreter",
                                    "container": {"type": "auto"},
                                }
                            ]
                            print(f"[{self.model} WITH tools] reasoning_effort={self.reasoning_effort}")
                        else:
                            print(f"[{self.model} NO tools] reasoning_effort={self.reasoning_effort}")

                        # Responses API does not support n like Chat Completions.
                        # High-reasoning GPT-5 vision calls often spend the entire
                        # token budget on hidden reasoning and return no text.
                        outputs = []
                        for _ in range(self.n_generations):
                            response = self.client.responses.create(**kwargs)
                            pred = self._response_text_from_responses_api(response)
                            status = getattr(response, "status", None)
                            incomplete = getattr(response, "incomplete_details", None)
                            usage = getattr(response, "usage", None)
                            print(
                                f"[{self.model}] status={status} incomplete={incomplete} "
                                f"max_output_tokens={kwargs.get('max_output_tokens')} usage={usage}"
                            )

                            # If the model only reasoned and never wrote an answer, retry once
                            # with a larger output budget.
                            incomplete_reason = getattr(incomplete, "reason", None) if incomplete else None
                            if (not pred) and incomplete_reason == "max_output_tokens":
                                retry_tokens = max(int(kwargs.get("max_output_tokens") or 0), 4096) * 4
                                retry_tokens = min(retry_tokens, 32768)
                                print(
                                    f"[{self.model}] Empty visible text after reasoning. "
                                    f"Retrying once with max_output_tokens={retry_tokens}"
                                )
                                retry_kwargs = dict(kwargs)
                                retry_kwargs["max_output_tokens"] = retry_tokens
                                response = self.client.responses.create(**retry_kwargs)
                                pred = self._response_text_from_responses_api(response)
                                status = getattr(response, "status", None)
                                incomplete = getattr(response, "incomplete_details", None)
                                usage = getattr(response, "usage", None)
                                print(
                                    f"[{self.model} retry] status={status} incomplete={incomplete} "
                                    f"usage={usage}"
                                )

                            outputs.append(pred if pred else "[EMPTY_RESPONSE]")

                    else:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=self.temperature,
                            max_completion_tokens=self.max_tokens,
                            n=self.n_generations,
                        )
                        outputs = [
                            choice.message.content.strip()
                            for choice in response.choices
                        ]

                    results.extend(outputs)
                    break

                except Exception as e:
                    print(f"Error during OpenAI API call: {e}")
                    traceback.print_exc()
                    if retries == 0:
                        results.extend(["[ERROR]"] * self.n_generations)

        return results
