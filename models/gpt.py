from openai import OpenAI
import time
import traceback
from typing import List, Dict, Any
from io import BytesIO
import base64
import os

def _require_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY before calling the OpenAI API.")
    return key

class GPT:
    def __init__(self, 
                 model_name_path='gpt-4o-mini',
                 args=None,
                 temperature=1,
                 max_tokens=1024,
                 n_generations=1,
                 patience=2):
        self.model = model_name_path
        self.temperature = None
        self.max_tokens = max_tokens
        self.n_generations = n_generations
        self.max_patience = patience
        self.client = OpenAI(api_key=_require_openai_key())
        self.args = args
        # Check if tools should be disabled for o3/o4
        self.no_tools = getattr(args, 'no_o3_tools', False) if args else False

    def extract_answer_from_raw_response(self, prompt):
        messages = [
            {"role": "user",
             "content": [{"type": "text", "text": prompt}]}
        ]
        
        extraction = "None"
        patience = self.max_patience

        while patience > 0:
            patience -= 1
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                    n=1
                )
                pred = response.choices[0].message.content.strip()

                if pred:
                    extraction = pred
                    break
                else:
                    print("Empty response from GPT.")

            except Exception as e:
                print(f"Exception during GPT call: {e.__class__.__name__}: {e}")
                traceback.print_exc()
                print(f"Exception during GPT call: {e}")

        return extraction

    def generate_response(self, batched_input: List[Dict[str, Any]]) -> List[str]:
        def encode_image(img):
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()

        results = []
        for item in batched_input:
            prompt = item["question_prompt"]
            if self.args and hasattr(self.args, 'gen_prompt_suffix'):
                prompt += self.args.gen_prompt_suffix
            img = item.get("decoded_image", None)

            # Build content differently for o3/o4 (responses API) vs others (chat API)
            is_o3 = "o3" in self.model or "o4" in self.model

            if is_o3:
                # responses.create uses input_text / input_image types
                content = []
                if img is not None:
                    try:
                        b64_img = encode_image(img)
                        content.append({
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{b64_img}"
                        })
                    except Exception as e:
                        print(f"Failed to encode image: {e}")
                content.append({"type": "input_text", "text": prompt})
                messages = [{"role": "user", "content": content}]
            else:
                # chat.completions.create uses image_url / text types
                content = []
                if img is not None:
                    try:
                        b64_img = encode_image(img)
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_img}"
                            }
                        })
                    except Exception as e:
                        print(f"Failed to encode image: {e}")
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]

            retries = self.max_patience
            while retries > 0:
                retries -= 1
                try:
                    if ("o3" in self.model or "o4" in self.model) and not self.no_tools:
                        # O3/O4 WITH tools (thinking with images) — original behavior
                        instructions = "You are a helpful assistant that analyzes an image and answers the question clearly. Use the code interpreter if needed."
                        response = self.client.responses.create(
                                    model=self.model,
                                    tools=[
                                        {
                                            "type": "code_interpreter",
                                            "container": {"type": "auto"}
                                        }
                                    ],
                                    instructions=instructions,
                                    input=messages,
                                    timeout=6000,
                                )
                        outputs = [response.output_text.strip()]
                        print(f"[o3 WITH tools] {outputs}")

                    elif ("o3" in self.model or "o4" in self.model) and self.no_tools:
                        # O3/O4 WITHOUT tools — no code interpreter, no thinking with images
                        instructions = "You are a helpful assistant that analyzes an image and answers the question clearly."
                        response = self.client.responses.create(
                                    model=self.model,
                                    instructions=instructions,
                                    input=messages,
                                    timeout=6000,
                                )
                        outputs = [response.output_text.strip()]
                        print(f"[o3 NO tools] {outputs}")

                    else:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=self.temperature,
                            max_completion_tokens=self.max_tokens,
                            n=self.n_generations,
                        )
                        outputs = [choice.message.content.strip() for choice in response.choices]

                    results.extend(outputs)
                    break
                except Exception as e:
                    print(f"Error during OpenAI API call: {e}")
                    if retries == 0:
                        results.extend(["[ERROR]"] * self.n_generations)

        return results
