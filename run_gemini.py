import os
import csv
import json
import re
import argparse
import random
import time
from collections import defaultdict

from tqdm import tqdm
from PIL import Image
from google import genai
from google.genai import types


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]
MAX_SIDE = 1200


SYSTEM_PROMPT = """You are an expert visual reasoning system.
Solve the puzzle carefully.
Return your response exactly in this format:

<think>
Briefly describe the visual pattern and reasoning.
</think>
<answer>X</answer>

Replace X with exactly one answer option: A, B, C, or D.
"""


def norm(s):
    return str(s or "").strip()


def resize_image(img):
    w, h = img.size
    scale = min(MAX_SIDE / w, MAX_SIDE / h, 1.0)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)))
    return img


def find_csv(folder):
    for fn in os.listdir(folder):
        if fn.lower().endswith(".csv"):
            return os.path.join(folder, fn)
    return None


def resolve_image(folder, pid):
    pid = norm(pid)
    base, ext = os.path.splitext(pid)
    files_lower = {fn.lower(): fn for fn in os.listdir(folder)}

    candidates = []
    if ext:
        candidates.append(pid)
        candidates.append(base)
    else:
        candidates.append(pid)
    for e in IMAGE_EXTS:
        candidates.append(base + e if ext else pid + e)

    for c in candidates:
        if c.lower() in files_lower:
            return os.path.join(folder, files_lower[c.lower()])
    return None


def load_dataset(dataset_root, split="abstract"):
    if not dataset_root or not os.path.isdir(dataset_root):
        from hf_data import VRIQ_REPO, load_vriq_light
        print(f"[HF] Loading {VRIQ_REPO} split={split}")
        return load_vriq_light(split)

    data = []
    missing = []

    for sub in sorted(os.listdir(dataset_root)):
        folder = os.path.join(dataset_root, sub)
        if not os.path.isdir(folder):
            continue

        csv_path = find_csv(folder)
        if csv_path is None:
            continue

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                pid = norm(row.get("PID"))
                category = norm(row.get("Category"))
                question = norm(row.get("Question"))
                gt = norm(row.get("Ground truth"))

                if not pid or not category or not question:
                    continue

                image_path = resolve_image(folder, pid)
                if image_path is None:
                    missing.append({"folder": sub, "pid": pid, "row_index": idx})
                    continue

                data.append({
                    "pid": pid,
                    "category": category,
                    "question": question,
                    "ground_truth": gt,
                    "image_path": image_path,
                    "source_folder": sub,
                    "row_index": idx,
                })

    return data, missing


def print_dataset_stats(data, missing):
    counts = defaultdict(int)
    for item in data:
        counts[item["category"]] += 1

    print("\n================ Dataset Stats ================")
    print(f"Loaded samples:  {len(data)}")
    print(f"Missing images:  {len(missing)}")
    print("\nSamples per category:")
    for c in sorted(counts):
        print(f"  {c}: {counts[c]}")
    print("================================================\n")


def sample_x_per_category(data, x, seed):
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for item in data:
        by_cat[item["category"]].append(item)

    sampled = []
    for cat in sorted(by_cat):
        items = by_cat[cat]
        k = min(x, len(items))
        chosen = rng.sample(items, k)
        sampled.extend(chosen)
        print(f"[SAMPLE] {cat}: {k}/{len(items)}")
    return sampled


def normalize_answer(ans):
    s = norm(ans).upper()
    if s in {"1", "2", "3", "4", "5"}:
        return {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}[s]
    m = re.search(r"\b([A-E])\b", s)
    if m:
        return m.group(1)
    return ""


def extract_answer(raw):
    raw = norm(raw)
    m = re.search(r"<answer>\s*([A-E1-5])\s*</answer>", raw, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return normalize_answer(m.group(1))
    matches = re.findall(r"\b([A-E])\b", raw, flags=re.IGNORECASE)
    if matches:
        return normalize_answer(matches[-1])
    matches = re.findall(r"\b([1-5])\b", raw)
    if matches:
        return normalize_answer(matches[-1])
    return ""


def run_gemini(client, item, model_name, max_tokens, max_retries=5):
    img = resize_image(Image.open(item["image_path"]).convert("RGB"))
    prompt = (
        SYSTEM_PROMPT
        + "\n\nQuestion:\n"
        + item["question"]
        + "\n\nSolve the visual reasoning puzzle. Use the image. "
        + "Return the final answer in <answer> tags."
    )

    last_err = None
    for attempt in range(max_retries):
        try:
            config_kwargs = {
                "temperature": 0,
                "max_output_tokens": max_tokens,
            }
            if "3.1" in model_name or "3-1" in model_name:
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="HIGH")
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, img],
                config=types.GenerateContentConfig(**config_kwargs),
            )
            raw = (response.text or "").strip()
            pred = extract_answer(raw)
            return raw, pred
        except Exception as e:
            last_err = e
            err = str(e).lower()
            if attempt < max_retries - 1 and ("429" in err or "rate" in err or "quota" in err):
                wait = 5 * (attempt + 1)
                print(f"[RETRY {attempt + 1}/{max_retries}] Rate hit, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise last_err


def compute_accuracy(results):
    total = defaultdict(int)
    correct = defaultdict(int)
    for r in results:
        cat = r["category"]
        total[cat] += 1
        if r["correct"]:
            correct[cat] += 1

    overall_correct = sum(correct.values())
    overall_total = sum(total.values())
    overall_acc = overall_correct / overall_total if overall_total else 0.0

    print("\n================ Results ================")
    print(f"Overall accuracy: {overall_acc:.4f} ({overall_correct}/{overall_total})")
    print("\n=== Accuracy by Category ===")
    for cat in sorted(total):
        acc = correct[cat] / total[cat] if total[cat] else 0.0
        print(f"{cat}: {acc:.4f} ({correct[cat]}/{total[cat]})")

    return {
        "overall_accuracy": overall_acc,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "accuracy_by_category": {
            cat: {
                "accuracy": correct[cat] / total[cat] if total[cat] else 0.0,
                "correct": correct[cat],
                "total": total[cat],
            }
            for cat in sorted(total)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="",
                        help="Local category-folder root. Empty loads tina-khezresmaeilzadeh/VRIQ.")
    parser.add_argument("--split", default="abstract", choices=["abstract", "natural"])
    parser.add_argument("--samples_per_category", type=int, default=0,
                        help="If >0, run only this many items per category. 0 runs every item.")
    parser.add_argument("--model", default="gemini-3.1-pro-preview",
                        help="gemini-2.5-pro or gemini-3.1-pro-preview. Gemini 3.1 Pro uses thinking level high.")
    parser.add_argument("--outputs_dir", default="./results")
    parser.add_argument("--tag", default="gemini_31pro")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if "GEMINI_API_KEY" not in os.environ:
        raise RuntimeError("Set GEMINI_API_KEY first. Example: export GEMINI_API_KEY=AIza...")

    os.makedirs(args.outputs_dir, exist_ok=True)
    data, missing = load_dataset(args.dataset_root, split=args.split)
    print_dataset_stats(data, missing)

    if args.samples_per_category > 0:
        sampled = sample_x_per_category(
            data=data,
            x=args.samples_per_category,
            seed=args.seed,
        )
    else:
        sampled = data

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    results = []

    print(f"\n[RUN] Model: {args.model}")
    print(f"[RUN] Total sampled items: {len(sampled)}\n")

    for item in tqdm(sampled, desc="Running Gemini"):
        gt = normalize_answer(item["ground_truth"])
        try:
            raw, pred = run_gemini(
                client=client,
                item=item,
                model_name=args.model,
                max_tokens=args.max_tokens,
            )
        except Exception as e:
            raw = f"[ERROR] {e}"
            pred = ""

        correct = bool(pred and gt and pred == gt)
        rec = {
            "pid": item["pid"],
            "category": item["category"],
            "question": item["question"],
            "ground_truth": gt,
            "prediction": pred,
            "correct": correct,
            "raw_response": raw,
            "image_path": item["image_path"],
            "source_folder": item["source_folder"],
            "row_index": item["row_index"],
        }
        results.append(rec)

        status = "OK" if correct else "MISS"
        preview = raw.replace("\n", " ")[:160]
        print(
            f"[GEN] {status} PID={item['pid']} | {item['category']} | "
            f"GT={gt} | pred={pred} | raw='{preview}...'"
        )

    scores = compute_accuracy(results)
    n_label = args.samples_per_category if args.samples_per_category > 0 else "all"
    out_path = os.path.join(
        args.outputs_dir,
        f"{args.model}_{args.tag}_x{n_label}_seed{args.seed}.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "parameters": vars(args),
                "scores": scores,
                "missing_images": missing,
                "result": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[OK] Saved results to: {out_path}")


if __name__ == "__main__":
    main()
