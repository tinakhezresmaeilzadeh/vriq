import os
import csv
import json
import re
import argparse
import random
import base64
import io
from collections import defaultdict

from tqdm import tqdm
from PIL import Image
import anthropic


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]
MAX_SIDE = 1200


def norm(s):
    return str(s or "").strip()


def resize_image(img):
    w, h = img.size
    scale = min(MAX_SIDE / w, MAX_SIDE / h, 1.0)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)))
    return img


def image_to_base64(path):
    img = Image.open(path).convert("RGB")
    img = resize_image(img)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def find_csv(folder):
    for fn in os.listdir(folder):
        if fn.lower().endswith(".csv"):
            return os.path.join(folder, fn)
    return None


def resolve_image(folder, pid):
    pid = norm(pid)
    base, ext = os.path.splitext(pid)

    candidates = []
    if ext:
        candidates.append(pid)
    for e in IMAGE_EXTS:
        candidates.append(base + e)

    files_lower = {f.lower(): f for f in os.listdir(folder)}

    for c in candidates:
        if c.lower() in files_lower:
            return os.path.join(folder, files_lower[c.lower()])

    return None


def load_dataset(dataset_root, split="abstract"):
    if not dataset_root or not os.path.isdir(dataset_root):
        from common.data import VRIQ_REPO, load_vriq_light
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
                    missing.append({
                        "folder": sub,
                        "pid": pid,
                        "row_index": idx,
                    })
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

    # map numeric MCQ labels
    if s in {"1", "2", "3", "4", "5"}:
        return {
            "1": "A",
            "2": "B",
            "3": "C",
            "4": "D",
            "5": "E",
        }[s]

    m = re.search(r"\b([A-E])\b", s)
    if m:
        return m.group(1)

    return ""


def extract_answer(raw):
    raw = norm(raw)

    # preferred: <answer>X</answer>
    m = re.search(
        r"<answer>\s*([A-E1-5])\s*</answer>",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return normalize_answer(m.group(1))

    # fallback: last standalone letter
    matches = re.findall(r"\b([A-E])\b", raw, flags=re.IGNORECASE)
    if matches:
        return normalize_answer(matches[-1])

    # fallback: last standalone number
    matches = re.findall(r"\b([1-5])\b", raw)
    if matches:
        return normalize_answer(matches[-1])

    return ""


SYSTEM_PROMPT = """You are an expert visual reasoning system.
Solve the puzzle carefully.
Return your response exactly in this format:

<think>
Briefly describe the visual pattern and reasoning.
</think>
<answer>X</answer>

Replace X with exactly one answer option: A, B, C, or D.
"""


def run_claude(client, item, model_name, max_tokens):
    img_b64 = image_to_base64(item["image_path"])

    user_text = f"""Question:
{item["question"]}

Solve the visual reasoning puzzle. Use the image. Return the final answer in <answer> tags."""

    extra = {}
    if "opus" in model_name.lower():
        extra["thinking"] = {"type": "adaptive"}
        extra["output_config"] = {"effort": "high"}

    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        **extra,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": user_text,
                    },
                ],
            }
        ],
    )

    text_parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)

    raw = "\n".join(text_parts).strip()
    pred = extract_answer(raw)

    return raw, pred


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
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="claude-sonnet-4-6 or claude-opus-5. Opus 5 uses adaptive thinking at effort high.")
    parser.add_argument("--outputs_dir", default="./results")
    parser.add_argument("--tag", default="claude_sonnet46_cot")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        raise RuntimeError("Set ANTHROPIC_API_KEY before calling the Claude API.")

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

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    results = []

    print(f"\n[RUN] Model: {args.model}")
    print(f"[RUN] Total sampled items: {len(sampled)}\n")

    for item in tqdm(sampled, desc="Running Claude"):
        gt = normalize_answer(item["ground_truth"])

        try:
            raw, pred = run_claude(
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

        status = "✅" if correct else "❌"
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