"""
DiagVRIQ Claude runner.

Uses the ground_truth_2.json subset (the items with written textual
descriptions, ~210 on Natural_Dataset).

Phases:
  two_stage         image + question; model writes a description, then reasons
  reason_augmented  image + question + GT description; model reasons and answers
"""

import os
import csv
import json
import re
import io
import base64
import argparse
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
from PIL import Image
import anthropic


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]
MAX_SIDE = 1200


TWO_STAGE_PROMPT = """
Approach the task in two stages.

Stage 1: Explain the entire question in pure text, as if you were communicating it to someone who cannot see the image. Describe each puzzle panel and each answer option in enough detail that a reader could solve the puzzle from your text alone. Include shapes, counts, positions, orientations, patterns, colors, and any changes across panels. Do not solve yet.

Stage 2: Reason about the underlying pattern or rule and choose the best answer.

Format your response exactly as:
<description>
Your text-only explanation of the question.
</description>
<think>
Your reasoning.
</think>
<answer>X</answer>

Replace X with exactly one letter: A, B, C, or D.
""".strip()


REASON_AUGMENTED_PROMPT = """
You are given:
1. The original visual question.
2. The image.
3. A ground-truth textual description of the image.

Use both the image and the ground-truth description.

First reason about the pattern or rule.
Then provide the final answer.

Format your response exactly as:
<think>
Your reasoning here.
</think>
<answer>X</answer>

Replace X with exactly one letter: A, B, C, or D.
""".strip()


def norm(s):
    return str(s or "").strip()


def pid_stem(pid):
    return Path(norm(pid)).stem


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
    files_lower = {fn.lower(): fn for fn in os.listdir(folder)}
    candidates = [pid, base] if ext else [pid]
    for e in IMAGE_EXTS:
        candidates.append((base if ext else pid) + e)
    for c in candidates:
        if c.lower() in files_lower:
            return os.path.join(folder, files_lower[c.lower()])
    return None


def load_gt_map(dataset_root):
    """(source_folder, pid_stem) -> GT_Description"""
    gt_map = {}
    for gt_path in Path(dataset_root).rglob("ground_truth_2.json"):
        source_folder = gt_path.parent.name
        with open(gt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ("result", "results", "data", "items"):
                if key in data:
                    data = data[key]
                    break
        if not isinstance(data, list):
            continue
        for item in data:
            pid = item.get("PID") or item.get("pid")
            desc = (
                item.get("GT_Description")
                or item.get("GT Description")
                or item.get("gt_description")
            )
            if pid and desc:
                gt_map[(source_folder, pid_stem(pid))] = norm(desc)
    return gt_map


def load_diagvriq(dataset_root):
    if not dataset_root or not os.path.isdir(dataset_root):
        from common.data import DIAG_REPO, load_diag
        print(f"[HF] Loading {DIAG_REPO}")
        return load_diag()
    gt_map = load_gt_map(dataset_root)
    print(f"[GT] Found {len(gt_map)} descriptions in ground_truth_2.json files.")

    data = []
    missing = []

    for sub in sorted(os.listdir(dataset_root)):
        folder = os.path.join(dataset_root, sub)
        if not os.path.isdir(folder):
            continue
        csv_path = find_csv(folder)
        if not csv_path:
            continue

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                pid_raw = norm(row.get("PID"))
                pid = pid_stem(pid_raw)
                desc = gt_map.get((sub, pid))
                if not desc:
                    continue

                img_path = resolve_image(folder, pid_raw)
                if not img_path:
                    missing.append({"folder": sub, "pid": pid_raw, "row_index": idx})
                    continue

                data.append({
                    "pid": pid,
                    "category": norm(row.get("Category")),
                    "question": norm(row.get("Question")),
                    "ground_truth": norm(row.get("Ground truth")),
                    "gt_description": desc,
                    "image_path": img_path,
                    "source_folder": sub,
                    "row_index": idx,
                })

    if not data:
        raise RuntimeError("No DiagVRIQ items loaded. Check ground_truth_2.json and CSVs.")
    return data, missing


def print_stats(data, missing):
    counts = defaultdict(int)
    for it in data:
        counts[it["category"] or "UNKNOWN"] += 1
    print("\n================ DiagVRIQ subset ================")
    print(f"Loaded samples:  {len(data)}")
    print(f"Missing images:  {len(missing)}")
    print("\nSamples per category:")
    for c in sorted(counts):
        print(f"  {c}: {counts[c]}")
    print("=================================================\n")


def normalize_answer(ans):
    s = norm(ans).upper()
    if s in {"1", "2", "3", "4", "5"}:
        return {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}[s]
    m = re.search(r"\b([A-E])\b", s)
    return m.group(1) if m else ""


def extract_answer(raw):
    raw = norm(raw)
    m = re.search(r"<answer>\s*([A-E1-5])\s*</answer>", raw, flags=re.I | re.S)
    if m:
        return normalize_answer(m.group(1))
    matches = re.findall(r"\b([A-E])\b", raw, flags=re.I)
    if matches:
        return normalize_answer(matches[-1])
    return ""


def build_user_text(item, phase):
    if phase == "two_stage":
        return (
            f"Question:\n{item['question']}\n\n"
            f"{TWO_STAGE_PROMPT}"
        )
    return (
        f"Original question:\n{item['question']}\n\n"
        f"Ground-truth image description:\n{item['gt_description']}\n\n"
        f"{REASON_AUGMENTED_PROMPT}"
    )


def run_claude(client, item, model_name, phase, max_tokens):
    img_b64 = image_to_base64(item["image_path"])
    user_text = build_user_text(item, phase)
    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
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
                    {"type": "text", "text": user_text},
                ],
            }
        ],
    )
    text_parts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ]
    raw = "\n".join(text_parts).strip()
    return raw, extract_answer(raw)


def compute_accuracy(results):
    total = defaultdict(int)
    correct = defaultdict(int)
    for r in results:
        cat = r["category"] or "UNKNOWN"
        total[cat] += 1
        if r["correct"]:
            correct[cat] += 1

    overall_c = sum(correct.values())
    overall_t = sum(total.values())
    overall = overall_c / overall_t if overall_t else 0.0

    print("\n================ Results ================")
    print(f"Overall accuracy: {overall:.4f} ({overall_c}/{overall_t})")
    print("\n=== Accuracy by Category ===")
    for cat in sorted(total):
        acc = correct[cat] / total[cat] if total[cat] else 0.0
        print(f"{cat}: {acc:.4f} ({correct[cat]}/{total[cat]})")

    return {
        "overall_accuracy": overall,
        "overall_correct": overall_c,
        "overall_total": overall_t,
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
    parser.add_argument(
        "--dataset_root",
        default=os.environ.get("DATASET_ROOT", ""),
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["two_stage", "reason_augmented"],
        help="two_stage = describe then reason; reason_augmented = GT text + reason",
    )
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--outputs_dir", default="./results")
    parser.add_argument("--tag", default="diagvriq")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument(
        "--sanity_check_n",
        type=int,
        default=0,
        help="If >0, run only the first N GT-subset items.",
    )
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        raise RuntimeError("Set ANTHROPIC_API_KEY first.")

    os.makedirs(args.outputs_dir, exist_ok=True)
    data, missing = load_diagvriq(args.dataset_root)
    print_stats(data, missing)

    if args.sanity_check_n > 0:
        data = data[: args.sanity_check_n]
        print(f"[SANITY] Running only first {len(data)} samples.")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    results = []

    print(f"\n[RUN] Model: {args.model}")
    print(f"[RUN] Phase: {args.phase}")
    print(f"[RUN] Items: {len(data)}\n")

    for item in tqdm(data, desc=f"Claude {args.phase}"):
        gt = normalize_answer(item["ground_truth"])
        try:
            raw, pred = run_claude(
                client=client,
                item=item,
                model_name=args.model,
                phase=args.phase,
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
            "gt_description": item["gt_description"],
            "prediction": pred,
            "correct": correct,
            "raw_response": raw,
            "image_path": item["image_path"],
            "source_folder": item["source_folder"],
            "row_index": item["row_index"],
            "phase": args.phase,
        }
        results.append(rec)

        status = "OK" if correct else "MISS"
        preview = raw.replace("\n", " ")[:160]
        print(
            f"[GEN] {status} PID={item['pid']} | {item['category']} | "
            f"GT={gt} | pred={pred} | raw='{preview}...'"
        )

    scores = compute_accuracy(results)
    out_path = os.path.join(
        args.outputs_dir,
        f"{args.model}_{args.tag}_{args.phase}.json",
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
