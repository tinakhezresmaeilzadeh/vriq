import os
import csv
import json
import re
import base64
import argparse
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
from PIL import Image as PILImage
from openai import OpenAI

from utils import save_response_to_json
from extract import Extractor


MAX_PIXELS = 250_000
MIN_SIDE = 28
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]


PROMPT_SUFFIX = """

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
"""


def _norm(s):
    return (s or "").strip()


def _pid_stem(pid):
    return Path(_norm(pid)).stem


def _resize_image(img):
    w, h = img.size
    total = w * h

    if total > MAX_PIXELS:
        scale = (MAX_PIXELS / total) ** 0.5
        w = max(1, int(w * scale))
        h = max(1, int(h * scale))
        img = img.resize((w, h), PILImage.LANCZOS)

    if w < MIN_SIDE or h < MIN_SIDE:
        w = max(w, MIN_SIDE)
        h = max(h, MIN_SIDE)
        img = img.resize((w, h), PILImage.LANCZOS)

    return img


def _list_files_lower(folder):
    return {fn.lower(): fn for fn in os.listdir(folder)}


def _resolve_image_path(folder, pid_raw):
    pid_raw = _norm(pid_raw)
    if not pid_raw:
        return None

    files_lower = _list_files_lower(folder)
    base, ext = os.path.splitext(pid_raw)
    ext = ext.lower()
    base_noext = base if ext else pid_raw

    candidates = []
    if ext:
        candidates.extend([pid_raw, base])
    else:
        candidates.append(pid_raw)

    for e in IMAGE_EXTS:
        candidates.append(base_noext + e)

    for c in candidates:
        p = os.path.join(folder, c)
        if os.path.isfile(p):
            return p
        lc = c.lower()
        if lc in files_lower:
            return os.path.join(folder, files_lower[lc])

    target = base_noext.lower()
    for lf, real in files_lower.items():
        b, e = os.path.splitext(lf)
        if b == target and e.lower() in IMAGE_EXTS:
            return os.path.join(folder, real)

    return None


def _find_csv(folder):
    for fn in os.listdir(folder):
        if fn.lower().endswith(".csv"):
            return os.path.join(folder, fn)
    return None


def load_gt_pids(dataset_root):
    allowed = set()

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
            if pid:
                allowed.add((source_folder, _pid_stem(pid)))

    return allowed


def load_local_dataset_gt_subset(root_dir):
    if not root_dir or not os.path.isdir(root_dir):
        from hf_data import DIAG_REPO, load_diag
        print(f"[HF] Loading {DIAG_REPO}")
        return load_diag()
    allowed = load_gt_pids(root_dir)
    print(f"[GT] Found {len(allowed)} PIDs from ground_truth_2.json files.")

    data = []
    missing = []

    for sub in sorted(os.listdir(root_dir)):
        folder = os.path.join(root_dir, sub)
        if not os.path.isdir(folder):
            continue

        csv_path = _find_csv(folder)
        if not csv_path:
            continue

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            for idx, row in enumerate(reader):
                pid = _norm(row.get("PID"))
                pid_stem = _pid_stem(pid)

                if (sub, pid_stem) not in allowed:
                    continue

                category = _norm(row.get("Category"))
                question = _norm(row.get("Question"))
                gt = _norm(row.get("Ground truth"))

                img_path = _resolve_image_path(folder, pid)
                if not img_path:
                    missing.append({"folder": sub, "pid": pid, "row_index": idx})
                    continue

                try:
                    img = PILImage.open(img_path).convert("RGB")
                    img = _resize_image(img)
                except Exception as e:
                    missing.append({
                        "folder": sub,
                        "pid": pid,
                        "row_index": idx,
                        "error": str(e),
                    })
                    continue

                data.append({
                    "pid": pid_stem,
                    "category": category,
                    "question_prompt": question,
                    "ground_truth": gt,
                    "decoded_image": img,
                    "image_path": img_path,
                    "source_folder": sub,
                    "row_index": idx,
                })

    if not data:
        raise RuntimeError("No GT-subset data loaded. Check ground_truth_2.json PID names.")

    return data, missing


def image_to_data_url(img):
    import io

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def gpt_generate(client, model_name, question, img, max_tokens):
    full_prompt = question + PROMPT_SUFFIX
    data_url = image_to_data_url(img)

    response = client.chat.completions.create(
        model=model_name,
        temperature=1,                        # <-- also change this (see note below)
        max_completion_tokens=max_tokens,     # <-- renamed parameter
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content.strip()


def normalize_option(ans):
    if ans is None:
        return ""

    s = str(ans).strip().upper()
    if not s:
        return ""

    m = re.search(r"\b([A-E])\b", s)
    if m:
        return m.group(1)

    m = re.search(r"\b([1-5])\b", s)
    if m:
        return {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}[m.group(1)]

    return s


def compute_acc_by_category(items):
    total = defaultdict(int)
    correct = defaultdict(int)

    for it in items:
        cat = it.get("category") or "UNKNOWN"
        gt = normalize_option(it.get("ground_truth"))

        pred_list = it.get("extracted_response", [])
        if isinstance(pred_list, list) and pred_list:
            pred_raw = pred_list[0]
        else:
            pred_raw = pred_list

        pred = normalize_option(pred_raw)
        ok = bool(pred and gt and pred == gt)

        it["pred"] = pred
        it["correct"] = ok

        total[cat] += 1
        if ok:
            correct[cat] += 1

    acc = {
        c: correct[c] / total[c] if total[c] else 0.0
        for c in total
    }
    overall = sum(correct.values()) / max(1, sum(total.values()))
    return overall, acc, total, correct


def print_stats(data, missing):
    counts = defaultdict(int)
    for ex in data:
        counts[ex.get("category") or "UNKNOWN"] += 1

    print("\n================ GT Subset Stats ================")
    print(f"Loaded GT-subset samples: {len(data)}")
    print(f"Missing images:           {len(missing)}")
    print("\nSamples per category:")
    for c in sorted(counts):
        print(f"  {c}: {counts[c]}")
    print("=================================================\n")


def run_all(args):
    args.task_name = "visreasoning"
    args.gen_engine = "openai"
    args.model_name_path = args.model
    args.gen_prompt_suffix_type = PROMPT_SUFFIX
    args.gen_prompt_suffix = PROMPT_SUFFIX
    args.n_generations = 1

    data, missing = load_local_dataset_gt_subset(args.dataset_root)
    print_stats(data, missing)

    if args.sanity_check_n > 0:
        data = data[:args.sanity_check_n]
        print(f"[SANITY] Running only first {len(data)} samples.")

    client = OpenAI()

    items_with_raw = []
    args.duty_type = "raw"

    for ex in tqdm(data, desc=f"Generating with {args.model}"):
        try:
            raw = gpt_generate(
                client=client,
                model_name=args.model,
                question=ex["question_prompt"],
                img=ex["decoded_image"],
                max_tokens=args.max_new_tokens,
            )
        except Exception as e:
            raw = f"[ERROR] {e}"

        rec = {
            k: v
            for k, v in ex.items()
            if k not in ("decoded_image", "decoded_images")
        }

        rec["raw_response"] = [raw]
        items_with_raw.append(rec)

        quick = re.search(r"<answer>\s*([A-D])\s*</answer>", raw, flags=re.I)
        quick_pred = quick.group(1).upper() if quick else ""
        gt = normalize_option(rec.get("ground_truth"))
        status = "OK" if quick_pred == gt else "MISS"
        print(f"[GEN] {status} PID={rec['pid']} | {rec['category']} | GT={gt} | pred={quick_pred}")

    save_response_to_json(args, items_with_raw)
    raw_path = args.file_with_raw_response

    args.duty_type = "extract"
    extractor = Extractor(
        items_with_raw,
        args,
        use_vanilla_extract=args.use_vanilla_extract,
        use_quick_extract_w_gpt=args.use_quick_extract_w_gpt,
        use_gpt_extract=args.use_gpt_extract,
        use_answer_tag_extract=args.use_answer_tag_extract,
    )
    extractor.extract_ans_and_save()
    extracted_path = args.file_with_extracted_response

    with open(extracted_path, "r", encoding="utf-8") as f:
        extracted_obj = json.load(f)

    extracted_items = extracted_obj.get("result", extracted_obj)

    args.duty_type = "score"
    overall, acc, total, correct = compute_acc_by_category(extracted_items)

    scores = {
        "overall_accuracy": overall,
        "accuracy_by_category": {
            c: {
                "accuracy": acc[c],
                "correct": correct[c],
                "total": total[c],
            }
            for c in sorted(acc)
        },
    }

    save_response_to_json(args, extracted_items, scores=scores)

    print("\n================ Results ================")
    print(f"Overall accuracy: {overall:.4f} ({sum(correct.values())}/{sum(total.values())})")
    for c in sorted(acc):
        print(f"{c}: {acc[c]:.4f} ({correct[c]}/{total[c]})")
    print("=========================================\n")

    print(f"[OK] Finished {args.model} GT-subset run.")
    print(f"Raw saved to:       {raw_path}")
    print(f"Extracted saved to: {extracted_path}")
    print(f"Score saved to:     {args.file_with_score}")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset_root", default="", help="Empty loads the Hugging Face dataset.")
    p.add_argument("--outputs_dir", default="./results")
    p.add_argument("--tag", default="gpt5.1_gt_subset")

    p.add_argument("--model", default="gpt-5.1")
    p.add_argument("--max_new_tokens", type=int, default=3072)
    p.add_argument("--sanity_check_n", type=int, default=0)

    p.add_argument("--debug", action="store_true", default=False)
    p.add_argument("--delete_prev_file", action="store_true", default=False)

    p.add_argument("--use_vanilla_extract", action="store_true", default=False)
    p.add_argument("--use_quick_extract_w_gpt", action="store_true", default=False)
    p.add_argument("--use_gpt_extract", action="store_true", default=False)
    p.add_argument("--use_answer_tag_extract", action="store_true", default=True)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(args)
