import os
import csv
import json
import argparse
from collections import defaultdict
import re

from tqdm import tqdm
from PIL import Image as PILImage

from common.utils import save_response_to_json
from common.extract import Extractor


MAX_PIXELS = 250_000
MIN_SIDE = 28
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]


ANSWER_PROMPT = """

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
"""


def _resize_image(img: PILImage.Image):
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


def _norm(s):
    return (s or "").strip()


def _pid_stem(pid):
    return os.path.splitext(_norm(pid))[0]


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
        c = _norm(c)
        if not c:
            continue

        p = os.path.join(folder, c)
        if os.path.isfile(p):
            return p

        if c.lower() in files_lower:
            return os.path.join(folder, files_lower[c.lower()])

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


def load_gt_descriptions(dataset_root):
    gt_map = {}

    for root, _, files in os.walk(dataset_root):
        if "ground_truth_2.json" not in files:
            continue

        gt_path = os.path.join(root, "ground_truth_2.json")
        source_folder = os.path.basename(root)

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
                gt_map[(source_folder, _pid_stem(pid))] = _norm(desc)

    return gt_map


def load_local_dataset(root_dir):
    if not root_dir or not os.path.isdir(str(root_dir)):
        from common.data import DIAG_REPO, load_diag
        print(f"[HF] Loading {DIAG_REPO}")

        def _prompt(q, desc):
            return (
                f"Original question:\n{q}\n\n"
                f"Ground-truth image description:\n{desc}\n\n"
                f"{ANSWER_PROMPT}"
            ).strip()

        return load_diag(make_prompt=_prompt)

    if not os.path.isdir(root_dir):
        raise ValueError(f"--dataset_root must be a directory, got: {root_dir}")

    gt_map = load_gt_descriptions(root_dir)
    print(f"[GT] Found {len(gt_map)} GT descriptions.")

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
                pid_raw = _norm(row.get("PID"))
                pid_stem = _pid_stem(pid_raw)

                gt_desc = gt_map.get((sub, pid_stem))
                if not gt_desc:
                    continue

                category = _norm(row.get("Category"))
                question = _norm(row.get("Question"))
                gt = _norm(row.get("Ground truth"))

                question_prompt = f"""
Original question:
{question}

Ground-truth image description:
{gt_desc}

{ANSWER_PROMPT}
""".strip()

                img_path = _resolve_image_path(folder, pid_raw)
                if not img_path:
                    missing.append({"folder": sub, "pid": pid_raw, "row_index": idx})
                    continue

                try:
                    img = PILImage.open(img_path).convert("RGB")
                    img = _resize_image(img)
                except Exception as e:
                    missing.append({
                        "folder": sub,
                        "pid": pid_raw,
                        "row_index": idx,
                        "error": str(e),
                    })
                    continue

                data.append({
                    "pid": pid_stem,
                    "category": category,
                    "question_prompt": question_prompt,
                    "original_question": question,
                    "ground_truth": gt,
                    "gt_description": gt_desc,
                    "decoded_image": img,
                    "image_path": img_path,
                    "source_folder": sub,
                    "row_index": idx,
                })

    if not data:
        raise RuntimeError("No GT-description subset loaded. Check PID names and ground_truth_2.json.")

    return data, missing


def get_model(model_name_path, args):
    if "llava" in model_name_path.lower():
        from models import llava_multi_image as llava
        return llava.LLaVA(args, model_name_path)

    raise RuntimeError(f"This script is for LLaVA. Got model: {model_name_path}")


def normalize_option(ans):
    if ans is None:
        return ""

    s = str(ans).strip().upper()
    if not s:
        return ""

    m = re.search(r"\b([A-D])\b", s)
    if m:
        return m.group(1)

    m = re.search(r"\b([1-4])\b", s)
    if m:
        return {"1": "A", "2": "B", "3": "C", "4": "D"}[m.group(1)]

    return s


def compute_acc_by_category(items):
    total = defaultdict(int)
    correct = defaultdict(int)

    for it in items:
        cat = it.get("category") or "UNKNOWN"
        gt = normalize_option(it.get("ground_truth"))

        pred_list = it.get("extracted_response", [])
        pred_raw = pred_list[0] if isinstance(pred_list, list) and pred_list else pred_list
        pred = normalize_option(pred_raw)

        ok = bool(pred and gt and pred == gt)

        it["pred"] = pred
        it["correct"] = ok

        total[cat] += 1
        if ok:
            correct[cat] += 1

    acc = {c: correct[c] / total[c] if total[c] else 0.0 for c in total}
    overall = sum(correct.values()) / max(1, sum(total.values()))
    return overall, acc, total, correct


def _ensure_multi_image_keys(batch):
    for it in batch:
        if "decoded_images" not in it:
            img = it.get("decoded_image")
            it["decoded_images"] = [img] if img is not None else []


def _extract_answer_quick(raw_text):
    if not raw_text:
        return ""

    m = re.search(r"<answer>\s*(.*?)\s*</answer>", raw_text, flags=re.I | re.S)
    if m:
        return normalize_option(m.group(1))

    m = re.findall(r"\b([A-D])\b", raw_text, flags=re.I)
    if m:
        return m[-1].upper()

    return ""


def run_all(args):
    args.task_name = "visreasoning"
    args.gen_prompt_suffix = ""
    args.n_generations = 1

    data, missing = load_local_dataset(args.dataset_root)

    print(f"\nLoaded samples: {len(data)}")
    print(f"Missing images: {len(missing)}\n")

    if args.sanity_check_n > 0:
        data = data[:args.sanity_check_n]
        print(f"[SANITY] Running first {len(data)} samples.")

    model = get_model(args.model_name_path, args)

    args.duty_type = "raw"
    items_with_raw = []

    try:
        for batch_idx in tqdm(range(0, len(data), args.bs), desc="Generating"):
            batch = data[batch_idx: batch_idx + args.bs]
            _ensure_multi_image_keys(batch)

            batch_outputs = model.generate_response(batch)

            for i, ex in enumerate(batch):
                out_slice = batch_outputs[
                    i * args.n_generations:
                    (i + 1) * args.n_generations
                ]

                rec = {
                    k: v
                    for k, v in ex.items()
                    if k not in ("decoded_image", "decoded_images")
                }

                rec["raw_response"] = out_slice
                items_with_raw.append(rec)

                raw0 = out_slice[0] if isinstance(out_slice, list) and out_slice else str(out_slice)
                quick_pred = _extract_answer_quick(raw0)
                gt = normalize_option(rec.get("ground_truth"))
                status = "✅" if quick_pred == gt else "❌"

                print(f"[GEN] {status} PID={rec['pid']} | {rec['category']} | GT={gt} | pred={quick_pred}")

    finally:
        if model and hasattr(model, "shutdown"):
            model.shutdown()

        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

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
            for c in sorted(acc.keys())
        },
    }

    save_response_to_json(args, extracted_items, scores=scores)

    print("\n================ Results ================")
    print(f"Overall accuracy: {overall:.4f} ({sum(correct.values())}/{sum(total.values())})")

    for c in sorted(acc.keys()):
        print(f"{c}: {acc[c]:.4f} ({correct[c]}/{total[c]})")

    print("=========================================\n")

    print("[OK] Finished LLaVA + GT-description run.")
    print(f"Raw saved to:       {raw_path}")
    print(f"Extracted saved to: {extracted_path}")
    print(f"Score saved to:     {args.file_with_score}")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset_root", default="", help="Empty loads the Hugging Face dataset.")
    p.add_argument("--outputs_dir", default="./results")
    p.add_argument("--tag", default="matrix_completion_llava_gt_description")

    p.add_argument("--bs", default=1, type=int)
    p.add_argument("--model_name_path", default="llava-hf/llava-v1.6-mistral-7b-hf")
    p.add_argument("--gen_engine", type=str, default="hf", choices=["hf", "openai", "vllm"])
    p.add_argument("--gen_prompt_suffix_type", default="", type=str)

    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--n_generations", type=int, default=1)

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