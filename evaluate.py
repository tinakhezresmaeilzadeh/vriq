import os
import csv
import json
import argparse
from collections import defaultdict
import re

from tqdm import tqdm
from PIL import Image as PILImage

from utils import save_response_to_json
from extract import Extractor  # uses prompts.py internally


# ----------------------------
# Image preprocessing
# ----------------------------
MAX_PIXELS = 250_000
MIN_SIDE = 28
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]


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


def _list_files_lower(folder):
    m = {}
    for fn in os.listdir(folder):
        m[fn.lower()] = fn
    return m


def _resolve_image_path(folder, pid_raw):
    pid_raw = _norm(pid_raw)
    if not pid_raw:
        return None

    files_lower = _list_files_lower(folder)
    base, ext = os.path.splitext(pid_raw)
    ext = ext.lower()

    candidates = []
    if ext:
        candidates.append(pid_raw)
        candidates.append(base)
        base_noext = base
    else:
        candidates.append(pid_raw)
        base_noext = pid_raw

    for e in IMAGE_EXTS:
        candidates.append(base_noext + e)

    for c in candidates:
        c = _norm(c)
        if not c:
            continue
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


def load_local_dataset(root_dir):
    """
    Returns:
      data: list[dict]
      missing: list[dict]
    """
    if not os.path.isdir(root_dir):
        raise ValueError(f"--dataset_root must be a directory, got: {root_dir}")

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
                category = _norm(row.get("Category"))
                question = _norm(row.get("Question"))
                gt = _norm(row.get("Ground truth"))

                img_path = _resolve_image_path(folder, pid)
                if not img_path:
                    missing.append({"folder": sub, "pid": pid, "row_index": idx})
                    continue

                try:
                    img = PILImage.open(img_path).convert("RGB")
                except Exception as e:
                    missing.append({"folder": sub, "pid": pid, "row_index": idx, "error": str(e)})
                    continue

                img = _resize_image(img)

                data.append({
                    "pid": pid,
                    "category": category,
                    "question_prompt": question,
                    "ground_truth": gt,
                    "decoded_image": img,
                    "image_path": img_path,
                    "source_folder": sub,
                    "row_index": idx,
                })

    if not data:
        raise RuntimeError("No data loaded. Check folder structure / CSV / image names.")
    return data, missing



def limit_per_category(data, max_per_category):
    """
    Keep at most max_per_category examples per category.
    Preserves original dataset order.
    """
    if not max_per_category or max_per_category <= 0:
        return data

    counts = defaultdict(int)
    kept = []

    for ex in data:
        cat = ex.get("category") or "UNKNOWN"
        if counts[cat] < max_per_category:
            kept.append(ex)
            counts[cat] += 1

    print(f"[LIMIT] Keeping at most {max_per_category} samples per category.")
    for cat in sorted(counts):
        print(f"  {cat}: {counts[cat]}")

    return kept


# ----------------------------
# Dataset stats
# ----------------------------
def dataset_stats(data, missing):
    counts = defaultdict(int)
    for ex in data:
        counts[ex.get("category") or "UNKNOWN"] += 1
    return {
        "total_loaded": len(data),
        "missing_images": len(missing),
        "per_category": {c: counts[c] for c in sorted(counts.keys())},
    }


def print_dataset_stats(stats):
    print("\n================ Dataset Stats ================")
    print(f"Total loaded samples: {stats['total_loaded']}")
    print(f"Missing images:       {stats['missing_images']}")
    print("\nSamples per category:")
    for c, n in stats["per_category"].items():
        print(f"  {c}: {n}")
    print("================================================\n")


def save_dataset_stats(stats, args):
    task_dir = os.path.join(args.outputs_dir, "visreasoning")
    os.makedirs(task_dir, exist_ok=True)
    out_path = os.path.join(task_dir, f"dataset_stats_{args.tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved dataset stats: {out_path}")


# ----------------------------
# Prompt suffix
# ----------------------------
def get_gen_prompt_suffix(gen_prompt_suffix_type):
    if gen_prompt_suffix_type == "dir":
        return "\n Output the answer directly, without anything else."
    if gen_prompt_suffix_type == "cot":
        return "\n First output your thinking process first and then output the answer at the end."
    if gen_prompt_suffix_type == "cot_tag":
        return "\n First output the thinking process in <think> </think> and then final answer (number) in <answer> </answer> tags."
    return gen_prompt_suffix_type


# ----------------------------
# Model selection
# ----------------------------
def get_model(model_name_path, args):
    model = None
    if "onevision-1.5" in model_name_path.lower() or "onevision1.5" in model_name_path.lower() or "one-vision-1.5" in model_name_path.lower():
        from models import llava_onevision15
        model = llava_onevision15.LLaVAOneVision15(args, model_name_path)
    elif "bee" in model_name_path.lower():
        from models import qwenvl_multi_image as qwenvl
        model = qwenvl.QwenVL(args, model_name_path)
    elif all(k in model_name_path.lower() for k in ("qwen", "vl")):
        from models import qwenvl_multi_image as qwenvl
        model = qwenvl.QwenVL(args, model_name_path)
    elif "qwen" in model_name_path.lower() or "llama" in model_name_path.lower():
        from models import qwen
        model = qwen.Qwen(args, model_name_path)
    elif "llava" in model_name_path.lower():
        from models import llava_multi_image as llava
        model = llava.LLaVA(args, model_name_path)
    elif "internvl" in model_name_path.lower():
        from models import internvl
        model = internvl.InternVL(args, model_name_path)
    elif "gpt" in model_name_path.lower() or "o3" in model_name_path.lower() or "o4" in model_name_path.lower():
        from models import openai_vlm
        model = openai_vlm.GPT(model_name_path, args)
    elif "deepseek" in model_name_path.lower():
        from models import deepseekvl
        model = deepseekvl.DeepSeekVL2(args, model_name_path)
    elif "keye" in model_name_path.lower():
        from models import keye_vl
        model = keye_vl.KeyeVL(args, model_name_path)
    elif "phi" in model_name_path.lower():
        from models import phi4_vl
        model = phi4_vl.Phi4Multimodal(args, model_name_path)
    return model


# ----------------------------
# Answer normalization
# ----------------------------
def normalize_option(ans: str) -> str:
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
        num = m.group(1)
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        return num_to_letter.get(num, num)

    if s in {"1", "2", "3", "4", "5"}:
        return {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}[s]

    return s


# ----------------------------
# Scoring
# ----------------------------
def compute_acc_by_category(items):
    tot = defaultdict(int)
    cor = defaultdict(int)

    for it in items:
        cat = it.get("category") or "UNKNOWN"

        gt_raw = _norm(it.get("ground_truth"))
        gt = normalize_option(gt_raw)

        pred_list = it.get("extracted_response", [])
        pred_raw = _norm(pred_list[0]) if isinstance(pred_list, list) and pred_list else _norm(pred_list)
        pred = normalize_option(pred_raw)

        ok = (pred == gt) if (pred and gt) else False

        it["pred"] = pred
        it["correct"] = ok

        tot[cat] += 1
        if ok:
            cor[cat] += 1

    acc = {c: (cor[c] / tot[c] if tot[c] else 0.0) for c in tot}
    overall = sum(cor.values()) / max(1, sum(tot.values()))
    return overall, acc, tot, cor


def _ensure_multi_image_keys(batch):
    for it in batch:
        if "decoded_images" not in it:
            img = it.get("decoded_image", None)
            it["decoded_images"] = [img] if img is not None else []


def _extract_answer_quick(raw_text: str):
    if not raw_text:
        return ""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", raw_text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return _norm(m.group(1))
    m = re.findall(r"\b([ABCD])\b", raw_text.strip(), flags=re.IGNORECASE)
    if m:
        return m[-1].upper()
    return ""


def _normalize_choice(pred: str):
    pred = _norm(pred).upper()
    pred = pred.replace("OPTION", "").strip()

    if pred in {"1", "2", "3", "4"}:
        return {"1": "A", "2": "B", "3": "C", "4": "D"}[pred]
    if pred in {"A", "B", "C", "D"}:
        return pred

    m = re.search(r"\b([ABCD])\b", pred)
    if m:
        return m.group(1)

    return pred


# ----------------------------
# ONE-SHOT pipeline
# ----------------------------
def run_all(args):
    args.task_name = "visreasoning"
    args.gen_prompt_suffix = get_gen_prompt_suffix(args.gen_prompt_suffix_type)

    # 1) load + stats
    if args.dataset_root and os.path.isdir(args.dataset_root):
        data, missing = load_local_dataset(args.dataset_root)
    else:
        from hf_data import VRIQ_REPO, load_vriq
        print(f"[HF] Loading {VRIQ_REPO} split={args.split}")
        data, missing = load_vriq(args.split)

    if args.max_per_category and args.max_per_category > 0:
        original_n = len(data)
        data = limit_per_category(data, args.max_per_category)
        print(f"[LIMIT] Reduced dataset from {original_n} to {len(data)} samples.")

    stats = dataset_stats(data, missing)
    print_dataset_stats(stats)
    if args.save_dataset_stats:
        save_dataset_stats(stats, args)
    if missing:
        print(f"[WARN] Missing images for {len(missing)} rows. First 5: {missing[:5]}")

    # sanity check
    if args.sanity_check_n and args.sanity_check_n > 0:
        n = min(args.sanity_check_n, len(data))
        print(f"[SANITY CHECK] Running only first {n} samples end-to-end.")
        data = data[:n]

    # 2) init VLM model
    model = get_model(args.model_name_path, args)
    if model is None:
        raise RuntimeError(f"Could not init model for: {args.model_name_path}")

    # 3) generate raw responses for main questions
    args.duty_type = "raw"
    items_with_raw = []
    try:
        for batch_idx in tqdm(range(0, len(data), args.bs), desc="Generating"):
            batch = data[batch_idx: batch_idx + args.bs]
            _ensure_multi_image_keys(batch)

            batch_outputs = model.generate_response(batch)

            for i, ex in enumerate(batch):
                out_slice = batch_outputs[i * args.n_generations:(i + 1) * args.n_generations]

                rec = {k: v for k, v in ex.items() if k not in ("decoded_image", "decoded_images")}
                rec["raw_response"] = out_slice
                items_with_raw.append(rec)

                # Stream log
                pid = rec.get("pid")
                cat = rec.get("category")
                gt = _normalize_choice(rec.get("ground_truth", ""))

                raw0 = out_slice[0] if (isinstance(out_slice, list) and out_slice) else str(out_slice)
                quick_pred = _normalize_choice(_extract_answer_quick(raw0))

                ok = (quick_pred == gt) if (quick_pred and gt) else False
                status = "✅" if ok else "❌"

                preview = raw0.replace("\n", " ")[:]
                print(f"[GEN] {status} PID={pid} | {cat} | GT={gt} | quick_pred={quick_pred} | raw_preview='{preview}...'")

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

    # 4) extract
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

    # 5) score
    args.duty_type = "score"
    overall, acc, tot, cor = compute_acc_by_category(extracted_items)

    scores = {
        "overall_accuracy": overall,
        "accuracy_by_category": {
            c: {"accuracy": acc[c], "correct": cor[c], "total": tot[c]}
            for c in sorted(acc.keys())
        }
    }

    save_response_to_json(args, extracted_items, scores=scores)

    print(f"\nOverall accuracy: {overall:.4f} ({sum(cor.values())}/{sum(tot.values())})")
    print("\n=== Accuracy by Category ===")
    for c in sorted(acc.keys()):
        print(f"{c}: {acc[c]:.4f} ({cor[c]}/{tot[c]})")

    print("\n[OK] Finished one-shot run.")
    print(f"Raw saved to:       {raw_path}")
    print(f"Extracted saved to: {extracted_path}")
    print(f"Score saved to:     {args.file_with_score}")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset_root", default="", type=str,
                   help="Local category-folder root. Empty loads tina-khezresmaeilzadeh/VRIQ.")
    p.add_argument("--split", default="abstract", choices=["abstract", "natural"],
                   help="VRIQ split when loading from the Hub.")

    # output
    p.add_argument("--outputs_dir", default="./results", type=str)
    p.add_argument("--tag", default="local", type=str)
    p.add_argument("--debug", action="store_true", default=False)
    p.add_argument("--delete_prev_file", action="store_true", default=False)

    # dataset stats
    p.add_argument("--save_dataset_stats", action="store_true", default=False)

    # sanity check
    p.add_argument("--sanity_check_n", type=int, default=0,
                   help="If >0, run only first N samples end-to-end.")
    p.add_argument("--max_per_category", type=int, default=0,
                   help="If >0, run at most N samples per category.")

    # model args
    p.add_argument("--bs", default=8, type=int)
    p.add_argument("--model_name_path", default="gpt-4o", type=str)
    p.add_argument("--gen_prompt_suffix_type", default="", type=str)
    p.add_argument("--gen_engine", type=str, default="openai", choices=["hf", "openai", "vllm"])
    p.add_argument("--reasoning_effort", type=str, default="high",
                   choices=["none", "low", "medium", "high"],
                   help="Reasoning effort for OpenAI reasoning-capable models. Use 'none' to disable passing this parameter.")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--n_generations", type=int, default=1)

    # extractor flags
    p.add_argument("--use_vanilla_extract", action="store_true", default=False)
    p.add_argument("--use_quick_extract_w_gpt", action="store_true", default=False)
    p.add_argument("--use_gpt_extract", action="store_true", default=False)
    p.add_argument("--use_answer_tag_extract", action="store_true", default=False)

    p.add_argument("--no_o3_tools", action="store_true", default=False,
                   help="Disable tool use (code interpreter) for o3/o4 models. "
                        "Runs o3 without 'thinking with images' capability.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_all(args)