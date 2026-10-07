import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
import os

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def norm(s):
    return (str(s) if s is not None else "").strip()


def normalize_pid(pid):
    pid = norm(pid)
    if not pid:
        return ""
    return Path(pid).stem


def normalize_answer(ans):
    if ans is None:
        return ""
    s = norm(ans).upper()
    if not s:
        return ""
    m = re.search(r"\b([A-D])\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\b([1-4])\b", s)
    if m:
        return {"1": "A", "2": "B", "3": "C", "4": "D"}[m.group(1)]
    return s


def extract_description(raw_response):
    if isinstance(raw_response, list):
        raw_response = raw_response[0] if raw_response else ""
    raw_response = norm(raw_response)
    if not raw_response:
        return ""
    match = re.search(
        r"<description>\s*(.*?)\s*</description>",
        raw_response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def extract_answer(raw_response, extracted_response=None):
    if isinstance(extracted_response, list) and extracted_response:
        if extracted_response[0] is not None:
            return normalize_answer(extracted_response[0])

    if extracted_response is not None and not isinstance(extracted_response, list):
        return normalize_answer(extracted_response)

    if isinstance(raw_response, list):
        raw_response = raw_response[0] if raw_response else ""

    raw_response = norm(raw_response)
    if not raw_response:
        return ""

    answer_block = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        raw_response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if answer_block:
        return normalize_answer(answer_block.group(1))

    matches = re.findall(r"\b([A-D])\b", raw_response, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    return ""


def load_model_outputs(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("result", "results", "data", "items"):
            if key in data:
                return data[key]

    if isinstance(data, list):
        return data

    raise ValueError(f"Unsupported model output format: {path}")


def load_ground_truth_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("result", "results", "data", "items"):
            if key in data:
                data = data[key]
                break

    gt_map = {}

    for item in data:
        pid = normalize_pid(item.get("PID") or item.get("pid"))
        gt_desc = (
            item.get("GT_Description")
            or item.get("GT Description")
            or item.get("gt_description")
        )
        if pid and gt_desc:
            gt_map[pid] = norm(gt_desc)

    return gt_map


def load_ground_truth_from_dataset(dataset_root):
    if not dataset_root or not Path(dataset_root).is_dir():
        from hf_data import DIAG_REPO, diag_gt_map
        print(f"[HF] Loading descriptions from {DIAG_REPO}")
        return diag_gt_map()
    dataset_root = Path(dataset_root)

    gt_map = {}
    found_files = []

    for gt_path in sorted(dataset_root.rglob("ground_truth_2.json")):
        found_files.append(str(gt_path))
        local_map = load_ground_truth_from_file(gt_path)

        for pid, desc in local_map.items():
            if pid in gt_map:
                print(f"[WARN] Duplicate GT description PID={pid}; overwriting with {gt_path}")
            gt_map[pid] = desc

    print(f"[GT] Found {len(found_files)} groundtruth_description.json files.")
    for p in found_files:
        print(f"  - {p}")

    return gt_map


JUDGE_PROMPT_TEMPLATE = """You are evaluating how well a model-generated description of a visual reasoning 
puzzle matches the ground-truth reference description.

The describe-then-solve protocol asks the model to report visual content (entities, 
panels, options) without performing rule inference. Your job is to evaluate whether 
the model's description faithfully captures what is in the image. Do NOT evaluate 
whether the description correctly identifies the puzzle's answer or rule — that is 
a separate reasoning step.

###Reference Description (Score 5 reference):
{gt_description}

###Model Description to Evaluate:
{model_description}

###Score on Three Dimensions (each 1-5)

COMPLETENESS — Does the description cover all panels and options described in the reference?
Score 5: All panels and options covered with appropriate detail
Score 4: All main elements covered, minor details missing
Score 3: Most main elements covered, some omissions
Score 2: Several important elements missing
Score 1: Major omissions; most elements absent

ACCURACY — Are entities and relationships correctly identified, without contradictions?
Score 5: Fully accurate; no factual errors or misidentifications
Score 4: Mostly accurate; minor imprecisions only
Score 3: Some inaccuracies but core entities correctly identified
Score 2: Multiple misidentifications affecting key elements
Score 1: Pervasive misidentification or contradictions with the reference

GROUNDING — Is the description faithful to image content, without hallucinations?
Score 5: Fully grounded; no content beyond what is in the image
Score 4: Almost fully grounded; minor unsupported additions
Score 3: Mix of grounded description and assumptions
Score 2: Substantial unsupported content
Score 1: Heavily hallucinated; describes content not in image

###Binary Perception Verdict
After scoring the three dimensions, give a final binary judgment:

PERCEPTION SUCCESS: Did the model accurately identify the entities, panels, and 
options that are present in the image?

- "yes" if the description correctly identifies what is shown in the image: 
  the entities are correctly named, the panels are correctly enumerated, the 
  options are correctly described. Minor omissions or imprecise wording are 
  acceptable as long as the core visual content is correctly identified.
  (Typically corresponds to ACCURACY ≥ 4 AND GROUNDING ≥ 3.)

- "no" if the description contains misidentified entities, missing panels, 
  hallucinated content, or incorrect relationships between visual elements. 
  Note: missing rule-relevant detail (e.g., "a sequence showing a frog growing" 
  without enumerating each stage) is NOT a perception failure if the entities 
  and structure are correctly identified — that is a description-precision 
  issue separate from perception.
  (Typically corresponds to ACCURACY < 4 OR GROUNDING < 3.)

The criterion is whether the model SAW what is in the image, not whether the 
description is detailed enough for reasoning to succeed downstream.
###Instructions
1. Briefly explain your scoring (1-2 sentences per dimension).
2. State the binary perception verdict with one-sentence justification.
3. Output as JSON.

Output format:
Feedback: <your feedback covering all three dimensions>
Verdict justification: <one sentence>

{{"completeness": <int>, "accuracy": <int>, "grounding": <int>, "perception_success": "<yes|no>"}}"""


def llm_judge_score(model_desc, gt_desc, client, model_name):
    if not model_desc or not gt_desc:
        return None

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        gt_description=gt_desc,
        model_description=model_desc,
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=350,
        )

        text = response.choices[0].message.content.strip()
        match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if not match:
            return None

        scores = json.loads(match.group(0))

        clean = {}
        for key in ("completeness", "accuracy", "grounding"):
            if key not in scores:
                return None
            clean[key] = max(1, min(5, int(scores[key])))

        ps = norm(scores.get("perception_success", "")).lower()
        if ps not in {"yes", "no"}:
            ps = "yes" if (clean["accuracy"] >= 4 and clean["grounding"] >= 3) else "no"

        clean["perception_success"] = ps
        return clean

    except Exception as e:
        print(f"[WARN] Judge error: {e}")
        return None


def simple_overlap_score(model_desc, gt_desc):
    if not model_desc or not gt_desc:
        return 0.0
    model_tokens = set(re.findall(r"\b[a-z]{3,}\b", model_desc.lower()))
    gt_tokens = set(re.findall(r"\b[a-z]{3,}\b", gt_desc.lower()))
    if not model_tokens or not gt_tokens:
        return 0.0
    return len(model_tokens & gt_tokens) / len(model_tokens | gt_tokens)


def pearson_correlation(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    den_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def preferred_category_order(categories):
    preferred = ["Matrix Prediction", "Odd One Out", "Sequence Completion"]
    ordered = [c for c in preferred if c in categories]
    ordered += sorted(c for c in categories if c not in preferred)
    return ordered


def build_outcome_attribution_table(by_cat):
    categories = preferred_category_order(by_cat.keys())

    row_defs = [
        ("perception_yes_answer_yes", "Perception ✓, Answer ✓", "yes", True),
        ("perception_yes_answer_no", "Perception ✓, Answer ✗", "yes", False),
        ("perception_no_answer_yes", "Perception ✗, Answer ✓", "no", True),
        ("perception_no_answer_no", "Perception ✗, Answer ✗", "no", False),
    ]

    table = {}
    for key, label, ps_value, answer_correct in row_defs:
        table[key] = {"label": label}
        for cat in categories:
            count = sum(
                1 for r in by_cat[cat]
                if r.get("perception_success") == ps_value
                and bool(r.get("is_correct")) == answer_correct
            )
            table[key][cat] = count

    return categories, table


def print_outcome_attribution_table(by_cat, model_name):
    categories, table = build_outcome_attribution_table(by_cat)

    print(f"\n========== Per-category outcome × perception attribution ({model_name}) ==========")

    first_col_width = 32
    col_width = 22

    header = " " * first_col_width
    for cat in categories:
        header += f"{cat:>{col_width}}"
    print(header)

    for row in table.values():
        line = f"{row['label']:<{first_col_width}}"
        for cat in categories:
            line += f"{row[cat]:>{col_width}}"
        print(line)

    return categories, table


def main():
    parser = argparse.ArgumentParser(
        description="Compare model-generated <description> blocks against groundtruth_description.json files."
    )

    parser.add_argument("--model_output", required=True, help="Path to model output JSON.")
    parser.add_argument("--dataset_root", default="", help="Empty loads tina-khezresmaeilzadeh/DiagVRIQ.")
    parser.add_argument("--out_dir", default="./description_analysis", help="Output directory.")
    parser.add_argument("--use_llm_judge", action="store_true", help="Use OpenAI LLM judge.")
    parser.add_argument("--judge_model", default="gpt-4o-mini", help="OpenAI judge model.")
    parser.add_argument("--max_items", type=int, default=None, help="Process only first N model items.")
    parser.add_argument("--save_unmatched", action="store_true", help="Save unmatched PID lists.")
    parser.add_argument("--table_model_name", default="Qwen2.5-VL-7B", help="Model name shown in attribution table title.")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] Model output: {args.model_output}")
    model_items = load_model_outputs(args.model_output)
    print(f"[LOAD] Loaded {len(model_items)} model items.")

    print(f"[LOAD] Dataset root: {args.dataset_root}")
    gt_map = load_ground_truth_from_dataset(args.dataset_root)
    print(f"[LOAD] Loaded {len(gt_map)} GT descriptions.")

    if args.max_items:
        model_items = model_items[:args.max_items]
        print(f"[DEBUG] Using first {len(model_items)} model items.")

    client = None
    if args.use_llm_judge:
        if not OPENAI_AVAILABLE:
            print("[WARN] openai package not installed. Falling back to token overlap.")
            args.use_llm_judge = False
        elif not os.environ.get("OPENAI_API_KEY"):
            print("[WARN] OPENAI_API_KEY not set. Falling back to token overlap.")
            args.use_llm_judge = False
        else:
            client = OpenAI()
            print(f"[JUDGE] Using LLM judge: {args.judge_model}")

    per_item_results = []
    skipped_no_gt = []
    skipped_no_desc = []

    for idx, item in enumerate(model_items):
        pid = normalize_pid(item.get("pid") or item.get("PID"))
        if not pid:
            continue

        gt_desc = gt_map.get(pid)
        if not gt_desc:
            skipped_no_gt.append(pid)
            continue

        raw_response = item.get("raw_response", "")
        model_desc = extract_description(raw_response)

        if not model_desc:
            skipped_no_desc.append(pid)
            continue

        if args.use_llm_judge and client:
            scores = llm_judge_score(model_desc, gt_desc, client, args.judge_model)
            if scores is None:
                scores = {
                    "completeness": None,
                    "accuracy": None,
                    "grounding": None,
                    "perception_success": None,
                }
                avg_quality = None
            else:
                avg_quality = (
                    scores["completeness"]
                    + scores["accuracy"]
                    + scores["grounding"]
                ) / 3.0
        else:
            overlap = simple_overlap_score(model_desc, gt_desc)
            scores = {
                "overlap": round(overlap, 4),
                "perception_success": None,
            }
            avg_quality = overlap

        gt_answer = normalize_answer(item.get("ground_truth"))
        model_answer = extract_answer(
            item.get("raw_response", ""),
            item.get("extracted_response"),
        )

        is_correct = bool(gt_answer and model_answer and gt_answer == model_answer)

        per_item_results.append({
            "pid": pid,
            "category": item.get("category"),
            "gt_answer": gt_answer,
            "model_answer": model_answer,
            "is_correct": is_correct,
            "model_description": model_desc,
            "gt_description": gt_desc,
            "scores": scores,
            "avg_quality": avg_quality,
            "perception_success": scores.get("perception_success"),
            "image_path": item.get("image_path"),
            "source_folder": item.get("source_folder"),
        })

        if len(per_item_results) % 10 == 0:
            print(f"[PROGRESS] Processed matched descriptions: {len(per_item_results)}")

    per_item_path = out_dir / "per_item_results.json"
    with open(per_item_path, "w", encoding="utf-8") as f:
        json.dump(per_item_results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Wrote per-item results to: {per_item_path}")

    if args.save_unmatched:
        unmatched_path = out_dir / "unmatched.json"
        with open(unmatched_path, "w", encoding="utf-8") as f:
            json.dump({
                "skipped_no_gt": sorted(set(skipped_no_gt)),
                "skipped_no_description": sorted(set(skipped_no_desc)),
            }, f, indent=2, ensure_ascii=False)
        print(f"[OK] Wrote unmatched info to: {unmatched_path}")

    matched = len(per_item_results)

    print("\n========== Summary ==========")
    print(f"Matched items: {matched}")
    print(f"Skipped no GT: {len(skipped_no_gt)}")
    print(f"Skipped no description: {len(skipped_no_desc)}")

    if matched == 0:
        print("No matched items. Check PID names. Because yes, naming strikes again.")
        return

    correct = sum(1 for r in per_item_results if r["is_correct"])
    overall_accuracy = correct / matched

    qualities = [
        r["avg_quality"]
        for r in per_item_results
        if r["avg_quality"] is not None
    ]

    correctness = [
        1 if r["is_correct"] else 0
        for r in per_item_results
        if r["avg_quality"] is not None
    ]

    mean_quality = sum(qualities) / len(qualities) if qualities else None
    corr = pearson_correlation(qualities, correctness) if qualities else None

    ps_items = [
        r for r in per_item_results
        if r.get("perception_success") in {"yes", "no"}
    ]
    ps_yes = sum(1 for r in ps_items if r["perception_success"] == "yes")
    ps_rate = ps_yes / len(ps_items) if ps_items else None

    print(f"Answer accuracy: {correct}/{matched} = {overall_accuracy * 100:.2f}%")

    if mean_quality is not None:
        print(f"Mean description quality: {mean_quality:.4f}")

    if corr is not None:
        print(f"Pearson correlation: description quality vs correctness = {corr:.4f}")
    else:
        print("Pearson correlation: undefined")

    if ps_rate is not None:
        print(f"Perception success rate: {ps_yes}/{len(ps_items)} = {ps_rate * 100:.2f}%")

    print("\n========== Per Category ==========")

    by_cat = defaultdict(list)
    for r in per_item_results:
        by_cat[r.get("category") or "UNKNOWN"].append(r)

    print(
        f"{'Category':<25} "
        f"{'N':>5} "
        f"{'Acc%':>8} "
        f"{'AvgQual':>10} "
        f"{'Corr':>10} "
        f"{'PercSucc%':>10}"
    )

    per_category = {}

    for cat in preferred_category_order(by_cat.keys()):
        items = by_cat[cat]
        n = len(items)

        cat_correct = sum(1 for r in items if r["is_correct"])
        cat_acc = cat_correct / n if n else 0.0

        cat_qs = [r["avg_quality"] for r in items if r["avg_quality"] is not None]
        cat_correctness = [
            1 if r["is_correct"] else 0
            for r in items
            if r["avg_quality"] is not None
        ]

        cat_mean_q = sum(cat_qs) / len(cat_qs) if cat_qs else None
        cat_corr = pearson_correlation(cat_qs, cat_correctness) if len(cat_qs) >= 2 else None

        cat_ps_items = [r for r in items if r.get("perception_success") in {"yes", "no"}]
        cat_ps_yes = sum(1 for r in cat_ps_items if r["perception_success"] == "yes")
        cat_ps_rate = cat_ps_yes / len(cat_ps_items) if cat_ps_items else None

        print(
            f"{cat:<25} "
            f"{n:>5} "
            f"{cat_acc * 100:>7.2f}% "
            f"{cat_mean_q if cat_mean_q is not None else float('nan'):>10.4f} "
            f"{cat_corr if cat_corr is not None else float('nan'):>10.4f} "
            f"{cat_ps_rate * 100 if cat_ps_rate is not None else float('nan'):>9.2f}%"
        )

        per_category[cat] = {
            "n": n,
            "correct": cat_correct,
            "accuracy": cat_acc,
            "mean_quality": cat_mean_q,
            "correlation_quality_vs_correctness": cat_corr,
            "perception_success_yes": cat_ps_yes,
            "perception_success_total": len(cat_ps_items),
            "perception_success_rate": cat_ps_rate,
        }

    attribution_categories, attribution_table = print_outcome_attribution_table(
        by_cat=by_cat,
        model_name=args.table_model_name,
    )

    correct_qs = [
        r["avg_quality"]
        for r in per_item_results
        if r["is_correct"] and r["avg_quality"] is not None
    ]
    wrong_qs = [
        r["avg_quality"]
        for r in per_item_results
        if not r["is_correct"] and r["avg_quality"] is not None
    ]

    correct_mean_q = sum(correct_qs) / len(correct_qs) if correct_qs else None
    wrong_mean_q = sum(wrong_qs) / len(wrong_qs) if wrong_qs else None

    print("\n========== Quality vs Correctness ==========")
    print(f"Correct items mean quality: {correct_mean_q}")
    print(f"Wrong items mean quality:   {wrong_mean_q}")

    summary = {
        "model_output": args.model_output,
        "dataset_root": args.dataset_root,
        "matched_items": matched,
        "skipped_no_gt": len(skipped_no_gt),
        "skipped_no_description": len(skipped_no_desc),
        "overall_accuracy": overall_accuracy,
        "mean_description_quality": mean_quality,
        "correlation_quality_vs_correctness": corr,
        "correct_items_mean_quality": correct_mean_q,
        "wrong_items_mean_quality": wrong_mean_q,
        "perception_success_yes": ps_yes,
        "perception_success_total": len(ps_items),
        "perception_success_rate": ps_rate,
        "per_category": per_category,
        "outcome_perception_attribution": {
            "categories": attribution_categories,
            "table": attribution_table,
        },
        "judge_method": "llm" if args.use_llm_judge else "token_overlap",
        "judge_model": args.judge_model if args.use_llm_judge else None,
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Wrote summary to: {summary_path}")


if __name__ == "__main__":
    main()