"""Load VRIQ and DiagVRIQ from the Hugging Face Hub.

Repos:
  tina-khezresmaeilzadeh/VRIQ
  tina-khezresmaeilzadeh/DiagVRIQ

Images are cached by huggingface_hub. Rows with a blank question or answer
are skipped.
"""

import csv

from huggingface_hub import hf_hub_download
from PIL import Image

VRIQ_REPO = "tina-khezresmaeilzadeh/VRIQ"
DIAG_REPO = "tina-khezresmaeilzadeh/DiagVRIQ"

MAX_PIXELS = 250_000
MIN_SIDE = 28


def use_hub(root) -> bool:
    """True when the caller did not pass a local dataset directory."""
    import os

    return not root or not os.path.isdir(str(root))


def _resize(img: Image.Image) -> Image.Image:
    w, h = img.size
    total = w * h
    if total > MAX_PIXELS:
        scale = (MAX_PIXELS / total) ** 0.5
        w = max(1, int(w * scale))
        h = max(1, int(h * scale))
        img = img.resize((w, h), Image.LANCZOS)
    if w < MIN_SIDE or h < MIN_SIDE:
        w = max(w, MIN_SIDE)
        h = max(h, MIN_SIDE)
        img = img.resize((w, h), Image.LANCZOS)
    return img


def _download(repo: str, filename: str) -> str:
    return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset")


def _read_metadata(repo: str, folder: str):
    path = _download(repo, f"{folder}/metadata.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _image_path(repo: str, folder: str, file_name: str) -> str:
    return _download(repo, f"{folder}/{file_name}")


def _clean_row(row):
    question = (row.get("question") or "").strip()
    answer = (row.get("answer") or "").strip()
    file_name = (row.get("file_name") or "").strip()
    item_id = (row.get("id") or "").strip()
    category = (row.get("category") or "").strip()
    return question, answer, file_name, item_id, category


def load_vriq(split: str, decode: bool = True):
    """End-to-end records for one VRIQ split: abstract or natural."""
    if split not in ("abstract", "natural"):
        raise ValueError(f"split must be abstract or natural, got {split!r}")

    data = []
    missing = []
    for idx, row in enumerate(_read_metadata(VRIQ_REPO, split)):
        question, answer, file_name, item_id, category = _clean_row(row)
        if not question or not answer or not file_name:
            missing.append({"id": item_id, "row_index": idx, "reason": "blank"})
            continue
        img_path = _image_path(VRIQ_REPO, split, file_name)
        img = None
        if decode:
            img = _resize(Image.open(img_path).convert("RGB"))
        data.append({
            "pid": item_id,
            "id": item_id,
            "category": category,
            "question": question,
            "question_prompt": question,
            "ground_truth": answer,
            "answer": answer,
            "split": split,
            "decoded_image": img,
            "image_path": img_path,
            "source_folder": split,
            "row_index": idx,
        })
    if not data:
        raise RuntimeError(f"No items loaded from {VRIQ_REPO} split={split}.")
    return data, missing


def load_vriq_light(split: str):
    """Same records as load_vriq, without decoding images."""
    return load_vriq(split, decode=False)


def load_diag(make_prompt=None):
    """DiagVRIQ records. make_prompt(question, gt_description) sets question_prompt."""
    data = []
    missing = []
    for idx, row in enumerate(_read_metadata(DIAG_REPO, "data")):
        question, answer, file_name, item_id, category = _clean_row(row)
        desc = (row.get("gt_description") or "").strip()
        if not question or not answer or not file_name or not desc:
            missing.append({"id": item_id, "row_index": idx, "reason": "blank"})
            continue
        img_path = _image_path(DIAG_REPO, "data", file_name)
        img = _resize(Image.open(img_path).convert("RGB"))
        prompt = make_prompt(question, desc) if make_prompt else question
        data.append({
            "pid": item_id,
            "id": item_id,
            "category": category,
            "question": question,
            "original_question": question,
            "question_prompt": prompt,
            "ground_truth": answer,
            "answer": answer,
            "gt_description": desc,
            "decoded_image": img,
            "image_path": img_path,
            "source_folder": "data",
            "row_index": idx,
            "split": "natural",
        })
    if not data:
        raise RuntimeError(f"No items loaded from {DIAG_REPO}.")
    return data, missing


def diag_gt_map():
    """pid -> verified description, for the perception judge."""
    gt_map = {}
    for row in _read_metadata(DIAG_REPO, "data"):
        item_id = (row.get("id") or "").strip()
        desc = (row.get("gt_description") or "").strip()
        if item_id and desc:
            gt_map[item_id] = desc
    return gt_map
