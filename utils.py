import os 
import json
from extract import Extractor
import re
from Levenshtein import distance

def save_response_to_json(args, items, scores=None):
    print("saving results...")

    if args.debug:
        task_dir = os.path.join(args.outputs_dir, "debug", args.task_name)
    else:
        task_dir = os.path.join(args.outputs_dir, args.task_name)
    os.makedirs(task_dir, exist_ok=True)

    # Determine model name
    path_parts = args.model_name_path.strip("/").split("/")
    if "checkpoint" in args.model_name_path:
        model_name = path_parts[-2]  # second-to-last part
    else:
        model_name = path_parts[-1]  # last part

    file_name = f"{model_name}_{args.duty_type}_{args.gen_engine}_{args.tag}.json"
    save_path = os.path.join(task_dir, file_name)

    base_name, ext = os.path.splitext(save_path)
    i = 0
    while os.path.exists(save_path):
        print(f"saving path exists, saving to 'original_file'_{i} instead")
        save_path = f"{base_name}_{i}{ext}"
        i += 1

    with open(save_path, "w", encoding="utf-8") as f:
        print(f"saving raw response to {save_path}")
        if hasattr(args, "save_path"):
            args.prev_path = args.save_path
            if args.delete_prev_file:
                print(f"deleting previous file {args.prev_path}")
                os.remove(args.save_path)
        args.save_path = save_path

        if args.duty_type == "raw":
            args.file_with_raw_response = args.save_path
        elif args.duty_type == "extract":
            args.file_with_extracted_response = args.save_path
        elif args.duty_type == "score":
            args.file_with_score = args.save_path

        json.dump({
            "score": scores,
            "parameters": vars(args),
            "result": items
        }, f, indent=2, ensure_ascii=False)
    
    # if args.duty_type == "raw":
    #     args.file_with_raw_response = args.save_path
    # elif args.duty_type == "extract":
    #     args.file_with_extracted_response = args.save_path
    # elif args.duty_type =="score":
    #     args.file_with_score = args.save_path

# for mathvista score calculation
def get_most_similar(prediction, choices):
    """
    Use the Levenshtein distance (or edit distance) to determine which of the choices is most similar to the given prediction
    """
    distances = [distance(prediction, choice) for choice in choices]
    ind = distances.index(min(distances))
    return choices[ind]

def normalize_extracted_answer(
    extraction, choices, question_type, answer_type, precision=1.0, ignore_empty_extractions=False
):
    """
    Normalize the extracted answer to match the answer type
    """
    if question_type == 'multi_choice':
        # make sure the extraction is a string
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            try:
                extraction = str(extraction)
            except Exception:
                extraction = ""

        # if the extraction is empty, return None
        if ignore_empty_extractions and not extraction:
            return None

        # extract "A" from "(A) text"
        letter = re.findall(r'\(([a-zA-Z])\)', extraction)
        if len(letter) > 0:
            extraction = letter[0].upper()

        sequential_characters = [chr(ord('A') + i) for i in range(len(choices))]

        # if model output a character, use it as index of available choices
        if extraction in sequential_characters:
            option_index = sequential_characters.index(extraction)
            normalized_extraction = choices[option_index]
        else:
            # select the most similar option
            # normalized_extraction = get_most_similar(extraction, choices)
            normalized_extraction = None #let's not allow this for precision
            
    elif answer_type == 'integer':
        try:
            normalized_extraction = str(int(float(extraction)))
        except Exception:
            normalized_extraction = None

    elif answer_type == 'float':
        try:
            normalized_extraction = str(round(float(extraction), int(precision)))
        except Exception:
            normalized_extraction = None

    elif answer_type == 'list':
        try:
            normalized_extraction = str(extraction)
        except Exception:
            normalized_extraction = None

    return normalized_extraction

def safe_equal(prediction, answer):
    """
    Check if the prediction is equal to the answer, even if they are of different types
    """
    
    try:
        # prediction = prediction.strip().lower().replace(" ", "_")
        # if str(answer).lower() in str(prediction).lower():
        #     return True
        if str(prediction).lower() == str(answer).lower():
            return True
        return False
    except Exception as e:
        return False