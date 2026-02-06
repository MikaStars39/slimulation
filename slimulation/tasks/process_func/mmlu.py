import logging
from pathlib import Path
from io import TextIOWrapper
import json

from datasets import load_dataset
from tqdm import tqdm

from slimulation.tasks.base import DATASETS, get_question_text, get_answer_text, load_dataset_from_hf


def load_mmlu(
    dataset_name: str,
    cache_dir: str,
    k: int,
    f_out: TextIOWrapper,
):
    """Load the MMLU dataset (subset: all, split: test)."""
    hf_name = DATASETS[dataset_name]["hf_name"]
    split = DATASETS[dataset_name]["split"]
    subset = "all"

    def _load_subset():
        if cache_dir is not None:
            cache_dataset_name = hf_name.split("/")[-1]
            cache_path = Path(cache_dir) / cache_dataset_name
            logging.info(f"Cache path: {cache_path}")
            logging.info(f"Cache path exists: {cache_path.exists()}")
            if cache_path.exists():
                try:
                    return load_dataset(str(cache_path), subset, split=split)
                except Exception as e:
                    logging.info(f"Cache loading failed, falling back to HF: {e}")
        return load_dataset(hf_name, subset, split=split, cache_dir=cache_dir)

    dataset = _load_subset()

    for idx, row in enumerate(tqdm(dataset, desc=f"Loading {dataset_name}/{subset}")):
        question = row["question"] if "question" in row else get_question_text(row)
        options = row.get("choices") or row.get("options") or []

        # build questions with lettered options:
        if options:
            lettered_options = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]
            prompt = f"{question}\n\nOptions:\n" + "\n".join(lettered_options)
        else:
            prompt = question

        answer = row.get("answer", get_answer_text(row))
        if isinstance(answer, int):
            label = chr(65 + answer)
        else:
            answer_str = str(answer).strip()
            if answer_str.isdigit():
                label = chr(65 + int(answer_str))
            else:
                label = answer_str

        for sample_idx in range(k):
            # Create a unique ID for each attempt
            # Format: {dataset}_{original_index}_{attempt_index}
            unique_id = f"{dataset_name}_{idx}_{sample_idx}"

            record = {
                "id": unique_id,
                "question_id": f"{dataset_name}_{idx}",
                "source": dataset_name,
                "subset": subset,
                "subject": row.get("subject"),
                "prompt": prompt,  # 'prompt' key matches the offline engine script
                "sample_index": sample_idx,
                "need_llm_extract": DATASETS[dataset_name]["need_llm_extract"],
                "label": label,
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
