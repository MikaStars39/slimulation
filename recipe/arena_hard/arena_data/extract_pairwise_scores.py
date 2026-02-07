import argparse
import json
import re
from collections import defaultdict
from typing import Dict, List, Optional


RESULT_RE = re.compile(r"<result>(.*?)</result>", re.DOTALL | re.IGNORECASE)
_DISQUALIFY_MARKERS = [
    "<|im_end|>",
    "<|im_start|>",
    "assistant\n\n",
    "<thinking>",
    "<|thinking|>",
    "<|think|>",
]
_ZH_RE = re.compile(r"[\u4e00-\u9fff]")
_NON_ZH_EN_RE = re.compile(
    r"[\u00C0-\u017F\u0370-\u03FF\u0400-\u04FF\u0590-\u05FF"
    r"\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\u0900-\u097F"
    r"\u0E00-\u0E7F\u1100-\u11FF\u3040-\u30FF\u3130-\u318F"
    r"\u0B80-\u0BFF\u0C00-\u0C7F\u0D00-\u0D7F\u10A0-\u10FF]"
)


def _extract_result_block(text: str) -> Optional[Dict]:
    if not text:
        return None
    m = RESULT_RE.search(text)
    if not m:
        return None
    block = m.group(1).strip()
    start = block.find("{")
    end = block.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    json_text = block[start : end + 1]
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return None


def _get_judge_text(data: Dict) -> str:
    for key in ("response", "output", "judgment", "result", "text"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _strip_thinking(text: Optional[str]) -> str:
    if not text:
        return ""
    if "</thinking>" in text:
        return text.split("</thinking>", 1)[1].strip()
    return text.strip()


def _has_disqualify_markers(text: Optional[str]) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _DISQUALIFY_MARKERS)


def _has_chinese(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(_ZH_RE.search(text))


def _has_non_zh_en_script(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(_NON_ZH_EN_RE.search(text))


def _get_score(parsed: Dict) -> Optional[int]:
    if not isinstance(parsed, dict):
        return None
    score = parsed.get("score")
    if isinstance(score, int):
        return max(0, min(100, score))
    if isinstance(score, str) and score.isdigit():
        return max(0, min(100, int(score)))
    return None


def extract_topk(
    input_file: str,
    output_file: str,
    topk: int,
):
    groups: Dict[str, List[Dict]] = defaultdict(list)
    total = 0
    parsed_count = 0

    with open(input_file, "r", encoding="utf-8") as f_in:
        for line in f_in:
            if not line.strip():
                continue
            data = json.loads(line)
            total += 1

            judge_text = _get_judge_text(data)
            parsed = _extract_result_block(judge_text)
            score = _get_score(parsed)
            if score is None or score == 0:
                continue

            parsed_count += 1
            seed_uid = data.get("seed_uid") or data.get("uid") or "unknown"
            candidate_answer = _strip_thinking(data.get("candidate_answer"))
            if _has_disqualify_markers(candidate_answer):
                continue
            question = data.get("question")
            if not _has_chinese(question) and _has_chinese(candidate_answer):
                continue
            if _has_non_zh_en_script(question) or _has_non_zh_en_script(candidate_answer):
                continue
            record = {
                "uid": data.get("uid"),
                "seed_uid": seed_uid,
                "category": data.get("category"),
                "subcategory": data.get("subcategory"),
                "rollout_idx": data.get("rollout_idx"),
                "question": question,
                "candidate_answer": candidate_answer,
                "qwen_answer": data.get("qwen_answer"),
                "score": score,
                "better_than_qwen": parsed.get("better_than_qwen") if isinstance(parsed, dict) else None,
                "brief_reason": parsed.get("brief_reason") if isinstance(parsed, dict) else None,
            }
            groups[seed_uid].append(record)

    selected_counts = {}
    with open(output_file, "w", encoding="utf-8") as f_out:
        for seed_uid, items in groups.items():
            items.sort(key=lambda x: x.get("score", -1), reverse=True)
            selected = items if topk <= 0 else items[:topk]
            selected_counts[seed_uid] = len(selected)
            for item in selected:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    distribution = defaultdict(int)
    for count in selected_counts.values():
        distribution[count] += 1
    if distribution:
        dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(distribution.items()))
        print(f"[ExtractScores] selected_per_seed_distribution={{{dist_str}}}")

    print(
        f"[ExtractScores] total={total}, parsed={parsed_count}, "
        f"seeds={len(groups)}, output={output_file}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract judge scores and keep top-k per seed_uid."
    )
    parser.add_argument("--input", required=True, help="Input judge output JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL with top-k per seed")
    parser.add_argument("--topk", type=int, default=1, help="Top-k per seed (0 = all)")
    args = parser.parse_args()

    extract_topk(args.input, args.output, args.topk)


if __name__ == "__main__":
    main()
