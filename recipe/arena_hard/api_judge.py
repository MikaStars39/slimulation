"""
API Judge: Compare two model outputs using LLM-as-a-judge.

This script performs pairwise comparison between two models:
1. Loads answers from both models
2. For each question, runs 2 rounds of judgment (swap positions to remove bias)
3. Calls judge API (OpenAI-compatible) to evaluate which answer is better
4. Saves judgment results in Arena-Hard-Auto format

Usage:
    python api_judge.py \
        --model-a /path/to/model_a.jsonl \
        --model-b /path/to/model_b.jsonl \
        --output /path/to/judgments.jsonl \
        --judge-model gpt-4.1 \
        --api-key YOUR_API_KEY \
        --api-base https://api.openai.com/v1 \
        --parallel 8
"""

import json
import argparse
import asyncio
import logging
import os
import re
import subprocess
import sys
try:
    import yaml
except Exception:  # Optional dependency for strict Arena-Hard alignment
    yaml = None
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm_asyncio

# Import judge settings from prompt.py
from prompt import JUDGE_SETTINGS


# ============================================================================
# Judge Prompt Template
# ============================================================================


JUDGE_USER_PROMPT_TEMPLATE = """<|User Prompt|>
{QUESTION}

<|The Start of Assistant A's Answer|>
{ANSWER_A}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{ANSWER_B}
<|The End of Assistant B's Answer|>"""


# Regex patterns to extract judgment score
SCORE_PATTERNS = [
    r"\[\[([AB<>=]+)\]\]",  # [[A>>B]]
    r"\[([AB<>=]+)\]",      # [A>>B]
]


def _iter_records_from_line(line: str):
    stripped = line.strip()
    if not stripped or stripped in ("[", "]"):
        return
    if stripped.endswith(","):
        stripped = stripped[:-1]
    try:
        yield json.loads(stripped)
        return
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    idx = 0
    length = len(stripped)
    while idx < length:
        while idx < length and stripped[idx].isspace():
            idx += 1
        if idx >= length:
            break
        if stripped[idx] == ",":
            idx += 1
            continue
        obj, end = decoder.raw_decode(stripped, idx)
        yield obj
        idx = end


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Answer:
    """Model answer with metadata."""
    uid: str
    model: str
    content: str
    
    @classmethod
    def from_jsonl(cls, data: Dict) -> 'Answer':
        """
        Load answer from JSONL format.
        Supports both inference output format and Arena-Hard-Auto format.
        """
        # Try inference output format first (has 'response' field)
        if "response" in data:
            content = data["response"]
        # Fall back to Arena-Hard-Auto format
        elif "messages" in data:
            content = data["messages"][-1]["content"]
            if isinstance(content, dict):
                content = content.get("answer", "")
        else:
            raise ValueError(f"Cannot extract answer from data: {data.keys()}")
        
        return cls(
            uid=data["uid"],
            model=data.get("model", "unknown"),
            content=content
        )


@dataclass
class Question:
    """Question with original prompt and category."""
    uid: str
    prompt: str
    category: str
    
    def get_baseline_model(self) -> str:
        """Get baseline model for this question's category."""
        return JUDGE_SETTINGS.get(self.category, JUDGE_SETTINGS["hard_prompt"])["baseline"]
    
    def get_system_prompt(self) -> str:
        """Get judge system prompt for this question's category."""
        return JUDGE_SETTINGS.get(self.category, JUDGE_SETTINGS["hard_prompt"])["system_prompt"]


# ============================================================================
# API Client
# ============================================================================

class JudgeAPIClient:
    """Async API client for judge model."""
    
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 60
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
    
    async def judge(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        system_prompt: str,
        session: aiohttp.ClientSession
    ) -> Optional[str]:
        """
        Call judge API to compare two answers.
        
        Args:
            question: User's question
            answer_a: First model's answer
            answer_b: Second model's answer
            system_prompt: Judge system prompt (category-specific)
            session: aiohttp session for connection pooling
        
        Returns:
            Judge's response text, or None if API call fails
        """
        # Build messages
        user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
            QUESTION=question,
            ANSWER_A=answer_a,
            ANSWER_B=answer_b
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # API request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Call API with retry logic
        for attempt in range(3):
            try:
                async with session.post(
                    f"{self.api_base}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, dict) and "error" in data:
                            logging.warning(
                                "API error payload (attempt %d/3): %s",
                                attempt + 1,
                                data.get("error")
                            )
                            continue
                        content = extract_content_from_response(data)
                        if content is not None:
                            return content
                        logging.warning(
                            "API response missing content (attempt %d/3): keys=%s",
                            attempt + 1,
                            list(data.keys())
                        )
                    else:
                        error_text = await response.text()
                        logging.warning(
                            f"API error (attempt {attempt+1}/3): "
                            f"status={response.status}, body={error_text}"
                        )
            except asyncio.TimeoutError:
                logging.warning(f"Timeout (attempt {attempt+1}/3)")
            except Exception as e:
                logging.warning(f"API call failed (attempt {attempt+1}/3): {e}")
            
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        logging.error(f"Failed to get judgment after 3 attempts")
        return None


# ============================================================================
# Score Extraction
# ============================================================================

def extract_score(judgment: str) -> Optional[str]:
    """
    Extract judgment score from judge's response.
    
    Args:
        judgment: Judge model's output text
    
    Returns:
        Score string like "A>>B", "A>B", "A=B", "B>A", "B>>A", or None
    """
    for pattern in SCORE_PATTERNS:
        matches = re.findall(pattern, judgment.upper())
        matches = [m for m in matches if m]  # Remove empty strings
        
        if matches:
            return matches[-1].strip()  # Return last match
    
    return None


def extract_content_from_response(data: Dict) -> Optional[str]:
    """
    Extract assistant content from various OpenAI-compatible response formats.
    """
    if not isinstance(data, dict):
        return None
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0] or {}
        message = choice.get("message")
        if isinstance(message, dict) and "content" in message:
            return message.get("content")
        if "text" in choice:
            return choice.get("text")
    if "output" in data:
        return data.get("output")
    if "answer" in data:
        return data.get("answer")
    return None


# ============================================================================
# Judgment Pipeline
# ============================================================================

async def judge_single_question(
    question: Question,
    answer_a: Answer,
    answer_b: Answer,
    client: JudgeAPIClient,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    system_prompt: str
) -> Optional[Dict]:
    """
    Judge a single question with 2 rounds (swap positions).
    
    Args:
        question: Question object
        answer_a: First model's answer
        answer_b: Second model's answer
        client: API client
        session: aiohttp session
        semaphore: Semaphore for rate limiting
        system_prompt: System prompt for judge
    
    Returns:
        Judgment result dict in Arena-Hard-Auto format, or None if failed
    """
    async with semaphore:
        # Round 1: A vs B
        judgment_1 = await client.judge(
            question.prompt,
            answer_a.content,
            answer_b.content,
            system_prompt,
            session
        )
        
        if judgment_1 is None:
            return None
        
        score_1 = extract_score(judgment_1)
        
        # Round 2: B vs A (swap positions)
        judgment_2 = await client.judge(
            question.prompt,
            answer_b.content,
            answer_a.content,
            system_prompt,
            session
        )
        
        if judgment_2 is None:
            return None
        
        score_2 = extract_score(judgment_2)
        
        # Build result in Arena-Hard-Auto format
        result = {
            "uid": question.uid,
            "category": question.category,
            "judge": client.model,
            "model": answer_b.model,
            "baseline": answer_a.model,
            "games": [
                {
                    "score": score_1,
                    "judgment": {"answer": judgment_1},
                    "prompt": None  # Optional: can save full prompt
                },
                {
                    "score": score_2,
                    "judgment": {"answer": judgment_2},
                    "prompt": None
                }
            ]
        }
        
        return result


async def run_judgments(
    questions: List[Question],
    answers_a: Dict[str, Answer],
    answers_b: Dict[str, Answer],
    client: JudgeAPIClient,
    output_file: str,
    parallel: int = 8,
    resume: bool = False,
    category: str = "hard_prompt"
):
    """
    Run all judgments with parallel API calls.
    
    Args:
        questions: List of questions
        answers_a: Model A answers (baseline), keyed by uid
        answers_b: Model B answers (test model), keyed by uid
        client: API client
        output_file: Output JSONL file path
        parallel: Max concurrent API calls
        resume: Skip already judged questions
        category: Question category for system prompt selection
    """
    # Get system prompt for this category
    system_prompt = JUDGE_SETTINGS.get(category, JUDGE_SETTINGS["hard_prompt"])["system_prompt"]
    # Load existing judgments if resuming
    existing_uids = set()
    if resume and Path(output_file).exists():
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    existing_uids.add(data["uid"])
        logging.info(f"Resuming: skipping {len(existing_uids)} existing judgments")
    
    # Filter questions that need judgment
    pending_questions = [
        q for q in questions
        if q.uid not in existing_uids
        and q.uid in answers_a
        and q.uid in answers_b
    ]
    pending_questions = [q for q in pending_questions if q.category == category]
    
    if not pending_questions:
        logging.info("No pending questions to judge")
        return
    
    logging.info(f"Judging {len(pending_questions)} questions with {parallel} parallel calls")
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(parallel)
    
    # Open output file in append mode
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            judge_single_question(
                question=q,
                answer_a=answers_a[q.uid],
                answer_b=answers_b[q.uid],
                client=client,
                session=session,
                semaphore=semaphore,
                system_prompt=system_prompt
            )
            for q in pending_questions
        ]
        
        # Run with progress bar
        with open(output_file, 'a', encoding='utf-8') as f_out:
            for coro in tqdm_asyncio(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Judging"
            ):
                result = await coro
                
                if result is not None:
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f_out.flush()


# ============================================================================
# Data Loading
# ============================================================================

def load_questions(question_file: str) -> List[Question]:
    """Load questions from JSONL file."""
    questions = []
    with open(question_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            for data in _iter_records_from_line(line):
                questions.append(Question(
                    uid=data["uid"],
                    prompt=data["prompt"],
                    category=data.get("category", "hard_prompt")
                ))
    logging.info(f"Loaded {len(questions)} questions from {question_file}")
    return questions


def load_answers(answer_file: str) -> Dict[str, Answer]:
    """Load model answers from JSONL file."""
    answers = {}
    with open(answer_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                answer = Answer.from_jsonl(data)
                answers[answer.uid] = answer
    
    model_name = list(answers.values())[0].model if answers else "unknown"
    logging.info(f"Loaded {len(answers)} answers for model: {model_name}")
    return answers


def run_show_result(
    judgment_file: str,
    baseline_model: str,
    output: str,
    answer_dir: str = None,
    control_features: List[str] = None
):
    """Call show_result.py as a subprocess."""
    script_path = os.path.join(os.path.dirname(__file__), "show_result.py")
    cmd = [
        sys.executable, script_path,
        "--judgment-file", judgment_file,
        "--baseline-model", baseline_model,
        "--output", output,
    ]
    if control_features:
        cmd += ["--answer-dir", answer_dir, "--control-features"] + control_features
    subprocess.run(cmd, check=True)


# ============================================================================
# Main Entry
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Judge two models using LLM-as-a-judge API"
    )
    
    # Input files
    parser.add_argument("--questions", type=str, required=True)
    parser.add_argument("--model-a", type=str, required=True)
    parser.add_argument("--model-b", type=str, required=True)
    parser.add_argument("--output", type=str, required=False,
                        help="Single output JSONL (when using --category)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for multi-category mode")
    parser.add_argument("--output-prefix", type=str, default=None,
                        help="Output file prefix for multi-category mode")
    parser.add_argument("--resume", action="store_true")
    
    # Judge API config
    parser.add_argument("--judge-model", type=str, default="gpt-4.1")
    parser.add_argument("--api-base", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    
    # Runtime config
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=60)
    
    # Category for system prompt selection
    parser.add_argument("--category", type=str, default="hard_prompt",
                        choices=["hard_prompt", "coding", "math", "creative_writing"],
                        help="Question category (determines system prompt)")
    parser.add_argument("--categories", nargs="+",
                        choices=["hard_prompt", "coding", "math", "creative_writing"],
                        help="Multiple categories to judge in one run")

    # Optional scoring with show_result.py
    parser.add_argument("--score-output-dir", type=str, default=None,
                        help="Output directory for per-category leaderboards")
    parser.add_argument("--baseline-model", type=str, default=None,
                        help="Baseline model name for scoring")
    parser.add_argument("--control-features", nargs="+", default=None,
                        choices=["length", "markdown"],
                        help="Enable style control features")
    parser.add_argument("--answer-dir", type=str, default=None,
                        help="Directory with model answer JSONL files")
    parser.add_argument("--arena-hard-config", type=str, default=None,
                        help="Arena-Hard-Auto config YAML for strict alignment")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Optional strict alignment with Arena-Hard-Auto config
    if args.arena_hard_config:
        if yaml is None:
            raise ImportError("pyyaml is required for --arena-hard-config")
        with open(args.arena_hard_config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        global JUDGE_USER_PROMPT_TEMPLATE, SCORE_PATTERNS
        JUDGE_USER_PROMPT_TEMPLATE = cfg.get("prompt_template", JUDGE_USER_PROMPT_TEMPLATE)
        SCORE_PATTERNS = cfg.get("regex_patterns", SCORE_PATTERNS)
        args.judge_model = cfg.get("judge_model", args.judge_model)
        args.temperature = cfg.get("temperature", args.temperature)
        args.max_tokens = cfg.get("max_tokens", args.max_tokens)
        logging.info(
            "Arena-Hard config loaded: model=%s, temperature=%s, max_tokens=%s",
            args.judge_model, args.temperature, args.max_tokens
        )
    
    # Load data
    questions = load_questions(args.questions)
    answers_a = load_answers(args.model_a)
    answers_b = load_answers(args.model_b)
    
    # Create API client
    client = JudgeAPIClient(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.judge_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout
    )
    
    if args.categories:
        if not args.output_dir or not args.output_prefix:
            raise ValueError("--output-dir and --output-prefix are required with --categories")
        if args.control_features and not args.answer_dir:
            raise ValueError("--answer-dir is required when using --control-features")
        if args.score_output_dir and not args.baseline_model:
            raise ValueError("--baseline-model is required when scoring")

        for cat in args.categories:
            output_file = os.path.join(args.output_dir, f"{args.output_prefix}_{cat}.jsonl")
            await run_judgments(
                questions=questions,
                answers_a=answers_a,
                answers_b=answers_b,
                client=client,
                output_file=output_file,
                parallel=args.parallel,
                resume=args.resume,
                category=cat
            )
            logging.info(f"✓ Judgments saved to: {output_file}")

            if args.score_output_dir:
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    leaderboard = os.path.join(
                        args.score_output_dir,
                        f"leaderboard_{args.judge_model}_{cat}.json"
                    )
                    run_show_result(
                        judgment_file=output_file,
                        baseline_model=args.baseline_model,
                        output=leaderboard,
                        answer_dir=args.answer_dir,
                        control_features=args.control_features
                    )
                else:
                    logging.warning(f"Skipping scoring for {cat}: no judgments found")
        return

    if not args.output:
        raise ValueError("--output is required when using single --category")

    await run_judgments(
        questions=questions,
        answers_a=answers_a,
        answers_b=answers_b,
        client=client,
        output_file=args.output,
        parallel=args.parallel,
        resume=args.resume,
        category=args.category
    )
    logging.info(f"✓ Judgments saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
