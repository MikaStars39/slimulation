"""
Calculate win rates and confidence intervals from judgment results.

This script implements the Arena-Hard-Auto scoring methodology:
1. Load judgment results
2. Convert judgment labels to win/loss scores
3. Bootstrap sampling to compute confidence intervals
4. Output leaderboard with scores and CI

Usage:
    python show_result.py \
        --judgment-dir /path/to/judgments \
        --baseline-model o3-mini-2025-01-31
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm


# ============================================================================
# Label to Score Mapping
# ============================================================================

def label_to_scores(label: str, weight: int = 3) -> List[float]:
    """
    Convert judgment label to score list.
    
    This matches Arena-Hard-Auto's exact mapping with all variants.
    
    Args:
        label: Judgment label (e.g., "A>>B", "A>B", "A=B")
        weight: Weight for strong preference (>>)
    
    Returns:
        List of scores (1.0 = win, 0.5 = tie, 0.0 = loss)
    """
    mapping = {
        # A wins variants
        "A>>B": [1.0] * weight,  # A significantly better
        "A>B": [1.0],            # A slightly better
        # Tie variants
        "A=B": [0.5],            # Tie
        "B=A": [0.5],            # Tie (alternative notation)
        # A loses variants
        "A<<B": [0.0] * weight,  # A significantly worse
        "A<B": [0.0],            # A slightly worse
        # B wins variants (same as A loses)
        "B>>A": [0.0] * weight,  # B significantly better
        "B>A": [0.0],            # B slightly better
        # B loses variants (same as A wins)
        "B<<A": [1.0] * weight,  # B significantly worse
        "B<A": [1.0],            # B slightly worse
    }
    
    return mapping.get(label, [0.5])  # Default to tie if unknown


# ============================================================================
# Load Judgments
# ============================================================================

def load_judgments(judgment_file: str) -> List[Dict]:
    """
    Load judgment results from JSONL file.
    
    Returns:
        List of judgment records
    """
    judgments = []
    with open(judgment_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                judgments.append(data)
    
    return judgments


def build_battles(judgments: List[Dict], baseline_model: str) -> List[Dict]:
    """
    Build expanded battles list from judgments.
    
    Each battle entry corresponds to one score (after expanding strong preferences).
    """
    battles = []
    for judgment in judgments:
        model = judgment["model"]
        baseline = judgment.get("baseline", baseline_model)
        uid = judgment.get("uid")
        games = judgment.get("games", [])
        
        if uid is None or len(games) != 2:
            continue
        
        round1_label = games[0].get("score")
        round2_label = games[1].get("score")
        
        if round1_label is None or round2_label is None:
            continue
        
        round1_scores = label_to_scores(round1_label)
        round2_scores = label_to_scores(round2_label)
        
        # Arena-Hard-Auto formula:
        # scores = round2_scores + [1 - s for s in round1_scores]
        flipped_round1 = [1.0 - s for s in round1_scores]
        all_scores = round2_scores + flipped_round1
        
        for score in all_scores:
            battles.append({
                "uid": uid,
                "model": model,
                "baseline": baseline,
                "score": float(score),
            })
    
    return battles


def extract_scores_from_judgments(judgments: List[Dict]) -> Dict[str, List[float]]:
    """
    Extract scores for each model from judgment results.
    
    This follows Arena-Hard-Auto's exact methodology:
    - Round 1: baseline (A) vs model (B) -> if A>B, baseline wins, so model score = 1-1 = 0
    - Round 2: model (A) vs baseline (B) -> if A>B, model wins, so model score = 1
    
    Formula: scores = round2_scores + [1 - s for s in round1_scores]
    
    Args:
        judgments: List of judgment records
    
    Returns:
        Dict mapping model_name -> list of scores
    """
    model_scores = {}
    
    for judgment in judgments:
        model = judgment["model"]
        games = judgment.get("games", [])
        
        if len(games) != 2:
            continue  # Skip incomplete judgments
        
        # Round 1: baseline (A) vs model (B)
        round1_label = games[0].get("score")
        # Round 2: model (A) vs baseline (B) - positions swapped
        round2_label = games[1].get("score")
        
        if round1_label is None or round2_label is None:
            continue
        
        # Convert labels to score lists
        round1_scores = label_to_scores(round1_label)
        round2_scores = label_to_scores(round2_label)
        
        # Apply Arena-Hard-Auto formula:
        # scores = round2_scores + [1 - s for s in round1_scores]
        # 
        # Why?
        # - Round 1: baseline is A, model is B
        #   If judge says "A>B", baseline wins, model should get 0
        #   So we need to flip: 1 - 1 = 0 ✓
        # - Round 2: model is A, baseline is B
        #   If judge says "A>B", model wins, model should get 1
        #   So we use directly: 1 ✓
        
        flipped_round1 = [1.0 - s for s in round1_scores]
        all_scores = round2_scores + flipped_round1
        
        if model not in model_scores:
            model_scores[model] = []
        
        model_scores[model].extend(all_scores)
    
    return model_scores


# ============================================================================
# Bootstrap Confidence Intervals
# ============================================================================

def bootstrap_mean(scores: List[float], num_rounds: int = 100) -> Tuple[float, float, float]:
    """
    Calculate mean and confidence intervals using bootstrap sampling.
    
    This matches Arena-Hard-Auto's methodology:
    1. For each round, sample with replacement from all scores
    2. Calculate mean of the sample
    3. Repeat num_rounds times
    4. Final mean = mean of bootstrap means
    5. CI = 5th and 95th percentiles of bootstrap means
    
    Args:
        scores: List of scores
        num_rounds: Number of bootstrap rounds
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    scores = np.array(scores)
    n = len(scores)
    
    bootstrap_means = []
    for _ in range(num_rounds):
        # Sample with replacement (frac=1.0 means sample n items)
        sample = np.random.choice(scores, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Mean of bootstrap means (matches pandas groupby.mean() on bootstraps)
    mean = np.mean(bootstrap_means)
    # Confidence intervals from bootstrap distribution
    lower = np.percentile(bootstrap_means, 5)   # 5th percentile
    upper = np.percentile(bootstrap_means, 95)  # 95th percentile
    
    return mean, lower, upper


# ============================================================================
# Leaderboard
# ============================================================================

def create_leaderboard(
    model_scores: Dict[str, List[float]],
    baseline_model: str
) -> List[Dict]:
    """
    Create leaderboard with scores and confidence intervals.
    
    Args:
        model_scores: Dict mapping model -> scores
        baseline_model: Baseline model name (fixed at 50%)
    
    Returns:
        List of leaderboard entries sorted by score
    """
    leaderboard = []
    
    for model, scores in model_scores.items():
        if len(scores) == 0:
            continue
        
        mean, lower, upper = bootstrap_mean(scores, num_rounds=100)
        
        leaderboard.append({
            "model": model,
            "score": mean * 100,  # Convert to percentage
            "ci_lower": (mean - lower) * 100,
            "ci_upper": (upper - mean) * 100
        })
    
    # Add baseline model with fixed 50%
    leaderboard.append({
        "model": baseline_model,
        "score": 50.0,
        "ci_lower": 0.0,
        "ci_upper": 0.0
    })
    
    # Sort by score descending
    leaderboard.sort(key=lambda x: x["score"], reverse=True)
    
    return leaderboard


def print_leaderboard(leaderboard: List[Dict]):
    """Pretty print leaderboard."""
    print("\n" + "="*70)
    print(f"{'Model':<40} {'Score (%)':<12} {'CI (%)':<15}")
    print("="*70)
    
    for idx, entry in enumerate(leaderboard):
        model = entry["model"]
        score = entry["score"]
        ci_lower = entry["ci_lower"]
        ci_upper = entry["ci_upper"]
        
        ci_str = f"(-{ci_lower:.1f} / +{ci_upper:.1f})"
        
        print(f"{idx:<3} {model:<37} {score:>6.1f}      {ci_str:<15}")
    
    print("="*70 + "\n")


# ============================================================================
# Style Control (Optional)
# ============================================================================

def load_model_metadata(answer_dir: str) -> Dict[str, Dict[str, Dict]]:
    """
    Load metadata from model answer files.
    
    Returns:
        model_metadata[model_name][uid] = metadata
    """
    model_metadata: Dict[str, Dict[str, Dict]] = {}
    answer_path = Path(answer_dir)
    
    for file_path in answer_path.glob("*.jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                uid = data.get("uid")
                metadata = data.get("metadata")
                if uid is None or metadata is None:
                    continue
                
                model_name = data.get("model", file_path.stem)
                if model_name not in model_metadata:
                    model_metadata[model_name] = {}
                
                model_metadata[model_name][uid] = metadata
    
    return model_metadata


def summarize_metadata(metadata: Dict) -> Tuple[float, float, float, float]:
    """
    Summarize metadata into numeric features.
    
    Returns:
        token_len, header_total, list_total, bold_total
    """
    token_len = float(metadata.get("token_len", 0))
    header_total = float(sum(metadata.get("header_count", {}).values()))
    list_total = float(sum(metadata.get("list_count", {}).values()))
    bold_total = float(sum(metadata.get("bold_count", {}).values()))
    
    return token_len, header_total, list_total, bold_total


def one_hot_encode(items: List[str], baseline: str) -> Tuple[np.ndarray, List[str]]:
    """
    One-hot encode model names with baseline as -1.
    """
    unique_items = sorted(set(items + [baseline]))
    item_to_index = {item: idx for idx, item in enumerate(unique_items)}
    
    matrix = []
    for item in items:
        vec = [0.0] * len(unique_items)
        vec[item_to_index[item]] = 1.0
        vec[item_to_index[baseline]] = -1.0
        matrix.append(vec)
    
    return np.array(matrix, dtype=np.float32), unique_items


def prepare_style_features(
    battles: List[Dict],
    model_metadata: Dict[str, Dict[str, Dict]],
    control_features: List[str],
    baseline_model: str
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """
    Prepare feature matrix and outcomes for style control.
    
    Returns:
        features, outcomes, unique_models, num_style_features
    """
    feature_rows = []
    outcomes = []
    models = []
    
    for row in battles:
        uid = row["uid"]
        model = row["model"]
        baseline = row.get("baseline", baseline_model)
        
        if model not in model_metadata or baseline not in model_metadata:
            continue
        if uid not in model_metadata[model] or uid not in model_metadata[baseline]:
            continue
        
        model_meta = model_metadata[model][uid]
        base_meta = model_metadata[baseline][uid]
        
        m_len, m_header, m_list, m_bold = summarize_metadata(model_meta)
        b_len, b_header, b_list, b_bold = summarize_metadata(base_meta)
        
        # Length difference (normalized)
        length_diff = 0.0
        if m_len + b_len > 0:
            length_diff = (m_len - b_len) / (m_len + b_len)
        
        # Markdown density differences (normalized)
        m_header_d = m_header / (m_len + 1.0)
        b_header_d = b_header / (b_len + 1.0)
        header_diff = (m_header_d - b_header_d) / (m_header_d + b_header_d + 1.0)
        
        m_list_d = m_list / (m_len + 1.0)
        b_list_d = b_list / (b_len + 1.0)
        list_diff = (m_list_d - b_list_d) / (m_list_d + b_list_d + 1.0)
        
        m_bold_d = m_bold / (m_len + 1.0)
        b_bold_d = b_bold / (b_len + 1.0)
        bold_diff = (m_bold_d - b_bold_d) / (m_bold_d + b_bold_d + 1.0)
        
        style_features = []
        if "length" in control_features:
            style_features.append(length_diff)
        if "markdown" in control_features:
            style_features.extend([header_diff, list_diff, bold_diff])
        
        if not style_features:
            continue
        
        feature_rows.append(style_features)
        outcomes.append(row["score"])
        models.append(model)
    
    if not feature_rows:
        raise ValueError("No valid style features could be constructed.")
    
    # Normalize style features (z-score)
    style_features = np.array(feature_rows, dtype=np.float32)
    mean = style_features.mean(axis=0)
    std = style_features.std(axis=0)
    std[std == 0] = 1.0
    style_features = (style_features - mean) / std
    
    # One-hot encode models
    model_features, unique_models = one_hot_encode(models, baseline_model)
    
    # Combine features
    all_features = np.concatenate([model_features, style_features], axis=1)
    outcomes = np.array(outcomes, dtype=np.float32)
    
    return all_features, outcomes, unique_models, style_features.shape[1]


def fit_bt(
    features: np.ndarray,
    outcomes: np.ndarray,
    indices: Optional[np.ndarray] = None,
    lr: float = 0.1,
    max_epochs: int = 50
) -> np.ndarray:
    """
    Fit Bradley-Terry model using logistic regression.
    """
    import torch
    import torch.nn.functional as F
    
    feats = features if indices is None else features[indices]
    outs = outcomes if indices is None else outcomes[indices]
    
    feats_t = torch.tensor(feats, dtype=torch.float32)
    outs_t = torch.tensor(outs, dtype=torch.float32)
    
    logits = torch.nn.Parameter(torch.zeros(feats_t.shape[1]) + 0.5)
    optimizer = torch.optim.LBFGS(
        [logits],
        lr=lr,
        max_iter=max_epochs,
        tolerance_grad=1e-9,
        tolerance_change=1e-9,
    )
    
    def closure():
        optimizer.zero_grad()
        pred = feats_t @ logits
        loss = F.binary_cross_entropy_with_logits(pred, outs_t, reduction="sum")
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    return logits.detach().cpu().numpy()


def bootstrap_bt(
    features: np.ndarray,
    outcomes: np.ndarray,
    num_rounds: int = 100
) -> np.ndarray:
    """
    Bootstrap Bradley-Terry coefficients.
    """
    n = features.shape[0]
    coefs = []
    
    for _ in tqdm(range(num_rounds), desc="BT bootstrap"):
        indices = np.random.randint(0, n, size=n)
        coef = fit_bt(features, outcomes, indices=indices)
        coefs.append(coef)
    
    return np.stack(coefs, axis=0)


def to_winrate_probabilities(
    coefs: np.ndarray,
    models: List[str],
    baseline_model: str
) -> np.ndarray:
    """
    Convert model coefficients to winrate probabilities vs baseline.
    """
    baseline_idx = models.index(baseline_model)
    
    exp_coefs = np.exp(coefs)
    denom = exp_coefs + exp_coefs[:, [baseline_idx]]
    probs = exp_coefs / denom
    probs[:, baseline_idx] = 0.5
    
    return probs


def create_leaderboard_style(
    probs: np.ndarray,
    models: List[str]
) -> List[Dict]:
    """
    Create leaderboard from bootstrap probability table.
    """
    median = np.quantile(probs, 0.5, axis=0)
    lower = np.quantile(probs, 0.05, axis=0)
    upper = np.quantile(probs, 0.95, axis=0)
    
    leaderboard = []
    for idx, model in enumerate(models):
        leaderboard.append({
            "model": model,
            "score": median[idx] * 100,
            "ci_lower": (median[idx] - lower[idx]) * 100,
            "ci_upper": (upper[idx] - median[idx]) * 100,
        })
    
    leaderboard.sort(key=lambda x: x["score"], reverse=True)
    return leaderboard


# ============================================================================
# Main Entry
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calculate win rates from judgment results"
    )
    
    parser.add_argument("--judgment-file", type=str, required=True,
                        help="Path to judgment JSONL file")
    parser.add_argument("--baseline-model", type=str, 
                        default="o3-mini-2025-01-31",
                        help="Baseline model name")
    parser.add_argument("--output", type=str, default=None,
                        help="Save leaderboard to JSON file")
    parser.add_argument("--control-features", nargs="+", default=[],
                        choices=["length", "markdown"],
                        help="Enable style control features")
    parser.add_argument("--answer-dir", type=str, default=None,
                        help="Directory with model answer JSONL files (required for style control)")
    
    args = parser.parse_args()
    
    print(f"Loading judgments from: {args.judgment_file}")
    judgments = load_judgments(args.judgment_file)
    print(f"Loaded {len(judgments)} judgments")
    
    if args.control_features:
        if not args.answer_dir:
            raise ValueError("--answer-dir is required when using --control-features")
        
        print(f"Style control enabled: {args.control_features}")
        print("Loading model metadata...")
        model_metadata = load_model_metadata(args.answer_dir)
        
        print("Building battles...")
        battles = build_battles(judgments, args.baseline_model)
        
        print("Preparing style features...")
        features, outcomes, models, num_style = prepare_style_features(
            battles=battles,
            model_metadata=model_metadata,
            control_features=args.control_features,
            baseline_model=args.baseline_model
        )
        
        print("Fitting Bradley-Terry model with bootstrap...")
        coefs = bootstrap_bt(features, outcomes, num_rounds=100)
        
        # Remove style coefficients and compute winrates
        model_coefs = coefs[:, :-num_style]
        probs = to_winrate_probabilities(model_coefs, models, args.baseline_model)
        
        leaderboard = create_leaderboard_style(probs, models)
    else:
        print("Extracting scores...")
        model_scores = extract_scores_from_judgments(judgments)
        
        print(f"Found {len(model_scores)} models")
        
        print("Computing confidence intervals with bootstrap sampling...")
        leaderboard = create_leaderboard(model_scores, args.baseline_model)
    
    print_leaderboard(leaderboard)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(leaderboard, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Leaderboard saved to: {args.output}")


if __name__ == "__main__":
    main()
