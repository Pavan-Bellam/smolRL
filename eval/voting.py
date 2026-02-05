"""
Voting strategies for test-time compute scaling evaluation.

Provides four voting strategies for aggregating multiple completions:
1. Naive Majority Vote - each completion = 1 vote
2. Logprob-Weighted Vote - weight by exp(cumulative_logprob)
3. Shortest Majority Vote (SMV) - vote among k shortest completions
4. Shortest + Weighted (SMV-Weighted) - weighted vote among k shortest
"""

import math
from dataclasses import dataclass

from grader import extract_raw_boxed, strip_string, math_equal


@dataclass
class ExtractedCompletion:
    """A completion with its extracted and normalized answer."""
    index: int
    text: str
    answer: str | None  # normalized answer (None if no boxed)
    cumulative_logprob: float
    num_tokens: int


def extract_answers(completions: list[dict]) -> list[ExtractedCompletion]:
    """Extract and normalize answers from all completions.

    Args:
        completions: List of completion dicts with text, cumulative_logprob, num_tokens

    Returns:
        List of ExtractedCompletion with normalized answers
    """
    extracted = []
    for i, comp in enumerate(completions):
        raw_answer = extract_raw_boxed(comp["text"])
        answer = strip_string(raw_answer) if raw_answer is not None else None
        extracted.append(ExtractedCompletion(
            index=i,
            text=comp["text"],
            answer=answer,
            cumulative_logprob=comp.get("cumulative_logprob", 0.0),
            num_tokens=comp.get("num_tokens", len(comp["text"])),
        ))
    return extracted


def group_by_equivalence(completions: list[ExtractedCompletion]) -> tuple[list[list[ExtractedCompletion]], list[ExtractedCompletion]]:
    """Group completions by answer equivalence using math_equal.

    Args:
        completions: List of ExtractedCompletion objects

    Returns:
        Tuple of (groups, null_group) where:
        - groups: List of lists, each containing equivalent completions
        - null_group: List of completions with no extractable answer
    """
    groups: list[list[ExtractedCompletion]] = []
    group_representatives: list[str] = []  # representative answer for each group
    null_group: list[ExtractedCompletion] = []

    for comp in completions:
        if comp.answer is None:
            null_group.append(comp)
            continue

        # Try to find an existing group
        found = False
        for i, rep in enumerate(group_representatives):
            if math_equal(comp.answer, rep):
                groups[i].append(comp)
                found = True
                break

        if not found:
            # Create new group
            groups.append([comp])
            group_representatives.append(comp.answer)

    return groups, null_group


def naive_majority_vote(
    completions: list[ExtractedCompletion],
    k: int,
    groups: list[list[ExtractedCompletion]] | None = None,
) -> tuple[str | None, dict]:
    """Naive majority vote: each completion gets 1 vote.

    Args:
        completions: List of ExtractedCompletion (will subsample first k)
        k: Number of completions to use
        groups: Pre-computed groups (optional, for efficiency)

    Returns:
        Tuple of (winning_answer, metadata)
    """
    # Subsample to first k completions
    subset = completions[:k]

    # Group by equivalence if not provided
    if groups is None:
        groups, null_group = group_by_equivalence(subset)
    else:
        # Filter groups to only include completions in subset
        subset_indices = {c.index for c in subset}
        groups = [[c for c in g if c.index in subset_indices] for g in groups]
        groups = [g for g in groups if g]  # remove empty groups

    if not groups:
        return None, {"vote_counts": {}, "null_count": k}

    # Count votes per group
    vote_counts = [(len(g), g[0].answer) for g in groups]
    vote_counts.sort(reverse=True)

    winning_answer = vote_counts[0][1]
    null_count = sum(1 for c in subset if c.answer is None)

    return winning_answer, {
        "vote_counts": {ans: cnt for cnt, ans in vote_counts},
        "null_count": null_count,
    }


def logprob_weighted_vote(
    completions: list[ExtractedCompletion],
    k: int,
    groups: list[list[ExtractedCompletion]] | None = None,
) -> tuple[str | None, dict]:
    """Logprob-weighted vote: weight = exp(cumulative_logprob).

    Args:
        completions: List of ExtractedCompletion (will subsample first k)
        k: Number of completions to use
        groups: Pre-computed groups (optional, for efficiency)

    Returns:
        Tuple of (winning_answer, metadata)
    """
    # Subsample to first k completions
    subset = completions[:k]

    # Group by equivalence if not provided
    if groups is None:
        groups, null_group = group_by_equivalence(subset)
    else:
        # Filter groups to only include completions in subset
        subset_indices = {c.index for c in subset}
        groups = [[c for c in g if c.index in subset_indices] for g in groups]
        groups = [g for g in groups if g]

    if not groups:
        return None, {"weighted_votes": {}, "null_count": k}

    # Compute weighted votes per group
    weighted_votes = []
    for g in groups:
        weight = sum(math.exp(c.cumulative_logprob) for c in g)
        weighted_votes.append((weight, g[0].answer))
    weighted_votes.sort(reverse=True)

    winning_answer = weighted_votes[0][1]
    null_count = sum(1 for c in subset if c.answer is None)

    return winning_answer, {
        "weighted_votes": {ans: w for w, ans in weighted_votes},
        "null_count": null_count,
    }


def shortest_majority_vote(
    completions: list[ExtractedCompletion],
    k: int,
) -> tuple[str | None, dict]:
    """Shortest Majority Vote: vote among k shortest completions.

    Args:
        completions: List of ExtractedCompletion
        k: Number of shortest completions to use

    Returns:
        Tuple of (winning_answer, metadata)
    """
    # Sort by num_tokens ascending
    sorted_by_length = sorted(completions, key=lambda c: c.num_tokens)

    # Take k shortest
    shortest_k = sorted_by_length[:k]

    # Run naive majority vote on shortest k
    groups, null_group = group_by_equivalence(shortest_k)
    answer, meta = naive_majority_vote(shortest_k, k, groups)

    meta["sorted_by"] = "num_tokens"
    return answer, meta


def shortest_weighted_vote(
    completions: list[ExtractedCompletion],
    k: int,
) -> tuple[str | None, dict]:
    """Shortest + Weighted: logprob-weighted vote among k shortest completions.

    Args:
        completions: List of ExtractedCompletion
        k: Number of shortest completions to use

    Returns:
        Tuple of (winning_answer, metadata)
    """
    # Sort by num_tokens ascending
    sorted_by_length = sorted(completions, key=lambda c: c.num_tokens)

    # Take k shortest
    shortest_k = sorted_by_length[:k]

    # Run logprob-weighted vote on shortest k
    groups, null_group = group_by_equivalence(shortest_k)
    answer, meta = logprob_weighted_vote(shortest_k, k, groups)

    meta["sorted_by"] = "num_tokens"
    return answer, meta
