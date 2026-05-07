"""Scoring utilities from the SynthWorlds codebase.

Gu et al., "SynthWorlds: Controlled Parallel Worlds for Disentangling Reasoning
and Knowledge in Language Models", ICLR 2026.
https://arxiv.org/abs/2510.24427
Code: https://github.com/behavioral-data/synthworlds

Copied verbatim from:
  normalize_answer    — synthworld_experiments/hipporag/utils/eval_utils.py
  compute_f1          — synthworld_experiments/datasets/qa_dataset.py
  extract_answer      — synthworld_experiments/datasets/qa_dataset.py
                        (originally a method on QuestionAnswerDataset;
                         extracted here as a standalone function)
  TimestampInfo       — synthworld_experiments/datasets/utils.py
  SubParser           — synthworld_experiments/datasets/utils.py
  parse_date_string   — synthworld_experiments/datasets/utils.py

Adapted from:
  score_instance      — get_instance_eval_metrics() in
                        synthworld_experiments/datasets/wikiqa.py
                        Extracted from a class method on QAWikiDataset to a
                        standalone function taking (pred, gold_answers,
                        expected_output_is_time) directly. extract_answer() is
                        not called inside — the caller handles extraction.
                        All retrieval recall metrics (recall@k, etc.) dropped —
                        not relevant here. Otherwise the logic is identical,
                        including: the double-computation of f1_scores, the
                        `if em_score == 1.0: f1_scores.append(1.0)` branch,
                        and np.max over all score lists.
                        One deliberate divergence: EM uses == (exact normalised
                        match) rather than the original's `in` (substring
                        containment). The original `in` check causes a silent
                        false-positive when extract_answer() returns "" —
                        empty string is a substring of every string.
"""
import re
import string
from collections import Counter
from typing import Any, Literal, Optional

import numpy as np
from dateutil import parser
from pydantic import BaseModel


def normalize_answer(answer: str) -> str:
    """Normalize: lowercase, remove punctuation and articles."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the|is)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def separate_concatenated_words(text):
        words = text.split()
        separated_words = []
        for word in words:
            # Common LaTeX commands that might get concatenated
            latex_commands = ['boxed', 'text', 'mathrm', 'mathbf', 'mathit', 'mathcal', 'mathbb']
            if len(word) > 8:
                for cmd in latex_commands:
                    if cmd in word:
                        parts = word.split(cmd)
                        if len(parts) == 2 and parts[0] == '':
                            separated_words.extend([cmd, parts[1]])
                            break
                else:
                    separated_words.append(word)
            else:
                separated_words.append(word)
        return " ".join(separated_words)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(separate_concatenated_words(remove_punc(lower(answer)))))


def compute_f1(gold: str, predicted: str) -> float:
    """Token-level F1 between gold and predicted strings (after normalization)."""
    gold_tokens = normalize_answer(gold).split()
    predicted_tokens = normalize_answer(predicted).split()
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(predicted_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)


def extract_answer(llm_output: str) -> str:
    """Parse model output — extracts text after 'Answer: ', 'Answer is: ', etc."""
    if "answer is:" in llm_output.lower():
        return llm_output.lower().split("answer is:")[1].strip()
    elif "answer:" in llm_output.lower():
        return llm_output.lower().split("answer:")[1].strip()
    elif "answer." in llm_output.lower():
        return llm_output.lower().split("answer.")[1].strip()
    elif "answer" in llm_output.lower():
        return llm_output.lower().split("answer")[1].strip()
    else:
        return llm_output.strip()


# ---------------------------------------------------------------------------
# Date scoring helpers
# Copied verbatim from synthworld_experiments/datasets/utils.py
# ---------------------------------------------------------------------------

class TimestampInfo(BaseModel):
    sign: Literal["+", "-"]
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    hour: Optional[int] = None
    minute: Optional[int] = None
    second: Optional[int] = None

    def model_post_init(self, __context: Any) -> None:
        if self.year is None and self.month is None and self.day is None:
            raise ValueError("At least one of year, month, or day must be provided")

    def to_string(self) -> str:
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        if self.sign == "-":
            sign_str = "BCE"
        else:
            sign_str = "CE"

        if self.year is not None and self.year <= 1000:
            return f"{self.year} {sign_str}"
        else:
            if self.month is None:
                return f"{self.year}"
            elif self.day is None:
                return f"{months[self.month - 1]}, {self.year}"
            else:
                return f"{months[self.month - 1]} {self.day}, {self.year}"

    def __hash__(self) -> int:
        return hash(
            (
                self.sign,
                self.year,
                self.month,
                self.day,
                self.hour,
                self.minute,
                self.second,
            )
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TimestampInfo):
            return False
        return (
            self.sign == other.sign
            and self.year == other.year
            and self.month == other.month
            and self.day == other.day
            and self.hour == other.hour
            and self.minute == other.minute
            and self.second == other.second
        )

    def to_date_string(self) -> str:
        return f"{self.sign}{self.year}-{self.month}-{self.day}T{self.hour}:{self.minute}:{self.second}Z"

    def to_parseable_string(self) -> str:
        """
        Convert this TimestampInfo back to a string format that parse_date_string can parse.

        Returns:
            str: A string representation that parse_date_string can successfully parse
        """
        # Handle BCE/CE format
        if self.sign == "+" and self.year is not None and self.year < 0:
            self.sign = "-"
            self.year = abs(self.year) + 1
        if self.sign == "-":
            era_suffix = " BCE"
        else:
            if self.year is not None:
                if self.year <= 1000:
                    era_suffix = " CE"
                else:
                    era_suffix = " (year)"

        # If we have hour, minute, second, use ISO datetime format
        if (
            self.hour is not None
            and self.minute is not None
            and self.second is not None
        ):
            return f"{self.year:04d}-{self.month:02d}-{self.day:02d}T{self.hour:02d}:{self.minute:02d}:{self.second:02d}Z"

        # If we have day, use ISO date format
        elif self.day is not None:
            return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"

        # If we have month, use year-month format
        elif self.month is not None:
            return f"{self.year:04d}-{self.month:02d}"

        # If we only have year, use year with era format
        else:
            return f"{self.year}{era_suffix}"


class SubParser(parser.parser):
    def _build_naive(self, res, default):
        _ = super()._build_naive(res, default)  # type: ignore
        return TimestampInfo(
            sign="+",
            year=res.year,
            month=res.month,
            day=res.day,
            hour=res.hour,
            minute=res.minute,
            second=res.second,
        )


def parse_date_string(date_str) -> TimestampInfo:
    """
    Parse a date string and return a DateInfo object.

    Args:
        date_str (str): Date string in various formats

    Returns:
        DateInfo: Parsed date information
    """
    # Initialize default values
    sign = "+"
    year = 0
    month = None
    day = None
    hour = None
    minute = None
    second = None

    # Strip whitespace
    date_str = date_str.strip()

    # Check for BCE
    sign = "+"
    if "BCE" in date_str:
        sign = "-"
        date_str = date_str.replace("BCE", "").strip()
        year = int(date_str)
    elif "CE" in date_str:
        date_str = date_str.replace("CE", "").strip()
        year = int(date_str)
    elif "(year)" in date_str:
        date_str = date_str.replace("(year)", "").strip()
        year = int(date_str)
    else:
        # Handle YYYY-MM format
        year_month_match = re.match(r"^(\d{4})-(\d{1,2})$", date_str)
        iso_datetime_match = re.match(
            r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z?$", date_str
        )
        iso_date_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", date_str)
        if year_month_match:
            year, month = map(int, year_month_match.groups())
        elif iso_date_match:
            year, month, day = map(int, iso_date_match.groups())
        elif iso_datetime_match:
            year, month, day, hour, minute, second = map(
                int, iso_datetime_match.groups()
            )
        else:
            raise ValueError(f"Could not parse date string: {date_str}")

    return TimestampInfo(
        sign=sign,
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        second=second,
    )


# ---------------------------------------------------------------------------
# Main scoring entry point
# ---------------------------------------------------------------------------

def score_instance(
    pred: str,
    gold_answers: list[str],
    expected_output_is_time: bool = False,
) -> tuple[float, float]:
    """Compute EM and token-level F1 for a single prediction against gold answers.

    Adapted from get_instance_eval_metrics() in
    synthworld_experiments/datasets/wikiqa.py.

    Returns:
        (em, f1) — both floats in [0, 1].
    """
    em_scores = [
        (1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0)
        for gold in gold_answers
    ]
    agent_date_output = None
    gold_date_outputs = []
    if expected_output_is_time:
        sub_parser = SubParser()
        try:
            pred_date = sub_parser.parse(pred)
        except Exception:
            try:
                pred_date = parse_date_string(pred)
            except Exception:
                pred_date = None
        if pred_date is not None and isinstance(pred_date, TimestampInfo):
            agent_date_output = pred_date.to_string()
        if agent_date_output is not None:
            for gold in gold_answers:
                try:
                    gold_date = sub_parser.parse(gold)
                except Exception:
                    try:
                        gold_date = parse_date_string(gold)
                    except Exception:
                        gold_date = None
                if gold_date is not None and isinstance(gold_date, TimestampInfo):
                    gold_date_outputs.append(gold_date.to_string())
                    if agent_date_output == gold_date.to_string():
                        em_scores.append(1.0)
                    else:
                        em_scores.append(0.0)

    em_score = np.max(em_scores)
    f1_scores = [
        compute_f1(gold, pred)
        for gold in gold_answers
    ]
    if em_score == 1.0:
        f1_scores.append(1.0)
    if agent_date_output is not None:
        for gold_date in gold_date_outputs:
            f1_scores.append(compute_f1(gold_date, agent_date_output))
    f1_score = np.max(f1_scores)

    return em_score, f1_score
