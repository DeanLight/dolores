# =============================================================================
# SOURCE: oolong/src/eval/eval_helpers.py
# Copied verbatim from the Oolong benchmark authors.
# Reference: Bertsch et al. (2025) — https://arxiv.org/abs/2511.02817
#            https://github.com/abertsch72/oolong
# This file is NOT authored by this project. Do not modify it here;
# if the authors update their code, update this copy accordingly.
#
# LOCAL MODIFICATION: both scorers accept a `numeric_scorer` callable (gold, pred) -> float
# to replace the default 0.75^|error|. Pass None to use the original behaviour.
# =============================================================================

#import litellm
#from datasets import load_dataset
#import jsonlines
#import tiktoken
import dateutil

from datetime import datetime
#import os
import ast
#import sys
import re
#from pathlib import Path

# try to parse the answer; super simple, idea is that we can go back and try harder on "low" or sometimes "med" confidence answers
def synth_attempt_answer_parse(answer):
    parse_confidence = "low"
    if ":" not in answer:  # bad start
        if len(answer) < 20:  # it's short, return the whole thing
            return answer, parse_confidence
        else:
            return answer.split()[-1], parse_confidence
    candidate_answer = answer.split(":")[-1].strip()
    candidate_answer = candidate_answer.replace(
        "*", ""
    )  # OpenAI models like bolding the answer


    candidate_answer = candidate_answer.replace(
        "[", ""
    )
    candidate_answer = candidate_answer.replace(
        "]", ""
    )  # Anthropic models like putting the answer in []
    parse_confidence = "med"
    if (
        "User:" in answer
        or "Answer:" in answer
        or "Date:" in answer
        or "Label" in answer
    ):
        parse_confidence = "high"
    if len(candidate_answer) < 20:
        parse_confidence = "vhigh"
    elif "more common" in candidate_answer:
        candidate_answer = "more common"
    elif "less common" in candidate_answer:
        candidate_answer = "less common"
    elif "same frequency" in candidate_answer:
        candidate_answer = "same frequency"

    return candidate_answer, parse_confidence


def synth_process_response(datapoint, output, model, numeric_scorer=None):
    # NOTE (local change): added numeric_scorer param. If provided, called as
    # numeric_scorer(gold, pred) -> float instead of 0.75^|error|.
    score = 0
    gold = (
        ast.literal_eval(datapoint["answer"])[0]
        if "datetime" not in datapoint["answer"]
        else datetime.strptime(datapoint["answer"], "[datetime.date(%Y, %m, %d)]")
    )

    trimmed_output, parse_confidence =  synth_attempt_answer_parse(output)
    if str(trimmed_output) == str(gold):
        score = 1
    elif str(trimmed_output) in ['more common', 'less common', 'same frequency']: # account for these being slightly different wordings
        if str(trimmed_output) in  str(gold):
            score = 1
    elif (
        datapoint["answer_type"] == "ANSWER_TYPE.NUMERIC"
    ):  # partial credit for numbers
        try:
            trimmed_output = int(trimmed_output)
            gold = int(gold)
            score = numeric_scorer(gold, trimmed_output) if numeric_scorer else 0.75 ** abs(gold - trimmed_output)
        except Exception:
            parse_confidence = "low"  # didn't parse as a number, that's a bad sign
    elif datapoint["answer_type"] == "ANSWER_TYPE.DATE":
        try:
            trimmed_output = dateutil.parser.parse(trimmed_output)
            score = trimmed_output == gold
        except Exception:
            parse_confidence = "low"  # didn't parse as a date, that's a bad sign


    this_output = {
        "id": datapoint["id"],
        "context_window_id": datapoint["context_window_id"],
        "dataset": datapoint["dataset"],
        "model": model,
        "attempted_parse": str(trimmed_output),
        "parse_confidence": parse_confidence,
        "full_answer": output,
        "score": score,
        "answer": str(gold),
    }

    return this_output


"""Eval helpers for DnD split."""
#from transformers import AutoTokenizer



def dnd_parse_answer(answer) -> int | str | list[str]:
    """Parse the answer into int, str, or list of str."""
    # Try to convert to int first
    try:
        return int(answer)
    except ValueError:
        pass

    # Check if it contains commas (list case)
    if "," in answer:
        return [item.strip() for item in answer.split(",") if item.strip()]

    # Otherwise return as string
    return answer


def dnd_parse_response(answer) -> tuple[str, str]:
    match = re.search(r"\\boxed\{\\text\{([^}]*)\}\}", answer) or re.search(
        r"\\boxed[\{]+([^}]*)[\}]+", answer
    )
    if match:
        answer = match.group(1)
    else:
        return answer, "low"
    return dnd_parse_answer(answer), "high"


def dnd_score(gold: int | str | list, pred: int | str | list, numeric_scorer=None) -> float:
    # NOTE (local change): extracted verbatim from the scoring block inside
    # dnd_process_response (original authors' logic, unchanged). numeric_scorer
    # is the only addition — if provided, called as numeric_scorer(gold, pred)
    # -> float instead of the original 0.75^|error| for int answers.
    if isinstance(gold, int) and isinstance(pred, int):
        return numeric_scorer(gold, pred) if numeric_scorer else 0.75 ** abs(gold - pred)
    elif isinstance(gold, str) and isinstance(pred, str):
        return float(gold.strip().lower() == pred.strip().lower())
    elif isinstance(gold, list) and isinstance(pred, list):
        overlap = set(gold) & set(pred)
        return len(overlap) / len(gold) if gold else 0.0
    return 0.0


def dnd_process_response(datapoint, output, model, numeric_scorer=None) -> dict:
    gold = dnd_parse_answer(datapoint["answer"])
    trimmed_output, parse_confidence = dnd_parse_response(output)
    score = dnd_score(gold, trimmed_output, numeric_scorer=numeric_scorer)
    # else:
    #     msg = f"unknown match, gold answer type: {type(gold)}, model answer type: {type(trimmed_output)}"
    #     raise ValueError(msg)
    return {
        "id": datapoint["id"],
        "context_window_id": datapoint["context_window_id"],
        "model": model,
        "attempted_parse": trimmed_output,
        "parse_confidence": parse_confidence,
        "full_answer": output,
        "score": score,
        "answer": gold,
    }
