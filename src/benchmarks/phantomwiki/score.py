# Source: phantom-wiki/src/phantom_eval/score.py
#   https://github.com/kilian-group/phantom-wiki/blob/main/src/phantom_eval/score.py
# Source: phantom-wiki/src/phantom_eval/utils.py  (normalize_pred)
#   https://github.com/kilian-group/phantom-wiki/blob/main/src/phantom_eval/utils.py
# Source: phantom-wiki/src/phantom_eval/constants.py  (answer_sep)
#   https://github.com/kilian-group/phantom-wiki/blob/main/src/phantom_eval/constants.py
#
# Modifications:
#   - Inlined `normalize_pred` (from utils.py) and `answer_sep` (from constants.py) to avoid
#     importing the full phantom_eval package, which has heavy optional dependencies.
#   - No logic changes.

answer_sep: str = ","


def normalize_pred(pred: str, sep: str = answer_sep) -> set[str]:
    return set(map(str.lower, map(str.strip, pred.split(sep))))


def exact_match(pred: str, true: str, sep: str = answer_sep) -> bool:
    """Check if the prediction is equal to the true answer."""
    return normalize_pred(pred, sep) == normalize_pred(true, sep)


def precision(pred: str, true: str, sep: str = answer_sep) -> float:
    normalized_preds: set[str] = normalize_pred(pred, sep)
    normalized_trues: set[str] = normalize_pred(true, sep)
    count = sum(word in normalized_trues for word in normalized_preds)
    return count / len(normalized_preds)


def recall(pred: str, true: str, sep: str = answer_sep) -> float:
    normalized_preds: set[str] = normalize_pred(pred, sep)
    normalized_trues: set[str] = normalize_pred(true, sep)
    count = sum(word in normalized_preds for word in normalized_trues)
    return count / len(normalized_trues)


def f1(pred: str, true: str, sep: str = answer_sep) -> float:
    pres = precision(pred, true, sep)
    rec = recall(pred, true, sep)
    if pres + rec == 0:
        return 0
    return 2 * pres * rec / (pres + rec)
