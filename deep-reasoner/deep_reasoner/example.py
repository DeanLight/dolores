# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example Notebook
#
# This is a paired notebook. The source of truth is this `.py` file.
# The `.ipynb` file is generated via `poe nb` and is not tracked in git.

# %%
from juplit import test

# %% [markdown]
# ## String utilities

# %%
def reverse(s: str) -> str:
    return s[::-1]

if test():
    assert reverse("hello") == "olleh"
    assert reverse("") == ""
    assert reverse("a") == "a"

# %%
def palindrome(s: str) -> bool:
    return s == reverse(s)

if test():
    assert palindrome("racecar")
    assert palindrome("a")
    assert not palindrome("hello")

# %% [markdown]
# ## Math utilities

# %%
def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))

if test():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(11, 0, 10) == 10

# %%
def running_mean(values: list[float]) -> list[float]:
    result = []
    total = 0.0
    for i, v in enumerate(values):
        total += v
        result.append(total / (i + 1))
    return result

if test():
    assert running_mean([1, 2, 3]) == [1.0, 1.5, 2.0]
    assert running_mean([4]) == [4.0]
    assert running_mean([]) == []
