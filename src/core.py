# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # core
#
# Fill in a module description here

# %%
from juplit import test

# %%
if test():
    import os
    from dotenv import load_dotenv

# %%
if test():
    load_dotenv('../.env.dev')

# %%
def foo(): return 'bar'
