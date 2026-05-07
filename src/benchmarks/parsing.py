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
# # Parsing
#
# Parse raw model output into structured data using OpenAI structured outputs.

# %%
from juplit import test

# %%
from openai import OpenAI
from pydantic import BaseModel

# %%
def _parse(user_prompt: str, system_prompt: str, schema: type[BaseModel], model: str, api_key: str) -> dict:
    """Parse a raw string into a dict using OpenAI structured outputs.

    Args:
        user_prompt:   The user message (e.g. question + model output to parse).
        system_prompt: The system message (instructions, examples, output format).
        schema:        A Pydantic BaseModel subclass defining the output shape.
        model:         OpenAI model name.
        api_key:       OpenAI API key.

    Returns:
        A dict matching the schema fields.

    Raises:
        ValueError: If the model refuses to respond or the response could not be parsed.
    """
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=schema,
    )
    message = completion.choices[0].message
    if message.refusal:
        raise ValueError(f"Model refused: {message.refusal}")
    if not message.parsed:
        raise ValueError("Could not parse response")
    return message.parsed.model_dump()
