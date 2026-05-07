# Source: phantom-wiki/src/phantom_eval/agents/react.py  (_step_observation)
#   https://github.com/kilian-group/phantom-wiki/blob/main/src/phantom_eval/agents/react.py
#
# Modifications:
#   - Extracted RetrieveArticle and Search logic from ReactAgent._step_observation into
#     standalone functions bound to the article corpus via closures (make_tools),
#     so callers receive plain callables retrieve_article(entity) and search(attribute)
#     without having to pass the corpus on every call.
#   - Input changed from pandas DataFrame to list[dict] (matching load_articles() output)
#     to avoid requiring pandas as a dependency. The filtering logic is equivalent:
#     pandas str.contains() with a simple keyword behaves identically to Python's `in`.
#   - format_pred (newline stripping) is intentionally omitted. The original strips newlines
#     because observations are inlined into a single-line ReAct scratchpad. Since we return
#     the string directly to the caller, preserving newlines is more useful.
#   - Closures are used instead of functools.partial so that inspect.signature() returns
#     the user-facing signature only (e.g. (entity: str) -> str), not the internal
#     `articles` argument. This is critical for agent frameworks that auto-generate tool
#     schemas from signatures (e.g. OpenAI Agents SDK, Anthropic tool use).
#   - No other logic changes to the tool behaviour or their return strings.
#
# Tool docstrings (retrieve_article, search) are taken verbatim from the authors'
# REACT_INSTRUCTION prompt in phantom_eval/prompts.py (ReactLLMPrompt.REACT_INSTRUCTION),
# because agent frameworks (smolagents, OpenAI Agents SDK, etc.) surface __doc__ as the
# tool description seen by the model.  Keeping the wording identical to the benchmark's
# own prompt ensures the agent's mental model of each tool matches the paper's setup.


def make_tools(articles: list[dict]) -> tuple:
    """Bind RetrieveArticle and Search tools to a fixed article corpus.

    Call once with the corpus returned by ``load_articles()``, then pass the
    returned callables to your agent — no need to pass ``articles`` on every tool call.

    Args:
        articles: Full article corpus as returned by ``phantomwiki.load_articles()``.
                  List of dicts with ``title`` and ``article`` keys.

    Returns:
        ``(retrieve_article, search)`` — two callables ready to hand to the model.
        Both have clean signatures (only user-facing params) and proper ``__doc__``/
        ``__name__`` so agent frameworks that inspect them for tool schemas work correctly.
    """

    # Docstring: verbatim from ReactLLMPrompt.REACT_INSTRUCTION (phantom_eval/prompts.py)
    def retrieve_article(entity: str) -> str:
        """Retrieve the article about the given entity, if it exists.

        Args:
            entity: The name of the entity to retrieve the article for.

        Returns:
            The full article text, or an error string if no article with that title exists.
        """
        for a in articles:
            if a["title"].lower() == entity.lower():
                return a["article"]
        return (
            "No article exists for the requested entity. "
            "Please try retrieving article for another entity."
        )

    # Docstring: verbatim from ReactLLMPrompt.REACT_INSTRUCTION (phantom_eval/prompts.py)
    def search(attribute: str) -> str:
        """Search the database for the given attribute and retrieve all articles that contain it.

        Args:
            attribute: The keyword or phrase to search for (e.g. a hobby, occupation, or name).

        Returns:
            A numbered list of matching article titles (e.g. "(1) Alice Smith\\n\\n(2) Bob Jones"),
            or an error string if no articles contain the attribute.
        """
        matching_titles = [
            a["title"] for a in articles
            if attribute.lower() in a["article"].lower()
        ]
        if not matching_titles:
            return (
                "No articles contain the requested attribute. "
                "Please try searching for another attribute."
            )
        return "\n\n".join(f"({i + 1}) {title}" for i, title in enumerate(matching_titles))

    return retrieve_article, search
