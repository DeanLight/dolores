from dolores.unified.agents.base import (
    Agent,
    AgentCfg,
    AgentResult,
    get_agent_cls,
    list_agents,
    register_agent,
)

# Import each agent module for its registration side effect.
from dolores.unified.agents import (  # noqa: E402,F401
    codeact,
    cot,
    deep_reasoner,
    deepresearch,
    react,
    rlm,
)

__all__ = [
    "Agent",
    "AgentCfg",
    "AgentResult",
    "register_agent",
    "get_agent_cls",
    "list_agents",
]
