"""awreason — A portable reasoning client.

Sessions, phases, thoughts, and the chain that produced the answer.
"""

__version__ = "0.1.0"

from awreason.client import (
    Depth,
    ReasoningClient,
    ReasoningError,
    SASEPhase,
    Session,
    SessionInfo,
    Thought,
    ThoughtType,
)

__all__ = [
    "__version__",
    "ReasoningClient",
    "ReasoningError",
    "Session",
    "SessionInfo",
    "Thought",
    "ThoughtType",
    "SASEPhase",
    "Depth",
]
