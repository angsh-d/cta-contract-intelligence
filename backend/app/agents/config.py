"""Agent configuration dataclass."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AgentConfig:
    """Immutable configuration for a single agent instance."""

    agent_name: str
    llm_role: str
    model_override: Optional[str] = None
    max_output_tokens: Optional[int] = None
    max_retries: int = 3
    timeout_seconds: int = 120
    temperature: float = 0.0
    verification_threshold: float = 0.75
