"""Tier 3: ReusabilityAnalyzerAgent — Phase 2 stub."""

from app.agents.base import BaseAgent
from app.models.agent_schemas import ReusabilityInput, ReusabilityOutput


class ReusabilityAnalyzerAgent(BaseAgent):
    """Phase 2 — Not yet implemented."""

    async def process(self, input_data: ReusabilityInput) -> ReusabilityOutput:
        raise NotImplementedError(
            "ReusabilityAnalyzerAgent is scheduled for Phase 2. "
            "Use the current single-stack analysis pipeline for now."
        )
