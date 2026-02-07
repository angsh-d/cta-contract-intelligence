"""Query Pipeline: TruthSynthesizer — answer synthesis with citations."""

import json
import logging
from typing import Any
from uuid import UUID

from app.agents.base import BaseAgent
from app.exceptions import LLMResponseError
from app.models.agent_schemas import (
    SourceCitation, TruthSynthesisInput, TruthSynthesisOutput,
)

logger = logging.getLogger(__name__)


class TruthSynthesizer(BaseAgent):
    """Synthesize answers from resolved clauses with full citations and caveats."""

    async def process(self, input_data: TruthSynthesisInput) -> TruthSynthesisOutput:
        system_prompt = self.prompts.get("truth_synthesizer_answer")

        clauses_text = "\n\n".join(
            f"Section {c.section_number} ({c.section_title}) "
            f"[Source: {c.source_document_label}, Effective: {c.effective_date}]:\n{c.current_text}"
            for c in input_data.relevant_clauses
        )

        conflicts_text = ""
        if input_data.conflicts:
            conflicts_text = "\n\n".join(
                f"Conflict: {c.description}\n"
                f"  Severity: {c.severity.value}\n"
                f"  Sections: {', '.join(c.affected_sections)}\n"
                f"  Recommendation: {c.recommendation}"
                for c in input_data.conflicts
            )

        user_prompt = self.prompts.get(
            "truth_synthesizer_input",
            query_text=input_data.query_text,
            query_type=input_data.query_type.value,
            relevant_clauses=clauses_text,
            known_conflicts=conflicts_text or "(none)",
        )
        result = await self.call_llm(system_prompt, user_prompt)

        answer = result.get("answer")
        if not answer:
            raise LLMResponseError("TruthSynthesizer: LLM response missing 'answer'")

        sources = [SourceCitation(**s) for s in result.get("sources", [])]

        return TruthSynthesisOutput(
            answer=answer,
            sources=sources,
            confidence=result.get("confidence", 0.9),
            caveats=result.get("caveats", []),
            llm_reasoning=result.get("reasoning", ""),
        )
