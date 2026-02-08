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
            f"[document_id: {c.source_document_id}, document_name: {c.source_document_label}, "
            f"effective_date: {c.effective_date}]:\n{c.current_text}"
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

        sources = []
        for s in result.get("sources", []):
            try:
                doc_id = s.get("document_id")
                if doc_id:
                    try:
                        UUID(str(doc_id))
                    except (ValueError, AttributeError):
                        doc_id = None
                eff = s.get("effective_date")
                if eff and not isinstance(eff, str):
                    eff = str(eff)
                if isinstance(eff, str):
                    try:
                        from datetime import date as dt_date
                        dt_date.fromisoformat(eff)
                    except (ValueError, TypeError):
                        eff = None
                sources.append(SourceCitation(
                    document_id=doc_id if doc_id else input_data.relevant_clauses[0].source_document_id if input_data.relevant_clauses else None,
                    document_name=s.get("document_name", ""),
                    section_number=s.get("section_number", ""),
                    relevant_text=s.get("relevant_text", ""),
                    effective_date=eff,
                ))
            except Exception as e:
                logger.warning("Skipping malformed source citation: %s — %s", s, e)

        return TruthSynthesisOutput(
            answer=answer,
            sources=sources,
            confidence=result.get("confidence", 0.9),
            caveats=result.get("caveats", []),
            llm_reasoning=result.get("reasoning", ""),
        )
