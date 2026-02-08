"""Tier 2: OverrideResolutionAgent — apply amendments chronologically to determine current clause text."""

import logging
from datetime import date
from typing import Any

from app.agents.base import BaseAgent
from app.exceptions import LLMResponseError
from app.models.agent_schemas import (
    AmendmentForSection, ClauseVersion, OverrideResolutionInput,
    OverrideResolutionOutput, SourceChainLink,
)

logger = logging.getLogger(__name__)


class OverrideResolutionAgent(BaseAgent):
    """For each clause, apply all amendments in order to determine current text with provenance."""

    async def process(self, input_data: OverrideResolutionInput) -> OverrideResolutionOutput:
        amendments = sorted(input_data.amendments, key=lambda a: a.effective_date or date.min)

        system_prompt = self.prompts.get("override_resolution_system")
        user_prompt = self.prompts.get(
            "override_resolution_apply",
            section_number=input_data.section_number,
            original_text=input_data.original_clause.text,
            original_source=input_data.original_document_label,
            amendment_chain=self._format_amendment_chain(amendments),
        )

        result = await self.call_llm(system_prompt, user_prompt)

        current_text = result.get("current_text")
        if current_text is None:
            current_text = input_data.original_clause.text  # Fall back to original

        source_chain_raw = result.get("source_chain") or []
        source_chain = []
        for link in source_chain_raw:
            if isinstance(link, dict):
                try:
                    source_chain.append(SourceChainLink(**link))
                except Exception as e:
                    logger.warning("Skipping invalid source chain link: %s", e)
            elif isinstance(link, str):
                source_chain.append(SourceChainLink(stage="unknown", document_label=link, text=link))

        if amendments:
            last_modified_by = amendments[-1].amendment_document_id
            last_modified_date = amendments[-1].effective_date
        else:
            last_modified_by = input_data.original_document_id
            last_modified_date = (
                date.fromisoformat(result["effective_date"])
                if result.get("effective_date") else None
            )

        return OverrideResolutionOutput(
            clause_version=ClauseVersion(
                section_number=input_data.section_number,
                section_title=input_data.original_clause.section_title,
                current_text=current_text,
                source_chain=source_chain,
                last_modified_by=last_modified_by,
                last_modified_date=last_modified_date,
                confidence=result.get("confidence", 0.8),
                clause_category=result.get("clause_category", "general"),
            ),
            llm_reasoning=result.get("reasoning", ""),
            confidence_factors=self._sanitize_confidence_factors(result.get("confidence_factors", {})),
        )

    @staticmethod
    def _sanitize_confidence_factors(raw: Any) -> dict[str, float]:
        """Coerce confidence_factors to dict[str, float], dropping non-numeric values."""
        if not isinstance(raw, dict):
            return {}
        cleaned = {}
        for k, v in raw.items():
            try:
                cleaned[str(k)] = float(v)
            except (ValueError, TypeError):
                continue
        return cleaned

    def _format_amendment_chain(self, amendments: list[AmendmentForSection]) -> str:
        if not amendments:
            return "(no amendments — original clause is current)"
        parts = []
        for a in amendments:
            parts.append(
                f"Amendment {a.amendment_number} (Effective {a.effective_date}):\n"
                f"  Document ID: {a.amendment_document_id}\n"
                f"  Type: {a.modification.modification_type.value}\n"
                f"  Original text: {a.modification.original_text or '[N/A]'}\n"
                f"  New text: {a.modification.new_text or '[DELETED]'}\n"
                f"  Change: {a.modification.change_description}"
            )
        return "\n\n".join(parts)
