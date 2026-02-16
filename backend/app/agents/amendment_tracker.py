"""Tier 1: AmendmentTrackerAgent — modification detection with buried change scanning."""

import json
import logging
from typing import Any

from app.agents.base import BaseAgent
from app.exceptions import LLMResponseError
from app.models.agent_schemas import (
    AmendmentTrackInput, AmendmentTrackOutput,
    Modification, ParsedSection, ParsedTable,
)
from app.models.enums import ModificationType

logger = logging.getLogger(__name__)


class AmendmentTrackerAgent(BaseAgent):
    """Identify exactly what each amendment modifies, with old/new text extraction."""

    async def process(self, input_data: AmendmentTrackInput) -> AmendmentTrackOutput:
        system_prompt = self.prompts.get("amendment_tracker_system")
        user_prompt = self.prompts.get(
            "amendment_tracker_analysis",
            amendment_number=str(input_data.amendment_number),
            amendment_text=self._format_sections(input_data.amendment_sections, input_data.amendment_tables),
            original_sections=self._format_sections(input_data.original_sections, input_data.original_tables),
            prior_modifications=self._format_prior_amendments(input_data.prior_amendments),
        )
        result = await self.call_llm(system_prompt, user_prompt)

        mods_raw = result.get("modifications")
        if mods_raw is None:
            raise LLMResponseError("AmendmentTrackerAgent: LLM response missing 'modifications'")
        if not isinstance(mods_raw, list):
            mods_raw = [mods_raw] if mods_raw else []
        valid_mod_types = {e.value for e in ModificationType}
        for m in mods_raw:
            if m.get("modification_type") not in valid_mod_types:
                m["modification_type"] = "selective_override"
            # Ensure original_text is set for types that require it
            if m.get("modification_type") != "addition" and not m.get("original_text"):
                m["original_text"] = m.get("change_description", "(not provided)")
        modifications = [Modification(**m) for m in mods_raw]

        # Buried change scan
        amendment_text = self._format_sections(input_data.amendment_sections, input_data.amendment_tables)
        missed = await self._scan_for_buried_changes(amendment_text, modifications)
        if missed:
            modifications.extend(missed)

        return AmendmentTrackOutput(
            amendment_document_id=input_data.amendment_document_id,
            amendment_number=input_data.amendment_number,
            effective_date=result.get("effective_date"),
            amendment_type=result.get("amendment_type") or "unknown",
            rationale=result.get("rationale") or "",
            modifications=modifications,
            sections_modified=[m.section_number for m in modifications],
            exhibits_affected=self._normalize_exhibits(result.get("exhibits_affected") or []),
            llm_reasoning=result.get("reasoning") or "",
            extraction_confidence=result.get("extraction_confidence", 0.9),
        )

    async def _verify_output(
        self, output: AmendmentTrackOutput, input_data: AmendmentTrackInput
    ) -> AmendmentTrackOutput:
        """Verify every modification references a real section from the original CTA.

        Allows subsection references: mod "7.2" is valid if CTA has "7.2" OR parent "7".
        """
        known_sections = {s.section_number for s in input_data.original_sections}
        for mod in output.modifications:
            if mod.modification_type == ModificationType.ADDITION:
                continue
            # Exact match
            if mod.section_number in known_sections:
                continue
            # Parent section match (7.2 → 7)
            if "." in mod.section_number:
                parent = mod.section_number.rsplit(".", 1)[0]
                if parent in known_sections:
                    continue
            logger.warning(
                "Modification references unknown section %s — may be hallucinated",
                mod.section_number,
            )
            output.extraction_confidence = min(output.extraction_confidence, 0.6)
        return output

    async def _scan_for_buried_changes(
        self, amendment_text: str, found_modifications: list[Modification]
    ) -> list[Modification]:
        """Adversarial scan for modifications missed in the initial extraction."""
        found_sections = [m.section_number for m in found_modifications]
        system_prompt = self.prompts.get("amendment_tracker_buried_scan")
        user_prompt = self.prompts.get(
            "amendment_tracker_buried_scan_input",
            amendment_text=amendment_text,
            already_found=json.dumps(found_sections),
        )
        result = await self.call_llm(system_prompt, user_prompt)

        missed = []
        valid_mod_types = {e.value for e in ModificationType}
        for mod in result.get("missed_modifications", []):
            if mod["section_number"] not in found_sections:
                if mod.get("modification_type") not in valid_mod_types:
                    mod["modification_type"] = "selective_override"
                if mod.get("modification_type") != "addition" and not mod.get("original_text"):
                    mod["original_text"] = mod.get("change_description", "(not provided)")
                try:
                    missed.append(Modification(**mod))
                except Exception as e:
                    logger.warning("Skipping invalid buried modification: %s", e)
                    continue

        if missed:
            logger.warning(
                "Buried change scan found %d missed modifications: %s",
                len(missed), [m.section_number for m in missed],
            )
        return missed

    @staticmethod
    def _normalize_exhibits(exhibits: list) -> list[str]:
        """Normalize exhibits_affected to list[str] — LLM may return dicts."""
        normalized = []
        for item in exhibits:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(item.get("exhibit_id", str(item)))
            else:
                normalized.append(str(item))
        return normalized

    def _format_sections(self, sections: list[ParsedSection], tables: list[ParsedTable] | None = None) -> str:
        """Format sections for prompt, including associated table data."""
        parts = []
        table_map: dict[str, list[ParsedTable]] = {}
        if tables:
            for t in tables:
                if t.source_section:
                    table_map.setdefault(t.source_section, []).append(t)

        for s in sections:
            part = f"Section {s.section_number} ({s.section_title}):\n{s.text}"
            for t in table_map.get(s.section_number, []):
                headers = " | ".join(t.headers)
                rows = "\n".join(" | ".join(row) for row in t.rows)
                part += f"\n\n[Table: {t.caption or t.table_id}]\n{headers}\n{rows}"
            parts.append(part)
        return "\n\n---\n\n".join(parts)

    def _format_prior_amendments(self, prior: list[AmendmentTrackOutput]) -> str:
        if not prior:
            return "(none)"
        parts = []
        for p in prior:
            mods = "; ".join(f"{m.section_number}: {m.change_description}" for m in p.modifications)
            parts.append(f"Amendment {p.amendment_number} ({p.amendment_type}): {mods}")
        return "\n".join(parts)
