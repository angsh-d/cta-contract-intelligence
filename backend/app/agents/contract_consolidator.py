"""Contract Consolidation Agent — assembles resolved clauses into a structured document."""

import json
import logging
from typing import Any

from app.agents.base import BaseAgent
from app.exceptions import LLMResponseError
from app.models.agent_schemas import (
    ConsolidatedSection,
    ConsolidationInput,
    ConsolidationOutput,
)

logger = logging.getLogger(__name__)


class ContractConsolidatorAgent(BaseAgent):
    """Organize resolved contract clauses into a hierarchical document mirroring the original CTA."""

    async def process(self, input_data: ConsolidationInput) -> ConsolidationOutput:
        system_prompt = self.prompts.get("contract_consolidator_system")

        # Build truncated clause summaries for the LLM (keep source_chain metadata, trim text)
        clause_summaries = []
        clause_lookup: dict[str, dict[str, Any]] = {}
        for c in input_data.clauses:
            section_num = c.get("section_number", "")
            clause_lookup[section_num] = c
            source_chain = c.get("source_chain", [])
            clause_summaries.append({
                "section_number": section_num,
                "section_title": c.get("section_title", ""),
                "clause_category": c.get("clause_category", "general"),
                "source_chain": source_chain,
                "text_preview": (c.get("current_text", "") or "")[:200],
            })

        user_prompt = self.prompts.get(
            "contract_consolidator_assemble",
            total_clauses=str(len(clause_summaries)),
            clauses_json=json.dumps(clause_summaries, indent=2, default=str),
        )

        result = await self.call_llm(system_prompt, user_prompt)

        raw_structure = result.get("document_structure")
        if not raw_structure or not isinstance(raw_structure, list):
            raise LLMResponseError("ContractConsolidator: LLM response missing 'document_structure'")

        appendices = result.get("appendices", [])

        # Build a lookup from the LLM response keyed by section_number
        llm_item_map: dict[str, dict] = {}
        for item in raw_structure:
            llm_item_map[item.get("section_number", "")] = item

        # Collect all section_numbers claimed as children so we can skip them at top level
        child_set: set[str] = set()
        for item in raw_structure:
            for child_num in item.get("child_sections", []):
                child_set.add(child_num)

        # Build top-level sections (those NOT claimed as a child of another section)
        sections = self._build_sections(raw_structure, clause_lookup, llm_item_map, child_set)

        # Verify all input clauses are represented — append any the LLM missed
        emitted: set[str] = set()
        self._collect_section_numbers(sections, emitted)
        missing = [sn for sn in clause_lookup if sn not in emitted]
        if missing:
            logger.warning("LLM omitted %d clauses — appending: %s", len(missing), missing)
            for sn in missing:
                sections.append(self._make_section(sn, clause_lookup[sn], level=1))

        # Count all sections recursively (including subsections)
        total_sections, amended_count = self._count_sections(sections)

        return ConsolidationOutput(
            contract_stack_id=input_data.contract_stack_id,
            document_structure=sections,
            metadata={
                "total_sections": total_sections,
                "amended_sections": amended_count,
                "appendices": appendices,
            },
            llm_reasoning=result.get("reasoning", ""),
        )

    def _build_sections(
        self,
        raw_structure: list[dict],
        clause_lookup: dict[str, dict[str, Any]],
        llm_item_map: dict[str, dict],
        child_set: set[str],
    ) -> list[ConsolidatedSection]:
        """Convert LLM-ordered structure into ConsolidatedSection objects enriched with full text.

        Skips items already consumed as children. Recurses into child_sections
        so arbitrarily deep hierarchies (7 -> 7.1 -> 7.1.1) are preserved.
        """
        sections: list[ConsolidatedSection] = []
        for item in raw_structure:
            section_num = item.get("section_number", "")
            # Skip sections that belong under a parent
            if section_num in child_set:
                continue
            sections.append(self._build_one(section_num, item, clause_lookup, llm_item_map))
        return sections

    def _build_one(
        self,
        section_num: str,
        item: dict,
        clause_lookup: dict[str, dict[str, Any]],
        llm_item_map: dict[str, dict],
    ) -> ConsolidatedSection:
        """Build a single ConsolidatedSection with recursive children."""
        clause_data = clause_lookup.get(section_num, {})
        source_chain = clause_data.get("source_chain", [])

        _chain_amended = len(source_chain) > 1
        if not _chain_amended and source_chain:
            last_link = source_chain[-1] if isinstance(source_chain[-1], dict) else {}
            mod_type = last_link.get("modification_type") or ""
            if mod_type and mod_type not in ("original", "none", "None"):
                _chain_amended = True
        is_amended = item.get("is_amended", _chain_amended)
        amendment_source = item.get("amendment_source")
        amendment_description = item.get("amendment_description")

        if is_amended and source_chain and not amendment_source:
            last_link = source_chain[-1] if isinstance(source_chain[-1], dict) else {}
            amendment_source = last_link.get("document_label", "")
            amendment_description = last_link.get("change_description", "")

        # Recurse into children
        subsections: list[ConsolidatedSection] = []
        for child_num in item.get("child_sections", []):
            child_item = llm_item_map.get(child_num)
            if child_item:
                subsections.append(self._build_one(child_num, child_item, clause_lookup, llm_item_map))
            elif child_num in clause_lookup:
                # LLM listed it as a child but didn't include it in document_structure — build from clause data
                subsections.append(self._make_section(child_num, clause_lookup[child_num], level=item.get("level", 1) + 1))

        return ConsolidatedSection(
            section_number=section_num,
            section_title=item.get("section_title", clause_data.get("section_title", "")),
            level=item.get("level", 1),
            content=clause_data.get("current_text", ""),
            is_amended=is_amended,
            amendment_source=amendment_source,
            amendment_description=amendment_description,
            subsections=subsections,
        )

    def _make_section(self, section_num: str, clause_data: dict[str, Any], level: int = 1) -> ConsolidatedSection:
        """Create a ConsolidatedSection directly from clause data (for LLM-missed clauses)."""
        source_chain = clause_data.get("source_chain", [])
        is_amended = len(source_chain) > 1
        if not is_amended and source_chain:
            last = source_chain[-1] if isinstance(source_chain[-1], dict) else {}
            mod_type = last.get("modification_type") or ""
            if mod_type and mod_type not in ("original", "none", "None"):
                is_amended = True
        amendment_source = None
        amendment_description = None
        if is_amended and source_chain:
            last = source_chain[-1] if isinstance(source_chain[-1], dict) else {}
            amendment_source = last.get("document_label", "")
            amendment_description = last.get("change_description", "")
        return ConsolidatedSection(
            section_number=section_num,
            section_title=clause_data.get("section_title", ""),
            level=level,
            content=clause_data.get("current_text", ""),
            is_amended=is_amended,
            amendment_source=amendment_source,
            amendment_description=amendment_description,
        )

    def _collect_section_numbers(self, sections: list[ConsolidatedSection], out: set[str]) -> None:
        """Recursively collect all section_numbers in the tree."""
        for s in sections:
            out.add(s.section_number)
            self._collect_section_numbers(s.subsections, out)

    def _count_sections(self, sections: list[ConsolidatedSection]) -> tuple[int, int]:
        """Recursively count total and amended sections."""
        total = 0
        amended = 0
        for s in sections:
            total += 1
            if s.is_amended:
                amended += 1
            child_total, child_amended = self._count_sections(s.subsections)
            total += child_total
            amended += child_amended
        return total, amended
