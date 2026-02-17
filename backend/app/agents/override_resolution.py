"""Tier 2: OverrideResolutionAgent — programmatic text assembly + LLM classification."""

import difflib
import logging
import re
from datetime import date
from typing import Any

from app.agents.base import BaseAgent
from app.models.agent_schemas import (
    AmendmentForSection, ClauseVersion, OverrideResolutionInput,
    OverrideResolutionOutput, SourceChainLink,
)
from app.models.enums import ModificationType

logger = logging.getLogger(__name__)


class OverrideResolutionAgent(BaseAgent):
    """Apply amendments programmatically to preserve verbatim text, then LLM-classify the result."""

    async def process(self, input_data: OverrideResolutionInput) -> OverrideResolutionOutput:
        amendments = sorted(input_data.amendments, key=lambda a: a.effective_date or date.min)

        # Step 1: Programmatic text assembly — no LLM rewriting
        assembled_text, source_chain, assembly_warnings = self._assemble_text(
            input_data.original_clause.text,
            input_data.original_document_label,
            str(input_data.original_document_id),
            amendments,
        )

        # Step 2: LLM classifies the assembled text (does NOT rewrite it)
        system_prompt = self.prompts.get("override_resolution_system")
        user_prompt = self.prompts.get(
            "override_resolution_verify",
            section_number=input_data.section_number,
            assembled_text=assembled_text,
            original_text=input_data.original_clause.text,
            amendment_chain=self._format_amendment_chain(amendments),
        )

        result = await self.call_llm(system_prompt, user_prompt)

        # Confidence: reduce if there were assembly warnings
        base_confidence = result.get("confidence", 0.9)
        if assembly_warnings:
            base_confidence = min(base_confidence, 0.7)
            for w in assembly_warnings:
                logger.warning("Section %s assembly: %s", input_data.section_number, w)

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
                current_text=assembled_text,
                source_chain=source_chain,
                last_modified_by=last_modified_by,
                last_modified_date=last_modified_date,
                confidence=base_confidence,
                clause_category=result.get("clause_category", "general"),
            ),
            llm_reasoning=result.get("reasoning", ""),
            confidence_factors=self._sanitize_confidence_factors(result.get("confidence_factors", {})),
        )

    # ── Programmatic Text Assembly ────────────────────────────────

    def _assemble_text(
        self,
        original_text: str,
        original_label: str,
        original_doc_id: str,
        amendments: list[AmendmentForSection],
    ) -> tuple[str, list[SourceChainLink], list[str]]:
        """Apply amendments in chronological order using code, not LLM.

        Returns (assembled_text, source_chain, warnings).
        """
        current_text = original_text
        warnings: list[str] = []

        source_chain = [
            SourceChainLink(
                stage="original",
                document_id=original_doc_id,
                document_label=original_label,
                text=original_text,
                change_description=None,
                modification_type=None,
            )
        ]

        for amendment in amendments:
            mod = amendment.modification
            mod_type = mod.modification_type
            stage_label = f"amendment_{amendment.amendment_number}"
            doc_label = f"Amendment {amendment.amendment_number} (Effective {amendment.effective_date})"

            new_text, warning = self._apply_modification(
                current_text, mod_type, mod.original_text, mod.new_text,
                amendment.amendment_number,
            )

            if warning:
                warnings.append(warning)

            current_text = new_text

            source_chain.append(
                SourceChainLink(
                    stage=stage_label,
                    document_id=str(amendment.amendment_document_id),
                    document_label=doc_label,
                    text=current_text,
                    change_description=mod.change_description,
                    modification_type=mod_type.value if hasattr(mod_type, 'value') else str(mod_type),
                )
            )

        return current_text, source_chain, warnings

    @staticmethod
    def _strip_section_prefix(text: str) -> str:
        """Remove LLM-added section header prefixes from clause text.

        Strips patterns like:
        - "Section 2.2 (Replacement of Section 2.2): ..."
        - "Section 7.2 (Payment Terms): ..."
        - "Section 10.1: ..."
        """
        # Pattern: "Section X.Y (anything):" at the start of text
        stripped = re.sub(
            r'^Section\s+[\d.]+(?:\s*\([^)]*\))?\s*:\s*',
            '',
            text.strip(),
            flags=re.IGNORECASE,
        )
        return stripped.strip() if stripped != text.strip() else text

    def _apply_modification(
        self,
        current_text: str,
        mod_type: ModificationType,
        original_text: str | None,
        new_text: str | None,
        amendment_number: int,
    ) -> tuple[str, str | None]:
        """Apply a single modification. Returns (result_text, optional_warning)."""
        # Strip any LLM-added section header prefixes from amendment text
        if new_text:
            new_text = self._strip_section_prefix(new_text)
        if original_text:
            original_text = self._strip_section_prefix(original_text)

        if mod_type == ModificationType.COMPLETE_REPLACEMENT:
            if new_text:
                return new_text, None
            # Don't wipe the section — keep original text if replacement is empty
            return current_text, f"Amendment {amendment_number}: complete_replacement but new_text is empty — keeping original"

        if mod_type == ModificationType.DELETION:
            return "", None

        if mod_type == ModificationType.ADDITION:
            if new_text:
                # Append with separator
                if current_text:
                    return current_text.rstrip() + "\n\n" + new_text, None
                return new_text, None
            return current_text, f"Amendment {amendment_number}: addition but new_text is empty"

        if mod_type in (ModificationType.SELECTIVE_OVERRIDE, ModificationType.EXHIBIT_REPLACEMENT):
            if not original_text or not new_text:
                return current_text, (
                    f"Amendment {amendment_number}: {mod_type.value} missing original_text or new_text"
                )
            return self._apply_selective_replacement(current_text, original_text, new_text, amendment_number)

        return current_text, f"Amendment {amendment_number}: unknown modification_type {mod_type}"

    def _apply_selective_replacement(
        self,
        current_text: str,
        find_text: str,
        replace_text: str,
        amendment_number: int,
    ) -> tuple[str, str | None]:
        """Replace find_text with replace_text in current_text. Falls back to fuzzy matching."""

        # Exact match first
        if find_text in current_text:
            return current_text.replace(find_text, replace_text, 1), None

        # Normalize whitespace and try again
        normalized_current = " ".join(current_text.split())
        normalized_find = " ".join(find_text.split())
        if normalized_find in normalized_current:
            # Find the approximate position in the original text
            idx = normalized_current.index(normalized_find)
            # Map back: count chars in original up to this position
            result = self._replace_normalized(current_text, find_text, replace_text)
            if result is not None:
                return result, f"Amendment {amendment_number}: whitespace-normalized match used"

        # Fuzzy matching with SequenceMatcher
        result, ratio = self._fuzzy_replace(current_text, find_text, replace_text)
        if result is not None:
            return result, (
                f"Amendment {amendment_number}: fuzzy match (ratio={ratio:.2f}) used for selective_override"
            )

        # Last resort: append the replacement text with a note
        warning = (
            f"Amendment {amendment_number}: could not locate original_text in current clause. "
            f"Treating as complete_replacement."
        )
        return replace_text, warning

    def _replace_normalized(self, text: str, find: str, replace: str) -> str | None:
        """Try to replace find in text by collapsing whitespace differences."""
        # Build a regex-free approach: walk through text matching word-by-word
        find_words = find.split()
        text_words = text.split()

        for start_idx in range(len(text_words) - len(find_words) + 1):
            if text_words[start_idx:start_idx + len(find_words)] == find_words:
                # Found word-level match — reconstruct with replacement
                before = " ".join(text_words[:start_idx])
                after = " ".join(text_words[start_idx + len(find_words):])
                parts = [p for p in [before, replace, after] if p]
                return " ".join(parts)
        return None

    def _fuzzy_replace(self, text: str, find: str, replace: str, threshold: float = 0.6) -> tuple[str | None, float]:
        """Use SequenceMatcher to find the closest substring match and replace it."""
        find_len = len(find)
        best_ratio = 0.0
        best_start = 0
        best_end = 0

        # Slide a window of approximately find_len over text
        for window_size in range(max(1, find_len - 50), find_len + 50):
            if window_size > len(text):
                break
            for start in range(0, len(text) - window_size + 1, max(1, window_size // 10)):
                candidate = text[start:start + window_size]
                ratio = difflib.SequenceMatcher(None, find, candidate).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = start
                    best_end = start + window_size

        if best_ratio >= threshold:
            return text[:best_start] + replace + text[best_end:], best_ratio

        return None, best_ratio

    # ── Helpers ────────────────────────────────────────────────────

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
