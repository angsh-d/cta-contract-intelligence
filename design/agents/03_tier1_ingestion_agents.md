# 03 — Tier 1: Document Ingestion Agents

> DocumentParserAgent, AmendmentTrackerAgent, TemporalSequencerAgent
> File locations: `backend/app/agents/document_parser.py`, `backend/app/agents/amendment_tracker.py`, `backend/app/agents/temporal_sequencer.py`

---

## Pipeline Flow

```
PDF files ──► DocumentParserAgent (parallel, per document)
                 │
                 ▼
          AmendmentTrackerAgent (sequential, per amendment in date order)
                 │
                 ▼
          TemporalSequencerAgent (single call, all documents)
                 │
                 ▼
          [Tier 2 agents]
```

- **Parsing** runs in parallel across all 6 documents (asyncio.gather).
- **Amendment tracking** runs sequentially because each amendment needs awareness of prior amendments' modifications.
- **Sequencing** runs once after all parsing/tracking is complete.

---

## 1. DocumentParserAgent

**Purpose:** Extract text, structure, tables, and metadata from PDFs.
**LLM:** Claude Sonnet (extraction role)
**File:** `backend/app/agents/document_parser.py`

### Config

```python
AgentConfig(
    agent_name="document_parser",
    llm_role="extraction",
    model_override="claude-sonnet-4-5-20250929",
    max_output_tokens=8192,
    max_retries=3,
    timeout_seconds=120,
    verification_threshold=0.80,
)
```

### Process Flow

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Raw Text Extraction (PyMuPDF)                    │
│   - fitz.open(file_path)                                │
│   - Extract text page-by-page                           │
│   - Track page boundaries                               │
├─────────────────────────────────────────────────────────┤
│ Step 2: Table Extraction (pdfplumber)                    │
│   - Open with pdfplumber                                │
│   - page.extract_tables() per page                      │
│   - Convert to ParsedTable models                       │
├─────────────────────────────────────────────────────────┤
│ Step 3: Chunking (for large documents)                   │
│   - If char_count > 100,000: split into ~50K chunks     │
│   - Each chunk overlaps by 2,000 chars                  │
│   - Process each chunk separately, merge results        │
├─────────────────────────────────────────────────────────┤
│ Step 4: LLM Structuring                                  │
│   - System prompt: document_parser_system.txt            │
│   - User prompt: document_parser_extraction.txt          │
│     with {document_type}, {raw_text} substituted        │
│   - LLM returns JSON → parse into ParsedSection list    │
├─────────────────────────────────────────────────────────┤
│ Step 5: Metadata Extraction                              │
│   - Effective date, parties, study protocol              │
│   - Extracted as part of the LLM structuring call        │
├─────────────────────────────────────────────────────────┤
│ Step 6: pgvector Checkpoint Embedding Upsert (NeonDB)      │
│   - Embed each section's text via Gemini                 │
│     (gemini-embedding-001, 768-dim, RETRIEVAL_DOCUMENT)  │
│   - Upsert to section_embeddings table keyed by          │
│     (contract_stack_id, document_id, section_number)     │
│   - Metadata includes:                                    │
│     {document_id, section_number, section_title,         │
│      effective_date, embedding_model, is_resolved=FALSE} │
│   - These are Stage 1 checkpoint embeddings — raw        │
│     per-document sections, NOT yet resolved across       │
│     amendments. Query-time search uses only              │
│     is_resolved=TRUE embeddings (written after Stage 4). │
└─────────────────────────────────────────────────────────┘
```

### Pseudocode

```python
class DocumentParserAgent(BaseAgent):
    async def process(self, input_data: DocumentParseInput) -> DocumentParseOutput:
        start = time.monotonic()

        # Step 1: Extract raw text
        raw_text, page_count = self._extract_text_pymupdf(input_data.file_path)
        await self._report_progress("text_extraction", 20, f"Extracted {len(raw_text)} chars from {page_count} pages")

        # Step 2: Extract tables
        tables = self._extract_tables_pdfplumber(input_data.file_path)
        await self._report_progress("table_extraction", 35, f"Found {len(tables)} tables")

        # Step 3: Chunk if needed
        chunks = self._chunk_text(raw_text) if len(raw_text) > 100_000 else [raw_text]

        # Step 4: LLM structuring (per chunk)
        all_sections: list[ParsedSection] = []
        metadata: Optional[DocumentMetadata] = None

        for i, chunk in enumerate(chunks):
            system_prompt = self.prompts.get("document_parser_system")
            user_prompt = self.prompts.get(
                "document_parser_extraction",
                document_type=input_data.document_type.value,
                raw_text=chunk,
            )
            result = await self.call_llm(system_prompt, user_prompt)
            chunk_sections = [ParsedSection(**s) for s in result["sections"]]
            all_sections.extend(chunk_sections)

            if metadata is None and "metadata" in result:
                metadata = DocumentMetadata(**result["metadata"])

            await self._report_progress("llm_structuring", 40 + int(40 * (i + 1) / len(chunks)), f"Structured chunk {i+1}/{len(chunks)}")

        # Step 5: Deduplicate sections from overlapping chunks
        sections = self._deduplicate_sections(all_sections)

        # Step 6: Embed and upsert to pgvector (section_embeddings table on NeonDB)
        # These are Stage 1 checkpoint embeddings with is_resolved=FALSE.
        # They capture raw per-document sections before override resolution.
        # Resolved embeddings (is_resolved=TRUE) are written after Stage 4 by
        # the orchestrator's _embed_resolved_clauses() method.
        await self._upsert_embeddings(input_data.contract_stack_id, sections, metadata)
        await self._report_progress("embedding", 95, "Checkpoint embeddings upserted to pgvector (is_resolved=FALSE)")

        return DocumentParseOutput(
            document_id=uuid4(),
            metadata=metadata,
            sections=sections,
            tables=tables,
            raw_text=raw_text,
            char_count=len(raw_text),
            page_count=page_count,
            extraction_model=self.config.model_override,
            extraction_latency_ms=int((time.monotonic() - start) * 1000),
            llm_reasoning=result.get("reasoning", ""),
            extraction_confidence=result.get("extraction_confidence", 0.9),
        )

    def _extract_text_pymupdf(self, file_path: str) -> tuple[str, int]:
        import fitz
        doc = fitz.open(file_path)
        try:
            pages_text = []
            for page in doc:
                pages_text.append(page.get_text())
            return "\n".join(pages_text), len(doc)
        finally:
            doc.close()

    def _extract_tables_pdfplumber(self, file_path: str) -> list[ParsedTable]:
        import pdfplumber
        tables = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                for table_data in (page.extract_tables() or []):
                    if table_data and len(table_data) > 1:
                        headers = [str(h or "") for h in table_data[0]]
                        rows = [[str(c or "") for c in row] for row in table_data[1:]]
                        tables.append(ParsedTable(
                            table_id=f"table_{page_num}_{len(tables)}",
                            headers=headers,
                            rows=rows,
                            page_number=page_num,
                        ))
        return tables

    def _chunk_text(self, text: str, chunk_size: int = 50_000, overlap: int = 2_000) -> list[str]:
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    def _deduplicate_sections(self, sections: list[ParsedSection]) -> list[ParsedSection]:
        seen = {}
        for section in sections:
            if section.section_number not in seen:
                seen[section.section_number] = section
            else:
                # Keep the version with more text
                if len(section.text) > len(seen[section.section_number].text):
                    seen[section.section_number] = section
        return list(seen.values())
```

### Prompt Files

**`prompt/document_parser_system.txt`**
```
You are a legal document parser specializing in clinical trial agreements (CTAs).

Your task is to extract structured data from a contract document. Return ONLY valid JSON.

Extract:
1. Document metadata (type, effective date, parties, study protocol)
2. All sections with their numbers, titles, and full text
3. Identify clause categories (payment, indemnification, insurance, data_retention, confidentiality, termination, etc.)

Before producing your final answer, reason step-by-step about your analysis.
Include your reasoning in the 'reasoning' field of the JSON output.

JSON schema:
{{
  "reasoning": "your step-by-step analysis...",
  "metadata": {{
    "document_type": "cta|amendment|exhibit",
    "effective_date": "YYYY-MM-DD",
    "title": "...",
    "amendment_number": null,
    "parties": [{{"name": "...", "role": "sponsor|institution|investigator"}}],
    "study_protocol": "..."
  }},
  "sections": [
    {{
      "section_number": "7.2",
      "section_title": "Payment Terms",
      "text": "full section text...",
      "page_numbers": [5, 6]
    }}
  ]
}}
```

**`prompt/document_parser_extraction.txt`**
```
Parse this {document_type} document and extract all sections and metadata.

Document text:
{raw_text}

Return structured JSON following the schema in your instructions.
```

---

## 2. AmendmentTrackerAgent

**Purpose:** Identify exactly what each amendment modifies, with old/new text extraction.
**LLM:** Claude Opus (complex reasoning — needs to understand legal amendment language)
**File:** `backend/app/agents/amendment_tracker.py`

### Config

```python
AgentConfig(
    agent_name="amendment_tracker",
    llm_role="complex_reasoning",
    model_override="claude-opus-4-5-20250514",
    max_output_tokens=8192,
    max_retries=3,
    timeout_seconds=180,
)
```

### Five Modification Patterns

The agent must detect these amendment language patterns (from HEARTBEAT-3):

| # | Pattern | Example |
|---|---------|---------|
| 1 | Complete replacement | "Section 7.2 is hereby deleted in its entirety and replaced with the following: ..." |
| 2 | Selective override | "Section 7.2 is amended by deleting 'Net 30' and substituting 'Net 45'" |
| 3 | Addition | "A new Section 14.5 is hereby added to the Agreement as follows: ..." |
| 4 | Deletion | "Section 8.3 is hereby deleted" |
| 5 | Exhibit replacement | "Exhibit B to the Agreement is hereby replaced by Exhibit B-1 attached hereto" |

### Process Flow

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: WHEREAS Rationale Extraction                     │
│   - Extract recitals/WHEREAS clauses                    │
│   - Summarize rationale for this amendment              │
├─────────────────────────────────────────────────────────┤
│ Step 2: Modification Identification                      │
│   - Feed amendment text + original CTA sections to LLM  │
│   - Also feed prior amendment outputs for context       │
│   - LLM identifies each section modified and how        │
├─────────────────────────────────────────────────────────┤
│ Step 3: Old/New Text Extraction                          │
│   - For each modification, extract:                     │
│     - original_text (from original CTA or prior amend)  │
│     - new_text (from this amendment)                    │
│     - modification_type (one of 5 patterns)             │
├─────────────────────────────────────────────────────────┤
│ Step 4: Exhibit Tracking                                 │
│   - Identify exhibit replacements (B → B-1 → B-2)      │
│   - Track which amendment introduced each exhibit       │
└─────────────────────────────────────────────────────────┘
```

### Pseudocode

```python
class AmendmentTrackerAgent(BaseAgent):
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

        modifications = [Modification(**m) for m in result["modifications"]]
        return AmendmentTrackOutput(
            amendment_document_id=input_data.amendment_document_id,
            amendment_number=input_data.amendment_number,
            effective_date=result.get("effective_date"),
            amendment_type=result["amendment_type"],
            rationale=result["rationale"],
            modifications=modifications,
            sections_modified=[m.section_number for m in modifications],
            exhibits_affected=result.get("exhibits_affected", []),
            llm_reasoning=result.get("reasoning", ""),
            extraction_confidence=result.get("extraction_confidence", 0.9),
        )

    def _format_sections(self, sections: list[ParsedSection], tables: list[ParsedTable] | None = None) -> str:
        """Format sections for prompt, including associated table data when available.

        Tables are critical for detecting budget exhibit changes (Pain Point #2).
        """
        parts = []
        # Build section_number → tables mapping
        table_map: dict[str, list[ParsedTable]] = {}
        if tables:
            for t in tables:
                if t.source_section:
                    table_map.setdefault(t.source_section, []).append(t)

        for s in sections:
            part = f"Section {s.section_number} ({s.section_title}):\n{s.text}"
            # Append any tables associated with this section
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
```

### Prompt Files

**`prompt/amendment_tracker_system.txt`**
```
You are an expert contract amendment analyst for clinical trial agreements.

Your task is to analyze an amendment and identify EXACTLY what it modifies. You must detect these modification patterns:

1. COMPLETE REPLACEMENT: "Section X is hereby deleted in its entirety and replaced with..."
2. SELECTIVE OVERRIDE: "Section X is amended by deleting 'Y' and substituting 'Z'"
3. ADDITION: "A new Section X.Y is hereby added..."
4. DELETION: "Section X is hereby deleted"
5. EXHIBIT REPLACEMENT: "Exhibit B is hereby replaced by Exhibit B-1"

For each modification you MUST extract:
- section_number: the exact section being modified
- modification_type: one of [complete_replacement, selective_override, addition, deletion, exhibit_replacement]
- original_text: the text BEFORE this amendment (from original or prior amendment)
- new_text: the text AFTER this amendment (null for deletions)
- change_description: a human-readable summary of what changed

CRITICAL: Do NOT miss buried changes. Payment term changes hidden inside COVID-related amendments, PI reference changes, and exhibit version updates are all common patterns.

Also extract:
- The WHEREAS rationale (why this amendment exists)
- The amendment_type category (protocol_change, budget_revision, pi_change, covid_protocol, study_extension, etc.)
- Any exhibit changes

Before producing your final answer, reason step-by-step about your analysis.
Include your reasoning in the 'reasoning' field of the JSON output.

Return ONLY valid JSON.
```

**`prompt/amendment_tracker_analysis.txt`**
```
Analyze Amendment {amendment_number}.

Amendment text:
{amendment_text}

Original CTA sections (for comparison):
{original_sections}

Prior amendment modifications (for context):
{prior_modifications}

Before producing your final answer, reason step-by-step about your analysis.
Include your reasoning in the 'reasoning' field of the JSON output.

Return JSON:
{{
  "reasoning": "your step-by-step analysis...",
  "amendment_type": "protocol_change|budget_revision|pi_change|covid_protocol|study_extension",
  "rationale": "extracted WHEREAS summary",
  "effective_date": "YYYY-MM-DD or null",
  "modifications": [
    {{
      "section_number": "7.2",
      "modification_type": "selective_override",
      "original_text": "...Net 30...",
      "new_text": "...Net 45...",
      "change_description": "Payment terms changed from Net 30 to Net 45"
    }}
  ],
  "exhibits_affected": ["Exhibit B-1"]
}}
```

### Why Sequential Processing

Amendment 3 might say "Section 7.2, as previously amended by Amendment 1, is hereby further amended..." The tracker needs Amendment 1's output to correctly identify what "previously amended" refers to. Processing sequentially by date order ensures this context is available.

### Buried Change Detection (Step 4)

The AmendmentTracker's most critical capability is detecting changes hidden in unexpected contexts (Pain Point #1: payment terms buried in a COVID amendment). After the initial LLM extraction, a dedicated adversarial scan checks for missed modifications:

```python
    async def _scan_for_buried_changes(
        self, amendment_text: str, found_modifications: list[Modification]
    ) -> list[Modification]:
        """
        Adversarial scan: ask the LLM to look specifically for changes that
        might have been missed in the initial extraction. This catches changes
        buried in unrelated sections (e.g., payment terms in a COVID protocol amendment).
        """
        found_sections = [m.section_number for m in found_modifications]
        system_prompt = self.prompts.get("amendment_tracker_buried_scan")
        user_prompt = self.prompts.get(
            "amendment_tracker_buried_scan_input",
            amendment_text=amendment_text,
            already_found=json.dumps(found_sections),
        )
        result = await self.call_llm(system_prompt, user_prompt)

        missed = []
        for mod in result.get("missed_modifications", []):
            if mod["section_number"] not in found_sections:
                missed.append(Modification(**mod))

        if missed:
            logger.warning(
                "Buried change scan found %d missed modifications: %s",
                len(missed), [m.section_number for m in missed],
            )
        return missed
```

**Prompt file: `prompt/amendment_tracker_buried_scan.txt`**
```
You are performing an ADVERSARIAL review of a contract amendment.

A prior analysis already identified modifications to specific sections. Your job is to find
modifications that were MISSED — particularly changes buried in unrelated contexts.

Common patterns for buried changes:
- Payment term changes hidden inside protocol modification sections
- Insurance requirement changes embedded in study extension language
- Indemnification modifications tucked into compliance updates
- Budget changes referenced only in exhibit replacement language

Look for ANY operative language that modifies terms, obligations, timelines, or financial
provisions that is NOT in the already-found list.

Think step-by-step: read each paragraph of the amendment and ask "does this change any
existing contract term?" If yes, check if it was already found.

Return JSON:
{{
  "reasoning": "Your step-by-step analysis of what you checked and why",
  "missed_modifications": [
    {{
      "section_number": "7.2",
      "modification_type": "selective_override",
      "original_text": "...",
      "new_text": "...",
      "change_description": "Payment terms changed from Net 30 to Net 45 — buried in COVID section"
    }}
  ]
}}

If no missed modifications are found, return: {{"reasoning": "...", "missed_modifications": []}}
```

### Self-Verification Override

```python
    async def _verify_output(
        self, output: AmendmentTrackOutput, input_data: AmendmentTrackInput
    ) -> AmendmentTrackOutput:
        """Verify every modification references a real section from the original CTA."""
        known_sections = {s.section_number for s in input_data.original_sections}
        for mod in output.modifications:
            if (mod.modification_type != ModificationType.ADDITION
                    and mod.section_number not in known_sections):
                logger.warning(
                    "Modification references unknown section %s — may be hallucinated",
                    mod.section_number,
                )
                # Flag for human review rather than silently dropping
                output.extraction_confidence = min(output.extraction_confidence, 0.6)
        return output
```

---

## 3. TemporalSequencerAgent

**Purpose:** Use LLM reasoning to determine document ordering, supersession relationships, and build the version tree — handling retroactive amendments, conditional effectiveness, and non-linear precedence. Deterministic date sorting serves as a validation check, not the primary logic.
**LLM:** Claude Sonnet (temporal reasoning — primary logic, not just date inference)
**File:** `backend/app/agents/temporal_sequencer.py`

### Config

```python
AgentConfig(
    agent_name="temporal_sequencer",
    llm_role="extraction",
    model_override="claude-sonnet-4-5-20250929",
    max_output_tokens=4096,
    max_retries=3,
    timeout_seconds=60,
    verification_threshold=0.80,
)
```

### Process Flow — LLM-First Temporal Reasoning

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Date Validation + LLM Inference                  │
│   - Check all documents have effective_date              │
│   - If missing: LLM infers from document text            │
│   - LLM also infers amendment_number from document title │
│     (never falls back to position-based numbering)       │
├─────────────────────────────────────────────────────────┤
│ Step 2: LLM Temporal Reasoning (PRIMARY)                 │
│   - Feed ALL documents with metadata to LLM              │
│   - LLM reasons about:                                   │
│     a) Execution date vs. effective date disagreements    │
│     b) Retroactive effectiveness clauses                 │
│     c) Conditional effectiveness ("upon IRB approval")   │
│     d) Non-chronological precedence language              │
│     e) Branching supersession (not just linear)          │
│   - LLM produces: ordering, supersession tree, warnings  │
├─────────────────────────────────────────────────────────┤
│ Step 3: Deterministic Validation                         │
│   - Sort by effective_date as a CROSS-CHECK              │
│   - Compare LLM ordering vs. deterministic ordering      │
│   - If they disagree: flag divergences as warnings       │
│     (LLM ordering is authoritative)                      │
├─────────────────────────────────────────────────────────┤
│ Step 4: Version Tree + Timeline Construction             │
│   - Build VersionTree from LLM's supersession output     │
│   - Supports branching supersession (not just linear)    │
│   - Create TimelineEvent for each document               │
├─────────────────────────────────────────────────────────┤
│ Step 5: PostgreSQL SUPERSEDES Relationships              │
│   - INSERT INTO document_supersessions                   │
│   - Based on LLM-determined supersession, not position   │
└─────────────────────────────────────────────────────────┘
```

### Pseudocode

```python
class TemporalSequencerAgent(BaseAgent):
    def __init__(self, config, llm_provider, prompt_loader, db_pool,
                 progress_callback=None, fallback_provider=None,
                 trace_context=None, llm_semaphore=None):
        super().__init__(config, llm_provider, prompt_loader, progress_callback,
                         fallback_provider, trace_context, llm_semaphore)
        self.db = db_pool

    async def process(self, input_data: TemporalSequenceInput) -> TemporalSequenceOutput:
        documents = [doc.model_copy() for doc in input_data.documents]
        dates_inferred: list[UUID] = []

        # Step 1: Infer missing dates AND amendment numbers via LLM
        for i, doc in enumerate(documents):
            if doc.effective_date is None or doc.amendment_number is None:
                inferred = await self._infer_metadata(doc)
                updates = {}
                if doc.effective_date is None and inferred.get("effective_date"):
                    updates["effective_date"] = date.fromisoformat(inferred["effective_date"])
                    dates_inferred.append(doc.document_id)
                if doc.amendment_number is None and inferred.get("amendment_number"):
                    updates["amendment_number"] = inferred["amendment_number"]
                if updates:
                    documents[i] = doc.model_copy(update=updates)

        # Step 2: LLM Temporal Reasoning (PRIMARY ordering logic)
        system_prompt = self.prompts.get("temporal_sequencer_system")
        user_prompt = self.prompts.get(
            "temporal_sequencer_ordering",
            documents_json=self._format_documents_for_llm(documents),
        )
        llm_result = await self.call_llm(system_prompt, user_prompt)

        llm_order = llm_result["chronological_order"]  # list of document_id strings
        llm_supersessions = llm_result["supersessions"]  # list of {predecessor, successor, reason}
        warnings = llm_result.get("warnings", [])
        reasoning = llm_result.get("reasoning", "")

        chronological_order = [UUID(doc_id) for doc_id in llm_order]

        # Step 3: Deterministic validation (cross-check, not primary)
        deterministic_order = sorted(documents, key=lambda d: (
            d.effective_date,
            0 if d.document_type == DocumentType.CTA else 1,
            d.amendment_number or 999,
        ))
        deterministic_ids = [d.document_id for d in deterministic_order]

        if chronological_order != deterministic_ids:
            divergences = []
            for i, (llm_id, det_id) in enumerate(zip(chronological_order, deterministic_ids)):
                if llm_id != det_id:
                    divergences.append(f"Position {i}: LLM={llm_id}, date-sort={det_id}")
            if divergences:
                warnings.append(
                    f"LLM ordering diverges from date-based sort at {len(divergences)} position(s): "
                    f"{'; '.join(divergences[:3])}. LLM ordering used (may indicate retroactive amendments)."
                )
                logger.warning("Temporal ordering divergence: %s", divergences)

        # Step 4: Build version tree from LLM supersession output
        doc_map = {str(d.document_id): d for d in documents}
        cta = next((d for d in documents if d.document_type == DocumentType.CTA), None)

        version_tree = VersionTree(
            root_document_id=cta.document_id,
            amendments=[
                VersionTreeNode(
                    document_id=UUID(s["successor"]),
                    amendment_number=doc_map[s["successor"]].amendment_number or 0,
                    effective_date=doc_map[s["successor"]].effective_date,
                    supersedes_document_id=UUID(s["predecessor"]),
                    label=f"Amendment {doc_map[s['successor']].amendment_number} ({doc_map[s['successor']].effective_date})",
                )
                for s in llm_supersessions
            ],
        )

        # Timeline
        timeline = [
            TimelineEvent(
                document_id=UUID(doc_id),
                event_date=doc_map[doc_id].effective_date,
                document_type=doc_map[doc_id].document_type,
                label=doc_map[doc_id].document_version or doc_map[doc_id].filename,
                amendment_number=doc_map[doc_id].amendment_number,
            )
            for doc_id in llm_order
        ]

        # Step 5: Write SUPERSEDES to PostgreSQL
        await self._write_supersedes(version_tree)

        return TemporalSequenceOutput(
            contract_stack_id=input_data.contract_stack_id,
            chronological_order=chronological_order,
            version_tree=version_tree,
            timeline=timeline,
            dates_inferred=dates_inferred,
            llm_reasoning=reasoning,
        )

    async def _infer_metadata(self, doc: DocumentSummary) -> dict:
        """Use LLM to infer effective date AND amendment number from document text/filename."""
        system_prompt = self.prompts.get("temporal_sequencer_date_inference")
        user_prompt = (
            f"Filename: {doc.filename}\n"
            f"Document version: {doc.document_version or 'unknown'}\n"
            f"Document type: {doc.document_type.value}"
        )
        return await self.call_llm(system_prompt, user_prompt)

    def _format_documents_for_llm(self, documents: list[DocumentSummary]) -> str:
        parts = []
        for d in documents:
            parts.append(
                f"Document ID: {d.document_id}\n"
                f"  Type: {d.document_type.value}\n"
                f"  Filename: {d.filename}\n"
                f"  Effective Date: {d.effective_date}\n"
                f"  Amendment Number: {d.amendment_number}\n"
                f"  Version: {d.document_version or 'N/A'}"
            )
        return "\n\n".join(parts)

    async def _write_supersedes(self, version_tree: VersionTree) -> None:
        """Write SUPERSEDES relationships atomically — single transaction for all-or-nothing."""
        async with self.db.acquire() as conn:
            async with conn.transaction():
                for node in version_tree.amendments:
                    await conn.execute(
                        """
                        INSERT INTO document_supersessions
                            (contract_stack_id, predecessor_document_id, successor_document_id)
                        VALUES ($1, $2, $3)
                        ON CONFLICT DO NOTHING
                        """,
                        str(version_tree.root_document_id),  # contract_stack_id from context
                        str(node.supersedes_document_id),
                        str(node.document_id),
                    )
```

### Prompt Files

**`prompt/temporal_sequencer_date_inference.txt`**
```
Given a clinical trial document filename and version string, infer:
1. The effective date (YYYY-MM-DD format)
2. The amendment number (integer, if this is an amendment)

Common patterns:
- "Amendment 1 - HEARTBEAT-3 (June 2022).pdf" → date: 2022-06-01, amendment_number: 1
- "Original CTA - HEARTBEAT-3 (January 2022).pdf" → date: 2022-01-01, amendment_number: null

If a specific day is not available, use the 1st of the month.
If no date can be inferred at all, return null for effective_date.
NEVER fall back to position-based numbering for amendment_number — infer from the document text or return null.

Before producing your final answer, reason step-by-step about your analysis.
Include your reasoning in the 'reasoning' field of the JSON output.

Return JSON: {{"reasoning": "your analysis...", "effective_date": "YYYY-MM-DD", "amendment_number": 1}}
```

**`prompt/temporal_sequencer_system.txt`**
```
You are a legal document temporal reasoning expert for clinical trial agreements.

Your task is to determine the correct chronological ordering and supersession relationships
for a set of contract documents. This is NOT a simple date sort — you must reason about:

1. RETROACTIVE AMENDMENTS: An amendment executed later may be effective from an earlier date
2. CONDITIONAL EFFECTIVENESS: "This amendment shall be effective upon IRB approval" — the effective
   date may differ from the execution date
3. NON-CHRONOLOGICAL PRECEDENCE: "Notwithstanding the date of this Amendment, it shall supersede
   Amendment 4 with respect to..." — explicit precedence language overrides date ordering
4. BRANCHING SUPERSESSION: Not all amendments form a linear chain. Amendment 5 might supersede
   Amendment 3 directly (not Amendment 4) for certain sections
5. PARTIAL OVERLAPS: Different sections of the same amendment may have different effective dates

The original CTA is always the root. Amendments supersede either the CTA directly or prior amendments.

Before producing your final answer, reason step-by-step about your analysis.
Include your reasoning in the 'reasoning' field of the JSON output.

Return ONLY valid JSON.
```

**`prompt/temporal_sequencer_ordering.txt`**
```
Determine the chronological ordering and supersession relationships for these documents:

{documents_json}

Analyze each document and determine:
1. The correct chronological order (considering retroactive dates, conditional effectiveness, etc.)
2. Which document each amendment supersedes (may not be the immediately preceding one)
3. Any warnings about non-obvious temporal relationships

Before producing your final answer, reason step-by-step about your analysis.

Return JSON:
{{
  "reasoning": "your step-by-step analysis of temporal relationships...",
  "chronological_order": ["doc-id-1", "doc-id-2", ...],
  "supersessions": [
    {{"predecessor": "doc-id-1", "successor": "doc-id-2", "reason": "Amendment 1 supersedes Original CTA"}}
  ],
  "warnings": ["Any non-obvious temporal relationships detected"]
}}
```

---

## 4. HEARTBEAT-3 Expected Processing

For the 6 HEARTBEAT-3 documents, the Tier 1 pipeline should produce:

| Document | Parser Output | Tracker Output | Sequencer Position |
|----------|--------------|----------------|-------------------|
| Original CTA (Jan 2022) | ~20 sections, 3-4 tables, metadata with parties | N/A (not an amendment) | Position 1 (root) |
| Amendment 1 (Jun 2022) | Protocol change sections | Modifications to protocol sections, new cardiac MRI visit | Position 2 |
| Amendment 2 (Feb 2023) | PI change + budget sections | PI name change, Exhibit B → B-1, **references old PI in budget** | Position 3 |
| Amendment 3 (Aug 2023) | COVID protocol + **buried payment change** | COVID changes + **Section 7.2: Net 30 → Net 45** (Pain Point #1) | Position 4 |
| Amendment 4 (Mar 2024) | Protocol revision, visit schedule changes | Removed follow-up visits, **cardiac MRI from Amend 1 survives** (Pain Point #4) | Position 5 |
| Amendment 5 (Nov 2024) | Study extension | Extended study timeline, **insurance coverage ambiguous** (Pain Point #3) | Position 6 |

---

## 5. Error Handling

| Error | Handling |
|-------|----------|
| PDF cannot be opened | Raise `DocumentExtractionError` with file path — never return partial results |
| No sections extracted | Raise `DocumentExtractionError` — document must have at least 1 section |
| LLM returns invalid JSON | 3-tier JSON parsing in BaseAgent handles this |
| LLM timeout | Retry with exponential backoff per BaseAgent policy |
| Missing effective date | TemporalSequencer infers via LLM — if that also fails, raise `DocumentExtractionError` |
| Duplicate section numbers | DocumentParser deduplicates by keeping longest text |

---

## 6. Exhibit Document Handling

**Important:** The `DocumentType` enum includes `EXHIBIT`, but in HEARTBEAT-3, exhibit documents (Budget Exhibit B, B-1, B-2) are typically embedded as appendices within their parent amendment PDFs rather than uploaded as standalone files. The pipeline handles this as follows:

| Scenario | Handling |
|----------|----------|
| Exhibit embedded in amendment PDF | DocumentParser extracts exhibit content as sections with `clause_category="budget"` or similar. AmendmentTracker detects exhibit replacements (B → B-1) via modification pattern #5. Tables within exhibits are extracted by `_extract_tables_pdfplumber` and associated with their parent sections. |
| Standalone exhibit file | DocumentParser classifies it as `DocumentType.EXHIBIT`. The orchestrator (doc 06) includes it in parsing but skips amendment tracking (exhibits are not amendments). Override resolution processes exhibit sections like any other clause. |
| Exhibit referenced but not present | ConflictDetection (doc 04) flags this as a `gap` conflict — a referenced exhibit is missing from the contract stack. |

If a standalone exhibit is uploaded and the orchestrator filters only for `CTA` and `AMENDMENT` types, the exhibit will be silently dropped. **Implementation note:** The orchestrator's `process_contract_stack()` must handle `DocumentType.EXHIBIT` explicitly — either treating exhibits as part of their parent amendment or processing them as standalone documents with appropriate sequencing.
