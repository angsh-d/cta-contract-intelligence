# ContractIQ Demo Story: Agentic AI for Clinical Trial Agreements

## The Setup (30 seconds)

> "Imagine you're a clinical operations manager at Memorial Medical Center. You've just been asked to confirm the current payment terms for the HEARTBEAT-3 cardiology trial before submitting your next invoice. The problem? The original contract has been amended five times over three years. Each amendment modifies, replaces, or extends different sections — and some changes are buried inside unrelated clauses. Today, answering this question takes your legal and finance teams days of cross-referencing. Let me show you how ContractIQ answers it in seconds."

---

## Act 1: The Contract Stack (1 minute)

**Action**: Navigate to the ContractIQ dashboard. Click "Create New Stack."

Fill in:
- **Name**: HEARTBEAT-3 Phase III CTA
- **Sponsor**: CardioPharm International
- **Site**: Memorial Medical Center
- **Therapeutic Area**: Cardiology

Upload the 6 PDFs in order:
1. Original CTA (January 2022)
2. Amendment 1 (June 2022) — Added cardiac MRI visits
3. Amendment 2 (February 2023) — PI change
4. Amendment 3 (August 2023) — COVID protocol adjustments
5. Amendment 4 (March 2024) — Visit schedule changes
6. Amendment 5 (November 2024) — Timeline extension

> "Six documents. Three years of amendments. Dozens of cross-references. This is typical for a Phase III clinical trial — and this is a *simple* one."

---

## Act 2: The Agentic Pipeline (1.5 minutes)

**Action**: Click **"Process Stack"**. The WebSocket progress bar begins streaming.

As the pipeline runs (~30 seconds), narrate each stage:

| Progress | Stage | What the Agent Does | Agentic Feature |
|----------|-------|---------------------|-----------------|
| 0-30% | **Document Parsing** | Extracts every section, table, exhibit, and metadata from all 6 PDFs | Structured extraction with LLM — not regex |
| 30-50% | **Amendment Tracking** | Identifies what each amendment changes: selective overrides vs. complete replacements. Flags changes **buried inside unrelated sections** | Buried change detection — human-level reading comprehension |
| 50-55% | **Temporal Sequencing** | Orders documents chronologically, handles retroactive amendments, builds a version tree | Date inference from context when explicit dates are missing |
| 55-80% | **Override Resolution** | For every clause, determines the **current truth** — tracing back through each amendment to find what applies today, with full source chain provenance | Multi-document reasoning with provenance tracking |
| 80-90% | **Dependency Mapping** | Builds a graph of which clauses depend on which others (e.g., Section 7 Payment depends on Exhibit B Budget) | 114 cross-clause dependencies discovered automatically |
| 90-100% | **Conflict Detection** | Analyzes the resolved clauses and dependency graph to detect contradictions, gaps, stale references, and buried changes | Pattern recognition across the full contract stack |

> "Nine AI agents just collaborated to do what would take a legal team days. No rules engine. No templates. Pure language understanding."

### Differentiating Feature: Self-Verification

> "Notice something critical — every agent has a confidence threshold. If the confidence drops below threshold, the agent automatically triggers a **self-critique loop**: it reviews its own output, identifies weaknesses, and re-processes with that critique as additional context. This is not retry-on-error — it's genuine self-reflection."

Point out in the logs if visible:
```
WARNING | Low confidence 0.78 (threshold 0.80) for truth_synthesizer — re-processing with self-critique
INFO    | Re-processing truth_synthesizer with self-critique feedback
```

---

## Act 3: Truth Reconstitution — "What is the current truth?" (2 minutes)

**Action**: Switch to the **Query** tab.

### Query 1: The Buried Payment Change

Type: **"What are the current payment terms?"**

> "This is the question that started it all. Let me show you what ContractIQ finds."

**Expected response**: The system returns a **fully formatted answer** — headings, bold text, structured bullet points — not raw text. Payment terms are **Net 45 days**, citing Amendment 3, Section 7.2, effective August 17, 2023.

**The reveal**:
> "Here's what makes this remarkable. The original contract said Net 30 days. Amendment 3 is titled 'COVID-19 Protocol Adjustments' — it's about safety monitoring and remote visits. But buried in Section 7.2 of that amendment, tucked between COVID-related changes, the payment terms were quietly changed from Net 30 to Net 45. A human reviewer focused on COVID changes would almost certainly miss this. Our Amendment Tracker agent flagged it as a **buried change** because the modification type doesn't match the amendment's stated purpose."

**Agentic differentiator**: Point out the source citation card:
- Document: Amendment 3
- Section: 7.2 (Payment Schedule)
- Effective Date: August 17, 2023
- Source Chain: Original CTA Section 7.1 -> Amendment 3 Section 7.2
- Confidence: 98%

> "Every answer comes with a complete audit trail — which document, which section, which date, and the full chain of how this clause evolved. This isn't a search result. It's a **reasoned conclusion with provenance**."

### Bonus: Document Detail View — See the Evidence

**Action**: Click on any document (e.g., Amendment 3) to open the split-panel detail view.

> "Let me show you what the system extracted from this amendment at the clause level."

Each clause card now shows **change tracking and conflict indicators**:

- **Modification badges**: Each clause shows whether it was **Modified** (orange), **Replaced** (red), or **Added** (green) by this amendment
- **Change descriptions**: An amber callout explains *what changed* — e.g., "Changed from Net 30 to Net 45"
- **Conflict indicators**: Clauses with conflicts show severity-coded cards with the conflict type, description, and pain point number. Click to expand the resolution recommendation
- **Amendment history**: Click "Show amendment history" at the bottom of any clause to see a vertical timeline of how that clause evolved across all amendments — which document changed it, what the modification was, and which version is current

> "This is the evidence layer. The Q&A gives you answers; the document view gives you proof. Every badge, every conflict indicator, every history entry traces back to the agent pipeline's analysis."

### Query 2: The Insurance Gap

Type: **"Is our insurance coverage adequate for the extended study period?"**

**Expected response**: The system identifies that Amendment 5 extended the study to June 2025, but the insurance obligation in Section 10 references "through study completion" based on the original December 2024 timeline. There is a potential **6-month coverage gap** (January–June 2025).

> "This is the kind of cross-reference gap that causes real liability exposure. Amendment 5 extended the timeline, but nobody went back to verify that the insurance clause — which is in a completely different section — still covers the new dates. Our agents traced the dependency chain and found the gap automatically."

### Query 3: The Cross-Reference Orphan

Type: **"Are there any visit requirements that conflict with each other?"**

**Expected response**: Amendment 1 added cardiac MRI visits at specific follow-up timepoints. Amendment 4 removed some follow-up visits. But the cardiac MRI requirement from Amendment 1 was never explicitly removed — it references visits that no longer exist. This is an **orphan obligation**.

> "This is multi-hop reasoning. The system needs to understand that (a) Amendment 1 added cardiac MRI at certain visits, (b) Amendment 4 removed those visits, and (c) nobody explicitly addressed the cardiac MRI requirement. Three documents, three years apart, creating an invisible conflict."

---

## Act 4: Conflict Detection — "What's wrong with this contract?" (1.5 minutes)

**Action**: Switch to the **Conflicts** tab.

> "While we were asking questions, the system had already detected these issues during processing. Let me show you everything it found."

Show the auto-detected conflicts:

| Severity | Conflict | Pain Point |
|----------|----------|------------|
| **HIGH** | Payment terms changed from Net 30 to Net 45 in Amendment 3, buried in COVID section | Buried Payment Change |
| **HIGH** | Study extended to June 2025 but insurance coverage references original Dec 2024 end date | Insurance Gap |
| **MEDIUM** | Exhibit B evolved to B-1 to B-2 with inconsistent payment term references | Budget Exhibit Evolution |
| **MEDIUM** | Cardiac MRI requirement from Amendment 1 references visits removed by Amendment 4 | Cross-Reference Orphan |
| **MEDIUM** | Amendment 2 changed PI but Exhibit B-1 still references the previous PI | PI Change Ambiguity |

> "Five material issues. Detected automatically. Each one has actionable evidence — the exact section numbers, the exact text, and a recommendation for resolution. In a traditional review, these would be found over days by different people at different times, if they're found at all."

### Differentiating Feature: Evidence-Based Detection

For each conflict, expand to show:
- **Evidence**: Exact text from multiple documents with inline citations
- **Affected Clauses**: Section numbers from both the modifying and modified documents
- **Recommendation**: Specific action to resolve (e.g., "Issue a standalone amendment to explicitly set payment terms to Net 45 in the Payment section")

> "This isn't pattern matching. Each conflict was detected by an LLM agent that read the actual clause text, understood the semantic meaning, and identified the logical contradiction. The evidence chain lets your legal team verify every finding in minutes."

---

## Act 5: Ripple Effect Analysis — "What breaks if we change this?" (1.5 minutes)

**Action**: Switch to the **Ripple Effects** tab.

> "Let's say CardioPharm wants to extend the data retention period from 15 years to 25 years. Before we agree, we need to know: what else in this contract is affected by that change?"

Enter the proposed change:
- **Section**: 12.1 (Data Retention)
- **Current**: "Site shall retain all study records for a period of 15 years"
- **Proposed**: "Site shall retain all study records for a period of 25 years"

**Expected response** — cascading impacts across multiple hops:

**Hop 1 (Direct):**
- Section 12.2 (Data Storage): Storage costs increase — facilities must be maintained for 25 years
- Section 12.3 (Data Access): Sponsor access rights extend to 25 years

**Hop 2 (Indirect):**
- Section 9.1 (Indemnification): Current indemnification period is 7 years post-study — creates a **gap** of 18 years with no liability coverage
- Exhibit C (Budget): Storage line item needs revision for 25-year horizon

**Hop 3 (Cascade):**
- Section 10.1 (Insurance): Insurance obligation references "through completion of data retention" — 25 years of insurance is not industry standard
- Section 15 (Termination): Early termination clause doesn't address fate of 25-year data retention

> "One clause change. Six downstream impacts across three levels of dependency. The system traced the clause dependency graph — 114 edges — to identify every section that's connected to data retention, directly or transitively. This is why we built the dependency mapper as a dedicated agent."

### Differentiating Feature: Multi-Hop Graph Traversal

> "Traditional contract review tools can find keyword matches. But ripple effect analysis requires understanding *semantic dependencies* — that changing data retention affects indemnification, which affects insurance, which affects the budget. Our Dependency Mapper agent builds this graph, and the Ripple Effect Analyzer traverses it up to 5 hops deep."

### Differentiating Feature: Intelligent Caching

**Action**: Click **"Analyze Impact"** a second time with the same inputs.

> "Notice how the result came back instantly this time? Ripple effect results are cached — the system hashes the section number and clause text to create a cache key. Identical analyses return from cache in milliseconds instead of re-running the LLM agents. The cache is automatically invalidated when the contract stack is reprocessed, so you always get fresh results after new amendments are uploaded."

---

## Act 6: The Architecture Story (1 minute)

> "Let me briefly explain why this works. ContractIQ isn't one large prompt. It's nine specialized AI agents organized in three tiers."

**Tier 1 — Document Ingestion** (what happened):
- Document Parser: Extracts structure
- Amendment Tracker: Identifies changes with buried change detection
- Temporal Sequencer: Orders the version history

**Tier 2 — Reasoning** (what it means):
- Override Resolution: Determines current truth with source chain
- Dependency Mapper: Builds the clause relationship graph
- Conflict Detection: Finds contradictions, gaps, and risks

**Tier 3 — Analysis** (what to do about it):
- Ripple Effect Analyzer: Predicts cascade impact of proposed changes
- Query Router + Truth Synthesizer: Natural language Q&A with citations

### Key Agentic Differentiators

1. **Checkpoint Resume**: If processing is interrupted at any stage, it resumes from the last completed checkpoint — not from scratch. Each stage checks the database for existing results before re-running.

2. **Self-Verification**: Every agent validates its own output against a confidence threshold. Below threshold triggers an automatic self-critique loop — the agent reviews its reasoning, identifies gaps, and re-processes.

3. **Fault Tolerance**: If one section fails during override resolution, the agent logs the failure and continues with remaining sections. Only if *all* sections fail does the pipeline abort.

4. **Inter-Agent Communication**: An in-memory blackboard allows agents to share results. Stage 4 (override resolution) consumes outputs from Stages 1-3. Stage 6 (conflict detection) consumes outputs from all prior stages.

5. **Provider Resilience**: Circuit breakers monitor each LLM provider. If Azure OpenAI fails repeatedly, the system automatically fails over to Gemini — per-agent, per-call, without manual intervention.

6. **Semantic Search**: All clauses are embedded using Gemini embeddings and stored in pgvector. When you ask a question, the system retrieves the most semantically relevant clauses — not just keyword matches — before reasoning over them.

7. **Intelligent Caching**: Query answers and ripple effect analyses are cached with SHA-256 hashed keys. Identical inputs return instantly from the in-memory cache (1-hour TTL). Cache is automatically invalidated per-stack when documents are reprocessed.

8. **Rich Evidence Layer**: Every clause in the document detail view shows its full provenance — modification type badges, change descriptions, conflict indicators with severity, and an expandable amendment history timeline. Q&A responses render as fully formatted markdown with headings, bold text, and structured lists.

---

## The Close (30 seconds)

> "Let me recap what just happened. We uploaded six documents — a real clinical trial contract stack. In 30 seconds, nine AI agents extracted every clause, tracked every amendment, resolved the current truth, mapped 114 dependencies, and detected five material conflicts that a manual review might miss entirely. Then we asked natural language questions and got cited, confidence-scored answers. And we predicted the cascade impact of a proposed change across three levels of dependencies."

> "This is agentic AI applied to contract intelligence. Not a chatbot. Not a search engine. A system of specialized agents that *reason* about legal documents the way your best contract manager does — but in seconds, not days."

---

## Backup Queries (if audience asks for more)

| Question | What It Demonstrates |
|----------|---------------------|
| "Who is the current Principal Investigator?" | PI change tracking across Amendment 2, with stale reference detection |
| "What is the current budget for per-patient costs?" | Exhibit B → B-1 → B-2 evolution with override resolution |
| "What happens if we add a new study site?" | Ripple effect on insurance, indemnification, IRB, budget |
| "Are there any amendments that contradict each other?" | Full conflict detection scan with severity ranking |
| "Show me the history of Section 7 (Payment)" | Clause evolution timeline with source chain provenance |
| "What are our obligations if the study is terminated early?" | Cross-reference analysis across termination, data retention, insurance |

---

## Technical Talking Points (for technical audiences)

- **LLM**: Azure OpenAI GPT-5.2 (primary), Gemini (fallback) — all agents, no model hardcoding
- **Database**: PostgreSQL on NeonDB with pgvector for semantic search + recursive CTEs for dependency graph traversal
- **Embeddings**: Gemini text-embedding-004 (768-dim) with two-tier storage (raw sections + resolved clauses)
- **Architecture**: FastAPI + asyncio background tasks + WebSocket progress streaming
- **Prompts**: 26 externalized prompt templates with runtime parameter substitution
- **Resilience**: Circuit breakers, exponential backoff, self-verification loops, checkpoint resume
- **Zero hardcoding**: Every extraction, classification, and analysis decision is made by an LLM — no regex, no keyword matching, no rule engines
