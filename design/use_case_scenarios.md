# ContractIQ - Use Case Scenarios

## Overview
ContractIQ supports clinical trial contract management through AI-powered analysis. These use cases map directly to the technical implementation and demonstrate real-world value.

---

## CATEGORY 1: DOCUMENT MANAGEMENT & PROCESSING

### UC-001: Upload and Process Contract Stack
**Actor:** Clinical Operations Manager  
**Goal:** Digitize and structure a complete contract stack for analysis

**Scenario:**
1. User creates a new contract stack: "HEARTBEAT-3 - Memorial Medical"
2. Uploads 6 documents: Original CTA + 5 amendments (PDFs)
3. System processes documents in background (30 seconds)
4. System extracts sections, identifies amendments, builds knowledge graph
5. User receives notification: "Contract stack ready for analysis"

**Technical Flow:**
- POST `/api/v1/contract-stacks` → Create stack
- POST `/api/v1/contract-stacks/{id}/documents` → Upload each document
- DocumentParserAgent → Extract text and structure
- AmendmentTrackerAgent → Identify modifications
- TemporalSequencerAgent → Order chronologically
- DependencyMapperAgent → Build knowledge graph

**Success Criteria:**
- All documents parsed successfully
- Sections identified with 95%+ accuracy
- Knowledge graph contains all clause relationships
- Processing time <30 seconds per stack

**Value:** Transforms 6 separate PDFs into a queryable, analyzable knowledge base

---

### UC-002: View Contract Timeline
**Actor:** Site Contract Manager  
**Goal:** Understand the evolution of a contract over time

**Scenario:**
1. User opens contract stack
2. Selects "Timeline View"
3. System displays interactive timeline showing:
   - Original CTA (Jan 2022)
   - Amendment 1 (Jun 2022) - Added MRI, extended follow-up
   - Amendment 2 (Feb 2023) - PI change
   - Amendment 3 (Aug 2023) - COVID provisions + payment change
   - Amendment 4 (Mar 2024) - Protocol changes
   - Amendment 5 (Nov 2024) - Timeline extension
4. User clicks on Amendment 3 to see details
5. System shows: sections modified, rationale, current vs. previous terms

**Technical Flow:**
- GET `/api/v1/contract-stacks/{id}/timeline`
- TemporalSequencerAgent data retrieved from database
- Frontend renders interactive timeline with D3.js

**Success Criteria:**
- Chronological accuracy 100%
- All amendments properly sequenced
- Visual clarity on what changed when

**Value:** Instant visibility into contract evolution vs. manual document review

---

## CATEGORY 2: TRUTH RECONSTITUTION

### UC-003: Query Current Contract Terms
**Actor:** Financial Analyst  
**Goal:** Determine current payment terms without reading all amendments

**Scenario:**
1. User asks: "What are the current payment terms for this site?"
2. System analyzes in real-time (10 seconds):
   - Searches all payment-related clauses
   - Applies amendment override logic
   - Identifies that Amendment 3 changed terms from Net 30 to Net 45
   - Verifies no later amendments modified payment terms
3. System responds: "Net 45 days per Amendment 3, Section 7.2, effective August 17, 2023"
4. System shows source: Direct link to Amendment 3, Section 7.2 with highlighted text
5. User clicks source to view original document context

**Technical Flow:**
- POST `/api/v1/contract-stacks/{id}/query`
- Query router classifies as "truth_reconstitution"
- Vector search finds relevant clauses
- OverrideResolutionAgent determines current state
- TruthConsolidationAgent synthesizes answer with provenance

**Success Criteria:**
- Answer provided in <15 seconds
- 100% accuracy on current terms
- Complete source citation with document + section + date
- Confidence score displayed

**Value:** 10 seconds vs. 2-3 days manual review

---

### UC-004: Compare Clause Evolution
**Actor:** Legal Counsel  
**Goal:** Understand how a specific clause changed over time

**Scenario:**
1. User queries: "Show me the history of Section 7.2 (Payment Terms)"
2. System displays clause evolution:
   - **Original CTA (Jan 2022):** "Net 30 days"
   - **Amendment 1 (Jun 2022):** No change
   - **Amendment 2 (Feb 2023):** No change
   - **Amendment 3 (Aug 2023):** Changed to "Net 45 days" ⚠️ Buried in COVID amendment
   - **Amendment 4 (Mar 2024):** No change
   - **Amendment 5 (Nov 2024):** No change
   - **Current State:** "Net 45 days"
3. User sees warning: "This change was buried in Amendment 3 (COVID provisions)"
4. User exports clause history to Word document for legal review

**Technical Flow:**
- POST `/api/v1/contract-stacks/{id}/clause-history`
- Query specific section number
- OverrideResolutionAgent traces all modifications
- Frontend displays as timeline with diff highlighting

**Success Criteria:**
- Complete modification history
- Highlight buried changes
- Exportable format for legal review

**Value:** Instant clause archaeology vs. manual cross-referencing

---

### UC-005: Answer Multi-Clause Questions
**Actor:** Study Manager  
**Goal:** Get comprehensive answers requiring multiple contract sections

**Scenario:**
1. User asks: "What are the complete payment terms including holdback and closeout?"
2. System identifies this requires multiple sections:
   - Section 7.2: Payment timing (Net 45)
   - Section 7.4: Holdback (10%)
   - Exhibit B-2: Per-patient amounts ($19,800)
   - Exhibit B-2: Closeout payment ($6,000)
3. System synthesizes comprehensive answer:
   ```
   Current Payment Structure:
   - Timing: Net 45 days (Amendment 3, Section 7.2)
   - Per-Patient: $19,800 (Exhibit B-2, Amendment 4)
   - Holdback: 10% retained until LPLV + closeout (Section 7.4)
   - Closeout: $6,000 upon completion (Exhibit B-2)
   - Total Budget: $442,900 for 20 patients
   ```
4. Each component includes clickable source links

**Technical Flow:**
- Multi-clause semantic search
- Cross-reference resolution
- TruthConsolidationAgent synthesizes from multiple sources
- Provenance tracking for each component

**Success Criteria:**
- Captures all relevant clauses
- Synthesizes coherent answer
- Complete source attribution

**Value:** Comprehensive answer from scattered sources in one query

---

## CATEGORY 3: CONFLICT & RISK DETECTION

### UC-006: Detect Hidden Conflicts
**Actor:** Compliance Officer  
**Goal:** Proactively identify contract inconsistencies before they cause problems

**Scenario:**
1. User clicks "Analyze Conflicts" for contract stack
2. System runs comprehensive conflict detection (15 seconds)
3. System identifies 6 conflicts:

   **🔴 CRITICAL: Insurance Coverage Gap**
   - Original agreement: Coverage through Dec 31, 2024
   - Amendment 5: Study extended to June 30, 2025
   - Risk: 6-month gap with no insurance coverage
   - Impact: Regulatory violation, site liability exposure
   - Recommendation: Extend insurance policy immediately

   **🟠 HIGH: Buried Payment Term Change**
   - Amendment 3: Payment terms changed from Net 30 to Net 45
   - Location: Hidden in COVID-related amendment (Section 7.2)
   - Risk: Finance team may not know, causing payment delays
   - Recommendation: Update accounting system to Net 45

   **🟡 MEDIUM: Cross-Reference Ambiguity**
   - Amendment 4: "Protocol as amended" (doesn't specify version)
   - Risk: Unclear whether refers to Protocol 2.0 or 3.0
   - Recommendation: Clarify in next amendment

4. User exports conflict report to PDF for legal review
5. User marks conflicts as "acknowledged" or "resolved"

**Technical Flow:**
- POST `/api/v1/contract-stacks/{id}/analyze/conflicts`
- ConflictDetectionAgent analyzes all clauses
- Multiple analysis passes: contradictions, gaps, ambiguities, buried changes
- RiskScoringAgent prioritizes by severity
- Results saved to conflicts table

**Success Criteria:**
- Detect 90%+ of material conflicts (validated vs. human review)
- <5% false positives
- Processing time <20 seconds
- Clear prioritization (critical → high → medium → low)

**Value:** Proactive risk detection vs. reactive crisis management

---

### UC-007: Monitor Contract Health
**Actor:** Contract Administrator  
**Goal:** Continuously monitor portfolio for emerging issues

**Scenario:**
1. User views "Contract Health Dashboard"
2. System displays real-time monitoring across portfolio:

   **Active Alerts (3):**
   - Site 301: Insurance expires Dec 31, 2024 but study runs to June 2025 (90 days)
   - Site 402: Payment cycle degraded from Net 45 to Net 68 (trending)
   - Site 505: IRB approval expires in 30 days, study still active

   **Upcoming Events (5):**
   - 3 contracts expire Q1 2025
   - 2 sites approaching enrollment targets (closeout pending)

   **Regulatory Changes (1):**
   - FDA updated ICF guidance → 47 contracts require review

3. User clicks on Site 301 alert
4. System shows detailed analysis and recommended actions
5. User assigns task to legal team to extend insurance

**Technical Flow:**
- Background monitoring jobs (Celery)
- Queries run daily checking for:
  - Expiration dates vs. study timelines
  - Payment trends (via queries table history)
  - Regulatory database updates (external API)
- Real-time dashboard via WebSocket updates

**Success Criteria:**
- 100% detection of expiration date conflicts
- 90-day advance warning for critical events
- Zero missed regulatory updates

**Value:** Prevents emergencies through proactive monitoring

---

## CATEGORY 4: RIPPLE EFFECT ANALYSIS

### UC-008: Analyze Amendment Impact Before Execution
**Actor:** Clinical Operations Director  
**Goal:** Understand full implications of proposed amendment before issuing to sites

**Scenario:**
1. User proposes amendment: "Extend data retention from 15 years to 25 years"
2. User clicks "Analyze Ripple Effects"
3. System performs multi-hop analysis (20 seconds):

   **Direct Impacts (Hop 1):**
   - Section 9.4: Archive storage costs increase by $3-5K
   - Section 9.5: Data destruction timeline extends +10 years

   **Indirect Impacts (Hop 2):**
   - Section 12.1: Indemnification covers 7 years but data retention is 25 years → GAP
   - Section 13.4: Insurance coverage ends year 5 but data retention is 25 years → GAP
   - Section 11.2: Vendor contracts specify 7-year retention → CONFLICT

   **Cascade Impacts (Hop 3):**
   - Section 9.3: Technology obsolescence risk (EDC may sunset in 10-12 years)
   - Section 10.1: Audit rights limited to 7 years, inconsistent with 25-year retention

4. System quantifies total impact:
   - Timeline: +8-12 weeks for vendor renegotiation
   - Cost: $1.8M - $2.3M (immediate + future)
   - Risk: 3 regulatory violations if not addressed
5. System recommends:
   - **CRITICAL:** Extend indemnification to 25 years
   - **CRITICAL:** Extend insurance to 25 years
   - **HIGH:** Renegotiate vendor contracts ($900K)
   - **MEDIUM:** Add technology migration plan

**Technical Flow:**
- POST `/api/v1/contract-stacks/{id}/analyze/ripple-effects`
- RippleEffectAnalyzerAgent queries Neo4j dependency graph
- Multi-hop traversal up to 5 levels deep
- Claude analyzes each hop for material impacts
- ActionRecommenderAgent prioritizes and synthesizes

**Success Criteria:**
- Detect 95%+ of material impacts
- Multi-hop reasoning up to 5 levels
- Quantified cost/timeline estimates
- Actionable prioritized recommendations

**Value:** $50K-150K rework prevention by finding issues before execution

---

### UC-009: Compare Amendment Scenarios
**Actor:** Budget Manager  
**Goal:** Evaluate multiple amendment options before deciding which to pursue

**Scenario:**
1. User creates three amendment scenarios:
   - **Scenario A:** Extend data retention to 25 years
   - **Scenario B:** Add cardiac MRI at Week 24
   - **Scenario C:** Remove Week 56/64 follow-up visits
2. User clicks "Compare Scenarios"
3. System analyzes each scenario and displays comparison:

   | Metric | Scenario A | Scenario B | Scenario C |
   |--------|-----------|-----------|-----------|
   | Cost Impact | +$1.8-2.3M | +$800/patient | -$1,400/patient |
   | Timeline | +8-12 weeks | +3 weeks | -2 weeks |
   | Sections Impacted | 11 | 4 | 6 |
   | Risk Level | High | Medium | Low |
   | Site Acceptance | 62% | 82% | 95% |

4. User selects Scenario C (cost reduction, high acceptance)
5. System generates draft amendment with ripple effects already analyzed

**Technical Flow:**
- Multiple calls to ripple-effects endpoint
- Comparison engine aggregates results
- Predictive intelligence estimates site acceptance (future feature)

**Success Criteria:**
- Side-by-side comparison of up to 5 scenarios
- Consistent metrics across scenarios
- Clear recommendation based on objectives

**Value:** Data-driven amendment decisions vs. gut feel

---

## CATEGORY 5: CONTRACT REUSABILITY

### UC-010: Assess Contract Reusability for New Study
**Actor:** Site Selection Manager  
**Goal:** Determine if existing site contract can be reused for new study

**Scenario:**
1. User selects existing contract: "HEARTBEAT-3 - Memorial Medical (Cardiology)"
2. User specifies new study: "ONC-402 - Phase II Oncology"
3. User clicks "Assess Reusability"
4. System performs deep comparison (15 seconds):

   **❌ BLOCKERS (Cannot Reuse):**
   - **PI Qualifications:** Current PI is cardiologist, requires oncologist
   - **Indemnification Scope:** Covers cardiovascular, not immunotherapy
   - **Budget Structure:** No line items for biopsies, CT/PET scans, immunophenotyping

   **⚠️ MAJOR MODIFICATIONS REQUIRED:**
   - **Insurance:** Current $1M/$3M, oncology requires $5M/$10M
   - **Data Retention:** 15 years may be insufficient (oncology often requires 25 years)

   **✅ REUSABLE CLAUSES:**
   - Payment Terms: Net 45 (reusable)
   - Confidentiality: Therapy-agnostic (reusable)
   - Publication Rights: Standard language (reusable)

5. System provides strategic recommendation:
   ```
   RECOMMENDATION: Start Fresh Contract
   
   Rationale:
   - 3 critical blockers require complete renegotiation
   - Modification approach: 8-10 weeks, 73% success rate
   - Fresh contract approach: 12-14 weeks, 95% success rate
   - Time savings of modification not worth increased risk
   
   Suggested Approach:
   - Use master oncology template as starting point
   - Reuse only administrative clauses (Articles 8, 10, 18-21)
   - Leverage existing site relationship for faster negotiation
   ```

**Technical Flow:**
- POST `/api/v1/contract-stacks/{id}/analyze/reusability`
- ReusabilityAnalyzerAgent compares requirements
- Cross-domain analysis (PI, budget, indemnification, insurance)
- Strategic recommendation based on historical success rates

**Success Criteria:**
- Identifies all critical blockers
- Provides accurate effort estimates
- Strategic recommendation with rationale

**Value:** Prevents 8-10 weeks wasted on incompatible contract modification

---

### UC-011: Find Best Template for New Site
**Actor:** Contracts Lead  
**Goal:** Identify the optimal contract template from portfolio for new site negotiation

**Scenario:**
1. User starting negotiation with new site: "University Hospital - Oncology Trial"
2. User clicks "Find Similar Contracts"
3. User specifies criteria:
   - Therapeutic area: Oncology
   - Site type: Academic medical center
   - Geography: US
4. System searches portfolio and recommends:

   **Best Match: Site 405 - Phase II Oncology**
   - Therapeutic area: ✅ Oncology (exact match)
   - Site type: ✅ Academic medical center
   - Negotiation cycle: 4.2 weeks (fastest in portfolio)
   - Acceptance rate: 94% (highest for oncology)
   - Key features:
     - Indemnification language accepted by 23/25 academic sites
     - Per-patient payment structure proven optimal for oncology
     - Insurance terms compliant with latest requirements

   **Alternative: Site 612 - Phase III Oncology**
   - 6.1 week cycle, 88% acceptance
   - More comprehensive but slower negotiation

5. System generates new contract using Site 405 as template
6. User reviews, makes site-specific adjustments, sends to site

**Technical Flow:**
- POST `/api/v1/portfolio/find-template`
- Semantic search across portfolio with filters
- Historical success analysis from queries table
- Template synthesis from best-performing clauses

**Success Criteria:**
- Relevant recommendations 90%+ of time
- Success metrics based on historical data
- One-click template generation

**Value:** Start negotiations with proven language vs. generic templates

---

## CATEGORY 6: PORTFOLIO INTELLIGENCE

### UC-012: Portfolio-Wide Payment Terms Analysis
**Actor:** Finance Director  
**Goal:** Standardize payment terms across entire site portfolio

**Scenario:**
1. User opens "Portfolio Analytics"
2. User runs query: "Show payment terms distribution across all contracts"
3. System analyzes 150 active contracts:

   **Payment Terms Distribution:**
   - Net 30: 42 sites (28%)
   - Net 45: 78 sites (52%)
   - Net 60: 24 sites (16%)
   - Net 90: 6 sites (4%)

   **Analysis:**
   - Average payment term: Net 47 days
   - Industry benchmark: Net 45 days ✅ Aligned
   - Outliers: 6 sites at Net 90 (investigate)
   - Trend: Shifting from Net 30 to Net 45 over past 2 years

4. User drills into Net 90 sites
5. System shows these are all European sites (regulatory requirement)
6. User exports report for CFO showing portfolio is optimized

**Technical Flow:**
- GET `/api/v1/portfolio/analytics/payment-terms`
- Aggregate query across all contract_stacks
- Statistical analysis and visualization
- Export to Excel/PDF

**Success Criteria:**
- Process 1000+ contracts in <30 seconds
- Accurate categorization
- Trend analysis over time

**Value:** Portfolio-level insights impossible to gather manually

---

### UC-013: Benchmark Against Industry
**Actor:** VP Clinical Operations  
**Goal:** Understand how our contracts compare to market

**Scenario:**
1. User views "Industry Benchmarking Dashboard"
2. System displays (anonymized cross-customer data):

   **Your Portfolio vs. Industry:**
   - **Per-Patient Payments (Phase III Cardiology)**
     - Your average: $19,200
     - Industry median: $21,500
     - Industry 75th percentile: $24,000
     - ⚠️ Alert: You may be underpaying by 11%

   - **Negotiation Cycle Time**
     - Your average: 4.8 weeks
     - Industry median: 6.8 weeks
     - ✅ You're 30% faster than average

   - **Payment Terms**
     - Your mode: Net 45
     - Industry mode: Net 45
     - ✅ Aligned with market

   - **Indemnification Language**
     - Your acceptance rate: 73%
     - Top performers: 91%
     - 💡 Recommendation: Review Site 405 language

3. User clicks on payment recommendation
4. System shows: "Consider increasing cardiology per-patient to $21,000 to improve enrollment competitiveness"
5. User shares with budget team for next protocol

**Technical Flow:**
- Industry intelligence requires opt-in data sharing
- Anonymized aggregation across customers
- Statistical analysis with privacy preservation
- Recommendation engine based on correlations

**Success Criteria:**
- Benchmarks based on 10,000+ contracts minimum
- Complete anonymization (no customer identification)
- Actionable recommendations

**Value:** Strategic intelligence for competitive positioning

---

### UC-014: Track Clause Performance Over Time
**Actor:** Legal Operations Manager  
**Goal:** Identify which contract clauses lead to fastest execution

**Scenario:**
1. User asks: "Which indemnification language gets accepted fastest?"
2. System analyzes 200 contracts with different indemnification variants:

   **Indemnification Performance:**
   - **Variant A** (Sponsor broad indemnification):
     - Used in: 80 contracts
     - Acceptance rate: 62%
     - Average negotiation: 8.2 weeks
     - Common pushback: Scope too broad

   - **Variant B** (Balanced with carve-outs):
     - Used in: 95 contracts
     - Acceptance rate: 91%
     - Average negotiation: 4.1 weeks ⚡ **FASTEST**
     - Rare pushback: Well-balanced

   - **Variant C** (Site-favorable):
     - Used in: 25 contracts
     - Acceptance rate: 96%
     - Average negotiation: 5.3 weeks
     - Note: Higher cost due to broader coverage

3. System recommends: "Adopt Variant B as standard - optimal balance of acceptance rate and timeline"
4. User updates master template with Variant B language
5. System tracks improvement over next 6 months

**Technical Flow:**
- Machine learning on historical negotiation outcomes
- Clause variant identification using embeddings
- Correlation analysis: clause → outcome
- Continuous learning from new negotiations

**Success Criteria:**
- Statistical significance (minimum 30 samples per variant)
- Control for confounding factors (site type, geography)
- Measurable improvement after template updates

**Value:** Evidence-based contract optimization vs. subjective preferences

---

## CATEGORY 7: COLLABORATION & WORKFLOW

### UC-015: Share Analysis with Team
**Actor:** Contract Manager  
**Goal:** Collaborate with legal and finance teams on contract review

**Scenario:**
1. User completes ripple effect analysis on proposed amendment
2. User clicks "Share Analysis"
3. User selects recipients: Legal team, Finance team, Study Manager
4. System generates shareable report with:
   - Executive summary
   - Detailed impacts by category
   - Recommendations with priority
   - Source documents attached
5. Recipients receive email with link to interactive report
6. Legal counsel adds comments: "Insurance gap is critical - must address"
7. Finance approves budget impact: "Approved for $2.1M"
8. Study Manager acknowledges timeline: "8-week delay acceptable"
9. All feedback consolidated in system
10. User proceeds with amendment knowing all stakeholders aligned

**Technical Flow:**
- POST `/api/v1/queries/{id}/share`
- Generate read-only shareable link
- Email notifications
- Comment system (future: stored in queries table)

**Success Criteria:**
- Shareable links with access control
- Email notifications
- Comments tracked and attributed

**Value:** Async collaboration vs. email chains and meetings

---

### UC-016: Export for Regulatory Audit
**Actor:** Quality Assurance Manager  
**Goal:** Prepare contract documentation for FDA audit

**Scenario:**
1. FDA announces inspection, requests contract documentation
2. User opens contract stack for site under audit
3. User clicks "Generate Audit Package"
4. System creates comprehensive package:
   - All source documents (PDFs)
   - Complete clause history with provenance
   - Amendment tracking documentation
   - Conflict resolution records
   - Query history showing due diligence
5. System generates audit report showing:
   - When each document was executed
   - What changed in each amendment
   - How conflicts were identified and resolved
   - Compliance verification for all terms
6. User exports as ZIP file with organized folders
7. Package submitted to FDA within 2 hours of request

**Technical Flow:**
- POST `/api/v1/contract-stacks/{id}/export/audit-package`
- Aggregates all data from database
- Generates compliance report
- Creates structured ZIP file
- Includes PDF rendering of all analysis

**Success Criteria:**
- Complete documentation in <5 minutes
- FDA-compliant formatting
- Full provenance chain
- No manual document gathering required

**Value:** 2 hours vs. 2-3 days manual preparation

---

## CATEGORY 8: PREDICTIVE INTELLIGENCE (Future Enhancements)

### UC-017: Predict Negotiation Outcome
**Actor:** Site Engagement Manager  
**Goal:** Understand likelihood of site accepting proposed terms

**Scenario:**
1. User proposes amendment to Site 301: "Add cardiac MRI, increase per-patient by $1,200"
2. User clicks "Predict Outcome"
3. System analyzes historical data:
   - Site 301 history: 5 prior amendments
   - Similar amendments at academic sites: 127 cases
   - Imaging capability at Site 301: High (MRI center on campus)
4. System predicts:
   ```
   Acceptance Probability: 82%
   
   Confidence: High (based on 127 similar cases)
   
   Predicted Timeline: 2.8 weeks
   
   Expected Counter-Proposal: $1,500 per patient (64% probability)
   
   Recommendation:
   - Propose $1,200 as starting point
   - Be prepared to accept up to $1,400
   - Highlight site's imaging center capacity in proposal
   - Reference similar pricing at comparable sites
   
   If proposal is rejected:
   - Most likely objection: Budget constraints (73%)
   - Alternative: Partner imaging with nearby facility (-$400/patient)
   ```

**Technical Flow:**
- POST `/api/v1/predictions/negotiation-outcome`
- Machine learning model trained on historical amendments
- Feature engineering: site characteristics, amendment type, therapeutic area
- Probabilistic prediction with confidence intervals

**Success Criteria:**
- Prediction accuracy within ±15% of actual outcome
- Continuous learning from new negotiations
- Transparent confidence scoring

**Value:** Strategic negotiation advantage vs. blind proposals

---

### UC-018: Recommend Optimal Amendment Timing
**Actor:** Portfolio Manager  
**Goal:** Decide when to issue amendments across multiple sites

**Scenario:**
1. User needs to issue protocol amendment to 50 sites
2. User clicks "Optimize Execution Plan"
3. System analyzes:
   - Current workload at each site
   - Historical response times
   - Upcoming site events (IRB meetings, enrollment milestones)
   - Dependencies between sites
4. System recommends phased approach:
   ```
   Phase 1 (Weeks 1-2): Fast sites (12 sites)
   - Sites 301, 405, 512... (avg 3.2 week response)
   - Start immediately while protocol is finalized
   
   Phase 2 (Weeks 3-4): Medium sites (23 sites)
   - Sites 203, 308, 421... (avg 5.1 week response)
   - Start after Phase 1 feedback incorporated
   
   Phase 3 (Weeks 5-6): Slow sites (15 sites)
   - Sites 107, 234, 567... (avg 8.3 week response)
   - Start last, allow maximum time
   
   Predicted Completion: 8.7 weeks (vs. 12.1 weeks if simultaneous)
   ```

**Technical Flow:**
- POST `/api/v1/portfolio/optimize-execution`
- Operations research optimization
- Constraint satisfaction (resource limits, dependencies)
- Monte Carlo simulation for timeline prediction

**Success Criteria:**
- 20-30% faster portfolio-wide execution
- Resource leveling across teams
- Risk mitigation through phasing

**Value:** Strategic portfolio management vs. ad-hoc execution

---

## SUMMARY TABLE: ALL USE CASES

| ID | Use Case | Actor | Time Saved | Key Value |
|----|----------|-------|------------|-----------|
| UC-001 | Upload & Process | Ops Manager | 2 hrs | Automated digitization |
| UC-002 | View Timeline | Contract Manager | 30 min | Visual evolution |
| UC-003 | Query Terms | Financial Analyst | 3 days | Instant truth |
| UC-004 | Clause Evolution | Legal Counsel | 4 hrs | Archaeology automation |
| UC-005 | Multi-Clause Query | Study Manager | 2 hrs | Comprehensive synthesis |
| UC-006 | Detect Conflicts | Compliance Officer | 2 days | Proactive risk detection |
| UC-007 | Monitor Health | Contract Admin | Daily | Prevents emergencies |
| UC-008 | Ripple Analysis | Ops Director | 2 weeks | $50K-150K rework prevention |
| UC-009 | Compare Scenarios | Budget Manager | 1 week | Data-driven decisions |
| UC-010 | Assess Reusability | Site Selection | 1 day | Prevents 8-week mistake |
| UC-011 | Find Template | Contracts Lead | 3 days | Proven language |
| UC-012 | Portfolio Analytics | Finance Director | 1 week | Strategic insights |
| UC-013 | Industry Benchmark | VP Ops | N/A | Competitive intelligence |
| UC-014 | Clause Performance | Legal Ops | N/A | Evidence-based optimization |
| UC-015 | Share Analysis | Contract Manager | 2 days | Async collaboration |
| UC-016 | Audit Export | QA Manager | 3 days | Compliance automation |
| UC-017 | Predict Outcome | Site Engagement | N/A | Negotiation advantage |
| UC-018 | Optimize Timing | Portfolio Manager | 30% | Resource optimization |

## TECHNICAL IMPLEMENTATION MAPPING

Each use case maps to specific technical components:

| Use Case Category | Primary Agents | Database Tables | API Endpoints |
|-------------------|----------------|-----------------|---------------|
| Document Management | Parser, TemporalSequencer | documents, contract_stacks | /contract-stacks, /documents |
| Truth Reconstitution | OverrideResolver, TruthConsolidator | clauses, amendments | /query |
| Conflict Detection | ConflictDetector, RiskScorer | conflicts, clauses | /analyze/conflicts |
| Ripple Analysis | RippleAnalyzer, DependencyMapper | Neo4j graph | /analyze/ripple-effects |
| Reusability | ReusabilityAnalyzer | clauses, contract_stacks | /analyze/reusability |
| Portfolio Intelligence | Portfolio analyzers | All tables | /portfolio/* |
| Collaboration | N/A | queries, users | /share, /export |
| Predictive | PredictionAgent (future) | queries, outcomes | /predictions/* |

---

## PRIORITIZATION FOR MVP

**Phase 1 (Weeks 1-8) - Core Demo:**
- ✅ UC-001: Upload & Process
- ✅ UC-003: Query Terms
- ✅ UC-006: Detect Conflicts
- ✅ UC-008: Ripple Analysis

**Phase 2 (Weeks 9-16) - Pilot:**
- ✅ UC-002: Timeline View
- ✅ UC-004: Clause Evolution
- ✅ UC-007: Monitor Health
- ✅ UC-010: Assess Reusability
- ✅ UC-015: Share Analysis

**Phase 3 (Months 5-8) - Production:**
- ✅ UC-012: Portfolio Analytics
- ✅ UC-016: Audit Export
- ✅ UC-009: Compare Scenarios
- ✅ UC-011: Find Template

**Phase 4 (Year 2) - Advanced:**
- ✅ UC-013: Industry Benchmark
- ✅ UC-014: Clause Performance
- ✅ UC-017: Predict Outcome
- ✅ UC-018: Optimize Timing

---

**This use case document provides clear scenarios that map directly to the technical implementation and demonstrate tangible user value at every level.**