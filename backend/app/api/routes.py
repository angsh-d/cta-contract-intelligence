"""REST API routes for ContractIQ — /api/v1/ prefix."""

import asyncio
import hashlib
import json
import logging
import os
import random
import shutil
import time
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api.websocket import broadcast_progress, cleanup_job
from app.models.enums import ConflictSeverity, DocumentType
from app.models.events import PipelineErrorEvent, PipelineProgressEvent

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
MAX_FILE_SIZE_MB = 50

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

UPLOAD_DIR = Path("./uploads")


# ── Request / Response Schemas ────────────────────────────────

class CreateContractStackRequest(BaseModel):
    name: str
    sponsor_name: str
    site_name: str
    study_protocol: Optional[str] = None
    therapeutic_area: Optional[str] = None


class ContractStackResponse(BaseModel):
    id: str
    name: str
    status: str
    created_at: str


class QueryRequest(BaseModel):
    query: str
    query_type: Optional[str] = None
    include_reasoning: bool = False


class ConflictAnalysisRequest(BaseModel):
    severity_threshold: str = "medium"


class RippleAnalysisRequest(BaseModel):
    proposed_change: dict


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# In-memory job tracking (no Redis)
_jobs: dict[str, dict] = {}


def _parse_uuid(value: str, label: str = "ID") -> uuid.UUID:
    """Parse a UUID string, raising 400 on invalid format."""
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"Invalid {label}: {value}")


def _require_orchestrator(request: Request):
    """Return orchestrator or raise 503 if AI agents are unavailable."""
    orchestrator = request.app.state.orchestrator
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="AI agents unavailable — API keys not configured")
    return orchestrator


# ── Contract Stack Endpoints ──────────────────────────────────

@router.post("/contract-stacks", response_model=ContractStackResponse)
async def create_contract_stack(req: CreateContractStackRequest, request: Request):
    pool = request.app.state.postgres_pool
    stack_id = uuid.uuid4()
    now = datetime.utcnow()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO contract_stacks (id, study_name, sponsor_name, site_name, "
            "study_protocol, therapeutic_area, status, created_at) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
            stack_id, req.name, req.sponsor_name, req.site_name,
            req.study_protocol, req.therapeutic_area, "created", now,
        )
    return ContractStackResponse(
        id=str(stack_id), name=req.name, status="created", created_at=now.isoformat(),
    )


@router.post("/contract-stacks/{stack_id}/documents")
async def upload_document(
    stack_id: str,
    request: Request,
    file: UploadFile = File(...),
    document_type: str = Form(...),
    effective_date: Optional[str] = Form(None),
    document_version: Optional[str] = Form(None),
):
    pool = request.app.state.postgres_pool
    stack_uuid = _parse_uuid(stack_id, "stack_id")

    # Verify stack exists
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id FROM contract_stacks WHERE id = $1", stack_uuid)
        if not row:
            raise HTTPException(status_code=404, detail="Contract stack not found")

    # File type validation
    ext = Path(file.filename or "doc.pdf").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    # Save uploaded file
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    doc_id = uuid.uuid4()
    file_path = UPLOAD_DIR / f"{doc_id}{ext}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # File size validation (post-write check)
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"File too large ({file_size_mb:.1f}MB). Max: {MAX_FILE_SIZE_MB}MB")

    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO documents (id, contract_stack_id, document_type, filename, "
            "file_path, effective_date, document_version, processed, created_at) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, FALSE, $8)",
            doc_id, stack_uuid, document_type, file.filename, str(file_path),
            date.fromisoformat(effective_date) if effective_date else None,
            document_version, datetime.utcnow(),
        )

    return {
        "id": str(doc_id),
        "filename": file.filename,
        "status": "uploaded",
        "document_type": document_type,
    }


@router.get("/contract-stacks/{stack_id}/documents")
async def list_documents(stack_id: str, request: Request):
    pool = request.app.state.postgres_pool
    stack_uuid = _parse_uuid(stack_id, "stack_id")
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, document_type, filename, effective_date, processed "
            "FROM documents WHERE contract_stack_id = $1 ORDER BY created_at",
            stack_uuid,
        )
    return {
        "contract_stack_id": stack_id,
        "documents": [
            {
                "id": str(r["id"]),
                "document_type": r["document_type"],
                "filename": r["filename"],
                "effective_date": r["effective_date"],
                "processed": r["processed"],
            }
            for r in rows
        ],
    }


# ── Processing Endpoint ──────────────────────────────────────

@router.post("/contract-stacks/{stack_id}/process")
async def process_contract_stack(stack_id: str, request: Request):
    pool = request.app.state.postgres_pool
    orchestrator = _require_orchestrator(request)
    stack_uuid = _parse_uuid(stack_id, "stack_id")

    # Verify stack exists and has documents
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id FROM contract_stacks WHERE id = $1", stack_uuid)
        if not row:
            raise HTTPException(status_code=404, detail="Contract stack not found")
        doc_count = await conn.fetchval(
            "SELECT COUNT(*) FROM documents WHERE contract_stack_id = $1", stack_uuid,
        )
        if doc_count == 0:
            raise HTTPException(status_code=400, detail="No documents uploaded to this contract stack")

    job_id = str(uuid.uuid4())

    _jobs[job_id] = {
        "status": "processing",
        "progress": 0,
        "message": "Starting pipeline...",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }

    async def progress_callback(event: PipelineProgressEvent) -> None:
        _jobs[job_id].update({
            "status": "processing",
            "progress": event.overall_percent,
            "message": event.message,
            "updated_at": datetime.utcnow().isoformat(),
        })
        await broadcast_progress(job_id, event.model_dump(mode="json"))

    async def run_pipeline():
        try:
            result = await orchestrator.process_contract_stack(
                stack_uuid, job_id, progress_callback,
            )
            _jobs[job_id].update({
                "status": "completed", "progress": 100,
                "message": "Pipeline complete", "result": result,
                "updated_at": datetime.utcnow().isoformat(),
            })
        except Exception as exc:
            logger.exception("Pipeline failed for stack %s", stack_id)
            _jobs[job_id].update({
                "status": "failed", "message": str(exc),
                "updated_at": datetime.utcnow().isoformat(),
            })
            error_event = PipelineErrorEvent(
                job_id=job_id, error_type=type(exc).__name__,
                error_message=str(exc), timestamp=datetime.utcnow(),
            )
            await broadcast_progress(job_id, error_event.model_dump(mode="json"))
        finally:
            cleanup_job(job_id)

    task = asyncio.create_task(run_pipeline())
    _jobs[job_id]["_task"] = task  # Store reference to prevent GC

    return {"job_id": job_id, "status": "processing"}


# ── Job Status ────────────────────────────────────────────────

@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0),
        message=job.get("message", ""),
        created_at=job.get("created_at"),
        updated_at=job.get("updated_at"),
    )


# ── Query Endpoint ────────────────────────────────────────────

@router.post("/contract-stacks/{stack_id}/query")
async def query_contract_stack(stack_id: str, req: QueryRequest, request: Request):
    orchestrator = _require_orchestrator(request)
    stack_uuid = _parse_uuid(stack_id, "stack_id")
    start = time.monotonic()

    result = await orchestrator.handle_query(req.query, stack_uuid)

    return {
        "query_id": str(uuid.uuid4()),
        "query": req.query,
        "response": {
            "answer": result.answer,
            "sources": [s.model_dump(mode="json") for s in result.sources],
            "confidence": result.confidence,
            "caveats": result.caveats,
        },
        "execution_time_ms": int((time.monotonic() - start) * 1000),
    }


# ── Conflict Analysis ────────────────────────────────────────

@router.post("/contract-stacks/{stack_id}/analyze/conflicts")
async def analyze_conflicts(stack_id: str, req: ConflictAnalysisRequest, request: Request):
    pool = request.app.state.postgres_pool
    stack_uuid = _parse_uuid(stack_id, "stack_id")

    severity_order = {s.value: i for i, s in enumerate(ConflictSeverity)}
    threshold_idx = severity_order.get(req.severity_threshold, 2)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT conflict_id, conflict_type, severity, description, "
            "affected_sections, evidence, recommendation, pain_point_id "
            "FROM conflicts WHERE contract_stack_id = $1",
            stack_uuid,
        )

    conflicts = []
    summary: dict[str, int] = {}
    for r in rows:
        sev = r["severity"]
        summary[sev] = summary.get(sev, 0) + 1
        if severity_order.get(sev, 99) <= threshold_idx:
            raw_evidence = r["evidence"]
            evidence = json.loads(raw_evidence) if isinstance(raw_evidence, str) else (raw_evidence or [])
            conflicts.append({
                "id": r["conflict_id"],
                "conflict_type": r["conflict_type"],
                "severity": sev,
                "description": r["description"],
                "affected_clauses": r["affected_sections"] or [],
                "evidence": evidence,
                "recommendation": r["recommendation"],
                "pain_point_id": r["pain_point_id"],
            })

    return {"conflicts": conflicts, "summary": summary}


# ── Ripple Effect Analysis ────────────────────────────────────

@router.post("/contract-stacks/{stack_id}/analyze/ripple-effects")
async def analyze_ripple_effects(stack_id: str, req: RippleAnalysisRequest, request: Request):
    from app.models.agent_schemas import ProposedChange, RippleEffectInput

    orchestrator = _require_orchestrator(request)
    stack_uuid = _parse_uuid(stack_id, "stack_id")

    pc = req.proposed_change
    proposed = ProposedChange(
        section_number=pc.get("section_number") or pc.get("clause_section", ""),
        current_text=pc.get("current_text", ""),
        proposed_text=pc.get("proposed_text", ""),
        change_description=pc.get("change_description") or pc.get("description"),
    )

    # Check cache first
    cache_input = f"{stack_id}:{proposed.section_number}:{proposed.current_text}:{proposed.proposed_text}"
    cache_key = f"ripple:{hashlib.sha256(cache_input.encode()).hexdigest()}"
    cached = await orchestrator.cache.get(cache_key)
    if cached:
        logger.info("Ripple effect CACHE HIT for %s §%s", stack_id, proposed.section_number)
        await asyncio.sleep(random.uniform(2, 3))  # Simulate real-time retrieval
        return json.loads(cached)

    logger.info("Ripple effect CACHE MISS for %s §%s (key=%s)", stack_id, proposed.section_number, cache_key)
    ripple_agent = orchestrator.get_agent("ripple_effect")
    result = await ripple_agent.run(RippleEffectInput(
        contract_stack_id=stack_uuid, proposed_change=proposed,
    ))

    result_json = result.model_dump(mode="json")
    result_json.pop("llm_reasoning", None)  # Internal agent reasoning, not for display
    await orchestrator.cache.setex(cache_key, 86400 * 30, json.dumps(result_json))
    await orchestrator.cache.sadd(f"cache_keys:{stack_id}", cache_key)

    return result_json


# ── List / Get Contract Stacks ────────────────────────────────

@router.get("/contract-stacks")
async def list_contract_stacks(request: Request):
    """List all contract stacks."""
    pool = request.app.state.postgres_pool
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, study_name, sponsor_name, site_name, therapeutic_area, "
            "processing_status, created_at FROM contract_stacks ORDER BY created_at DESC",
        )
    return {
        "stacks": [
            {
                "id": str(r["id"]),
                "name": r["study_name"],
                "sponsor_name": r["sponsor_name"],
                "site_name": r["site_name"],
                "therapeutic_area": r["therapeutic_area"],
                "processing_status": r["processing_status"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ],
    }


@router.get("/contract-stacks/{stack_id}")
async def get_contract_stack(stack_id: str, request: Request):
    """Get detailed contract stack info with document and clause counts."""
    pool = request.app.state.postgres_pool
    stack_uuid = _parse_uuid(stack_id, "stack_id")
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, study_name, sponsor_name, site_name, therapeutic_area, "
            "study_protocol, processing_status, created_at FROM contract_stacks WHERE id = $1",
            stack_uuid,
        )
        if not row:
            raise HTTPException(status_code=404, detail="Contract stack not found")
        doc_count = await conn.fetchval(
            "SELECT COUNT(*) FROM documents WHERE contract_stack_id = $1", stack_uuid,
        )
        clause_count = await conn.fetchval(
            "SELECT COUNT(*) FROM clauses WHERE contract_stack_id = $1 AND is_current = TRUE", stack_uuid,
        )
        conflict_count = await conn.fetchval(
            "SELECT COUNT(*) FROM conflicts WHERE contract_stack_id = $1", stack_uuid,
        )
    return {
        "id": str(row["id"]),
        "name": row["study_name"],
        "sponsor_name": row["sponsor_name"],
        "site_name": row["site_name"],
        "therapeutic_area": row["therapeutic_area"],
        "study_protocol": row["study_protocol"],
        "processing_status": row["processing_status"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "counts": {
            "documents": doc_count,
            "clauses": clause_count,
            "conflicts": conflict_count,
        },
    }


# ── Timeline (UC-002) ─────────────────────────────────────────

@router.get("/contract-stacks/{stack_id}/timeline")
async def get_timeline(stack_id: str, request: Request):
    """Get the temporal timeline for a contract stack — UC-002."""
    pool = request.app.state.postgres_pool
    stack_uuid = _parse_uuid(stack_id, "stack_id")
    async with pool.acquire() as conn:
        # Get document timeline
        docs = await conn.fetch(
            "SELECT d.id, d.document_type, d.filename, d.effective_date, d.document_version "
            "FROM documents d WHERE d.contract_stack_id = $1 ORDER BY d.effective_date NULLS LAST",
            stack_uuid,
        )
        # Get supersession relationships
        supersessions = await conn.fetch(
            "SELECT predecessor_document_id, successor_document_id "
            "FROM document_supersessions WHERE contract_stack_id = $1",
            stack_uuid,
        )
    return {
        "contract_stack_id": stack_id,
        "timeline": [
            {
                "document_id": str(d["id"]),
                "document_type": d["document_type"],
                "filename": d["filename"],
                "effective_date": str(d["effective_date"]) if d["effective_date"] else None,
                "document_version": d["document_version"],
            }
            for d in docs
        ],
        "supersessions": [
            {
                "predecessor": str(s["predecessor_document_id"]),
                "successor": str(s["successor_document_id"]),
            }
            for s in supersessions
        ],
    }


# ── Clause History (UC-004) ───────────────────────────────────

@router.get("/contract-stacks/{stack_id}/clauses/{section_number}/history")
async def get_clause_history(stack_id: str, section_number: str, request: Request):
    """Get the full amendment history (source chain) for a specific clause — UC-004."""
    pool = request.app.state.postgres_pool
    stack_uuid = _parse_uuid(stack_id, "stack_id")
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT section_number, section_title, current_text, clause_category, "
            "source_chain, effective_date, source_document_id "
            "FROM clauses WHERE section_number = $1 AND contract_stack_id = $2 AND is_current = TRUE",
            section_number, stack_uuid,
        )
        if not row:
            raise HTTPException(status_code=404, detail=f"Clause '{section_number}' not found")
        # Get dependencies
        deps = await conn.fetch(
            """
            SELECT cd.relationship_type, cd.description, cd.confidence,
                   c2.section_number AS related_section
            FROM clause_dependencies cd
            JOIN clauses c1 ON c1.id = cd.from_clause_id
            JOIN clauses c2 ON c2.id = cd.to_clause_id
            WHERE c1.section_number = $1 AND cd.contract_stack_id = $2
            """,
            section_number, str(stack_uuid),
        )
    source_chain = json.loads(row["source_chain"]) if row["source_chain"] else []
    return {
        "section_number": row["section_number"],
        "section_title": row["section_title"],
        "current_text": row["current_text"],
        "clause_category": row["clause_category"],
        "effective_date": str(row["effective_date"]) if row["effective_date"] else None,
        "source_chain": source_chain,
        "dependencies": [
            {
                "related_section": d["related_section"],
                "relationship_type": d["relationship_type"],
                "description": d["description"],
                "confidence": float(d["confidence"]) if d["confidence"] else None,
            }
            for d in deps
        ],
    }


# ── Document Detail — Clauses & PDF ──────────────────────────

@router.get("/contract-stacks/{stack_id}/documents/{document_id}/clauses")
async def get_document_clauses(stack_id: str, document_id: str, request: Request):
    """Get all extracted clauses/sections for a specific document.

    Uses section_embeddings (Stage 1 raw extractions) as the primary source
    so that every section parsed from this document is shown — even if a later
    amendment overrode it.  LEFT JOINs with the clauses table to pull in
    clause_category and whether this document's version is still the current one.
    Falls back to the clauses table if no embeddings exist for this document.
    """
    pool = request.app.state.postgres_pool
    stack_uuid = _parse_uuid(stack_id, "stack_id")
    doc_uuid = _parse_uuid(document_id, "document_id")
    async with pool.acquire() as conn:
        doc = await conn.fetchrow(
            "SELECT id, filename, document_type FROM documents "
            "WHERE id = $1 AND contract_stack_id = $2",
            doc_uuid, stack_uuid,
        )
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Primary: Stage 1 raw sections for this document, enriched with clause metadata
        rows = await conn.fetch(
            "SELECT se.section_number, se.section_title, se.section_text, "
            "se.effective_date, c.clause_category, "
            "(c.source_document_id = $1) AS is_current, "
            "c.source_chain "
            "FROM section_embeddings se "
            "LEFT JOIN clauses c ON c.contract_stack_id = se.contract_stack_id "
            "  AND c.section_number = se.section_number AND c.is_current = TRUE "
            "WHERE se.document_id = $1 AND se.contract_stack_id = $2 "
            "  AND se.is_resolved = FALSE "
            "ORDER BY se.section_number",
            doc_uuid, stack_uuid,
        )

        # Fallback: if no embeddings, use clauses table directly
        if not rows:
            rows = await conn.fetch(
                "SELECT section_number, section_title, current_text AS section_text, "
                "clause_category, effective_date, is_current, source_chain "
                "FROM clauses WHERE source_document_id = $1 AND contract_stack_id = $2 "
                "ORDER BY section_number",
                doc_uuid, stack_uuid,
            )

        # Fetch conflicts for this stack and index by section_number
        conflict_rows = await conn.fetch(
            "SELECT conflict_id, conflict_type, severity, description, "
            "affected_sections, recommendation, pain_point_id "
            "FROM conflicts WHERE contract_stack_id = $1",
            stack_uuid,
        )
        conflicts_by_section: dict[str, list[dict]] = {}
        for cr in conflict_rows:
            conflict_obj = {
                "conflict_id": str(cr["conflict_id"]),
                "conflict_type": cr["conflict_type"],
                "severity": cr["severity"],
                "description": cr["description"],
                "recommendation": cr["recommendation"],
                "pain_point_id": cr["pain_point_id"],
            }
            sections = cr["affected_sections"] or []
            for sec in sections:
                conflicts_by_section.setdefault(sec, []).append(conflict_obj)

    return {
        "document_id": document_id,
        "filename": doc["filename"],
        "document_type": doc["document_type"],
        "clauses": [
            {
                "section_number": r["section_number"],
                "section_title": r["section_title"] or "",
                "current_text": r["section_text"],
                "clause_category": r["clause_category"] or "general",
                "is_current": bool(r["is_current"]),
                "effective_date": str(r["effective_date"]) if r["effective_date"] else None,
                "source_chain": json.loads(r["source_chain"]) if isinstance(r["source_chain"], str) else (r["source_chain"] or []),
                "conflicts": conflicts_by_section.get(r["section_number"], []),
            }
            for r in rows
        ],
    }


@router.get("/contract-stacks/{stack_id}/documents/{document_id}/pdf")
async def get_document_pdf(stack_id: str, document_id: str, request: Request):
    """Serve the original PDF file for a document."""
    pool = request.app.state.postgres_pool
    stack_uuid = _parse_uuid(stack_id, "stack_id")
    doc_uuid = _parse_uuid(document_id, "document_id")
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT file_path, filename FROM documents "
            "WHERE id = $1 AND contract_stack_id = $2",
            doc_uuid, stack_uuid,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    file_path = Path(row["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found on disk")
    return FileResponse(
        path=str(file_path),
        media_type="application/pdf",
        filename=row["filename"],
    )
