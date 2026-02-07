"""Progress and event models for WebSocket / pipeline updates."""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class AgentProgressEvent(BaseModel):
    """Progress update from a single agent."""
    job_id: str
    agent_name: str
    stage: str
    percent_complete: int
    message: str
    timestamp: datetime


class PipelineProgressEvent(BaseModel):
    """Overall pipeline progress event."""
    job_id: str
    pipeline_stage: str
    overall_percent: int
    message: str
    current_agent: Optional[str] = None
    timestamp: datetime


class PipelineCompleteEvent(BaseModel):
    """Emitted when full pipeline finishes."""
    job_id: str
    success: bool
    total_documents: int
    total_clauses: int
    total_conflicts: int
    total_dependencies: int
    total_duration_ms: int
    timestamp: datetime


class PipelineErrorEvent(BaseModel):
    """Emitted when pipeline encounters a fatal error."""
    job_id: str
    pipeline_stage: str = "failed"
    error_type: str
    error_message: str
    agent_name: Optional[str] = None
    stage: Optional[str] = None
    timestamp: datetime
