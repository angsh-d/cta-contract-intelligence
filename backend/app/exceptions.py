"""ContractIQ exception hierarchy."""


class ContractIQError(Exception):
    """Base exception for all ContractIQ errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class DocumentExtractionError(ContractIQError):
    """PDF/DOCX could not be opened, parsed, or structured."""
    pass


class LLMResponseError(ContractIQError):
    """LLM returned invalid/unparseable content after all retries."""
    pass


class LLMProviderError(ContractIQError):
    """LLM provider API error (auth, rate limit, timeout, server error)."""

    def __init__(self, message: str, provider: str, status_code: int | None = None, **kw):
        super().__init__(message, **kw)
        self.provider = provider
        self.status_code = status_code


class AgentProcessingError(ContractIQError):
    """An agent's process() method failed."""

    def __init__(self, message: str, agent_name: str, **kw):
        super().__init__(message, **kw)
        self.agent_name = agent_name


class DatabaseError(ContractIQError):
    """PostgreSQL (NeonDB) or Redis operation failed."""

    def __init__(self, message: str, database: str, **kw):
        super().__init__(message, **kw)
        self.database = database


class PipelineError(ContractIQError):
    """Pipeline-level failure (orchestrator, task queue)."""

    def __init__(self, message: str, stage: str, **kw):
        super().__init__(message, **kw)
        self.stage = stage


class PromptTemplateError(ContractIQError):
    """Prompt file missing or variable placeholder cannot be resolved."""
    pass
