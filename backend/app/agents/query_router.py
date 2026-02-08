"""Query Pipeline: QueryRouter — classify and decompose user queries."""

import logging
import time

from app.agents.base import BaseAgent
from app.models.agent_schemas import (
    QueryRouteInput, QueryRouteOutput, SubQuery,
)
from app.models.enums import QueryType

logger = logging.getLogger(__name__)


class QueryRouter(BaseAgent):
    """Classify user queries into types and extract entities."""

    async def process(self, input_data: QueryRouteInput) -> QueryRouteOutput:
        start = time.monotonic()

        system_prompt = self.prompts.get("query_router_classify")
        user_prompt = self.prompts.get(
            "query_router_input",
            query_text=input_data.query_text,
        )
        result = await self.call_llm(system_prompt, user_prompt)

        raw_sqs = result.get("sub_queries", [])
        sub_queries: list[SubQuery] = []
        for sq in raw_sqs:
            if isinstance(sq, dict):
                qt = sq.get("query_text") or sq.get("text") or sq.get("query") or ""
                if not qt:
                    continue
                qtype = sq.get("query_type", "truth_reconstitution")
                ents = sq.get("entities", [])
                try:
                    sub_queries.append(SubQuery(query_type=qtype, query_text=qt, entities=ents))
                except Exception:
                    logger.warning("Skipping malformed sub_query: %s", sq)
                    continue

        query_type_raw = result.get("query_type", "truth_reconstitution")
        try:
            query_type = QueryType(query_type_raw)
        except ValueError:
            logger.warning("Unknown query_type '%s' — defaulting to truth_reconstitution", query_type_raw)
            query_type = QueryType.TRUTH_RECONSTITUTION

        return QueryRouteOutput(
            query_type=query_type,
            extracted_entities=result.get("entities", []),
            confidence=result.get("confidence", 0.9),
            routing_latency_ms=int((time.monotonic() - start) * 1000),
            llm_reasoning=result.get("reasoning", ""),
            sub_queries=sub_queries,
        )
