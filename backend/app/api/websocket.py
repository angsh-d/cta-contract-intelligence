"""WebSocket endpoint for real-time pipeline progress (asyncio.Queue, no Redis)."""

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()

# Global registry: job_id → list[asyncio.Queue]
_job_queues: dict[str, list] = {}


def get_progress_queue(job_id: str):
    """Get or create the queue list for a job. Returns the list of queues."""
    if job_id not in _job_queues:
        _job_queues[job_id] = []
    return _job_queues[job_id]


async def broadcast_progress(job_id: str, event_data: dict) -> None:
    """Push a progress event to all connected WebSocket clients for a job."""
    queues = _job_queues.get(job_id, [])
    for q in queues:
        await q.put(event_data)


def cleanup_job(job_id: str) -> None:
    """Remove all queues for a completed/failed job."""
    _job_queues.pop(job_id, None)


@router.websocket("/ws/jobs/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str):
    """Stream real-time pipeline progress to a WebSocket client."""
    import asyncio

    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()

    # Register this client's queue
    queues = get_progress_queue(job_id)
    queues.append(queue)

    try:
        while True:
            event = await queue.get()
            await websocket.send_text(json.dumps(event))

            # Close on completion or failure
            if event.get("pipeline_stage") in ("complete", "failed"):
                await websocket.close()
                return
    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected for job %s", job_id)
    except Exception:
        logger.exception("WebSocket error for job %s", job_id)
    finally:
        if queue in queues:
            queues.remove(queue)
