import json

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from backend.app.core.logging import get_logger
from backend.app.rag.chain import answer_stream
from backend.app.rag.schemas import ChatRequest

log = get_logger("api.chat")

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat(request: ChatRequest) -> EventSourceResponse:
    """Stream the answer as server-sent events.

    Retrieval metadata is sent first so the UI can render citation cards and the
    retrieval trace while the model is still writing.
    """

    async def event_source():
        try:
            async for event in answer_stream(
                request.question,
                history=request.history,
                mode=request.mode,
                explain=request.explain,
            ):
                yield {"event": event["type"], "data": json.dumps(event)}
        except Exception as exc:
            log.exception("chat stream failed")
            yield {"event": "error", "data": json.dumps({"type": "error", "message": str(exc)})}

    return EventSourceResponse(event_source())
