"""Agent entrypoint for one-episode elevator-dispatch tool-call rollouts.

The policy returns a full dispatch episode as a single ``dispatch`` tool call.
The agent itself stays a thin model caller -- exactly like the Tic-Tac-Toe and
2048 examples -- and AReno replays the actions deterministically in the reward
function.
"""

from __future__ import annotations

import asyncio
import logging

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are an elevator dispatcher. Call the dispatch tool ONCE with the COMPLETE "
    "action string for the whole episode and stop.\n\n"
    "Actions (one letter each, concatenated, NO spaces): U move up one floor, D move "
    "down one floor, O open the door (let passengers off then on), C close the door. "
    "Door must be OPEN to exchange passengers, CLOSED to move. Invalid actions (wrong "
    "door state, moving past top/bottom floor) are penalized.\n\n"
    "LENGTH IS CRITICAL. A correct dispatch for a building with 6 arrivals is about "
    "40-55 letters, NEVER just 3-6 letters. Two-letter or three-letter answers always "
    "fail: the car stops before the first passenger even arrives and nobody is "
    "delivered. Plan the ENTIRE route first -- every pickup and dropoff across all "
    "floors, from the first arrival tick to the last -- then emit one long string.\n\n"
    "WORKED EXAMPLE (a different building): the dispatch "
    "\"UUUUUDDUOCDDDDOCUUUOCUUOCDDOCDDDOCUUUOCUOCUO\" is 44 letters and delivers all "
    "6 passengers. Mimic this LENGTH and structure, not the short examples below. "
    "Inside the dispatch tool call, put the real long string for YOUR building.\n\n"
    "The Building lists 'pending arrivals tN:F..->F..': passengers appear on tick tN. "
    "The car must keep running until the clock passes the LAST tN AND every passenger "
    "has been picked up and dropped off. Keep issuing U/D/O/C until done."
)

DISPATCH_TOOL = {
    "type": "function",
    "function": {
        "name": "dispatch",
        "description": "Submit a full elevator-dispatch episode as an ordered action sequence.",
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "string",
                    "description": "The COMPLETE action string for the whole episode, one letter each from U/D/O/C concatenated with no spaces. Usually 40-55 letters for a typical building -- a correct dispatch is long, never just a few letters.",
                    "pattern": "^[UDOC]+$",
                }
            },
            "required": ["actions"],
            "additionalProperties": False,
        },
    },
}


async def run_agent(ctx, batch):
    """Run one tool-call model request for each building."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The elevator agentic example requires `openai` and `httpx`. Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info("elevator agent start requests=%d max_running_prompts=%d", len(items), ctx.max_running_prompts)
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)

    async def run_one(item):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]
        tool_choice = {"type": "function", "function": {"name": "dispatch"}}
        response = await client.chat.completions.create(
            model="policy",
            messages=messages,
            tools=[DISPATCH_TOOL],
            tool_choice=tool_choice,
            stream=False,
        )
        return AgentTrajectoryTurn(
            item=item,
            messages=messages,
            response=response,
            tools=[DISPATCH_TOOL],
            tool_choice=tool_choice,
        )

    try:
        return AgentTrajectory(turns=list(await asyncio.gather(*(run_one(item) for item in items))))
    finally:
        await client.close()
