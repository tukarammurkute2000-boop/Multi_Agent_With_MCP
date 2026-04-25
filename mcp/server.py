"""
MCP (Model Control Protocol) Server — secure execution layer.
Validates requests, monitors calls, and applies circuit breaking
before any agent touches an external API.
"""
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import structlog
import time
from collections import defaultdict
from jose import jwt, JWTError
from config.settings import get_settings

settings = get_settings()
log = structlog.get_logger()
app = FastAPI(title="Travel AI MCP Server")
_bearer = HTTPBearer()

# ─── Circuit Breaker state (in-memory; use Redis for multi-instance) ─────────
_failures: dict[str, int] = defaultdict(int)
_open_until: dict[str, float] = {}


def _check_circuit(tool_name: str) -> None:
    if tool_name in _open_until:
        if time.time() < _open_until[tool_name]:
            raise HTTPException(503, f"Circuit open for {tool_name!r}. Try later.")
        else:
            del _open_until[tool_name]
            _failures[tool_name] = 0


def _record_failure(tool_name: str) -> None:
    _failures[tool_name] += 1
    if _failures[tool_name] >= settings.mcp_circuit_breaker_threshold:
        _open_until[tool_name] = time.time() + settings.mcp_circuit_breaker_timeout
        log.warning("circuit_breaker.opened", tool=tool_name)


# ─── Auth ─────────────────────────────────────────────────────────────────────
def verify_token(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    try:
        return jwt.decode(creds.credentials, settings.mcp_secret_key, algorithms=["HS256"])
    except JWTError as exc:
        raise HTTPException(401, "Invalid MCP token") from exc


# ─── Request / Response models ───────────────────────────────────────────────
class ToolCall(BaseModel):
    tool: str
    params: dict = {}
    session_id: str = ""


class ToolResult(BaseModel):
    tool: str
    result: dict | list | str | None = None
    error: str | None = None
    latency_ms: float = 0.0


# ─── Tool registry ────────────────────────────────────────────────────────────
_tools: dict = {}


def register_tool(name: str, fn):
    _tools[name] = fn


# ─── Main endpoint ────────────────────────────────────────────────────────────
@app.post("/execute", response_model=ToolResult)
async def execute_tool(call: ToolCall, claims: dict = Depends(verify_token)):
    _check_circuit(call.tool)

    if call.tool not in _tools:
        raise HTTPException(404, f"Unknown tool: {call.tool!r}")

    log.info("mcp.execute", tool=call.tool, session=call.session_id)
    start = time.perf_counter()

    try:
        result = await _tools[call.tool](**call.params)
        latency = (time.perf_counter() - start) * 1000
        log.info("mcp.success", tool=call.tool, latency_ms=round(latency, 2))
        return ToolResult(tool=call.tool, result=result, latency_ms=latency)
    except Exception as exc:
        _record_failure(call.tool)
        log.error("mcp.error", tool=call.tool, error=str(exc))
        raise HTTPException(500, str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok", "open_circuits": list(_open_until.keys())}
