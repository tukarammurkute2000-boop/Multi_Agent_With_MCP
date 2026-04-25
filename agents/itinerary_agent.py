"""
Itinerary Agent — Autonomous RAG (Self-RAG) for day-by-day trip planning.

Pipeline:
  1. Retrieve travel knowledge from Pinecone
  2. Grade relevance; refine query if score < threshold
  3. Fetch real travel times via Google Distance Matrix
  4. Generate structured itinerary with Claude
  5. Verify for hallucinations / unrealistic logistics
  6. Refine once if verification fails
"""
import json
import httpx
import anthropic
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from config.settings import get_settings

settings = get_settings()
_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_pc = Pinecone(api_key=settings.pinecone_api_key)
_index = _pc.Index(settings.pinecone_index_name)

# ─── Prompts ─────────────────────────────────────────────────────────────────

_GRADE_PROMPT = """
Rate how relevant this context is for planning a trip to {destination}.
Return JSON only: {{"score": <1-10>, "gap": "<what is missing>"}}
Context: {context}
"""

_GENERATE_PROMPT = """
You are an expert travel planner. Create a detailed {duration}-day itinerary for {destination}.

User preferences: {preferences}
Budget per day (INR): {daily_budget}
Avoid: {avoid}
Travel distances/times between key spots: {distances}
Knowledge base context: {context}

Return structured JSON:
{{
  "destination": "{destination}",
  "duration_days": {duration},
  "daily_budget_inr": {daily_budget},
  "days": [
    {{
      "day": 1,
      "theme": "<theme>",
      "morning": {{"activity": "", "location": "", "duration_hrs": 0, "cost_inr": 0}},
      "afternoon": {{"activity": "", "location": "", "duration_hrs": 0, "cost_inr": 0}},
      "evening": {{"activity": "", "location": "", "duration_hrs": 0, "cost_inr": 0}},
      "accommodation": "",
      "travel_notes": "",
      "estimated_daily_spend_inr": 0
    }}
  ],
  "total_estimated_cost_inr": 0,
  "packing_tips": [],
  "best_season": ""
}}
"""

_VERIFY_PROMPT = """
Verify this itinerary for:
1. Unrealistic travel times between locations (use provided distances)
2. Attractions that don't exist or are in wrong city
3. Budget figures that are obviously wrong
4. Days that pack too many activities (>4 spots is unrealistic)

Return JSON: {{"valid": true/false, "issues": ["..."], "severity": "ok"|"minor"|"major"}}

Itinerary: {itinerary}
Distances: {distances}
"""

_REFINE_PROMPT = """
Fix the following issues in the travel itinerary. Keep all valid days unchanged.
Issues: {issues}
Original itinerary: {itinerary}
Return the corrected full itinerary JSON only.
"""


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _embed(text: str) -> list[float]:
    return _embedder.encode(text).tolist()


def _retrieve(query: str, top_k: int = 6) -> list[str]:
    results = _index.query(vector=_embed(query), top_k=top_k, include_metadata=True)
    return [m["metadata"].get("text", "") for m in results.get("matches", [])]


def _grade(destination: str, context: str) -> tuple[int, str]:
    resp = _client.messages.create(
        model=settings.claude_model,
        max_tokens=128,
        messages=[{
            "role": "user",
            "content": _GRADE_PROMPT.format(destination=destination, context=context[:2000]),
        }],
    )
    try:
        data = json.loads(resp.content[0].text)
        return data["score"], data.get("gap", "")
    except Exception:
        return 5, ""


async def _fetch_distances(spots: list[str], api_key: str) -> dict:
    """
    Call Google Distance Matrix to get travel times between attraction pairs.
    Returns {("A","B"): "25 mins by car"}.
    """
    if not spots or not api_key:
        return {}

    origins = "|".join(spots[:-1])
    destinations = "|".join(spots[1:])
    url = settings.google_distance_matrix_url

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params={
                "origins": origins,
                "destinations": destinations,
                "key": api_key,
                "mode": "driving",
            })
            resp.raise_for_status()
            data = resp.json()

        result = {}
        rows = data.get("rows", [])
        for i, row in enumerate(rows):
            elements = row.get("elements", [])
            if i < len(elements):
                elem = elements[i]
                if elem.get("status") == "OK":
                    result[(spots[i], spots[i + 1])] = (
                        elem["duration"]["text"] + " / " + elem["distance"]["text"]
                    )
        return result
    except Exception:
        return {}


def _extract_spots(itinerary_json: dict) -> list[str]:
    spots = []
    for day in itinerary_json.get("days", []):
        for slot in ("morning", "afternoon", "evening"):
            loc = day.get(slot, {}).get("location", "")
            if loc and loc not in spots:
                spots.append(loc)
    return spots[:8]  # cap at 8 to stay within Distance Matrix free tier


def _call_claude(prompt: str, max_tokens: int = 4096) -> str:
    resp = _client.messages.create(
        model=settings.claude_model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def _parse_json_block(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


# ─── Main entry point ─────────────────────────────────────────────────────────

async def run_itinerary_agent(
    destination: str,
    duration_days: int,
    preferences: dict,
    budget_total_inr: float = 0,
    avoid: list[str] = None,
) -> dict:
    """
    Autonomous RAG itinerary agent.

    Returns:
        {
            "itinerary": <structured dict>,
            "iterations": int,
            "relevance_score": int,
            "distances": dict,
        }
    """
    avoid = avoid or []
    daily_budget = round(budget_total_inr / duration_days) if budget_total_inr else 3000

    base_query = f"{duration_days}-day trip to {destination}"
    refined_query = base_query
    best_context = ""
    best_score = 0

    # ── Self-RAG loop ──────────────────────────────────────────────────────────
    for iteration in range(1, settings.max_itinerary_retries + 1):
        docs = _retrieve(refined_query)
        context = "\n\n".join(d for d in docs if d)
        score, gap = _grade(destination, context)

        if score > best_score:
            best_score = score
            best_context = context

        if score >= 7:
            break

        # Narrow the retrieval query towards the identified gap
        if gap:
            refined_query = f"{base_query} — {gap}"

    # ── Generate itinerary ────────────────────────────────────────────────────
    itinerary_text = _call_claude(
        _GENERATE_PROMPT.format(
            destination=destination,
            duration=duration_days,
            preferences=json.dumps(preferences),
            daily_budget=daily_budget,
            avoid=", ".join(avoid) if avoid else "none",
            distances="(fetching after generation)",
            context=best_context[:3000],
        )
    )

    try:
        itinerary_json = _parse_json_block(itinerary_text)
    except Exception:
        itinerary_json = {"raw": itinerary_text}
        return {
            "itinerary": itinerary_json,
            "iterations": iteration,
            "relevance_score": best_score,
            "distances": {},
        }

    # ── Fetch real distances for extracted spots ───────────────────────────────
    spots = _extract_spots(itinerary_json)
    distances = await _fetch_distances(spots, settings.google_maps_api_key)
    distances_str = json.dumps({f"{k[0]} → {k[1]}": v for k, v in distances.items()})

    # ── Verify ────────────────────────────────────────────────────────────────
    verify_text = _call_claude(
        _VERIFY_PROMPT.format(
            itinerary=json.dumps(itinerary_json),
            distances=distances_str,
        ),
        max_tokens=512,
    )

    try:
        verification = _parse_json_block(verify_text)
    except Exception:
        verification = {"valid": True, "issues": [], "severity": "ok"}

    # ── Refine if major issues found ──────────────────────────────────────────
    if not verification.get("valid") and verification.get("severity") == "major":
        refined_text = _call_claude(
            _REFINE_PROMPT.format(
                issues=verification["issues"],
                itinerary=json.dumps(itinerary_json),
            )
        )
        try:
            itinerary_json = _parse_json_block(refined_text)
        except Exception:
            pass  # keep original if refinement fails to parse

    return {
        "itinerary": itinerary_json,
        "iterations": iteration,
        "relevance_score": best_score,
        "distances": {f"{k[0]} → {k[1]}": v for k, v in distances.items()},
        "verification": verification,
    }
