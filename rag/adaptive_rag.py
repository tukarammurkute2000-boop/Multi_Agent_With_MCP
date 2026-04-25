"""
Adaptive RAG for hotel recommendations.
A query router decides: vector DB only, live API only, or both.
"""
import json
import httpx
import anthropic
from config.settings import get_settings
from rag.self_rag import _embed, _index

settings = get_settings()
_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

ROUTE_PROMPT = """
Classify the hotel search query into a routing strategy.
Return JSON: {"strategy": "vector" | "live_api" | "hybrid", "reason": "<short>"}
- "vector": generic info request, no real-time availability needed
- "live_api": specific dates + availability critical
- "hybrid": both quality context AND real-time availability needed
Query: {query}
"""


async def _get_amadeus_token() -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.amadeus_base_url}/v1/security/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": settings.amadeus_api_key,
                "client_secret": settings.amadeus_api_secret,
            },
        )
        resp.raise_for_status()
        return resp.json()["access_token"]


async def _search_hotels_live(city_code: str, check_in: str, check_out: str) -> list[dict]:
    """Call Amadeus Hotel Search API for real-time availability."""
    token = await _get_amadeus_token()
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{settings.amadeus_base_url}/v3/shopping/hotel-offers",
            headers={"Authorization": f"Bearer {token}"},
            params={
                "cityCode": city_code,
                "checkInDate": check_in,
                "checkOutDate": check_out,
                "adults": 1,
                "max": settings.hotel_max_results if hasattr(settings, "hotel_max_results") else 20,
            },
        )
        resp.raise_for_status()
        return resp.json().get("data", [])


def _search_hotels_vector(query: str, top_k: int = 5) -> list[dict]:
    results = _index.query(
        vector=_embed(f"hotel {query}"),
        top_k=top_k,
        include_metadata=True,
        filter={"type": "hotel"},
    )
    return [m["metadata"] for m in results.get("matches", [])]


async def recommend_hotels(
    query: str, city_code: str, check_in: str = "", check_out: str = ""
) -> dict:
    """Route to vector, live API, or both based on query characteristics."""
    resp = _client.messages.create(
        model=settings.claude_model,
        max_tokens=256,
        messages=[{"role": "user", "content": ROUTE_PROMPT.format(query=query)}],
    )
    try:
        route = json.loads(resp.content[0].text)
        strategy = route["strategy"]
    except Exception:
        strategy = "hybrid"

    vector_results, live_results = [], []

    if strategy in ("vector", "hybrid"):
        vector_results = _search_hotels_vector(query)

    if strategy in ("live_api", "hybrid") and check_in and check_out:
        live_results = await _search_hotels_live(city_code, check_in, check_out)

    # Merge and rank with Claude
    merge_prompt = f"""
Rank and summarize the best hotel options for this request.
Request: {query}
Vector DB results: {json.dumps(vector_results[:5])}
Live API results: {json.dumps(live_results[:5])}
Return top 3 hotels with name, price, highlights, and booking_reference.
"""
    final = _client.messages.create(
        model=settings.claude_model,
        max_tokens=1024,
        messages=[{"role": "user", "content": merge_prompt}],
    )

    return {
        "strategy_used": strategy,
        "recommendations": final.content[0].text,
        "live_count": len(live_results),
        "vector_count": len(vector_results),
    }
