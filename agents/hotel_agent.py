"""
Hotel Recommendation & Suggestion Agent — budget-aware.

Strategy:
  1. Query router: vector DB / live Amadeus API / hybrid  (Adaptive RAG)
  2. Filter results against budget_per_night
  3. Score options (value = amenities / price ratio)
  4. If nothing fits budget → suggest cheapest available + budget gap warning
  5. Return ranked recommendations with offer_id ready for booking
"""
import json
import httpx
import anthropic
from config.settings import get_settings
from rag.self_rag import _embed, _index   # reuse shared embedder + Pinecone index

settings = get_settings()
_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

# ─── Prompts ─────────────────────────────────────────────────────────────────

_ROUTE_PROMPT = """
Classify this hotel search to decide the data source.
Return JSON only: {{"strategy": "vector"|"live_api"|"hybrid", "reason": "<10 words>"}}
Rules:
- "vector"   → generic question, no specific dates, just inspiration
- "live_api" → specific dates + city, real-time availability matters
- "hybrid"   → need both quality context AND real-time pricing
Query: {query}
Has dates: {has_dates}
"""

_RANK_PROMPT = """
You are a hotel concierge. Rank these hotels for the traveler.

Request: {query}
Budget per night (INR): {budget_per_night}
Total budget for stay (INR): {total_budget}
Nights: {nights}
Preferences: {preferences}

Vector DB options: {vector_options}
Live API options: {live_options}

Rules:
- Exclude hotels where price_per_night > budget_per_night * 1.15  (allow 15% flex)
- If ALL options exceed budget, include cheapest 2 with a budget_warning flag
- Score each on: location(30%) + amenities(30%) + value_for_money(40%)

Return JSON:
{{
  "recommendations": [
    {{
      "rank": 1,
      "name": "",
      "stars": 0,
      "price_per_night_inr": 0,
      "total_price_inr": 0,
      "location": "",
      "amenities": [],
      "highlights": "",
      "offer_id": "",          // Amadeus offerId if available, else ""
      "within_budget": true,
      "score": 0.0
    }}
  ],
  "budget_warning": "",        // empty string if all within budget
  "cheapest_option": {{}}      // populated only when budget_warning is set
}}
"""

_SUGGEST_BUDGET_PROMPT = """
The traveler's hotel budget (INR {budget_per_night}/night) is too low for {destination}.
Suggest: (1) realistic budget range, (2) budget-friendly alternatives, (3) ways to reduce cost.
Keep response under 150 words.
"""


# ─── Amadeus helpers ──────────────────────────────────────────────────────────

async def _get_token() -> str:
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"{settings.amadeus_base_url}/v1/security/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": settings.amadeus_api_key,
                "client_secret": settings.amadeus_api_secret,
            },
        )
        r.raise_for_status()
        return r.json()["access_token"]


async def _live_hotel_search(city_code: str, check_in: str, check_out: str) -> list[dict]:
    try:
        token = await _get_token()
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(
                f"{settings.amadeus_base_url}/v3/shopping/hotel-offers",
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "cityCode": city_code,
                    "checkInDate": check_in,
                    "checkOutDate": check_out,
                    "adults": 1,
                    "max": 20,
                    "currency": settings.razorpay_currency,
                },
            )
            r.raise_for_status()
            return r.json().get("data", [])
    except Exception:
        return []


def _live_to_normalized(offers: list[dict]) -> list[dict]:
    result = []
    for item in offers:
        hotel = item.get("hotel", {})
        for offer in item.get("offers", [])[:1]:
            price = offer.get("price", {})
            result.append({
                "name": hotel.get("name", ""),
                "stars": hotel.get("rating", 0),
                "price_per_night_inr": float(price.get("total", 0)),
                "location": hotel.get("cityCode", ""),
                "amenities": hotel.get("amenities", []),
                "offer_id": offer.get("id", ""),
                "source": "live_api",
            })
    return result


def _vector_hotel_search(destination: str, top_k: int = 8) -> list[dict]:
    try:
        results = _index.query(
            vector=_embed(f"hotel {destination} accommodation stay"),
            top_k=top_k,
            include_metadata=True,
            filter={"type": {"$in": ["hotel", "accommodation"]}},
        )
        return [
            {**m["metadata"], "source": "vector", "offer_id": ""}
            for m in results.get("matches", [])
        ]
    except Exception:
        return []


# ─── Route decision ────────────────────────────────────────────────────────────

def _decide_route(query: str, has_dates: bool) -> str:
    resp = _client.messages.create(
        model=settings.claude_model,
        max_tokens=128,
        messages=[{
            "role": "user",
            "content": _ROUTE_PROMPT.format(query=query, has_dates=str(has_dates)),
        }],
    )
    try:
        return json.loads(resp.content[0].text)["strategy"]
    except Exception:
        return "hybrid"


# ─── Budget suggestion ────────────────────────────────────────────────────────

def _budget_suggestion(destination: str, budget_per_night: float) -> str:
    resp = _client.messages.create(
        model=settings.claude_model,
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": _SUGGEST_BUDGET_PROMPT.format(
                budget_per_night=budget_per_night, destination=destination
            ),
        }],
    )
    return resp.content[0].text


# ─── Main entry point ─────────────────────────────────────────────────────────

async def run_hotel_agent(
    destination: str,
    city_code: str,
    check_in: str = "",
    check_out: str = "",
    budget_per_night_inr: float = 0,
    total_budget_inr: float = 0,
    nights: int = 1,
    preferences: dict = None,
    raw_query: str = "",
) -> dict:
    """
    Budget-aware hotel recommendation.

    Returns:
        {
            "strategy": str,
            "recommendations": list[dict],
            "budget_warning": str,
            "budget_suggestion": str,   # non-empty only when budget is too low
            "cheapest_option": dict,
        }
    """
    preferences = preferences or {}
    has_dates = bool(check_in and check_out)
    strategy = _decide_route(raw_query or f"hotels in {destination}", has_dates)

    vector_results, live_results = [], []

    if strategy in ("vector", "hybrid"):
        vector_results = _vector_hotel_search(destination)

    if strategy in ("live_api", "hybrid") and has_dates:
        raw_live = await _live_hotel_search(city_code, check_in, check_out)
        live_results = _live_to_normalized(raw_live)

    # ── Rank & filter by budget ────────────────────────────────────────────────
    rank_resp = _client.messages.create(
        model=settings.claude_model,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": _RANK_PROMPT.format(
                query=raw_query or f"hotels in {destination}",
                budget_per_night=budget_per_night_inr,
                total_budget=total_budget_inr,
                nights=nights,
                preferences=json.dumps(preferences),
                vector_options=json.dumps(vector_results[:6]),
                live_options=json.dumps(live_results[:6]),
            ),
        }],
    )

    try:
        ranked = json.loads(rank_resp.content[0].text)
    except Exception:
        ranked = {"recommendations": [], "budget_warning": "", "cheapest_option": {}}

    # ── Add budget advisory if all options exceed budget ──────────────────────
    budget_suggestion = ""
    if ranked.get("budget_warning") and budget_per_night_inr > 0:
        budget_suggestion = _budget_suggestion(destination, budget_per_night_inr)

    return {
        "strategy": strategy,
        "recommendations": ranked.get("recommendations", []),
        "budget_warning": ranked.get("budget_warning", ""),
        "cheapest_option": ranked.get("cheapest_option", {}),
        "budget_suggestion": budget_suggestion,
    }
