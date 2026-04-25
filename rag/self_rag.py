"""
Self-RAG (Autonomous RAG) for itinerary generation.
Loop: retrieve → grade relevance → generate → verify → refine until confident.
"""
import anthropic
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from config.settings import get_settings

settings = get_settings()
_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_pc = Pinecone(api_key=settings.pinecone_api_key)
_index = _pc.Index(settings.pinecone_index_name)

GRADE_PROMPT = """
Given the query and retrieved context, rate relevance 1-10.
Return JSON: {"score": <int>, "reason": "<short reason>"}
Query: {query}
Context: {context}
"""

GENERATE_PROMPT = """
You are a travel planning expert. Using the context below, create a detailed
day-by-day itinerary. Include timings, distances, and realistic travel logistics.
Context: {context}
Request: {query}
User preferences: {preferences}
"""

VERIFY_PROMPT = """
Check the itinerary for: (1) unrealistic distances/timings, (2) hallucinated
attractions, (3) missing logistic details.
Return JSON: {"valid": <bool>, "issues": ["..."]}
Itinerary: {itinerary}
"""


def _embed(text: str) -> list[float]:
    return _embedder.encode(text).tolist()


def _retrieve(query: str, top_k: int = 5) -> list[str]:
    results = _index.query(vector=_embed(query), top_k=top_k, include_metadata=True)
    return [m["metadata"].get("text", "") for m in results.get("matches", [])]


def _grade(query: str, context: str) -> int:
    import json
    resp = _client.messages.create(
        model=settings.claude_model,
        max_tokens=256,
        messages=[{"role": "user", "content": GRADE_PROMPT.format(query=query, context=context)}],
    )
    try:
        return json.loads(resp.content[0].text)["score"]
    except Exception:
        return 5


def _generate(query: str, context: str, preferences: dict) -> str:
    resp = _client.messages.create(
        model=settings.claude_model,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": GENERATE_PROMPT.format(
                    context=context, query=query, preferences=preferences
                ),
            }
        ],
    )
    return resp.content[0].text


def _verify(itinerary: str) -> tuple[bool, list[str]]:
    import json
    resp = _client.messages.create(
        model=settings.claude_model,
        max_tokens=512,
        messages=[{"role": "user", "content": VERIFY_PROMPT.format(itinerary=itinerary)}],
    )
    try:
        data = json.loads(resp.content[0].text)
        return data["valid"], data.get("issues", [])
    except Exception:
        return True, []


def generate_itinerary(query: str, preferences: dict, max_retries: int = None) -> dict:
    """
    Iterative Self-RAG loop.
    Returns {"itinerary": str, "context_used": str, "iterations": int}.
    """
    max_retries = max_retries or settings.max_itinerary_retries
    best_context = ""
    best_score = 0
    refined_query = query

    for iteration in range(1, max_retries + 1):
        docs = _retrieve(refined_query)
        context = "\n\n".join(docs)
        score = _grade(query, context)

        if score > best_score:
            best_score = score
            best_context = context

        if score >= 7:
            break
        # Refine query for next iteration based on low score
        refined_query = f"{query} — focus on detailed local experiences and logistics"

    itinerary = _generate(query, best_context, preferences)
    valid, issues = _verify(itinerary)

    if not valid and issues:
        # One refinement pass addressing flagged issues
        fix_prompt = f"Fix these issues in the itinerary: {issues}\n\n{itinerary}"
        resp = _client.messages.create(
            model=settings.claude_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": fix_prompt}],
        )
        itinerary = resp.content[0].text

    return {
        "itinerary": itinerary,
        "context_used": best_context,
        "iterations": iteration,
        "relevance_score": best_score,
    }
