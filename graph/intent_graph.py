"""
Intent Graph — captures multi-intent queries as a graph of relationships.
Nodes: intents (plan_trip, book_flight, book_hotel, …)
Edges: dependencies and constraints between them.
"""
import networkx as nx
from dataclasses import dataclass, field
from typing import Any
import anthropic
import json
from config.settings import get_settings

settings = get_settings()
_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

INTENT_EXTRACTION_PROMPT = """
Analyze the travel query and extract all intents, entities, and constraints.
Return JSON with this exact structure:
{
  "intents": ["plan_trip", "book_flight", "book_hotel"],
  "entities": {
    "destinations": ["Goa"],
    "duration_days": 6,
    "budget_inr": null,
    "travelers": 1,
    "travel_dates": null,
    "preferences": [],
    "avoid": []
  },
  "dependencies": [
    {"from": "book_flight", "to": "plan_trip", "type": "requires"},
    {"from": "book_hotel", "to": "plan_trip", "type": "requires"}
  ],
  "missing_slots": ["travel_dates", "budget_inr"]
}
Query: {query}
"""


@dataclass
class ParsedIntent:
    intents: list[str] = field(default_factory=list)
    entities: dict[str, Any] = field(default_factory=dict)
    dependencies: list[dict] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)


def build_intent_graph(query: str) -> tuple[nx.DiGraph, ParsedIntent]:
    """Parse a raw user query into a dependency graph of intents."""
    response = _client.messages.create(
        model=settings.claude_model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": INTENT_EXTRACTION_PROMPT.format(query=query),
            }
        ],
    )
    raw = response.content[0].text.strip()
    # Claude may wrap JSON in a code block
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw)

    parsed = ParsedIntent(
        intents=data.get("intents", []),
        entities=data.get("entities", {}),
        dependencies=data.get("dependencies", []),
        missing_slots=data.get("missing_slots", []),
    )

    G = nx.DiGraph()
    for intent in parsed.intents:
        G.add_node(intent, **parsed.entities)
    for dep in parsed.dependencies:
        G.add_edge(dep["from"], dep["to"], type=dep["type"])

    return G, parsed


def execution_order(graph: nx.DiGraph) -> list[str]:
    """Topological sort — run intents in dependency-safe order."""
    return list(reversed(list(nx.topological_sort(graph))))
