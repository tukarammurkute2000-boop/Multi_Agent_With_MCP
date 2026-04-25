"""
LangGraph Supervisor — fully dynamic routing based on user intent.

Route plans (set by decide_route node):
  "full"             → plan itinerary + hotels + transport + book both
  "trip_plan_only"   → generate itinerary only, no booking
  "hotel_search"     → recommend hotels, no booking
  "hotel_book"       → recommend + book hotel only
  "transport_search" → search transport options, no booking
  "transport_book"   → search + book transport only
  "availability_only"→ read-only seat/room availability check → END

Graph edges:
  parse_query → validate_slots → decide_route
      ├─ "availability_only" → check_availability → END
      └─ all others          → plan_itinerary (skips if route not relevant)
                                    ↓
                               recommend_hotels (skips if route not relevant)
                                    ↓
                               find_transport   (skips if route not relevant)
                                    ↓
                               human_confirmation
                                    ├─ booking_needed=False → END
                                    └─ booking_needed=True  → execute_bookings → END
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
import asyncio
import structlog

from graph.intent_graph import build_intent_graph
from graph.memory_graph import load_memory, update_memory, get_avoided_destinations
from agents.itinerary_agent import run_itinerary_agent
from agents.hotel_agent import run_hotel_agent
from agents.booking_agent import (
    run_booking_agent, check_availability_standalone,
    BookingRequest, AvailabilityRequest,
)
from booking.flight_booking import search_flights

log = structlog.get_logger()

# ─── Route plan constants ─────────────────────────────────────────────────────

ROUTE_FULL              = "full"
ROUTE_TRIP_PLAN_ONLY    = "trip_plan_only"
ROUTE_HOTEL_SEARCH      = "hotel_search"
ROUTE_HOTEL_BOOK        = "hotel_book"
ROUTE_TRANSPORT_SEARCH  = "transport_search"
ROUTE_TRANSPORT_BOOK    = "transport_book"
ROUTE_AVAILABILITY_ONLY = "availability_only"

# Nodes that participate in each route
_USES_ITINERARY  = {ROUTE_FULL, ROUTE_TRIP_PLAN_ONLY}
_USES_HOTELS     = {ROUTE_FULL, ROUTE_HOTEL_SEARCH, ROUTE_HOTEL_BOOK}
_USES_TRANSPORT  = {ROUTE_FULL, ROUTE_TRANSPORT_SEARCH, ROUTE_TRANSPORT_BOOK}
_NEEDS_BOOKING   = {ROUTE_FULL, ROUTE_HOTEL_BOOK, ROUTE_TRANSPORT_BOOK}

# ─── IATA helpers ─────────────────────────────────────────────────────────────

_IATA_MAP = {
    "goa": "GOI", "mumbai": "BOM", "delhi": "DEL", "bangalore": "BLR",
    "chennai": "MAA", "hyderabad": "HYD", "kolkata": "CCU", "jaipur": "JAI",
    "pune": "PNQ", "ahmedabad": "AMD",
}


def _city_to_iata(city: str) -> str:
    return _IATA_MAP.get(city.lower(), city.upper()[:3])


# ─── State ────────────────────────────────────────────────────────────────────

class TravelState(TypedDict):
    user_id: str
    raw_query: str
    parsed: dict
    memory: dict
    route_plan: str             # one of ROUTE_* constants

    # Agent outputs
    itinerary: dict
    hotel_recommendations: dict
    flight_options: list
    train_options: list
    bus_options: list
    availability_results: dict  # for availability_only route

    # Selections from human_confirmation
    selected_transport_type: str
    selected_transport: dict
    selected_hotel_offer_id: str
    selected_hotel_price_inr: float
    booking_needed: bool

    # Final booking results
    transport_booking: dict
    hotel_booking: dict
    payment_orders: list

    confirmed: bool
    errors: Annotated[list, operator.add]


# ─── Node: parse_query ────────────────────────────────────────────────────────

def parse_query(state: TravelState) -> TravelState:
    _, parsed = build_intent_graph(state["raw_query"])
    memory = load_memory(state["user_id"])
    avoided = get_avoided_destinations(state["user_id"])

    destinations = parsed.entities.get("destinations", [])
    filtered = [d for d in destinations if d not in avoided]
    if len(filtered) < len(destinations):
        log.info("supervisor.filtered_destinations",
                 removed=list(set(destinations) - set(filtered)))
        parsed.entities["destinations"] = filtered

    return {**state, "parsed": vars(parsed), "memory": memory}


# ─── Node: validate_slots ─────────────────────────────────────────────────────

def validate_slots(state: TravelState) -> TravelState:
    missing = state["parsed"].get("missing_slots", [])
    if missing:
        log.warning("supervisor.missing_slots", slots=missing)
    return state


# ─── Node: decide_route ───────────────────────────────────────────────────────

def decide_route(state: TravelState) -> TravelState:
    """
    Reads parsed intents and sets route_plan.
    This is the single decision point — every branch flows from here.
    """
    intents: list[str] = state["parsed"].get("intents", [])
    intent_str = " ".join(intents).lower()

    wants_plan           = any(w in intent_str for w in ("plan_trip", "itinerary", "plan_itinerary"))
    wants_hotel_search   = any(w in intent_str for w in ("search_hotel", "find_hotel", "hotel_recommend", "suggest_hotel"))
    wants_hotel_book     = any(w in intent_str for w in ("book_hotel", "reserve_hotel", "hotel_booking"))
    wants_transport_book = any(w in intent_str for w in ("book_flight", "book_train", "book_bus", "book_transport"))
    wants_transport_search = any(w in intent_str for w in ("search_flight", "search_train", "search_bus", "find_flight", "find_train", "find_bus"))
    wants_availability   = any(w in intent_str for w in ("check_availability", "seat_availability", "check_seats", "available_seats", "room_availability"))

    # Availability-only: explicit availability check with no booking intent
    if wants_availability and not (wants_hotel_book or wants_transport_book):
        route = ROUTE_AVAILABILITY_ONLY

    # Full pipeline: trip planning + some form of booking
    elif wants_plan and (wants_hotel_book or wants_transport_book):
        route = ROUTE_FULL

    # Trip plan with no booking
    elif wants_plan and not (wants_hotel_book or wants_hotel_search or wants_transport_book or wants_transport_search):
        route = ROUTE_TRIP_PLAN_ONLY

    # Hotel only
    elif (wants_hotel_book or wants_hotel_search) and not (wants_transport_book or wants_transport_search or wants_plan):
        route = ROUTE_HOTEL_BOOK if wants_hotel_book else ROUTE_HOTEL_SEARCH

    # Transport only
    elif (wants_transport_book or wants_transport_search) and not (wants_hotel_book or wants_hotel_search or wants_plan):
        route = ROUTE_TRANSPORT_BOOK if wants_transport_book else ROUTE_TRANSPORT_SEARCH

    # Fallback: if any booking intent exists go full, else full (default for travel queries)
    else:
        route = ROUTE_FULL

    log.info("supervisor.route_decided", route=route, intents=intents)
    return {**state, "route_plan": route}


# ─── Node: check_availability (availability_only route only) ─────────────────

async def check_availability(state: TravelState) -> TravelState:
    """
    Read-only availability check — does NOT book anything.
    Handles: flight seat search, train class availability, bus seats, hotel rooms.
    """
    intents: list[str] = state["parsed"].get("intents", [])
    entities = state["parsed"].get("entities", {})
    intent_str = " ".join(intents).lower()
    destinations = entities.get("destinations", [])
    origin = state["memory"].get("home_city", "mumbai")
    travel_date = entities.get("travel_dates") or ""

    results: dict = {}

    # Determine which transport/accommodation type to check
    check_flight = "flight" in intent_str
    check_train  = "train" in intent_str
    check_bus    = "bus" in intent_str
    check_hotel  = "hotel" in intent_str or "room" in intent_str

    # Default to flight if nothing specific
    if not any([check_flight, check_train, check_bus, check_hotel]):
        check_flight = True

    tasks = []

    if check_flight and destinations:
        req = AvailabilityRequest(
            resource_type="flight",
            origin=origin,
            destination=destinations[0],
            travel_date=travel_date,
        )
        tasks.append(("flight", check_availability_standalone(req)))

    if check_train and destinations:
        req = AvailabilityRequest(
            resource_type="train",
            origin=origin,
            destination=destinations[0],
            travel_date=travel_date,
            train_number=entities.get("train_number", ""),
            train_class=entities.get("travel_class", "3A"),
        )
        tasks.append(("train", check_availability_standalone(req)))

    if check_bus and destinations:
        req = AvailabilityRequest(
            resource_type="bus",
            origin=origin,
            destination=destinations[0],
            travel_date=travel_date,
        )
        tasks.append(("bus", check_availability_standalone(req)))

    if check_hotel and destinations:
        req = AvailabilityRequest(
            resource_type="hotel",
            destination=destinations[0],
            city_code=_city_to_iata(destinations[0]),
            check_in=entities.get("check_in", ""),
            check_out=entities.get("check_out", ""),
        )
        tasks.append(("hotel", check_availability_standalone(req)))

    if tasks:
        labels, coros = zip(*tasks)
        raw_results = await asyncio.gather(*coros, return_exceptions=True)
        for label, result in zip(labels, raw_results):
            if isinstance(result, Exception):
                log.warning("supervisor.availability_check_failed",
                            type=label, error=str(result))
                results[label] = {"error": str(result)}
            else:
                results[label] = result

    log.info("supervisor.availability_done", checked=list(results.keys()))
    return {**state, "availability_results": results}


# ─── Node: plan_itinerary ─────────────────────────────────────────────────────

async def plan_itinerary(state: TravelState) -> TravelState:
    if state["route_plan"] not in _USES_ITINERARY:
        return state   # skip — not needed for this route

    entities = state["parsed"]["entities"]
    destinations = entities.get("destinations", [])
    if not destinations:
        return {**state, "itinerary": {}, "errors": ["No destination in query"]}

    avoided = list(get_avoided_destinations(state["user_id"]).keys())
    result = await run_itinerary_agent(
        destination=destinations[0],
        duration_days=entities.get("duration_days", 3),
        preferences=state["memory"],
        budget_total_inr=float(entities.get("budget_inr") or 0),
        avoid=avoided,
    )
    log.info("supervisor.itinerary_done",
             score=result["relevance_score"], iters=result["iterations"])
    return {**state, "itinerary": result["itinerary"]}


# ─── Node: recommend_hotels ───────────────────────────────────────────────────

async def recommend_hotels(state: TravelState) -> TravelState:
    if state["route_plan"] not in _USES_HOTELS:
        return state   # skip

    entities = state["parsed"]["entities"]
    destinations = entities.get("destinations", [])
    if not destinations:
        return {**state, "hotel_recommendations": {}, "errors": ["No destination for hotels"]}

    duration = int(entities.get("duration_days") or 1)
    total_budget = float(entities.get("budget_inr") or 0)
    hotel_budget = total_budget * 0.30
    per_night = round(hotel_budget / duration) if hotel_budget else 3000

    result = await run_hotel_agent(
        destination=destinations[0],
        city_code=_city_to_iata(destinations[0]),
        check_in=entities.get("check_in", ""),
        check_out=entities.get("check_out", ""),
        budget_per_night_inr=per_night,
        total_budget_inr=hotel_budget,
        nights=duration,
        preferences=state["memory"],
        raw_query=state["raw_query"],
    )

    if result.get("budget_warning"):
        log.warning("supervisor.hotel_budget_warning", msg=result["budget_warning"])

    return {**state, "hotel_recommendations": result}


# ─── Node: find_transport ─────────────────────────────────────────────────────

async def find_transport(state: TravelState) -> TravelState:
    if state["route_plan"] not in _USES_TRANSPORT:
        return state   # skip

    entities = state["parsed"]["entities"]
    intents: list[str] = state["parsed"].get("intents", [])
    intent_str = " ".join(intents).lower()
    destinations = entities.get("destinations", [])
    origin = state["memory"].get("home_city", "mumbai")
    travel_date = entities.get("travel_dates") or "2025-06-01"

    wants_flight = "flight" in intent_str
    wants_train  = "train"  in intent_str
    wants_bus    = "bus"    in intent_str
    if not any([wants_flight, wants_train, wants_bus]):
        wants_flight = True   # default to flight

    flight_opts, train_opts, bus_opts = [], [], []

    if destinations:
        dest = destinations[0]
        tasks, labels = [], []

        if wants_flight:
            tasks.append(search_flights(_city_to_iata(origin), _city_to_iata(dest), travel_date))
            labels.append("flight")
        if wants_train:
            from booking.train_booking import search_trains
            tasks.append(search_trains(origin, dest, travel_date))
            labels.append("train")
        if wants_bus:
            from booking.bus_booking import search_buses
            tasks.append(search_buses(origin, dest, travel_date))
            labels.append("bus")

        raw = await asyncio.gather(*tasks, return_exceptions=True)
        for label, result in zip(labels, raw):
            if isinstance(result, Exception):
                log.warning("supervisor.transport_search_failed",
                            mode=label, error=str(result))
                continue
            if label == "flight":   flight_opts = result
            elif label == "train":  train_opts  = result
            elif label == "bus":    bus_opts    = result

    return {**state,
            "flight_options": flight_opts,
            "train_options":  train_opts,
            "bus_options":    bus_opts}


# ─── Node: human_confirmation ─────────────────────────────────────────────────

def human_confirmation(state: TravelState) -> TravelState:
    """
    Selects best option(s) and decides whether a booking is needed.

    - search-only routes:    sets booking_needed=False → graph ends after this node
    - hotel-only booking:    picks hotel, skips transport selection
    - transport-only booking: picks transport, skips hotel selection
    - full booking:          picks both

    Production: suspend graph here, emit selections to UI, resume on user callback.
    Dev/test:   auto-select cheapest.
    """
    route = state["route_plan"]

    # Search-only and trip-plan-only paths — nothing to book
    if route not in _NEEDS_BOOKING:
        log.info("supervisor.no_booking_needed", route=route)
        return {**state, "booking_needed": False, "confirmed": False}

    transport_type = ""
    selected_transport: dict = {}
    hotel_offer_id = ""
    hotel_price = 0.0

    # ── Pick transport (skip for hotel_book route) ────────────────────────────
    if route in {ROUTE_FULL, ROUTE_TRANSPORT_BOOK}:
        for mode, opts, price_key in [
            ("flight", state.get("flight_options", []), "price"),
            ("train",  state.get("train_options",  []), "fare_inr"),
            ("bus",    state.get("bus_options",    []), "fare_inr"),
        ]:
            if opts:
                best = min(opts, key=lambda x: float(x.get(price_key) or 0))
                transport_type = mode
                selected_transport = best
                log.info("supervisor.transport_selected", mode=mode,
                         name=best.get("flight_number") or best.get("train_name") or best.get("operator", ""))
                break

        if not transport_type:
            return {**state, "confirmed": False, "booking_needed": False,
                    "errors": ["No transport options found — check origin/destination/date"]}

    # ── Pick hotel (skip for transport_book route) ────────────────────────────
    if route in {ROUTE_FULL, ROUTE_HOTEL_BOOK}:
        recs = state.get("hotel_recommendations", {}).get("recommendations", [])
        if recs:
            top = recs[0]
            hotel_offer_id = top.get("offer_id", "")
            hotel_price = float(top.get("total_price_inr") or top.get("price_per_night_inr") or 0)
            log.info("supervisor.hotel_selected",
                     hotel=top.get("name", ""), price=hotel_price)
        elif route == ROUTE_HOTEL_BOOK:
            return {**state, "confirmed": False, "booking_needed": False,
                    "errors": ["No hotel offers found — try different dates or budget"]}

    return {**state,
            "selected_transport_type":  transport_type,
            "selected_transport":       selected_transport,
            "selected_hotel_offer_id":  hotel_offer_id,
            "selected_hotel_price_inr": hotel_price,
            "booking_needed": True,
            "confirmed": True}


# ─── Node: execute_bookings ───────────────────────────────────────────────────

async def execute_bookings(state: TravelState) -> TravelState:
    route   = state["route_plan"]
    entities = state["parsed"]["entities"]
    memory  = state["memory"]
    traveler = memory.get("traveler_profile", {})
    contact_email = traveler.get("contact", {}).get("emailAddress", "user@example.com")
    contact_phone = memory.get("phone", "9999999999")
    passengers = memory.get("passengers", [
        {"name": "Traveler User", "age": 30, "gender": "M",
         "id_type": "AADHAR", "id_number": "XXXX-XXXX-XXXX"}
    ])

    destinations = entities.get("destinations", [])
    origin = memory.get("home_city", "mumbai")
    dest = destinations[0] if destinations else ""
    travel_date = entities.get("travel_dates") or "2025-06-01"
    errors = list(state.get("errors", []))
    tasks: list = []
    labels: list[str] = []

    # ── Queue transport booking if needed ─────────────────────────────────────
    if route in {ROUTE_FULL, ROUTE_TRANSPORT_BOOK} and state.get("selected_transport_type"):
        transport = state["selected_transport"]
        t_req = BookingRequest(
            booking_type=state["selected_transport_type"],
            origin=origin,
            destination=dest,
            travel_date=travel_date,
            offer_id=transport.get("offer_id", ""),
            train_number=transport.get("train_number", ""),
            train_class=memory.get("preferred_train_class", "3A"),
            bus_id=transport.get("bus_id", ""),
            seat_numbers=memory.get("preferred_seats", ["L1"]),
            boarding_point_id=(transport.get("boarding_points") or [{}])[0].get("id", ""),
            dropping_point_id=(transport.get("dropping_points") or [{}])[0].get("id", ""),
            passengers=passengers,
            quoted_price_inr=float(transport.get("price") or transport.get("fare_inr") or 0),
            contact_phone=contact_phone,
            contact_email=contact_email,
            traveler_profile=traveler,
        )
        tasks.append(run_booking_agent(t_req))
        labels.append("transport")

    # ── Queue hotel booking if needed ─────────────────────────────────────────
    if route in {ROUTE_FULL, ROUTE_HOTEL_BOOK} and state.get("selected_hotel_offer_id"):
        h_req = BookingRequest(
            booking_type="hotel",
            offer_id=state["selected_hotel_offer_id"],
            check_in=entities.get("check_in", travel_date),
            check_out=entities.get("check_out", ""),
            passengers=[{
                "firstName": "Traveler", "lastName": "User",
                "email": contact_email, "phone": contact_phone,
            }],
            quoted_price_inr=state.get("selected_hotel_price_inr", 0),
            contact_phone=contact_phone,
            contact_email=contact_email,
        )
        tasks.append(run_booking_agent(h_req))
        labels.append("hotel")

    if not tasks:
        return {**state, "errors": errors + ["Nothing to book — no valid selections"]}

    # ── Run all queued bookings concurrently ──────────────────────────────────
    raw = await asyncio.gather(*tasks, return_exceptions=True)
    transport_detail: dict = {}
    hotel_detail: dict = {}
    payment_orders: list = []

    for label, result in zip(labels, raw):
        if isinstance(result, Exception):
            errors.append(f"{label} booking error: {result}")
            continue
        if not result.success:
            errors.append(f"{label} booking failed: {result.error}")
            continue
        if label == "transport":
            transport_detail = result.details
            payment_orders.append(result.payment_order)
        elif label == "hotel":
            hotel_detail = result.details
            payment_orders.append(result.payment_order)

    if transport_detail or hotel_detail:
        update_memory(state["user_id"], {
            "last_booking": {"transport": transport_detail, "hotel": hotel_detail, "destination": dest}
        })

    return {**state,
            "transport_booking": transport_detail,
            "hotel_booking":     hotel_detail,
            "payment_orders":    payment_orders,
            "errors":            errors}


# ─── Conditional routing functions ────────────────────────────────────────────

def _route_after_decide(state: TravelState) -> str:
    if state["route_plan"] == ROUTE_AVAILABILITY_ONLY:
        return "check_availability"
    return "plan_itinerary"


def _route_after_confirm(state: TravelState) -> str:
    if state.get("booking_needed"):
        return "execute_bookings"
    return END


# ─── Graph assembly ───────────────────────────────────────────────────────────

def build_supervisor() -> StateGraph:
    sg = StateGraph(TravelState)

    sg.add_node("parse_query",        parse_query)
    sg.add_node("validate_slots",     validate_slots)
    sg.add_node("decide_route",       decide_route)
    sg.add_node("check_availability", check_availability)
    sg.add_node("plan_itinerary",     plan_itinerary)
    sg.add_node("recommend_hotels",   recommend_hotels)
    sg.add_node("find_transport",     find_transport)
    sg.add_node("human_confirmation", human_confirmation)
    sg.add_node("execute_bookings",   execute_bookings)

    sg.set_entry_point("parse_query")
    sg.add_edge("parse_query",    "validate_slots")
    sg.add_edge("validate_slots", "decide_route")

    # Branch: availability-only skips all planning nodes
    sg.add_conditional_edges("decide_route", _route_after_decide, {
        "check_availability": "check_availability",
        "plan_itinerary":     "plan_itinerary",
    })

    sg.add_edge("check_availability", END)
    sg.add_edge("plan_itinerary",     "recommend_hotels")
    sg.add_edge("recommend_hotels",   "find_transport")
    sg.add_edge("find_transport",     "human_confirmation")

    # Branch: search-only routes end here; booking routes continue
    sg.add_conditional_edges("human_confirmation", _route_after_confirm, {
        "execute_bookings": "execute_bookings",
        END: END,
    })

    sg.add_edge("execute_bookings", END)

    return sg.compile()
