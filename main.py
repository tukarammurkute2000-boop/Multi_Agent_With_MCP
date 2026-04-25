"""
Entry point — demonstrates all dynamic routing paths.
Usage: python main.py
"""
import asyncio
import json
import structlog
from agents.supervisor import build_supervisor, TravelState

log = structlog.get_logger()


def _empty_state(user_id: str, query: str) -> TravelState:
    return TravelState(
        user_id=user_id,
        raw_query=query,
        parsed={},
        memory={},
        route_plan="",
        itinerary={},
        hotel_recommendations={},
        flight_options=[],
        train_options=[],
        bus_options=[],
        availability_results={},
        selected_transport_type="",
        selected_transport={},
        selected_hotel_offer_id="",
        selected_hotel_price_inr=0.0,
        booking_needed=False,
        transport_booking={},
        hotel_booking={},
        payment_orders=[],
        confirmed=False,
        errors=[],
    )


async def run(user_id: str, query: str) -> dict:
    supervisor = build_supervisor()
    initial = _empty_state(user_id, query)
    log.info("pipeline.start", user=user_id, query=query)
    result = await supervisor.ainvoke(initial)
    log.info("pipeline.done", route=result.get("route_plan"))
    return result


def _section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def _print_result(result: dict) -> None:
    route = result.get("route_plan", "unknown")
    _section(f"ROUTE: {route.upper()}")

    # ── Availability results ──────────────────────────────────────────────────
    avail = result.get("availability_results", {})
    if avail:
        _section("AVAILABILITY RESULTS")
        for res_type, data in avail.items():
            print(f"  [{res_type.upper()}]")
            print(f"    Available : {data.get('available')}")
            print(f"    Count     : {data.get('count', 0)}")
            if data.get("availability_status"):
                print(f"    Status    : {data['availability_status']}")
                print(f"    Fare      : ₹{data.get('fare_inr', 0)}")
                print(f"    Seats     : {data.get('available_seats', 0)}")
            for opt in data.get("options", [])[:3]:
                name = opt.get("train_name") or opt.get("flight_number") or opt.get("operator") or opt.get("name", "")
                price = opt.get("fare_inr") or opt.get("price") or opt.get("price_per_night_inr", "")
                print(f"    • {name}  ₹{price}")
        return   # availability path ends here

    # ── Itinerary ─────────────────────────────────────────────────────────────
    itin = result.get("itinerary", {})
    if itin:
        _section("ITINERARY")
        if isinstance(itin, dict) and itin.get("days"):
            for day in itin["days"]:
                print(f"  Day {day['day']} — {day.get('theme', '')}")
                for slot in ("morning", "afternoon", "evening"):
                    act = day.get(slot, {})
                    if act.get("activity"):
                        print(f"    {slot.capitalize()}: {act['activity']} @ {act.get('location', '')}")
        else:
            print(json.dumps(itin, indent=2)[:800])

    # ── Hotel recommendations ─────────────────────────────────────────────────
    hotels = result.get("hotel_recommendations", {})
    if hotels.get("recommendations"):
        _section("HOTEL RECOMMENDATIONS")
        for r in hotels["recommendations"][:3]:
            flag = "✓" if r.get("within_budget") else "✗ over budget"
            print(f"  [{r.get('rank', '?')}] {r.get('name', 'N/A')} — ₹{r.get('price_per_night_inr', '?')}/night ({flag})")
        if hotels.get("budget_warning"):
            print(f"  Warning  : {hotels['budget_warning']}")
        if hotels.get("budget_suggestion"):
            print(f"  Tip      : {hotels['budget_suggestion']}")

    # ── Transport options (search-only routes) ────────────────────────────────
    for mode, key in [("FLIGHTS", "flight_options"), ("TRAINS", "train_options"), ("BUSES", "bus_options")]:
        opts = result.get(key, [])
        if opts and not result.get("transport_booking"):
            _section(f"{mode} FOUND")
            for o in opts[:3]:
                name = o.get("flight_number") or o.get("train_name") or o.get("operator", "")
                price = o.get("price") or o.get("fare_inr", "")
                print(f"  • {name}  ₹{price}  {o.get('departure', '')} → {o.get('arrival', '')}")

    # ── Booking results ───────────────────────────────────────────────────────
    tb = result.get("transport_booking", {})
    if tb:
        _section("TRANSPORT BOOKING CONFIRMED")
        print(f"  Type      : {result.get('selected_transport_type', '').upper()}")
        print(f"  Booking ID: {tb.get('booking_id')}")
        print(f"  PNR       : {tb.get('pnr') or tb.get('ticket_number', 'N/A')}")
        print(f"  Status    : {tb.get('status')}")
        print(f"  Fare      : ₹{tb.get('total_fare_inr') or tb.get('fare_inr', 'N/A')}")

    hb = result.get("hotel_booking", {})
    if hb:
        _section("HOTEL BOOKING CONFIRMED")
        print(f"  Hotel     : {hb.get('hotel_name')}")
        print(f"  Booking ID: {hb.get('booking_id')}")
        print(f"  Dates     : {hb.get('check_in')} → {hb.get('check_out')}")
        print(f"  Status    : {hb.get('status')}")
        print(f"  Total     : ₹{hb.get('total_price_inr')}")

    po = result.get("payment_orders", [])
    if po:
        _section("PAYMENT ORDERS")
        for o in po:
            print(f"  {o.get('order_id')}  ₹{int(o.get('amount', 0)) // 100}  [{o.get('status')}]")

    if result.get("errors"):
        _section("ERRORS")
        for e in result["errors"]:
            print(f"  • {e}")


# ─── Demo queries — each exercises a different route ─────────────────────────

DEMO_QUERIES = [
    # Full pipeline
    ("user_001", "Plan a 6-day Goa trip and book flights from Mumbai, budget 40000 INR"),
    # Hotel only — search + book
    ("user_002", "Book a hotel in Goa for 5 nights check-in 2025-06-10 check-out 2025-06-15 budget 8000 per night"),
    # Train availability check — read-only
    ("user_003", "Check seat availability on train 12051 from Mumbai to Goa on 2025-06-10 for 3A class"),
    # Transport only — search flights, no booking
    ("user_004", "Search flights from Delhi to Bangalore on 2025-07-01"),
    # Trip plan only
    ("user_005", "Give me a 4-day itinerary for Jaipur"),
]


if __name__ == "__main__":
    import sys
    # Run first demo query by default; pass index as arg to try others
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    user_id, query = DEMO_QUERIES[idx]
    print(f"Query: {query!r}")
    result = asyncio.run(run(user_id, query))
    _print_result(result)
