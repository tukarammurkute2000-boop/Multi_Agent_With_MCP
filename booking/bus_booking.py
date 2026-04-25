"""
Bus booking via RedBus API (RapidAPI).
API: https://rapidapi.com/redbus/api/redbus

Covers:
  - Search bus routes between cities
  - Check seat layout and availability
  - Select seats
  - Book ticket
"""
import httpx
from datetime import date, datetime
from config.settings import get_settings

settings = get_settings()

_HEADERS = {
    "X-RapidAPI-Key": settings.redbus_api_key,
    "X-RapidAPI-Host": "redbus.p.rapidapi.com",
}
_BASE = settings.redbus_base_url


class BusBookingError(Exception):
    pass


# ─── City ID lookup (RedBus uses numeric city IDs) ───────────────────────────

_CITY_IDS: dict[str, int] = {
    "mumbai": 122,
    "delhi": 77,
    "goa": 212,
    "bangalore": 2,
    "chennai": 3,
    "hyderabad": 5,
    "pune": 25,
    "jaipur": 72,
    "ahmedabad": 50,
    "kolkata": 4,
    "surat": 56,
    "nagpur": 40,
}


def city_to_id(city: str) -> int:
    cid = _CITY_IDS.get(city.lower())
    if cid is None:
        raise BusBookingError(f"Unknown city for bus booking: {city!r}. Add it to _CITY_IDS.")
    return cid


# ─── Search ───────────────────────────────────────────────────────────────────

async def search_buses(
    origin: str,
    destination: str,
    travel_date: str,   # YYYY-MM-DD
) -> list[dict]:
    """
    Returns list of available buses with operator, type, departure, arrival, fare, seats.
    """
    travel_date_obj = datetime.strptime(travel_date, "%Y-%m-%d").date()
    if travel_date_obj < date.today():
        raise BusBookingError(f"Travel date {travel_date} is in the past")

    origin_id = city_to_id(origin)
    dest_id = city_to_id(destination)

    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            f"{_BASE}/bus/search",
            headers=_HEADERS,
            params={
                "srcId": origin_id,
                "destId": dest_id,
                "doj": travel_date,   # YYYY-MM-DD
            },
        )
        r.raise_for_status()
        buses = r.json().get("buses", [])

    return [_normalize_bus(b) for b in buses]


def _normalize_bus(b: dict) -> dict:
    return {
        "bus_id": b.get("id", ""),
        "operator": b.get("travels", ""),
        "bus_type": b.get("busType", ""),   # AC Sleeper, Non-AC Seater, etc.
        "departure": b.get("departureTime", ""),
        "arrival": b.get("arrivalTime", ""),
        "duration": b.get("duration", ""),
        "fare_inr": float(b.get("fare", {}).get("minFare", 0)),
        "available_seats": int(b.get("availableSeats", 0)),
        "rating": float(b.get("rating", 0)),
        "boarding_points": b.get("boardingPoints", []),
        "dropping_points": b.get("droppingPoints", []),
        "raw": b,
    }


# ─── Seat layout ──────────────────────────────────────────────────────────────

async def get_seat_layout(bus_id: str, travel_date: str) -> dict:
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            f"{_BASE}/bus/seats",
            headers=_HEADERS,
            params={"busId": bus_id, "doj": travel_date},
        )
        r.raise_for_status()
        return r.json()


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_bus_booking(
    bus: dict,
    seats_requested: int,
    passengers: list[dict],
) -> None:
    if seats_requested < 1:
        raise BusBookingError("At least 1 seat required")

    if bus["available_seats"] < seats_requested:
        raise BusBookingError(
            f"Only {bus['available_seats']} seats available, {seats_requested} requested"
        )

    if len(passengers) != seats_requested:
        raise BusBookingError(
            f"Passenger count ({len(passengers)}) must match seats_requested ({seats_requested})"
        )

    for p in passengers:
        if not p.get("name"):
            raise BusBookingError("Passenger name is required")
        age = int(p.get("age", 0))
        if not (1 <= age <= 120):
            raise BusBookingError(f"Invalid age for passenger {p.get('name', '?')}: {age}")


# ─── Book ─────────────────────────────────────────────────────────────────────

async def book_bus(
    bus_id: str,
    travel_date: str,
    seat_numbers: list[str],          # e.g. ["L1", "U2"] (lower/upper)
    passengers: list[dict],           # [{name, age, gender, id_type, id_number}]
    boarding_point_id: str,
    dropping_point_id: str,
    contact_phone: str,
    contact_email: str,
) -> dict:
    """
    Validate then book the bus ticket.
    Returns: {booking_id, ticket_number, operator, departure, arrival,
              seats, fare_inr, status}
    """
    if not seat_numbers:
        raise BusBookingError("No seats selected")
    if len(seat_numbers) != len(passengers):
        raise BusBookingError("seat_numbers count must equal passengers count")

    # Re-check live seat availability before booking
    layout = await get_seat_layout(bus_id, travel_date)
    available = {
        s["seatNumber"]
        for s in layout.get("seats", [])
        if s.get("available")
    }
    unavailable = [s for s in seat_numbers if s not in available]
    if unavailable:
        raise BusBookingError(f"Seats no longer available: {unavailable}")

    payload = {
        "busId": bus_id,
        "doj": travel_date,
        "seats": [
            {
                "seatNumber": sn,
                "passenger": passengers[i],
            }
            for i, sn in enumerate(seat_numbers)
        ],
        "boardingPointId": boarding_point_id,
        "droppingPointId": dropping_point_id,
        "contactPhone": contact_phone,
        "contactEmail": contact_email,
    }

    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post(
            f"{_BASE}/bus/book",
            headers={**_HEADERS, "Content-Type": "application/json"},
            json=payload,
        )
        r.raise_for_status()
        data = r.json()

    return {
        "booking_id": data.get("bookingId", ""),
        "ticket_number": data.get("ticketNumber", ""),
        "bus_id": bus_id,
        "travel_date": travel_date,
        "seats": seat_numbers,
        "passengers": len(passengers),
        "fare_inr": data.get("totalFare", 0),
        "status": data.get("status", "CONFIRMED"),
        "boarding_point": boarding_point_id,
        "dropping_point": dropping_point_id,
    }
