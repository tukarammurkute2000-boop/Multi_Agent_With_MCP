"""
Train booking via RailAPI (Indian Railways).
API docs: https://rapidapi.com/indianrailways/api/indian-railway1
Endpoint base: https://indian-railway1.p.rapidapi.com

Covers:
  - Search trains between two stations
  - Check seat availability by class
  - Get PNR status
  - Book ticket (creates a booking record; actual IRCTC booking requires IRCTC credentials)
"""
import httpx
from config.settings import get_settings

settings = get_settings()

_HEADERS = {
    "X-RapidAPI-Key": settings.railapi_key,
    "X-RapidAPI-Host": "indian-railway1.p.rapidapi.com",
}
_BASE = settings.railapi_base_url


class TrainBookingError(Exception):
    pass


# ─── Station code lookup ──────────────────────────────────────────────────────

_STATION_MAP = {
    "mumbai": "CSTM",       # Mumbai CST
    "mumbai central": "BCT",
    "delhi": "NDLS",
    "new delhi": "NDLS",
    "goa": "MAO",           # Madgaon
    "bangalore": "SBC",
    "chennai": "MAS",
    "hyderabad": "SC",
    "kolkata": "HWH",
    "jaipur": "JP",
    "pune": "PUNE",
    "ahmedabad": "ADI",
    "surat": "ST",
    "nagpur": "NGP",
    "bhopal": "BPL",
}


def city_to_station(city: str) -> str:
    return _STATION_MAP.get(city.lower(), city.upper()[:4])


# ─── Train search ─────────────────────────────────────────────────────────────

async def search_trains(
    origin: str,
    destination: str,
    travel_date: str,    # YYYY-MM-DD
) -> list[dict]:
    """
    Returns list of trains with name, number, departure, arrival, duration, classes.
    origin/destination: station codes (e.g. "CSTM", "MAO") or city names.
    """
    origin_code = city_to_station(origin)
    dest_code = city_to_station(destination)
    date_fmt = travel_date.replace("-", "")  # YYYYMMDD for RailAPI

    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            f"{_BASE}/trains/betweenstations",
            headers=_HEADERS,
            params={
                "fromStationCode": origin_code,
                "toStationCode": dest_code,
                "dateOfJourney": date_fmt,
            },
        )
        r.raise_for_status()
        trains = r.json().get("data", [])

    return [_normalize_train(t) for t in trains]


def _normalize_train(t: dict) -> dict:
    return {
        "train_number": t.get("train_number", ""),
        "train_name": t.get("train_name", ""),
        "origin_code": t.get("from", ""),
        "destination_code": t.get("to", ""),
        "departure": t.get("depart_time", ""),
        "arrival": t.get("arrive_time", ""),
        "duration": t.get("duration", ""),
        "classes": t.get("class_type", []),
        "runs_on": t.get("run_days", []),
        "raw": t,
    }


# ─── Seat availability ────────────────────────────────────────────────────────

async def check_availability(
    train_number: str,
    origin_code: str,
    destination_code: str,
    travel_date: str,
    travel_class: str = "3A",   # SL, 3A, 2A, 1A, CC, EC
) -> dict:
    """
    Returns availability and fare for a specific class.
    """
    date_fmt = travel_date.replace("-", "")
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            f"{_BASE}/trains/availability",
            headers=_HEADERS,
            params={
                "trainNumber": train_number,
                "fromStationCode": origin_code,
                "toStationCode": destination_code,
                "dateOfJourney": date_fmt,
                "classType": travel_class,
            },
        )
        r.raise_for_status()
        data = r.json().get("data", {})

    return {
        "train_number": train_number,
        "class": travel_class,
        "availability": data.get("availability", "UNKNOWN"),
        "fare_inr": float(data.get("fare", 0)),
        "available_seats": data.get("available_seats", 0),
    }


# ─── PNR status ───────────────────────────────────────────────────────────────

async def get_pnr_status(pnr: str) -> dict:
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            f"{_BASE}/pnr/status",
            headers=_HEADERS,
            params={"pnrNumber": pnr},
        )
        r.raise_for_status()
        return r.json().get("data", {})


# ─── Book ticket ──────────────────────────────────────────────────────────────

async def book_train_ticket(
    train_number: str,
    origin_code: str,
    destination_code: str,
    travel_date: str,
    travel_class: str,
    passengers: list[dict],   # [{name, age, gender, id_type, id_number}]
    contact_phone: str,
    contact_email: str,
) -> dict:
    """
    Validates availability then creates a booking record.
    In production this integrates with IRCTC agent API or a licensed partner.

    Returns: {booking_id, pnr, train_number, class, passengers, fare_inr, status}
    """
    if not passengers:
        raise TrainBookingError("At least one passenger required")
    if len(passengers) > 6:
        raise TrainBookingError("Maximum 6 passengers per booking")

    # Validate every passenger has required ID
    for p in passengers:
        if not p.get("id_number"):
            raise TrainBookingError(f"Government ID missing for passenger {p.get('name', '?')}")
        if not (1 <= int(p.get("age", 0)) <= 120):
            raise TrainBookingError(f"Invalid age for passenger {p.get('name', '?')}")

    # Check live availability
    avail = await check_availability(train_number, origin_code, destination_code,
                                     travel_date, travel_class)
    if avail["availability"] in ("NOT AVAILABLE", "REGRET"):
        raise TrainBookingError(
            f"No seats in {travel_class} on train {train_number} for {travel_date}"
        )

    # Seat availability check passes → call booking endpoint
    payload = {
        "trainNumber": train_number,
        "fromStation": origin_code,
        "toStation": destination_code,
        "dateOfJourney": travel_date.replace("-", ""),
        "classType": travel_class,
        "passengers": passengers,
        "contactPhone": contact_phone,
        "contactEmail": contact_email,
        "quota": "GN",          # General quota
    }

    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post(
            f"{_BASE}/trains/book",
            headers={**_HEADERS, "Content-Type": "application/json"},
            json=payload,
        )
        r.raise_for_status()
        data = r.json().get("data", {})

    total_fare = avail["fare_inr"] * len(passengers)
    return {
        "booking_id": data.get("booking_id", ""),
        "pnr": data.get("pnr", ""),
        "train_number": train_number,
        "train_name": data.get("train_name", ""),
        "class": travel_class,
        "from": origin_code,
        "to": destination_code,
        "travel_date": travel_date,
        "passengers": len(passengers),
        "fare_per_passenger_inr": avail["fare_inr"],
        "total_fare_inr": total_fare,
        "status": data.get("status", "CONFIRMED"),
    }
