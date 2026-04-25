"""
Flight search + booking via Amadeus API.
"""
import httpx
from config.settings import get_settings

settings = get_settings()


async def _get_token() -> str:
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


async def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    adults: int = 1,
    max_results: int = 10,
) -> list[dict]:
    """
    Search available flights.
    origin/destination: IATA airport codes (e.g. "BOM", "GOI").
    departure_date: "YYYY-MM-DD".
    """
    token = await _get_token()
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{settings.amadeus_base_url}/v2/shopping/flight-offers",
            headers={"Authorization": f"Bearer {token}"},
            params={
                "originLocationCode": origin,
                "destinationLocationCode": destination,
                "departureDate": departure_date,
                "adults": adults,
                "max": max_results,
                "currencyCode": settings.razorpay_currency,
            },
        )
        resp.raise_for_status()
        offers = resp.json().get("data", [])
        return [_normalize_offer(o) for o in offers]


def _normalize_offer(offer: dict) -> dict:
    seg = offer["itineraries"][0]["segments"][0]
    return {
        "offer_id": offer["id"],
        "airline": seg.get("carrierCode", ""),
        "flight_number": seg.get("number", ""),
        "departure": seg["departure"]["at"],
        "arrival": seg["arrival"]["at"],
        "duration": offer["itineraries"][0]["duration"],
        "price": offer["price"]["total"],
        "currency": offer["price"]["currency"],
        "raw": offer,
    }


async def book_flight(offer: dict, traveler: dict) -> dict:
    """
    Confirm a flight booking.
    traveler: {"id": "1", "name": {"firstName": ..., "lastName": ...},
                "contact": {"emailAddress": ...}, "documents": [...]}
    """
    token = await _get_token()
    payload = {
        "data": {
            "type": "flight-order",
            "flightOffers": [offer["raw"]],
            "travelers": [traveler],
        }
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.amadeus_base_url}/v1/booking/flight-orders",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return {
            "booking_id": data["id"],
            "pnr": data.get("associatedRecords", [{}])[0].get("reference", ""),
            "status": "CONFIRMED",
            "flight": offer,
        }
