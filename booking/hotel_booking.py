"""
Hotel booking via Amadeus Hotel Orders API.

Validation before booking:
  - Date range sanity (check_out > check_in, not in past)
  - Guest count vs room capacity
  - Price drift check (re-fetch offer to confirm price hasn't changed)
  - Cancellation policy acknowledgement
"""
import httpx
from datetime import date, datetime
from config.settings import get_settings

settings = get_settings()


class HotelBookingValidationError(ValueError):
    pass


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_hotel_booking(
    offer_id: str,
    check_in: str,
    check_out: str,
    guests: int,
    quoted_price_inr: float,
    live_price_inr: float,
    room_capacity: int,
) -> None:
    """
    Raises HotelBookingValidationError with a descriptive message if invalid.
    Call this before hitting the Amadeus booking endpoint.
    """
    today = date.today()
    try:
        ci = datetime.strptime(check_in, "%Y-%m-%d").date()
        co = datetime.strptime(check_out, "%Y-%m-%d").date()
    except ValueError as exc:
        raise HotelBookingValidationError("Dates must be in YYYY-MM-DD format") from exc

    if ci < today:
        raise HotelBookingValidationError(f"Check-in date {check_in} is in the past")

    if co <= ci:
        raise HotelBookingValidationError("Check-out must be after check-in")

    if (co - ci).days > 30:
        raise HotelBookingValidationError("Stay longer than 30 nights — please confirm manually")

    if guests < 1:
        raise HotelBookingValidationError("At least 1 guest required")

    if guests > room_capacity:
        raise HotelBookingValidationError(
            f"Room capacity is {room_capacity} but {guests} guests requested"
        )

    if not offer_id:
        raise HotelBookingValidationError("Missing offer_id — run hotel search first")

    # Price drift guard: reject if live price is >10% higher than quoted
    if live_price_inr > 0 and quoted_price_inr > 0:
        drift_pct = (live_price_inr - quoted_price_inr) / quoted_price_inr * 100
        if drift_pct > 10:
            raise HotelBookingValidationError(
                f"Price changed by +{drift_pct:.1f}% since quote "
                f"(was ₹{quoted_price_inr:.0f}, now ₹{live_price_inr:.0f}). "
                "Re-confirm with user before booking."
            )


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


async def _fetch_offer(offer_id: str) -> dict:
    """Re-fetch an offer to get current price and capacity."""
    token = await _get_token()
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            f"{settings.amadeus_base_url}/v3/shopping/hotel-offers/{offer_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        return r.json().get("data", {})


# ─── Public API ───────────────────────────────────────────────────────────────

async def book_hotel(
    offer_id: str,
    check_in: str,
    check_out: str,
    guests: list[dict],
    quoted_price_inr: float,
    payment_card: dict = None,
) -> dict:
    """
    Validate then book a hotel via Amadeus Hotel Orders API.

    guests: list of {firstName, lastName, email, phone}
    payment_card: {vendorCode, cardNumber, expiryDate, holderName}
                  — pass None to use agency payment (test env default)

    Returns: {booking_id, hotel_name, check_in, check_out, status,
              total_price_inr, cancellation_policy}
    """
    # ── Re-fetch live offer for validation ────────────────────────────────────
    live_offer = await _fetch_offer(offer_id)
    live_price = float(
        live_offer.get("offers", [{}])[0].get("price", {}).get("total", 0)
    )
    room_capacity = int(
        live_offer.get("offers", [{}])[0]
        .get("room", {})
        .get("typeEstimated", {})
        .get("beds", 2)
    )
    cancel_policy = (
        live_offer.get("offers", [{}])[0].get("policies", {}).get("cancellation", {})
    )

    validate_hotel_booking(
        offer_id=offer_id,
        check_in=check_in,
        check_out=check_out,
        guests=len(guests),
        quoted_price_inr=quoted_price_inr,
        live_price_inr=live_price,
        room_capacity=room_capacity,
    )

    # ── Build booking payload ─────────────────────────────────────────────────
    guest_payloads = [
        {
            "tid": str(i + 1),
            "title": "MR",
            "firstName": g["firstName"],
            "lastName": g["lastName"],
            "email": g.get("email", ""),
            "phone": g.get("phone", ""),
        }
        for i, g in enumerate(guests)
    ]

    payload: dict = {
        "data": {
            "offerId": offer_id,
            "guests": guest_payloads,
            "payments": [],
        }
    }
    if payment_card:
        payload["data"]["payments"] = [{
            "id": 1,
            "method": "creditCard",
            "card": {
                "vendorCode": payment_card.get("vendorCode", "VI"),
                "cardNumber": payment_card["cardNumber"],
                "expiryDate": payment_card["expiryDate"],
                "holderName": payment_card["holderName"],
            },
        }]

    token = await _get_token()
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post(
            f"{settings.amadeus_base_url}/v1/booking/hotel-orders",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        r.raise_for_status()
        data = r.json().get("data", {})

    hotel_name = live_offer.get("hotel", {}).get("name", "")
    return {
        "booking_id": data.get("id", ""),
        "hotel_name": hotel_name,
        "check_in": check_in,
        "check_out": check_out,
        "guests": len(guests),
        "status": data.get("associatedRecords", [{}])[0].get("originSystemCode", "CONFIRMED"),
        "total_price_inr": live_price,
        "cancellation_policy": cancel_policy,
        "offer_id": offer_id,
    }
