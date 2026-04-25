"""
Payment integration via Razorpay.
Flow: create_order → frontend collects card/UPI → verify_payment
"""
import hmac
import hashlib
import razorpay
from config.settings import get_settings

settings = get_settings()

_rz = razorpay.Client(
    auth=(settings.razorpay_key_id, settings.razorpay_key_secret)
)


def create_order(amount_inr: float, receipt: str, notes: dict = None) -> dict:
    """
    Create a Razorpay order.
    amount_inr: amount in Indian Rupees (converted to paise internally).
    receipt: your internal booking/transaction ID.
    """
    order = _rz.order.create(
        {
            "amount": int(amount_inr * 100),  # paise
            "currency": settings.razorpay_currency,
            "receipt": receipt,
            "notes": notes or {},
            "payment_capture": 1,  # auto-capture
        }
    )
    return {
        "order_id": order["id"],
        "amount": order["amount"],
        "currency": order["currency"],
        "status": order["status"],
        "razorpay_key": settings.razorpay_key_id,
    }


def verify_payment(order_id: str, payment_id: str, signature: str) -> bool:
    """Verify Razorpay webhook signature to prevent fraud."""
    expected = hmac.new(
        settings.razorpay_key_secret.encode(),
        f"{order_id}|{payment_id}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def fetch_payment(payment_id: str) -> dict:
    return _rz.payment.fetch(payment_id)


def refund_payment(payment_id: str, amount_inr: float = None) -> dict:
    """Full refund if amount_inr is None, else partial refund."""
    params = {}
    if amount_inr:
        params["amount"] = int(amount_inr * 100)
    return _rz.payment.refund(payment_id, params)
