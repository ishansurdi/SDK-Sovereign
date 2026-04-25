from twilio.rest import Client


def send_otp(phone: str, code: str) -> dict:
    """Send a one-time password via SMS. Returns {sid, status}."""
    client = Client("ACxxx", "auth_xxx")
    msg = client.messages.create(
        to=phone, from_="+1555TWILIO", body=f"OTP: {code}"
    )
    return {"sid": msg.sid, "status": msg.status}
