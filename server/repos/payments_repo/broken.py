import stripe

stripe.api_key = "sk_test_REPLACE_ME"


def charge_customer(amount_inr: int, customer_id: str) -> dict:
    """Charge an INR amount to a customer. Returns {id, status}."""
    charge = stripe.Charge.create(
        amount=amount_inr * 100,   # paise
        currency="inr",
        customer=customer_id,
    )
    return {"id": charge.id, "status": charge.status}
