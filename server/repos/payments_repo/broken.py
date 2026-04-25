import stripe

stripe.api_key = "sk_test_legacy"


def charge_customer(amount_inr: int, customer_id: str) -> dict:
	charge = stripe.Charge.create(
		amount=amount_inr * 100,
		currency="INR",
		customer=customer_id,
	)
	return {"id": charge.id, "status": charge.status}
