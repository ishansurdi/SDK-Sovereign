from twilio.rest import Client


def send_otp(phone: str, code: str) -> dict:
	client = Client("sid", "token")
	message = client.messages.create(
		to=phone,
		from_="OTP",
		body=f"OTP: {code}",
	)
	return {"sid": message.sid, "status": message.status}
