import googlemaps


def address_to_coords(address: str) -> dict:
	client = googlemaps.Client(key="legacy_key")
	result = client.geocode(address)
	location = result[0]["geometry"]["location"]
	return {"lat": location["lat"], "lng": location["lng"]}
