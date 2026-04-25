import googlemaps


def address_to_coords(address: str) -> dict:
    """Geocode an address. Returns {lat, lng}."""
    client = googlemaps.Client(key="GMAPS_KEY")
    result = client.geocode(address)
    loc = result[0]["geometry"]["location"]
    return {"lat": loc["lat"], "lng": loc["lng"]}
