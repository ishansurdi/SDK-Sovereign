"""Utility: rewrite all 9 repo fixture files to exact PRD content."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ── payments_repo ─────────────────────────────────────────────────────────────

(ROOT / "server/repos/payments_repo/broken.py").write_text(
    "import stripe\n"
    "\n"
    'stripe.api_key = "sk_test_REPLACE_ME"\n'
    "\n"
    "\n"
    "def charge_customer(amount_inr: int, customer_id: str) -> dict:\n"
    '    """Charge an INR amount to a customer. Returns {id, status}."""\n'
    "    charge = stripe.Charge.create(\n"
    "        amount=amount_inr * 100,   # paise\n"
    '        currency="inr",\n'
    "        customer=customer_id,\n"
    "    )\n"
    '    return {"id": charge.id, "status": charge.status}\n',
    encoding="utf-8",
)

(ROOT / "server/repos/payments_repo/tests.json").write_text(
    json.dumps(
        {
            "test_basic_charge": {
                "args": [100, "cust_001"],
                "expected": {
                    "id": {"type": "str", "contains": "cust_001"},
                    "status": {"type": "str"},
                },
            },
            "test_zero_amount": {
                "args": [0, "cust_002"],
                "expected": {
                    "status": {"type": "str", "equals": "failed"},
                },
            },
            "test_large_amount": {
                "args": [50000, "cust_003"],
                "expected": {
                    "id": {"type": "str", "contains": "5000000"},
                    "status": {"type": "str"},
                },
            },
        },
        indent=2,
    ),
    encoding="utf-8",
)

(ROOT / "server/repos/payments_repo/meta.json").write_text(
    json.dumps(
        {
            "repo_id": "payments_repo",
            "deprecated_sdk": "stripe",
            "ground_truth_replacement": "razorpay",
            "category": "payments",
            "entrypoint": "charge_customer",
            "error_log": (
                "ImportError: stripe SDK suspended for IN region "
                "(sanctions notice 2026-04). All inbound traffic blocked at gateway."
            ),
        },
        indent=2,
    ),
    encoding="utf-8",
)

# ── maps_repo ─────────────────────────────────────────────────────────────────

(ROOT / "server/repos/maps_repo/broken.py").write_text(
    "import googlemaps\n"
    "\n"
    "\n"
    "def address_to_coords(address: str) -> dict:\n"
    '    """Geocode an address. Returns {lat, lng}."""\n'
    '    client = googlemaps.Client(key="GMAPS_KEY")\n'
    "    result = client.geocode(address)\n"
    '    loc = result[0]["geometry"]["location"]\n'
    '    return {"lat": loc["lat"], "lng": loc["lng"]}\n',
    encoding="utf-8",
)

(ROOT / "server/repos/maps_repo/tests.json").write_text(
    json.dumps(
        {
            "test_bangalore": {
                "args": ["MG Road, Bangalore"],
                "expected": {"lat": {"type": "float"}, "lng": {"type": "float"}},
            },
            "test_mumbai": {
                "args": ["Marine Drive, Mumbai"],
                "expected": {"lat": {"type": "float"}, "lng": {"type": "float"}},
            },
            "test_chennai": {
                "args": ["Marina Beach, Chennai"],
                "expected": {"lat": {"type": "float"}, "lng": {"type": "float"}},
            },
        },
        indent=2,
    ),
    encoding="utf-8",
)

(ROOT / "server/repos/maps_repo/meta.json").write_text(
    json.dumps(
        {
            "repo_id": "maps_repo",
            "deprecated_sdk": "googlemaps",
            "ground_truth_replacement": "mmi_sdk",
            "category": "maps",
            "entrypoint": "address_to_coords",
            "error_log": (
                "googlemaps.exceptions.ApiError: 403 \u2014 Maps Platform key "
                "revoked for IN region per regulatory action 2026-Q2."
            ),
        },
        indent=2,
    ),
    encoding="utf-8",
)

# ── comms_repo ────────────────────────────────────────────────────────────────

(ROOT / "server/repos/comms_repo/broken.py").write_text(
    "from twilio.rest import Client\n"
    "\n"
    "\n"
    "def send_otp(phone: str, code: str) -> dict:\n"
    '    """Send a one-time password via SMS. Returns {sid, status}."""\n'
    '    client = Client("ACxxx", "auth_xxx")\n'
    "    msg = client.messages.create(\n"
    '        to=phone, from_="+1555TWILIO", body=f"OTP: {code}"\n'
    "    )\n"
    '    return {"sid": msg.sid, "status": msg.status}\n',
    encoding="utf-8",
)

(ROOT / "server/repos/comms_repo/tests.json").write_text(
    json.dumps(
        {
            "test_indian_number": {
                "args": ["+919876543210", "123456"],
                "expected": {"sid": {"type": "str"}, "status": {"type": "str"}},
            },
            "test_short_code": {
                "args": ["+919999000011", "00000"],
                "expected": {"sid": {"type": "str"}},
            },
            "test_long_otp": {
                "args": ["+918000000001", "987654"],
                "expected": {"status": {"type": "str"}},
            },
        },
        indent=2,
    ),
    encoding="utf-8",
)

(ROOT / "server/repos/comms_repo/meta.json").write_text(
    json.dumps(
        {
            "repo_id": "comms_repo",
            "deprecated_sdk": "twilio",
            "ground_truth_replacement": "kaleyra",
            "category": "messaging",
            "entrypoint": "send_otp",
            "error_log": (
                "TwilioRestException: HTTP 451 Unavailable For Legal Reasons. "
                "Cross-border SMS gateway disabled."
            ),
        },
        indent=2,
    ),
    encoding="utf-8",
)

print("All 9 fixture files written successfully.")
