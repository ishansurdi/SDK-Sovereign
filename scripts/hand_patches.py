"""Golden 'good' patches for each repo. Run them through the verifier;
all 9 tests (3 per repo × 3 repos) MUST pass."""
from __future__ import annotations
from pathlib import Path
from server.verifier import Verifier


GOOD_PATCHES = {
    "payments_repo": '''
import razorpay

_client = razorpay.Client(auth=("key", "secret"))

def charge_customer(amount_inr: int, customer_id: str) -> dict:
    payment = _client.payment.create({
        "amount": amount_inr * 100,
        "currency": "INR",
        "customer_id": customer_id,
    })
    return {"id": payment["id"], "status": payment["status"]}
''',
    "maps_repo": '''
import mmi_sdk

def address_to_coords(address: str) -> dict:
    client = mmi_sdk.Client(api_key="MMI_KEY")
    loc = client.get_location(address)
    return {"lat": loc["lat"], "lng": loc["lng"]}
''',
    "comms_repo": '''
import kaleyra

def send_otp(phone: str, code: str) -> dict:
    client = kaleyra.Client(api_key="KLR_KEY")
    resp = client.send_sms(to=phone, sender="OTP", message=f"OTP: {code}")
    return {"sid": resp["message_id"], "status": resp["status"]}
''',
}


def main() -> int:
    """Run all golden patches through verifier and report results."""
    repos_root = Path(__file__).resolve().parent.parent / "server" / "repos"
    v = Verifier(repos_root)
    all_pass = True
    for repo_id, patch in GOOD_PATCHES.items():
        results = v.run_parity_tests(patch, repo_id)
        ok = all(results.values())
        all_pass &= ok
        print(f"{repo_id}: {results} → {'PASS' if ok else 'FAIL'}")
    print(f"\nOverall: {'ALL GOOD' if all_pass else 'BROKEN'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
