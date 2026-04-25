"""Rule-based agents used for deterministic smoke tests and demos."""
from __future__ import annotations

import re
from typing import Optional

from models import ActionType, Role, SDKAction, SDKObservation


def auditor_rule_agent(obs: SDKObservation) -> SDKAction:
	"""Approve allowlisted proposals and reject everything else."""
	if obs.current_proposal is None:
		return SDKAction(
			role=Role.AUDITOR.value,
			action_type=ActionType.PASS.value,
			reasoning="Waiting for Lead to propose a replacement.",
		)
	if obs.visible_allowlist and obs.current_proposal in obs.visible_allowlist:
		return SDKAction(
			role=Role.AUDITOR.value,
			action_type=ActionType.APPROVE.value,
			reasoning=f"{obs.current_proposal} is on the sovereign allow-list.",
		)
	return SDKAction(
		role=Role.AUDITOR.value,
		action_type=ActionType.REJECT.value,
		rejection_reason=f"{obs.current_proposal} not on allow-list.",
		reasoning="Per sovereignty registry, this SDK is not approved.",
	)


_PATCH_TEMPLATES = {
	"razorpay": {
		"stripe": (
			'import razorpay\n\n'
			'_client = razorpay.Client(auth=("key", "secret"))\n\n'
			'def charge_customer(amount_inr: int, customer_id: str) -> dict:\n'
			'    payment = _client.payment.create({\n'
			'        "amount": amount_inr * 100,\n'
			'        "currency": "INR",\n'
			'        "customer_id": customer_id,\n'
			'    })\n'
			'    return {"id": payment["id"], "status": payment["status"]}\n'
		)
	},
	"mmi_sdk": {
		"googlemaps": (
			'import mmi_sdk\n\n'
			'def address_to_coords(address: str) -> dict:\n'
			'    client = mmi_sdk.Client(api_key="MMI_KEY")\n'
			'    loc = client.get_location(address)\n'
			'    return {"lat": loc["lat"], "lng": loc["lng"]}\n'
		)
	},
	"kaleyra": {
		"twilio": (
			'import kaleyra\n\n'
			'def send_otp(phone: str, code: str) -> dict:\n'
			'    client = kaleyra.Client(api_key="KLR_KEY")\n'
			'    resp = client.send_sms(to=phone, sender="OTP", message=f"OTP: {code}")\n'
			'    return {"sid": resp["message_id"], "status": resp["status"]}\n'
		)
	},
}

_GROUND_TRUTH = {"stripe": "razorpay", "googlemaps": "mmi_sdk", "twilio": "kaleyra"}


def lead_rule_agent(obs: SDKObservation) -> SDKAction:
	"""Propose and then submit a deterministic migration patch."""
	history = obs.conversation_history
	last_auditor = next((item for item in reversed(history) if item.get("role") == "auditor"), None)

	if obs.approved_replacement and last_auditor and last_auditor.get("action_type") == "approve":
		deprecated = _detect_deprecated(obs.visible_codebase or "")
		template = _PATCH_TEMPLATES.get(obs.approved_replacement, {}).get(deprecated)
		if template is None:
			template = (obs.visible_codebase or "").replace(
				f"import {deprecated}",
				f"import {obs.approved_replacement}",
			)
		return SDKAction(
			role=Role.LEAD.value,
			action_type=ActionType.SUBMIT_PATCH.value,
			patched_code=template,
			reasoning=f"Submitting migration to {obs.approved_replacement}.",
		)

	deprecated = _detect_deprecated(obs.visible_codebase or "")
	candidate = _GROUND_TRUTH.get(deprecated, "razorpay")
	return SDKAction(
		role=Role.LEAD.value,
		action_type=ActionType.PROPOSE_REPLACEMENT.value,
		proposed_sdk=candidate,
		reasoning=(
			f"The codebase imports {deprecated}, which is the sanctioned SDK. "
			f"Proposing {candidate} as the sovereign replacement."
		),
	)


def _detect_deprecated(code: str) -> Optional[str]:
	"""Infer deprecated SDK name from imports in a code string."""
	for sdk in ("stripe", "googlemaps", "twilio"):
		if re.search(rf"\bimport\s+{sdk}\b|\bfrom\s+{sdk}", code):
			return sdk
	return None


def get_rule_agent(role: str):
	"""Return the callable rule agent for a given role."""
	return {"auditor": auditor_rule_agent, "lead": lead_rule_agent}[role]
