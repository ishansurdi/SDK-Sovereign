"""Smoke loop over multiple episodes using a random policy."""
from __future__ import annotations

import random
import statistics

from models import SDKAction
from server.environment import SDKSovereignEnvironment


def _random_action(obs, rng: random.Random) -> SDKAction:
    """Sample a random valid action for the current role."""
    if obs.current_role == "auditor":
        choice = rng.choice(["pass", "approve", "reject"])
        if choice == "reject":
            return SDKAction(
                role="auditor",
                action_type="reject",
                rejection_reason="Random auditor rejection for exploration.",
                reasoning="Exploration policy.",
            )
        return SDKAction(role="auditor", action_type=choice, reasoning="Exploration policy.")

    choice = rng.choice(["pass", "propose_replacement", "submit_patch"])
    if choice == "propose_replacement":
        candidate = rng.choice(["razorpay", "mmi_sdk", "kaleyra", "unknown_sdk"])
        return SDKAction(
            role="lead",
            action_type="propose_replacement",
            proposed_sdk=candidate,
            reasoning="Random proposal for exploration.",
        )
    if choice == "submit_patch":
        snippet = rng.choice(
            [
                "import razorpay\ndef charge_customer(a, c):\n    return {'id': 'x', 'status': 'captured'}",
                "import mmi_sdk\ndef address_to_coords(a):\n    return {'lat': 12.0, 'lng': 77.0}",
                "import kaleyra\ndef send_otp(p, c):\n    return {'sid': 'k1', 'status': 'sent'}",
                "def malformed(:\n    pass",
            ]
        )
        return SDKAction(
            role="lead",
            action_type="submit_patch",
            patched_code=snippet,
            reasoning="Random patch submission for exploration.",
        )
    return SDKAction(role="lead", action_type="pass", reasoning="Exploration policy.")


def main() -> None:
    """Run ten episodes and print reward summary statistics."""
    env = SDKSovereignEnvironment(seed=1)
    rng = random.Random(11)
    rewards: list[float] = []

    for index in range(10):
        observation = env.reset()
        while not observation.done:
            observation = env.step(_random_action(observation, rng))
        total = sum(env.state().cumulative_reward_by_role.values())
        rewards.append(total)
        print(
            f"Episode {index + 1:02d}: total_reward={total:+6.2f} "
            f"repo={env.state().repo_id} "
            f"reason={env.state().terminated_reason}"
        )

    stdev = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
    print(f"\nMean: {statistics.mean(rewards):+.2f}  Stdev: {stdev:.2f}")


if __name__ == "__main__":
	main()
