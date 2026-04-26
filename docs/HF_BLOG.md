# SDK-Sovereign: Training Two Agents for a Real Migration Crisis

SDK-Sovereign started from a simple question: what if the environment was not a board game, not a chatbot benchmark, and not another toy coding task? What if the environment looked like the kind of engineering incident a real team would actually wake up to?

That is the hook behind the project:

> SDK-Sovereign trains two LLM policies to coordinate under partial information on a problem that isn't a game — it's the kind of crisis a real engineering team faces when geopolitics moves faster than their tech stack.

In the environment, it is 2026. A product team in India is suddenly cut off from sanctioned external SDKs. Stripe, Google Maps, and Twilio are no longer available for the affected workload. The system has to keep working, but the migration cannot be done by one agent acting alone. The Integration Lead can see the codebase and the failure log, but not the sovereign allow-list. The Security Auditor can see the approved replacement SDKs and the negotiation history, but not the source code. Together, they must coordinate inside a seven-turn OpenEnv episode.

## Why This Matters

The project is framed around digital sovereignty rather than abstract agent cooperation. That makes the coordination problem materially different. The Lead is not only solving a code transformation task. The Lead is solving under policy uncertainty. The Auditor is not only classifying a proposal. The Auditor is acting as a governance constraint inside the loop. That asymmetry is the core of the environment.

For India-focused infrastructure, that framing is useful because the operational question is rarely just “can a model patch code?” The real question is closer to: can models coordinate while each only sees the part of the system they are supposed to control?

## Environment Design

SDK-Sovereign combines four mechanisms that together make the environment feel more like a real engineering workflow than a game benchmark.

First, the environment uses partial observability. Auditor and Lead do not share the same state view. They can only coordinate through structured actions and free-text reasoning that becomes part of the next turn’s conversation history.

Second, the task is grounded in executable verification. Submitted patches are not accepted because they “look right.” The verifier stubs deprecated and sovereign SDKs, executes the patched code, and runs repo-specific parity checks. This is what keeps the reward tied to behavior instead of style.

Third, the reward is composable and inspectable. Format validity, proposal quality, approval behavior, syntax correctness, import choice, parity test passes, and terminal success are all exposed in the reward breakdown. That matters for training because reward hacking is the main silent failure mode in this kind of setup.

Fourth, the repos are scenario-specific. Payments, maps, and communications each fail differently and require different replacements. That lets us measure whether a policy learned a pattern or merely overfit one repo.

## Two-Policy Training Instead of One Monolithic Agent

The current training pipeline uses a shared Qwen 2.5-0.5B base with two role-specific LoRA adapters. One adapter is specialized for Lead behavior and the other for Auditor behavior. The agents are trained separately so credit assignment stays role-conditional.

The updated notebook keeps the setup practical for hackathon time constraints. It now prioritizes:

- short supervised bootstrap from expert trajectories
- tiny RL refinement instead of long unstable runs
- per-phase JSONL logging
- automatic Hugging Face backup of logs and adapters
- baseline-vs-trained evaluation artifacts that survive Colab disconnects

That last point turned out to be important. In practice, the hardest part was not just producing weights. It was producing evidence that survives GPU preemption and can still be shown to judges later.

## What We Measure

The evaluation path records both baseline and trained runs across all repos, then generates plots intended to answer judge-facing questions directly:

- did pass rate improve?
- did mean reward improve?
- did behavior change across repos?
- did negotiation patterns change?
- do the training traces show movement rather than random noise?

The repository now includes a broader smoke path for this as well: local route tests, remote OpenEnv smoke tests, and a standalone inference smoke runner that checks `/health`, OpenEnv reset and step flow, and the public `/play` endpoints.

## Current Scope and Honesty

This is still a hackathon build, not a production orchestration system. The goal is to make the environment legitimate, the training story inspectable, and the demo honest about what is running. Rule mode remains the reliability floor. Baseline and trained modes are surfaced only when the deployment is actually configured to serve them.

That honesty matters. It is easy to produce a good-looking demo. It is harder, and more useful, to produce one where the environment, the evaluation path, and the deployment behavior all line up.

## What Comes Next

The next step is not just “more epochs.” The next step is better curriculum quality and stronger repo-balanced evaluation. The environment is already in the right shape for that: executable verification, asymmetric roles, real scenario variation, and logs that make failure modes visible.

That is the real point of SDK-Sovereign. It is a compact testbed for agent coordination under operational constraints, built around a crisis that engineering teams can immediately understand.
