"""LLM-backed agents. Loads Qwen 2.5-0.5B with two LoRA adapters and
swaps between them based on whose turn it is.

The adapter-swap is the linchpin of the two-policy claim. Verify with
`assert model.active_adapter == 'auditor_adapter'` before every Auditor
generation, same for Lead.
"""
from __future__ import annotations
import json
import re
from typing import Any, Optional

from models import SDKAction, SDKObservation, ActionType
from server.prompts import SYSTEM_AUDITOR, SYSTEM_LEAD


def load_model_with_two_adapters(
    base_name: str = "unsloth/Qwen2.5-0.5B-Instruct",
    max_seq_length: int = 2048,
):
    """Load base model in 4-bit + apply two LoRA adapters."""
    from unsloth import FastLanguageModel
    from peft import LoraConfig, get_peft_model

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )

    # Apply Auditor LoRA first (creates PEFT model wrapper)
    model = get_peft_model(model, cfg, adapter_name="auditor_adapter")
    # Add Lead LoRA on the same base
    model.add_adapter(adapter_name="lead_adapter", peft_config=cfg)

    return model, tokenizer


class LLMAgent:
    """Single agent instance — swaps to its adapter on each call."""

    def __init__(self, model: Any, tokenizer: Any, role: str):
        self.model = model
        self.tokenizer = tokenizer
        self.role = role
        self.adapter_name = f"{role}_adapter"
        self.system_prompt = SYSTEM_AUDITOR if role == "auditor" else SYSTEM_LEAD

    def __call__(self, obs: SDKObservation) -> SDKAction:
        self.model.set_adapter(self.adapter_name)
        prompt = self._build_prompt(obs)
        text = self._generate(prompt)
        return self._parse_action(text)

    def _build_prompt(self, obs: SDKObservation) -> str:
        user_content = self._render_observation(obs)
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )

    def _render_observation(self, obs: SDKObservation) -> str:
        history_str = "\n".join(
            f"  turn {h.get('turn', '?')}: {h.get('role')} → {h.get('action_type')}"
            f" {h.get('proposed_sdk') or h.get('rejection_reason') or ''}"
            for h in obs.conversation_history[-6:]
        ) or "  (no history yet)"

        if self.role == "auditor":
            allow = ", ".join(obs.visible_allowlist or [])
            return (
                f"ERROR LOG: {obs.error_log}\n"
                f"TURN: {obs.turn_index} of {obs.max_turns}\n"
                f"CURRENT PROPOSAL: {obs.current_proposal or '(none yet)'}\n"
                f"ALLOW-LIST: {allow}\n"
                f"HISTORY:\n{history_str}\n\n"
                f"Choose an action. Respond as a single JSON object."
            )

        # Lead view
        return (
            f"ERROR LOG: {obs.error_log}\n"
            f"TURN: {obs.turn_index} of {obs.max_turns}\n"
            f"BROKEN CODE ({obs.visible_filename}):\n```\n{obs.visible_codebase}\n```\n"
            f"APPROVED REPLACEMENT: {obs.approved_replacement or '(not yet approved)'}\n"
            f"HISTORY:\n{history_str}\n\n"
            f"Choose an action. Respond as a single JSON object."
        )

    def _generate(self, prompt: str) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512 if self.role == "lead" else 200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def _parse_action(self, text: str) -> SDKAction:
        parsed = self._extract_json(text)
        if parsed is None:
            return SDKAction(role=self.role, action_type=ActionType.PASS.value,
                             reasoning=f"PARSE_FAIL: {text[:200]}")
        action_type = parsed.get("action_type", "pass")
        try:
            ActionType(action_type)
        except ValueError:
            action_type = "pass"
        return SDKAction(
            role=self.role,
            action_type=action_type,
            proposed_sdk=parsed.get("proposed_sdk"),
            rejection_reason=parsed.get("rejection_reason"),
            patched_code=parsed.get("patched_code"),
            hint_request=parsed.get("hint_request"),
            hint_response=parsed.get("hint_response"),
            reasoning=str(parsed.get("reasoning", ""))[:500],
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            # Trim from end until valid JSON
            s = m.group()
            for i in range(len(s), 0, -1):
                try:
                    return json.loads(s[:i] + "}")
                except json.JSONDecodeError:
                    continue
        return None


def make_agent_pair(model: Any, tokenizer: Any) -> dict:
    return {
        "auditor": LLMAgent(model, tokenizer, "auditor"),
        "lead": LLMAgent(model, tokenizer, "lead"),
    }
