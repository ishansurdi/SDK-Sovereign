"""LLM-backed agents with deterministic evaluation defaults."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from models import ActionType, SDKAction, SDKObservation
from server.prompts import SYSTEM_AUDITOR, SYSTEM_LEAD


@dataclass(frozen=True)
class GenerationProfile:
    """Small immutable generation profile for a policy role."""

    max_new_tokens: int
    do_sample: bool
    temperature: Optional[float] = None


DETERMINISTIC_PROFILE = {
    "lead": GenerationProfile(max_new_tokens=384, do_sample=False),
    "auditor": GenerationProfile(max_new_tokens=96, do_sample=False),
}
EXPLORATION_PROFILE = {
    "lead": GenerationProfile(max_new_tokens=384, do_sample=True, temperature=0.2),
    "auditor": GenerationProfile(max_new_tokens=96, do_sample=True, temperature=0.2),
}


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
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, cfg, adapter_name="auditor_adapter")
    model.add_adapter(adapter_name="lead_adapter", peft_config=cfg)
    return model, tokenizer


class LLMAgent:
    """Single agent instance that swaps to its adapter on each call."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        role: str,
        *,
        deterministic: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.role = role
        self.adapter_name = f"{role}_adapter"
        self.system_prompt = SYSTEM_AUDITOR if role == "auditor" else SYSTEM_LEAD
        self.profile = (DETERMINISTIC_PROFILE if deterministic else EXPLORATION_PROFILE)[role]

    def __call__(self, obs: SDKObservation) -> SDKAction:
        self.model.set_adapter(self.adapter_name)
        prompt = self._build_prompt(obs)
        text = self._generate(prompt)
        return self._parse_action(text)

    def _build_prompt(self, obs: SDKObservation) -> str:
        user_content = self._render_observation(obs)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _render_observation(self, obs: SDKObservation) -> str:
        history_str = "\n".join(
            f"  turn {item.get('turn', '?')}: {item.get('role')} -> {item.get('action_type')}"
            f" {item.get('proposed_sdk') or item.get('rejection_reason') or ''}"
            for item in obs.conversation_history[-6:]
        ) or "  (no history yet)"

        if self.role == "auditor":
            allowlist = ", ".join(obs.visible_allowlist or [])
            return (
                f"ERROR LOG: {obs.error_log}\n"
                f"TURN: {obs.turn_index} of {obs.max_turns}\n"
                f"CURRENT PROPOSAL: {obs.current_proposal or '(none yet)'}\n"
                f"ALLOW-LIST: {allowlist}\n"
                f"HISTORY:\n{history_str}\n\n"
                "Choose an action. Respond as a single JSON object."
            )

        return (
            f"ERROR LOG: {obs.error_log}\n"
            f"TURN: {obs.turn_index} of {obs.max_turns}\n"
            f"BROKEN CODE ({obs.visible_filename}):\n```\n{obs.visible_codebase}\n```\n"
            f"APPROVED REPLACEMENT: {obs.approved_replacement or '(not yet approved)'}\n"
            f"HISTORY:\n{history_str}\n\n"
            "Choose an action. Respond as a single JSON object."
        )

    def _generate(self, prompt: str) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generate_kwargs = {
            "max_new_tokens": self.profile.max_new_tokens,
            "do_sample": self.profile.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if self.profile.do_sample and self.profile.temperature is not None:
            generate_kwargs["temperature"] = self.profile.temperature
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _parse_action(self, text: str) -> SDKAction:
        parsed = self._extract_json(text)
        if parsed is None:
            return SDKAction(
                role=self.role,
                action_type=ActionType.PASS.value,
                reasoning=f"PARSE_FAIL: {text[:200]}",
            )
        action_type = parsed.get("action_type", ActionType.PASS.value)
        try:
            ActionType(action_type)
        except ValueError:
            action_type = ActionType.PASS.value
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
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            snippet = match.group()
            for index in range(len(snippet), 0, -1):
                try:
                    return json.loads(snippet[:index] + "}")
                except json.JSONDecodeError:
                    continue
        return None


def make_agent_pair(model: Any, tokenizer: Any, *, deterministic: bool = True) -> dict[str, LLMAgent]:
    """Return Lead and Auditor agents sharing one base model."""
    return {
        "auditor": LLMAgent(model, tokenizer, "auditor", deterministic=deterministic),
        "lead": LLMAgent(model, tokenizer, "lead", deterministic=deterministic),
    }
