"""Write all 4 Colab notebooks (run once from repo root)."""
import json
from pathlib import Path

HF_USER = "ishansurdi"
ENV_URL = f"https://{HF_USER}-sdk-sovereign.hf.space"
SPACE_REPO = f"{HF_USER}/SDK-Sovereign"

def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
            "colab": {"provenance": [], "gpuType": "T4"},
            "accelerator": "GPU"
        },
        "cells": cells
    }

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src}

# ── 01_smoke_test.ipynb ──────────────────────────────────────────────────────

nb01 = nb([
    md("# 01 — SDK-Sovereign Smoke Test\n\n"
       "**Purpose:** Load Qwen 2.5-0.5B with two LoRA adapters, run a smoke episode "
       "against the live HF Space env, verify adapter swap.\n\n"
       "**Runtime:** Colab T4 GPU (free tier) · ~10 min\n\n"
       "> Before running: set `HF_TOKEN` in **Colab Secrets** (key icon on the left sidebar)"),

    code("# Cell 1 — Install dependencies\n"
         "!pip install -q \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"\n"
         "!pip install -q --no-deps \"trl<0.13\" peft accelerate bitsandbytes\n"
         "!pip install -q \"openenv-core[core] @ git+https://github.com/meta-pytorch/OpenEnv@v0.2.3\"\n"
         "!pip install -q wandb huggingface_hub\n"
         "print('✓ Dependencies installed')"),

    code(f"# Cell 2 — Clone and install the SDK-Sovereign env package\n"
         f"import os, sys\n\n"
         f"HF_USER   = '{HF_USER}'\n"
         f"SPACE_REPO = '{SPACE_REPO}'\n"
         f"ENV_URL    = '{ENV_URL}'\n\n"
         f"!git clone https://huggingface.co/spaces/{{SPACE_REPO}} sdk_sovereign_pkg\n"
         f"!pip install -q -e sdk_sovereign_pkg/\n"
         f"sys.path.insert(0, 'sdk_sovereign_pkg')\n"
         f"print(f'✓ Env package installed from {{SPACE_REPO}}')"),

    code("# Cell 3 — HF auth\n"
         "from google.colab import userdata\n"
         "from huggingface_hub import login\n\n"
         "HF_TOKEN = userdata.get('HF_TOKEN')\n"
         "login(token=HF_TOKEN)\n"
         "print('✓ Logged in to HF Hub')"),

    code("# Cell 4 — Load Qwen 2.5-0.5B with two LoRA adapters\n"
         "from server.llm_agents import load_model_with_two_adapters, make_agent_pair\n\n"
         "model, tokenizer = load_model_with_two_adapters()\n"
         "print('Adapters:', list(model.peft_config.keys()))\n"
         "model.print_trainable_parameters()\n\n"
         "agents = make_agent_pair(model, tokenizer)\n"
         "print('✓ Agent pair created')"),

    code("# Cell 5 — Run smoke episode against live HF Space\n"
         "from client import SDKSovereignEnv\n\n"
         "print(f'Connecting to: {ENV_URL}')\n\n"
         "with SDKSovereignEnv(base_url=ENV_URL).sync() as env:\n"
         "    obs = env.reset()\n"
         "    print(f'Episode started | role={obs.current_role}')\n"
         "    for turn in range(7):\n"
         "        agent = agents[obs.current_role]\n"
         "        action = agent(obs)\n"
         "        print(f'Turn {turn:2d} | {obs.current_role:7s} | {action.action_type}')\n"
         "        obs = env.step(action)\n"
         "        if obs.done:\n"
         "            print(f'Done! reward={obs.reward:.2f} | breakdown={obs.reward_breakdown}')\n"
         "            break\n"
         "print('✓ Smoke episode complete')"),

    code("# Cell 6 — Verify adapter swap\n"
         "from models import SDKObservation\n\n"
         "dummy_aud = SDKObservation(\n"
         "    current_role='auditor', turn_index=0, max_turns=7, error_log='test',\n"
         "    conversation_history=[], visible_allowlist=['razorpay'],\n"
         "    visible_codebase=None, visible_filename=None, current_proposal=None,\n"
         "    approved_replacement=None, done=False, reward=0.0, reward_breakdown={},\n"
         ")\n"
         "agents['auditor'](dummy_aud)\n"
         "assert model.active_adapter == 'auditor_adapter'\n"
         "print('✓ auditor_adapter swaps correctly')\n\n"
         "dummy_lead = SDKObservation(\n"
         "    current_role='lead', turn_index=1, max_turns=7, error_log='test',\n"
         "    conversation_history=[], visible_allowlist=None,\n"
         "    visible_codebase='import stripe', visible_filename='payment.py',\n"
         "    current_proposal=None, approved_replacement=None,\n"
         "    done=False, reward=0.0, reward_breakdown={},\n"
         ")\n"
         "agents['lead'](dummy_lead)\n"
         "assert model.active_adapter == 'lead_adapter'\n"
         "print('✓ lead_adapter swaps correctly')\n"
         "print('\\n✅ Phase 6 acceptance criteria met!')"),
])

# ── 02_train_lead.ipynb ──────────────────────────────────────────────────────

nb02 = nb([
    md("# 02 — Train Lead Adapter (GRPO)\n\n"
       "**Purpose:** Fine-tune the Lead LoRA adapter using GRPO on rollout data collected "
       "from the live env. Upload trained adapter to HF Hub.\n\n"
       "**Runtime:** Colab T4 · ~3h\n\n"
       "> Requires notebook 01 to have run in the same session (or re-run cells 1-4 here)."),

    code("# Cell 1 — Install (skip if already done in 01)\n"
         "!pip install -q \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"\n"
         "!pip install -q --no-deps \"trl<0.13\" peft accelerate bitsandbytes\n"
         "!pip install -q \"openenv-core[core] @ git+https://github.com/meta-pytorch/OpenEnv@v0.2.3\"\n"
         "!pip install -q wandb huggingface_hub datasets\n"
         "import sys; sys.path.insert(0, 'sdk_sovereign_pkg')\n"
         "print('✓ Ready')"),

    code(f"# Cell 2 — Config\n"
         f"HF_USER  = '{HF_USER}'\n"
         f"ENV_URL  = '{ENV_URL}'\n"
         f"N_ROLLOUT_EPISODES = 80\n"
         f"WANDB_PROJECT = 'sdk-sovereign'"),

    code("# Cell 3 — Load model + agents (re-run if kernel is fresh)\n"
         "from server.llm_agents import load_model_with_two_adapters, make_agent_pair\n"
         "from google.colab import userdata\n"
         "from huggingface_hub import login\n"
         "import wandb, os\n\n"
         "login(token=userdata.get('HF_TOKEN'))\n"
         "os.environ['WANDB_API_KEY'] = userdata.get('WANDB_API_KEY', '')\n\n"
         "model, tokenizer = load_model_with_two_adapters()\n"
         "agents = make_agent_pair(model, tokenizer)\n"
         "print('✓ Model + agents loaded')"),

    code("# Cell A — Collect rollouts\n"
         "import json\n"
         "from pathlib import Path\n"
         "from client import SDKSovereignEnv\n"
         "from models import SDKAction\n\n"
         "rollout_data = {'auditor': [], 'lead': []}\n\n"
         "for ep in range(N_ROLLOUT_EPISODES):\n"
         "    with SDKSovereignEnv(base_url=ENV_URL).sync() as env:\n"
         "        obs = env.reset()\n"
         "        per_role_buffer = []\n"
         "        while not obs.done and obs.turn_index < 7:\n"
         "            agent = agents[obs.current_role]\n"
         "            agent.model.set_adapter(agent.adapter_name)\n"
         "            prompt = agent._build_prompt(obs)\n"
         "            completion = agent._generate(prompt)\n"
         "            action = agent._parse_action(completion)\n"
         "            new_obs = env.step(action)\n"
         "            per_role_buffer.append({\n"
         "                'role': obs.current_role,\n"
         "                'prompt': prompt,\n"
         "                'completion': completion,\n"
         "                'step_reward': new_obs.reward,\n"
         "            })\n"
         "            obs = new_obs\n"
         "        for entry in per_role_buffer:\n"
         "            rollout_data[entry['role']].append({\n"
         "                'prompt': entry['prompt'],\n"
         "                'reward': entry['step_reward'],\n"
         "            })\n"
         "    if ep % 10 == 0:\n"
         "        print(f'  rollout {ep}/{N_ROLLOUT_EPISODES}')\n\n"
         "print(f'Lead samples:    {len(rollout_data[\"lead\"])}')\n"
         "print(f'Auditor samples: {len(rollout_data[\"auditor\"])}')\n"
         "Path('rollout_lead.jsonl').write_text('\\n'.join(json.dumps(r) for r in rollout_data['lead']))\n"
         "Path('rollout_auditor.jsonl').write_text('\\n'.join(json.dumps(r) for r in rollout_data['auditor']))\n"
         "print('✓ Rollout data saved')"),

    code("# Cell B — GRPO train Lead adapter\n"
         "from trl import GRPOTrainer, GRPOConfig\n"
         "from datasets import Dataset\n\n"
         "wandb.init(project=WANDB_PROJECT, name='lead-grpo-round1')\n\n"
         "lead_prompts = [r['prompt'] for r in rollout_data['lead']]\n"
         "ds_lead = Dataset.from_dict({'prompt': lead_prompts})\n\n"
         "def lead_reward_fn(completions, **kwargs):\n"
         "    rewards = []\n"
         "    for completion in completions:\n"
         "        action = agents['lead']._parse_action(completion)\n"
         "        if action.action_type == 'submit_patch':\n"
         "            with SDKSovereignEnv(base_url=ENV_URL).sync() as env:\n"
         "                obs = env.reset()\n"
         "                env.step(SDKAction(role='auditor', action_type='pass'))\n"
         "                env.step(SDKAction(role='lead', action_type='propose_replacement',\n"
         "                                   proposed_sdk='razorpay'))\n"
         "                env.step(SDKAction(role='auditor', action_type='approve'))\n"
         "                final = env.step(action)\n"
         "                rewards.append(float(final.reward))\n"
         "        elif action.action_type == 'propose_replacement':\n"
         "            rewards.append(1.0)\n"
         "        elif action.action_type == 'request_hint':\n"
         "            rewards.append(0.3)\n"
         "        else:\n"
         "            rewards.append(-0.5)  # pass\n"
         "    return rewards\n\n"
         "config = GRPOConfig(\n"
         "    output_dir='checkpoints/lead',\n"
         "    num_generations=4,\n"
         "    max_completion_length=512,\n"
         "    per_device_train_batch_size=1,\n"
         "    gradient_accumulation_steps=4,\n"
         "    learning_rate=5e-6,\n"
         "    num_train_epochs=1,\n"
         "    logging_steps=2,\n"
         "    save_steps=50,\n"
         "    report_to='wandb',\n"
         ")\n\n"
         "model.set_adapter('lead_adapter')\n"
         "for n, p in model.named_parameters():\n"
         "    if 'auditor_adapter' in n:\n"
         "        p.requires_grad = False\n\n"
         "trainer = GRPOTrainer(\n"
         "    model=model,\n"
         "    reward_funcs=lead_reward_fn,\n"
         "    args=config,\n"
         "    train_dataset=ds_lead.select(range(min(60, len(ds_lead)))),\n"
         "    tokenizer=tokenizer,\n"
         ")\n"
         "trainer.train()\n"
         "wandb.finish()\n"
         "print('✓ Lead GRPO training complete')"),

    code("# Cell C — Save and push Lead adapter to HF Hub\n"
         "from huggingface_hub import HfApi\n\n"
         "model.save_pretrained('checkpoints/lead_adapter_v1', selected_adapters=['lead_adapter'])\n"
         "HfApi().upload_folder(\n"
         "    folder_path='checkpoints/lead_adapter_v1',\n"
         "    repo_id=f'{HF_USER}/sdk-sovereign-lead-adapter',\n"
         "    repo_type='model',\n"
         "    commit_message='Lead LoRA adapter v1 (GRPO round 1)',\n"
         ")\n"
         "print(f'✓ Lead adapter pushed to HF Hub: {HF_USER}/sdk-sovereign-lead-adapter')"),
])

# ── 03_train_auditor.ipynb ───────────────────────────────────────────────────

nb03 = nb([
    md("# 03 — Train Auditor Adapter (GRPO)\n\n"
       "**Purpose:** Fine-tune the Auditor LoRA adapter. Mirrors notebook 02 but with "
       "an auditor-specific reward function (allow-list correctness).\n\n"
       "**Runtime:** Colab T4 · ~2h\n\n"
       "> Run after notebook 02 (or re-run cells 1-3 to reload model)."),

    code("# Cell 1 — Install (skip if already done)\n"
         "!pip install -q \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"\n"
         "!pip install -q --no-deps \"trl<0.13\" peft accelerate bitsandbytes\n"
         "!pip install -q \"openenv-core[core] @ git+https://github.com/meta-pytorch/OpenEnv@v0.2.3\"\n"
         "!pip install -q wandb huggingface_hub datasets\n"
         "import sys; sys.path.insert(0, 'sdk_sovereign_pkg')\n"
         "print('✓ Ready')"),

    code(f"# Cell 2 — Config\n"
         f"HF_USER  = '{HF_USER}'\n"
         f"ENV_URL  = '{ENV_URL}'\n"
         f"WANDB_PROJECT = 'sdk-sovereign'"),

    code("# Cell 3 — Load model + agents (re-run if kernel is fresh)\n"
         "from server.llm_agents import load_model_with_two_adapters, make_agent_pair\n"
         "from google.colab import userdata\n"
         "from huggingface_hub import login\n"
         "import wandb, os, json\n"
         "from pathlib import Path\n\n"
         "login(token=userdata.get('HF_TOKEN'))\n"
         "os.environ['WANDB_API_KEY'] = userdata.get('WANDB_API_KEY', '')\n\n"
         "model, tokenizer = load_model_with_two_adapters()\n"
         "agents = make_agent_pair(model, tokenizer)\n"
         "print('✓ Model + agents loaded')\n\n"
         "# Load auditor rollout data (written by notebook 02)\n"
         "auditor_data = [json.loads(l) for l in Path('rollout_auditor.jsonl').read_text().splitlines()]\n"
         "print(f'Auditor samples: {len(auditor_data)}')"),

    code("# Cell B — GRPO train Auditor adapter\n"
         "from trl import GRPOTrainer, GRPOConfig\n"
         "from datasets import Dataset\n"
         "from server.environment import SDKSovereignEnvironment\n\n"
         "wandb.init(project=WANDB_PROJECT, name='auditor-grpo-round1')\n\n"
         "ds_auditor = Dataset.from_dict({'prompt': [r['prompt'] for r in auditor_data]})\n\n"
         "_local_env = SDKSovereignEnvironment()\n\n"
         "def auditor_reward_fn(completions, **kwargs):\n"
         "    rewards = []\n"
         "    for completion in completions:\n"
         "        action = agents['auditor']._parse_action(completion)\n"
         "        if action.action_type == 'approve':\n"
         "            mentioned = next(\n"
         "                (sdk for sdk in _local_env.allowlist\n"
         "                 if sdk in (action.reasoning or '').lower()), None\n"
         "            )\n"
         "            rewards.append(1.5 if mentioned else -2.0)\n"
         "        elif action.action_type == 'reject':\n"
         "            rewards.append(0.5)  # cautious, usually fine\n"
         "        elif action.action_type == 'give_hint':\n"
         "            rewards.append(0.3)\n"
         "        elif action.action_type == 'pass':\n"
         "            rewards.append(-0.5)\n"
         "        else:\n"
         "            rewards.append(-1.0)\n"
         "    return rewards\n\n"
         "config = GRPOConfig(\n"
         "    output_dir='checkpoints/auditor',\n"
         "    num_generations=4,\n"
         "    max_completion_length=200,\n"
         "    per_device_train_batch_size=1,\n"
         "    gradient_accumulation_steps=4,\n"
         "    learning_rate=5e-6,\n"
         "    num_train_epochs=1,\n"
         "    logging_steps=2,\n"
         "    save_steps=50,\n"
         "    report_to='wandb',\n"
         ")\n\n"
         "model.set_adapter('auditor_adapter')\n"
         "for n, p in model.named_parameters():\n"
         "    if 'lead_adapter' in n:\n"
         "        p.requires_grad = False\n\n"
         "trainer = GRPOTrainer(\n"
         "    model=model,\n"
         "    reward_funcs=auditor_reward_fn,\n"
         "    args=config,\n"
         "    train_dataset=ds_auditor.select(range(min(60, len(ds_auditor)))),\n"
         "    tokenizer=tokenizer,\n"
         ")\n"
         "trainer.train()\n"
         "wandb.finish()\n"
         "print('✓ Auditor GRPO training complete')"),

    code("# Cell C — Save and push Auditor adapter to HF Hub\n"
         "from huggingface_hub import HfApi\n\n"
         "model.save_pretrained('checkpoints/auditor_adapter_v1', selected_adapters=['auditor_adapter'])\n"
         "HfApi().upload_folder(\n"
         "    folder_path='checkpoints/auditor_adapter_v1',\n"
         "    repo_id=f'{HF_USER}/sdk-sovereign-auditor-adapter',\n"
         "    repo_type='model',\n"
         "    commit_message='Auditor LoRA adapter v1 (GRPO round 1)',\n"
         ")\n"
         "print(f'✓ Auditor adapter pushed to HF Hub: {HF_USER}/sdk-sovereign-auditor-adapter')"),
])

# ── 04_eval_and_plots.ipynb ──────────────────────────────────────────────────

nb04 = nb([
    md("# 04 — Eval + Plots\n\n"
       "**Purpose:** Compare random baseline vs trained adapters over 30 episodes each. "
       "Generate 6 publication-ready PNGs in `plots/`.\n\n"
       "**Runtime:** Colab T4 · ~1h\n\n"
       "> Requires notebooks 02 + 03 to have completed (adapters on HF Hub)."),

    code("# Cell 1 — Install\n"
         "!pip install -q \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"\n"
         "!pip install -q --no-deps \"trl<0.13\" peft accelerate bitsandbytes\n"
         "!pip install -q \"openenv-core[core] @ git+https://github.com/meta-pytorch/OpenEnv@v0.2.3\"\n"
         "!pip install -q wandb huggingface_hub matplotlib numpy\n"
         "import sys; sys.path.insert(0, 'sdk_sovereign_pkg')\n"
         "!mkdir -p plots\n"
         "print('✓ Ready')"),

    code(f"# Cell 2 — Config\n"
         f"HF_USER  = '{HF_USER}'\n"
         f"ENV_URL  = '{ENV_URL}'\n"
         f"N_EPISODES = 30"),

    code("# Cell 3 — Auth\n"
         "from google.colab import userdata\n"
         "from huggingface_hub import login\n"
         "login(token=userdata.get('HF_TOKEN'))\n"
         "print('✓ Auth OK')"),

    code("# Cell 4 — Load baseline model (fresh untrained adapters)\n"
         "import server.llm_agents as la\n"
         "baseline_model, baseline_tok = la.load_model_with_two_adapters()\n"
         "baseline_agents = la.make_agent_pair(baseline_model, baseline_tok)\n"
         "print('✓ Baseline model loaded (untrained adapters)')"),

    code("# Cell 5 — Load trained model with Hub adapters\n"
         "trained_model, trained_tok = la.load_model_with_two_adapters()\n"
         "trained_model.load_adapter(\n"
         "    f'{HF_USER}/sdk-sovereign-lead-adapter', adapter_name='lead_adapter_trained'\n"
         ")\n"
         "trained_model.load_adapter(\n"
         "    f'{HF_USER}/sdk-sovereign-auditor-adapter', adapter_name='auditor_adapter_trained'\n"
         ")\n"
         "trained_agents = la.make_agent_pair(trained_model, trained_tok)\n"
         "trained_agents['lead'].adapter_name    = 'lead_adapter_trained'\n"
         "trained_agents['auditor'].adapter_name = 'auditor_adapter_trained'\n"
         "print('✓ Trained model loaded')"),

    code("# Cell 6 — Eval loop\n"
         "import json\n"
         "from pathlib import Path\n"
         "from client import SDKSovereignEnv\n\n"
         "def run_eval(agents, label, n=N_EPISODES):\n"
         "    results = []\n"
         "    for ep in range(n):\n"
         "        with SDKSovereignEnv(base_url=ENV_URL).sync() as env:\n"
         "            obs = env.reset()\n"
         "            total = 0.0; transcript = []\n"
         "            while not obs.done and obs.turn_index < 7:\n"
         "                action = agents[obs.current_role](obs)\n"
         "                transcript.append({'turn': obs.turn_index,\n"
         "                                   'role': obs.current_role,\n"
         "                                   'action_type': action.action_type})\n"
         "                obs = env.step(action)\n"
         "                total += obs.reward\n"
         "            state = env.state()\n"
         "            tr = state.test_results or {}\n"
         "            results.append({\n"
         "                'total_reward': total,\n"
         "                'tests_passed': sum(tr.values()),\n"
         "                'tests_total':  len(tr) or 3,\n"
         "                'success': bool(tr and all(tr.values())),\n"
         "                'turns':   state.step_count,\n"
         "                'repo':    state.repo_id,\n"
         "                'terminated': state.terminated_reason,\n"
         "                'transcript': transcript,\n"
         "            })\n"
         "        if ep % 5 == 0:\n"
         "            print(f'  [{label}] ep {ep}/{n}')\n"
         "    return results\n\n"
         "baseline_results = run_eval(baseline_agents, 'baseline')\n"
         "trained_results  = run_eval(trained_agents,  'trained')\n"
         "Path('eval_results.json').write_text(json.dumps(\n"
         "    {'baseline': baseline_results, 'trained': trained_results}, indent=2\n"
         "))\n"
         "print('✓ Eval complete')"),

    code("# Cell 7 — Plot 1: pass rate bar chart\n"
         "import matplotlib.pyplot as plt\n\n"
         "b_rate = sum(r['success'] for r in baseline_results) / len(baseline_results)\n"
         "t_rate = sum(r['success'] for r in trained_results)  / len(trained_results)\n\n"
         "plt.figure(figsize=(6,4))\n"
         "plt.bar(['Baseline (untrained)', 'Trained (two LoRAs)'], [b_rate, t_rate],\n"
         "        color=['#bbbbbb', '#1f77b4'])\n"
         "plt.ylabel('Pass rate (all tests passed)')\n"
         "plt.title('SDK-Sovereign — pass rate, n=30 each')\n"
         "plt.ylim(0, 1)\n"
         "for i, v in enumerate([b_rate, t_rate]):\n"
         "    plt.text(i, v + 0.02, f'{v:.0%}', ha='center', fontweight='bold')\n"
         "plt.tight_layout()\n"
         "plt.savefig('plots/pass_rate_baseline_vs_trained.png', dpi=150, bbox_inches='tight')\n"
         "plt.show(); plt.close()\n"
         "print(f'Baseline: {b_rate:.0%}  |  Trained: {t_rate:.0%}')"),

    code("# Cell 8 — Plot 2: mean episode reward\n"
         "import statistics\n\n"
         "b_r = statistics.mean(r['total_reward'] for r in baseline_results)\n"
         "t_r = statistics.mean(r['total_reward'] for r in trained_results)\n\n"
         "plt.figure(figsize=(6,4))\n"
         "plt.bar(['Baseline', 'Trained'], [b_r, t_r], color=['#bbbbbb', '#1f77b4'])\n"
         "plt.ylabel('Mean episode reward')\n"
         "plt.axhline(0, color='k', lw=0.5)\n"
         "plt.title('Mean total reward per episode (n=30)')\n"
         "plt.tight_layout()\n"
         "plt.savefig('plots/mean_reward.png', dpi=150, bbox_inches='tight')\n"
         "plt.show(); plt.close()"),

    code("# Cell 9 — Plot 3 & 4: WandB reward curves\n"
         "import wandb\n\n"
         "api = wandb.Api()\n"
         "for run_name in ['lead-grpo-round1', 'auditor-grpo-round1']:\n"
         "    try:\n"
         "        run = api.runs(f'{HF_USER}/sdk-sovereign', {'display_name': run_name})[0]\n"
         "        h = run.history()\n"
         "        plt.figure(figsize=(8,4))\n"
         "        col = next((c for c in h.columns if 'reward' in c.lower()), None)\n"
         "        if col:\n"
         "            plt.plot(h['_step'], h[col], label=col)\n"
         "        plt.xlabel('GRPO step'); plt.ylabel('Reward')\n"
         "        plt.title(f'GRPO training — {run_name}')\n"
         "        plt.legend(); plt.grid(alpha=0.3)\n"
         "        role = run_name.split('-')[0]\n"
         "        plt.savefig(f'plots/reward_curve_{role}.png', dpi=150, bbox_inches='tight')\n"
         "        plt.show(); plt.close()\n"
         "    except Exception as e:\n"
         "        print(f'WandB run {run_name} not found: {e}')"),

    code("# Cell 10 — Plot 5: per-repo pass rate\n"
         "import numpy as np\n"
         "from collections import defaultdict\n\n"
         "b_per = defaultdict(list); t_per = defaultdict(list)\n"
         "for r in baseline_results: b_per[r['repo']].append(r['success'])\n"
         "for r in trained_results:  t_per[r['repo']].append(r['success'])\n"
         "repos = sorted(set(b_per) | set(t_per))\n"
         "b_vals = [sum(b_per[r])/len(b_per[r]) if b_per[r] else 0 for r in repos]\n"
         "t_vals = [sum(t_per[r])/len(t_per[r]) if t_per[r] else 0 for r in repos]\n\n"
         "x = np.arange(len(repos)); w = 0.35\n"
         "plt.figure(figsize=(8,4))\n"
         "plt.bar(x - w/2, b_vals, w, label='Baseline', color='#bbbbbb')\n"
         "plt.bar(x + w/2, t_vals, w, label='Trained',  color='#1f77b4')\n"
         "plt.xticks(x, repos, rotation=15)\n"
         "plt.ylabel('Pass rate'); plt.legend()\n"
         "plt.title('Pass rate by repo')\n"
         "plt.tight_layout()\n"
         "plt.savefig('plots/per_repo_pass_rate.png', dpi=150, bbox_inches='tight')\n"
         "plt.show(); plt.close()"),

    code("# Cell 11 — Plot 6: turns to completion distribution\n"
         "b_turns = [r['turns'] for r in baseline_results if r['success']]\n"
         "t_turns = [r['turns'] for r in trained_results  if r['success']]\n\n"
         "plt.figure(figsize=(7,4))\n"
         "bins = list(range(1, 9))\n"
         "plt.hist([b_turns, t_turns], bins=bins, label=['Baseline', 'Trained'],\n"
         "         color=['#bbbbbb', '#1f77b4'], edgecolor='white')\n"
         "plt.xlabel('Turns to completion (successful episodes only)')\n"
         "plt.ylabel('Count'); plt.legend()\n"
         "plt.title('Distribution of completion turns')\n"
         "plt.tight_layout()\n"
         "plt.savefig('plots/completion_turns.png', dpi=150, bbox_inches='tight')\n"
         "plt.show(); plt.close()\n\n"
         "print('\\n✅ All 6 plots saved to plots/')\n"
         "import os; print('Files:', sorted(os.listdir('plots/')))"),
])

# ── Write all notebooks ───────────────────────────────────────────────────────

notebooks_dir = Path("notebooks")
notebooks_dir.mkdir(exist_ok=True)

for fname, data in [
    ("01_smoke_test.ipynb",    nb01),
    ("02_train_lead.ipynb",    nb02),
    ("03_train_auditor.ipynb", nb03),
    ("04_eval_and_plots.ipynb", nb04),
]:
    out = notebooks_dir / fname
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"✓ Written {out} ({out.stat().st_size:,} bytes)")

print("\n✅ All 4 notebooks written.")
