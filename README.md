---
title: SDK-Sovereign
emoji: 🤖
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 8000
tags:
  - multi-agent
  - openenv
  - code-generation
  - sovereignty
  - india
license: mit
---

# SDK-Sovereign

Multi-agent OpenEnv environment for digital sovereignty SDK migrations.

This repo now supports a teacher-first training loop:
- `teacher` and `rule` policies for deterministic successful traces
- JSONL trace export for offline analysis
- chat-format SFT export for role-specific Lead and Auditor tuning
- deterministic live model evaluation for stable pass-rate comparisons

## Quickstart

```bash
pip install -e .
pytest tests/ -v
python scripts/generate_teacher_traces.py --episodes-per-repo 25
python scripts/export_sft_dataset.py --input logs/teacher_traces.jsonl --output logs/teacher_sft.jsonl
```
