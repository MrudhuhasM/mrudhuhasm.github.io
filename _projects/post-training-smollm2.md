---
title: Post-training SmolLM2-135M from scratch
key: post-training-smollm2
order: 1
status: in-progress
stack: [PyTorch, TRL, RLHF, DPO, RLVR]
summary: Instruction tuning, reward modeling, DPO-style methods, RL and verifiable-reward reasoning on a small model — each stage evaluated against the one before.
repo: ""   # add the GitHub URL when the repo is public
---

The full post-training pipeline, built stage by stage on SmolLM2-135M. The model is small on purpose: cheap, repeatable experiments make ablations practical.

## Stages

1. Base model and evaluation setup
2. Instruction tuning
3. Preference data and reward modeling
4. Direct alignment (DPO-style methods)
5. Reinforcement learning / policy optimization
6. RL with verifiable rewards and reasoning

Each stage's result becomes the baseline for the next, with explicit evaluation at every step.

## Status

Currently on instruction tuning. Write-ups will be linked here as each stage lands.
