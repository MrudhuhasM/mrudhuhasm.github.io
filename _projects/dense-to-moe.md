---
title: Dense to mixture-of-experts transformer
key: dense-to-moe
order: 3
status: in-progress
stack: [PyTorch, GQA, MoE]
summary: A GQA transformer with its own inference path, then the feed-forward block swapped for routed experts.
repo: ""
glyph: moe
---

A study of what changes when a dense transformer becomes a mixture-of-experts model.

## Approach

1. Implement a dense transformer with grouped-query attention, starting from attention itself.
2. Build a full inference path for it.
3. Replace the feed-forward block with routed experts and compare.

The goal is understanding the architecture, not training a usable model.
