---
title: Fused softmax in Triton
key: fused-softmax-triton
order: 4
status: completed
stack: [Triton, CUDA, PyTorch]
summary: Naive PyTorch, torch.compile, a Triton kernel and a persistent variant, benchmarked across shapes.
repo: ""
---

Four implementations of row-wise softmax — naive PyTorch, `torch.compile`, a hand-written Triton kernel with one block per row, and a persistent Triton kernel — benchmarked across matrix shapes, with a memory-traffic analysis of why fusion matters.

The full write-up is linked alongside.
