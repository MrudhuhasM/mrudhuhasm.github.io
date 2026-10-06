---
title: Autodiff engine
key: autodiff-engine
order: 2
status: in-progress
stack: [Python, NumPy]
summary: Reverse-mode autodiff from scalars to tensors, with layers, losses and optimizers. Done when an MLP trains on it and gradients match PyTorch.
repo: ""
---

A reverse-mode automatic differentiation engine, built from scratch.

## Plan

1. Scalar autodiff
2. Cleaner engine semantics
3. Tensors
4. Vector–Jacobian product framing
5. Neural-network layers, losses and optimizers
6. Train something small, then compare against PyTorch

**Done when:** a small MLP trains entirely on the engine and its gradients match PyTorch's.
