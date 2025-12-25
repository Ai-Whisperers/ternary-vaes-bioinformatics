<!-- SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

---
title: "Smoke Test Suite"
date: 2025-12-24
authors:
  - AI Whisperers
version: "0.1"
license: PolyForm-Noncommercial-1.0.0
---

# Smoke Test Suite

**Quick sanity checks to run on every commit.**

1.  **Environment Check**: Can we import `torch` and `src`?
2.  **Model Init**: Can `TernaryVAEV5_11(config)` be instantiated?
3.  **Inference**: Can the model run `forward(zeros)` without error?
