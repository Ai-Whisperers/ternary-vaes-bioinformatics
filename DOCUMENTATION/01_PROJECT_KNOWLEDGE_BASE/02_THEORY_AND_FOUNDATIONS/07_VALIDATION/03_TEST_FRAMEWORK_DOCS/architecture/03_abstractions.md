<!-- SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

---
title: "Abstractions"
date: 2025-12-24
authors:
  - AI Whisperers
version: "0.1"
license: PolyForm-Noncommercial-1.0.0
---

# Abstractions

## Drivers (`tests/core/drivers/`)

Wrappers around external tools.

- `api_driver`: Wraps `requests` or `httpx`. Handles auth tokens automatically.
- `db_driver`: Wraps `sqlalchemy` or raw SQL. Handles rapid setup/teardown.
- `browser_driver`: Wraps `selenium` or `playwright`.

## Matchers (`tests/core/matchers/`)

Custom assertions for domain-specific checks.

- `expect(z_hyp).toBeOnPoincareDisk()`
- `expect(model).toHaveGradientFlow()`
