<!-- SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

---
title: "Data Strategy"
date: 2025-12-24
authors:
  - AI Whisperers
version: "0.1"
license: PolyForm-Noncommercial-1.0.0
---

# Data Strategy

## 1. Factories over Fixtures

We prefer generating data programmatically (`Factory`) over static files (`Fixture`).

- **Pros**: Dynamic, robust to schema changes, clear intent.
- **Cons**: Slower than loading JSON.

## 2. Database Handling

- **Unit Tests**: Use `sqlite:///:memory:` or mocking.
- **Integration Tests**: Use a dedicated test database container.
- **Teardown**: Transaction rollback preference over `TRUNCATE`.

## 3. Seed Data

Global reference data (e.g., genetic codes, ternary logic tables) should be mocked or loaded once via a "singleton" factory.
