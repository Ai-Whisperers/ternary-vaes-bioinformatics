# Summary Code Health Report

**Generated:** 2025-12-24 00:25
**Full Data**: [View Raw Dashboard](./_raw_data/CODE_HEALTH_DASHBOARD.md)

## 1. Tool Status Summary

| Tool     | Status     | Findings                                  |
| :------- | :--------- | :---------------------------------------- |
| ruff     | ✅ success | 941                                       |
| mypy     | ✅ success | 1                                         |
| radon_cc | ✅ success | See `audit_data/radon_results.json`       |
| radon_mi | ✅ success | See `audit_data/radon_results.json`       |
| bandit   | ✅ success | See `audit_data/bandit_results.json`      |
| vulture  | ✅ success | 43                                        |
| pygount  | ✅ success | See `audit_data/code_health_metrics.json` |

## 2. Actionable Tasks

Based on the audit data, the following tasks have been generated:

- **Security**: [P1_SECURITY_FIXES](../00_TASKS/03_INFRASTRUCTURE/P1_SECURITY_FIXES.md)
- **Complexity**: [P2_COMPLEXITY_REFACTOR](../00_TASKS/03_INFRASTRUCTURE/P2_COMPLEXITY_REFACTOR.md)

## 4. Linting & Types

- **Ruff**: 941 issues found.
- **Mypy**: 1 issues found.
