# Code Health Summary

**Generated:** 2025-12-24
**Scope:** Full Repository Audit

---

## Quick Status

| Tool | Status | Findings |
|:-----|:-------|:---------|
| ruff | Success | 941 issues |
| mypy | Success | 1 issue |
| radon_cc | Success | See JSON |
| bandit | Success | 143 issues (5 high, 54 medium) |
| vulture | Success | 43 dead code items |

---

## Priority Actions

### P0 - Security (High Severity)
- 3 subprocess calls with `shell=True`
- 1 weak MD5 hash usage

### P1 - Security (Medium Severity)
- 45+ unsafe PyTorch load calls (use `weights_only=True`)
- SQL injection vector (string query construction)
- HTTP requests without timeout

### P2 - Code Quality
- 941 linting issues (ruff)
- 43 unused code items (vulture)

---

## Actionable Tasks

| Priority | Task | File |
|:---------|:-----|:-----|
| P1 | Security fixes | `../00_TASKS/03_INFRASTRUCTURE/P1_SECURITY_FIXES.md` |
| P2 | Complexity refactor | `../00_TASKS/03_INFRASTRUCTURE/P2_COMPLEXITY_REFACTOR.md` |

---

## Technical Debt Overview

See `TECHNICAL_DEBT_INVENTORY.md` for detailed inventory including:
- Performance bottlenecks
- Code duplication
- Dead code
- Complexity issues

---

## Regenerating Reports

```bash
# Run full audit
python scripts/analysis/comprehensive_audit.py

# Individual tools
ruff check src/
mypy src/
bandit -r src/
radon cc src/ -a
vulture src/
```

---

*For detailed findings, run audit tools directly or see raw JSON in `_raw_data/`*
