# Email Draft: PyAMPA Authors (UAB)

**Status:** NEEDS EMAIL RESEARCH

**Email:** (check UAB or GitHub for maintainers)

---

## Recipient Research

| Field | Details |
|-------|---------|
| **Name** | PyAMPA Authors |
| **Position** | Universitat Autonoma de Barcelona |
| **Key Paper** | PyAMPA - Python AMP prediction library (2024) |
| **Recent Focus** | High-throughput prediction, open source tools |
| **Code** | Likely on PyPI/GitHub |

---

## EMAIL

**To:** (corresponding author or GitHub maintainer)

**Subject:** Contribution offer for PyAMPA - alternative predictor

---

Hi,

PyAMPA is a clean implementation for AMP prediction. We've built an alternative predictor using hyperbolic embeddings that might be worth adding as an optional backend.

Our predictor achieves similar accuracy (Spearman ~0.66 for MIC) but uses completely different features - codon-based p-adic geometry vs. physicochemical descriptors. The error modes are likely independent, so users could ensemble both for better predictions.

If there's interest, I could submit a PR adding our predictor as an optional module. Would maintain API compatibility and add minimal dependencies.

Alternatively, if PyAMPA has a plugin system, we could package separately.

Best,
Ivan Weiss Van Der Pol
Jonathan Verdun
Kyrian Weiss Van Der Pol
AI Whisperers | CONACYT Paraguay
github.com/Ai-Whisperers/ternary-vaes-bioinformatics

---

## Notes

- **Word count:** 112
- **Why this works:** Offers concrete contribution (PR), respects their architecture, developer-focused
- **Tone:** Open source contributor
- **Alternative:** Could open GitHub issue instead of email

---

*Last Updated: 2026-01-26*
