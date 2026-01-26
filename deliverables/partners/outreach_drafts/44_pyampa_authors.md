# Email Draft: PyAMPA Authors (UAB)

**Status:** READY TO SEND

**Email:** marc.torrent@uab.cat (verified via SYSBIOLAB UAB - corresponding author)

---

## Recipient Research

| Field | Details |
|-------|---------|
| **Name** | Marc Torrent Burgas |
| **Position** | Principal Investigator, SYSBIOLAB, Universitat Autònoma de Barcelona |
| **Key Paper** | PyAMPA: a high-throughput prediction and optimization tool for AMPs (mSystems 2024) |
| **Recent Focus** | Systems biology of infection, AMP design, host-pathogen interactions |
| **Lab** | sites.google.com/site/marctorrentburgas |

---

## EMAIL

**To:** marc.torrent@uab.cat

**Subject:** Contribution offer for PyAMPA - alternative predictor

---

Dr. Torrent,

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
