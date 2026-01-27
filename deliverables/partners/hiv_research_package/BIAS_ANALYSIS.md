# HIV Research Package - Bias Analysis

**Date:** 2026-01-26
**Analyst:** AI Whisperers Team
**Status:** API-BASED - DIFFERENT VALIDATION APPROACH NEEDED

---

## Executive Summary

This package is fundamentally different from the DDG and AMP packages:
- **NOT an internally-trained ML model** - relies on Stanford HIVdb API
- **External validation** - Stanford HIVdb is peer-reviewed and clinically validated
- **Demo mode is simulation** - local fallback uses random simulation, not actual sequence analysis

| Issue | Severity | Impact |
|-------|:--------:|--------|
| No scaler leakage | N/A | No ML training |
| Demo mode limitation | **HIGH** | Demo does NOT do real analysis |
| Stanford API dependency | **MEDIUM** | Requires internet access |
| No patient data testing | **HIGH** | Not validated on real sequences |

---

## Package Architecture

```
HIV Package
    │
    ├── Stanford Mode (--use-stanford)
    │   └── Stanford HIVdb API
    │       └── Externally validated, peer-reviewed
    │
    └── Local Mode (default)
        └── Random simulation based on prevalence
            └── NOT REAL ANALYSIS
```

---

## Critical Issue: Demo Mode is Simulation

**File:** `scripts/H6_tdr_screening.py` (lines 157-183)

```python
def detect_mutations(sequence: str, reference: Optional[str] = None) -> list[dict]:
    """Detect TDR mutations in sequence.

    For demo: checks for presence of known mutation amino acids at positions.
    Real implementation would align to HXB2 reference.
    """
    detected = []

    # For demo, we'll simulate mutation detection
    # In reality, this would compare to HXB2 reference

    # Simulate by randomly detecting some mutations based on prevalence
    np.random.seed(hash(sequence[:20]) % 2**32)

    for drug_class, mutations in TDR_MUTATIONS.items():
        for mut_name, mut_info in mutations.items():
            # Probability of detection based on prevalence
            if np.random.random() * 100 < mut_info["prevalence"]:
                detected.append({...})
```

**Problem:** The local analysis does NOT actually detect mutations. It:
1. Seeds random number generator with sequence hash
2. Randomly selects mutations based on population prevalence
3. Does NOT align to HXB2 reference
4. Does NOT actually parse the sequence for mutations

**Impact:** Any demo output is SIMULATED, not analyzed.

---

## Stanford Mode: Externally Validated

When `--use-stanford` is used:
- Real sequence is sent to Stanford HIVdb API
- API performs actual sequence alignment to HXB2
- Returns clinically validated resistance predictions
- API version: v10.1 (verified 2026-01-26)

**Stanford HIVdb is:**
- Peer-reviewed (published methodology)
- Clinically validated
- Used in clinical practice worldwide
- Regularly updated with new resistance data

This is the CORRECT mode for real analysis.

---

## What's Actually Validated

| Component | Validated | Method |
|-----------|:---------:|--------|
| Stanford API connectivity | YES | Direct API test |
| Stanford API response parsing | YES | Fixed 2026-01-26 |
| Local mutation detection | **NO** | Simulated |
| Drug resistance scoring | Partial | Uses Stanford when available |
| Real patient sequences | **NO** | Only demo data tested |

---

## Recommendations

### For Emails

1. **DO NOT** claim the package "detects mutations" without Stanford mode
2. **CAN claim** Stanford HIVdb integration works (verified)
3. **CAN claim** the wrapper provides WHO regimen recommendations
4. **MUST clarify** that local mode is for demo/testing only

### For Production Use

```bash
# CORRECT: Use Stanford HIVdb for real analysis
python scripts/H6_tdr_screening.py --use-stanford --sequence patient.fasta

# INCORRECT: Local mode is DEMO ONLY
python scripts/H6_tdr_screening.py --demo  # This is simulation!
```

### Before Claiming Validation

To properly validate this package, we need:
1. Real HIV patient sequences (not demo)
2. Known resistance profiles (ground truth)
3. Compare Stanford predictions to known profiles
4. Report sensitivity/specificity

---

## No ML Bias Concerns

Unlike DDG and AMP packages:
- No StandardScaler leakage (no ML training)
- No cross-validation bias (no internal model)
- External validation provided by Stanford HIVdb

The validation concern is different: **Does the package correctly use and parse Stanford's validated predictions?**

Answer: YES (after fixes applied 2026-01-26)
- Fixed GraphQL query type (mutation → query)
- Fixed drug_scores parsing (list not dict)
- API connectivity verified

---

## Comparison with Other Packages

| Package | Type | Internal Validation | External Validation |
|---------|------|:-------------------:|:-------------------:|
| protein_stability_ddg | ML (Ridge) | LOO-CV, Bootstrap | N/A |
| antimicrobial_peptides | ML (GBR/VAE) | 5-fold CV, Bootstrap | N/A |
| arbovirus_surveillance | Computational | Skeptical validation | N/A |
| **hiv_research_package** | **API wrapper** | **Demo only** | **Stanford HIVdb** |

---

## Action Items

1. **Documentation:** Clarify that local mode is demo-only
2. **Testing:** Obtain real HIV sequences for validation
3. **Emails:** Only claim Stanford integration, not local analysis
4. **Production:** Always use `--use-stanford` flag

---

*This package is ready for outreach claiming Stanford HIVdb integration, not local analysis capabilities.*
