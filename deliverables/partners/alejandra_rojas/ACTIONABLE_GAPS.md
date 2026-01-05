# Rojas Package: Actionable Gap Analysis

**Doc-Type:** Gap Analysis · Version 1.0 · 2026-01-05 · AI Whisperers

---

## Executive Summary

The Rojas package has **excellent scientific validation** but **critical implementation failures**. The p-adic analysis successfully identified novel primer targets, but no actual usable primers have been generated.

| Category | Status | Evidence |
|----------|--------|----------|
| Scientific analysis | COMPLETE | 4 conjectures tested, root cause identified |
| P-adic integration | COMPLETE | E gene (2400) identified as best target |
| Phylogenetic analysis | COMPLETE | 270 genomes, 5 clades mapped |
| **Primer design** | **FAILED** | 0 usable primers generated |
| **In silico validation** | **NOT DONE** | 0% coverage verified |
| **Pan-arbovirus library** | **FAILED** | 0 specific primers |

---

## Critical Failures

### 1. Degenerate Primer Design - FAILED

**File:** `results/phylogenetic/degenerate_primer_results.json`

```json
{
  "designed_primers": [],
  "coverage_pct": 0.0,
  "error": "No primers with acceptable degeneracy found"
}
```

**Root cause:** DENV-4 diversity too high - all candidate windows have degeneracy >10^28 (unusable).

**Action required:** Design primers for individual clades, not pan-DENV-4.

---

### 2. Pan-Arbovirus Library - FAILED

**File:** `results/pan_arbovirus_primers/library_summary.json`

| Virus | Total Primers | Specific Primers | Pairs |
|-------|---------------|------------------|-------|
| DENV-1 | 10 | **0** | **0** |
| DENV-2 | 10 | **0** | **0** |
| DENV-3 | 10 | **0** | **0** |
| DENV-4 | 10 | **0** | **0** |
| ZIKV | 10 | **0** | **0** |
| CHIKV | 10 | **0** | **0** |
| MAYV | 10 | **0** | **0** |

**All primers have `is_specific: False`** - cross-reactivity not tested.

**Action required:** Re-run with proper cross-reactivity filtering against non-target genomes.

---

### 3. Tiered Detection - NOT VALIDATED

**File:** `results/tiered_detection/tiered_detection_results.json`

```json
"validation": {
  "tier1": {
    "primers_tested": 2,
    "sequences_matched": 0,     // <-- CRITICAL: 0% binding
    "coverage_pct": 0.0
  }
}
```

**Tier 1 primers (DENV4_E32_NS5_F/R) designed but NEVER validated in silico.**

**Action required:** Run in silico PCR against all 270 sequences.

---

### 4. E Gene Primers - NOT DESIGNED

**Discovery:** P-adic analysis identified E gene position 2400 as best target (hyp_var = 0.0183).

**Current state:** No actual primer sequences generated for this position.

**Action required:** Design primers for E gene 2400-2500 region.

---

### 5. Subclade Coverage - 13.3% ONLY

**File:** `results/phylogenetic/subclade_analysis_results.json`

| Subclade | Size | Has Primers | Conserved Windows |
|----------|------|-------------|-------------------|
| Clade_E.3.2 | 36 | Yes | 1 |
| Clade_E.1.3.2.1 | 88 | **No** | **0** |
| Clade_E.1.3.2.3 | 58 | **No** | **0** |
| Clade_D.* | 42 | **No** | **0** |
| Others | 46 | **No** | **0** |

**Only 36/270 sequences (13.3%) have primers.** 234 sequences (86.7%) remain uncovered.

---

## Data Integration Gaps

### Not Integrated

| Data Source | Status | Impact |
|-------------|--------|--------|
| E gene position 2400 | Identified but unused | Best p-adic candidate ignored |
| NS1 position 3000 | Identified but unused | Second-best candidate ignored |
| Codon pair bias table | Generated but unused | 4,052 pairs not leveraged |
| Per-clade hyperbolic variance | Not computed | Could identify clade-specific targets |

### Partially Validated

| Analysis | Validated | Missing Validation |
|----------|-----------|-------------------|
| P-adic vs Shannon | Correlation tested | No predictive power test |
| Hyperbolic variance | Genome-wide scan | No per-clade breakdown |
| Conjecture testing | 4 hypotheses | No positive conjecture found |

---

## Actionable Roadmap

### Priority 1: Design E Gene Primers (1-2 hours)

```python
# Target: E gene position 2400-2500
# Rationale: Lowest hyperbolic variance (0.0183)
# Expected: Better cross-clade coverage than NS5

ACTION:
1. Extract consensus sequence at position 2400-2500
2. Design forward primer (2400-2420)
3. Design reverse primer (2475-2495)
4. Allow 2-3 degenerate positions
5. Validate Tm, GC, no hairpins
```

### Priority 2: In Silico PCR Validation (2-3 hours)

```python
# Test ALL primer candidates against 270 genomes
# Include: Tier 1, E gene, NS1, NS5

ACTION:
1. Implement in silico PCR function
2. Allow 0-2 mismatches
3. Report: exact matches, 1mm, 2mm, no match
4. Generate coverage matrix per clade
```

### Priority 3: Clade-Specific Primers (3-4 hours)

```python
# DENV-4 is too diverse for pan-serotype primers
# Strategy: Design primers for each major subclade

ACTION:
1. For each subclade with >10 sequences:
   - Compute entropy within subclade
   - Find conserved windows (entropy < 0.3)
   - Design subclade-specific primers
2. Create multiplex cocktail
```

### Priority 4: Pan-Arbovirus Re-validation (2-3 hours)

```python
# Current library has 0 specific primers
# Need proper cross-reactivity testing

ACTION:
1. Download representative sequences: DENV-1/2/3, ZIKV, CHIKV, MAYV
2. For each primer candidate:
   - Test binding against target virus
   - Test binding against non-targets
   - Mark specific only if: target >90%, non-target <20%
```

---

## Scripts Needed

| Script | Purpose | Priority |
|--------|---------|----------|
| `design_egene_primers.py` | Design primers for E gene 2400 | HIGH |
| `insilico_pcr_validation.py` | Validate all primers against 270 genomes | HIGH |
| `clade_specific_primer_design.py` | Design primers per subclade | HIGH |
| `revalidate_pan_arbovirus.py` | Fix pan-arbovirus library | MEDIUM |
| `compute_clade_hyperbolic_var.py` | P-adic analysis per clade | MEDIUM |

---

## Success Metrics

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Usable DENV-4 primers | 0 | >5 pairs | CRITICAL |
| Strain coverage | 13.3% | >50% | CRITICAL |
| In silico validated | 0% | 100% | CRITICAL |
| Pan-arbovirus specific | 0/70 | >35/70 | HIGH |
| E gene primers designed | 0 | 2 pairs | HIGH |

---

## Quick Wins (Can Do Now)

1. **Extract E gene 2400-2500 consensus** - 15 min
2. **Design E gene primer pair manually** - 30 min
3. **Run simple string matching for Tier 1 primers** - 30 min
4. **Generate clade-stratified hyperbolic variance** - 1 hour

---

## Conclusion

The scientific foundation is solid, but **no usable primers exist**. The immediate priority is:

1. Design E gene primers (position 2400) - our novel finding
2. Validate ALL primer candidates in silico
3. Accept that pan-DENV-4 is impossible; design clade-specific cocktails

**Minimum viable deliverable:** E gene primer pair + Tier 1 primer pair, both validated against 270 genomes.

---

*Gap analysis: 2026-01-05*
*IICS-UNA Arbovirus Surveillance Program*
