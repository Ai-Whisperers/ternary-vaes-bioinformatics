# Sentinel Glycans: Inverse Goldilocks Model

**Doc-Type:** Discovery Module | Version 1.0 | Updated 2025-12-24

---

## Overview

The HIV glycan shield masks conserved epitopes from broadly neutralizing antibodies (bnAbs). Using the **Inverse Goldilocks Model**, we identified 7 "sentinel glycans" whose removal optimally exposes bnAb epitopes - shifting them into the immunogenic Goldilocks Zone (15-30% centroid shift).

---

## The Inverse Goldilocks Concept

```mermaid
flowchart LR
    subgraph STANDARD["Standard Goldilocks<br/>(Autoimmunity)"]
        direction TB
        S1["Native Protein"] -->|"+PTM"| S2["Modified Protein"]
        S2 --> S3["Goldilocks Zone<br/>(Immunogenic)"]
    end

    subgraph INVERSE["Inverse Goldilocks<br/>(HIV Vaccine)"]
        direction TB
        I1["Glycosylated Env<br/>(Shielded)"] -->|"-Glycan"| I2["Deglycosylated Env"]
        I2 --> I3["Goldilocks Zone<br/>(bnAb Accessible)"]
    end

    STANDARD -.->|"Opposite<br/>Direction"| INVERSE

    style S3 fill:#ff6b6b,stroke:#c92a2a,color:#fff
    style I3 fill:#69db7c,stroke:#2f9e44,stroke-width:2px
```

---

## Goldilocks Zone Classification

```mermaid
flowchart TB
    subgraph ZONES["P-Adic Shift Classification"]
        BELOW["<b>BELOW GOLDILOCKS</b><br/>&lt; 15% shift<br/>Still shielded"]
        GOLD["<b>GOLDILOCKS ZONE</b><br/>15-30% shift<br/>Optimal exposure"]
        ABOVE["<b>ABOVE GOLDILOCKS</b><br/>&gt; 30% shift<br/>Destabilizing"]
    end

    INPUT["Deglycosylation<br/>(N→Q mutation)"] --> ZONES

    style GOLD fill:#69db7c,stroke:#2f9e44,stroke-width:3px
    style BELOW fill:#74c0fc,stroke:#1864ab
    style ABOVE fill:#ff6b6b,stroke:#c92a2a,color:#fff
```

---

## Sentinel Glycan Results

```mermaid
xychart-beta
    title "Centroid Shift by Glycan Site (%)"
    x-axis ["N58", "N429", "N103", "N204", "N107", "N271", "N265"]
    y-axis "Centroid Shift (%)" 0 --> 35
    bar [22.4, 22.6, 23.7, 25.1, 17.0, 28.4, 29.1]
    line [30, 30, 30, 30, 30, 30, 30]
```

*Green bars = Goldilocks Zone (15-30%). Red line = Upper boundary.*

---

## Sentinel Glycan Map on gp120

```mermaid
flowchart TB
    subgraph GP120["BG505 gp120 Structure"]
        direction LR

        subgraph V1V2["V1/V2 Region"]
            N58["<b>N58</b><br/>22.4%<br/>Score: 1.19"]
            N103["<b>N103</b><br/>23.7%<br/>Score: 1.04"]
            N107["<b>N107</b><br/>17.0%<br/>Score: 0.46"]
        end

        subgraph V3["V3 Region"]
            N204["<b>N204</b><br/>25.1%<br/>Score: 0.85"]
        end

        subgraph C3["C3 Core"]
            N265["<b>N265</b><br/>29.1%<br/>Score: 0.32"]
            N271["<b>N271</b><br/>28.4%<br/>Score: 0.42"]
        end

        subgraph C5["C5 Region"]
            N429["<b>N429</b><br/>22.6%<br/>Score: 1.19"]
        end
    end

    BNAB["bnAb Epitopes<br/>Exposed"] --> GP120

    style N58 fill:#69db7c,stroke:#2f9e44,stroke-width:3px
    style N429 fill:#69db7c,stroke:#2f9e44,stroke-width:3px
    style N103 fill:#69db7c,stroke:#2f9e44,stroke-width:2px
    style N204 fill:#69db7c,stroke:#2f9e44,stroke-width:2px
    style N107 fill:#b2f2bb,stroke:#2f9e44
    style N265 fill:#b2f2bb,stroke:#2f9e44
    style N271 fill:#b2f2bb,stroke:#2f9e44
```

---

## Detailed Sentinel Table

| Rank | Site | Region | Shift | Score | bnAb Relevance | Priority |
|:-----|:-----|:-------|:------|:------|:---------------|:---------|
| 1 | **N58** | V1 | 22.4% | 1.19 | V1/V2 shield | HIGH |
| 2 | **N429** | C5 | 22.6% | 1.19 | Structural | HIGH |
| 3 | **N103** | V2 | 23.7% | 1.04 | PG9/PG16 apex | HIGH |
| 4 | **N204** | V3 | 25.1% | 0.85 | PGT121/128 supersite | HIGH |
| 5 | N107 | V2 | 17.0% | 0.46 | V1/V2 bnAbs | MEDIUM |
| 6 | N271 | C3 | 28.4% | 0.42 | Core glycan | MEDIUM |
| 7 | N265 | C3 | 29.1% | 0.32 | Core glycan | MEDIUM |

---

## Goldilocks Score Calculation

```mermaid
flowchart TB
    subgraph CALC["Goldilocks Score Formula"]
        SHIFT["Centroid Shift<br/>(Δ%)"] --> ZONE
        ZONE["Zone Classification"]

        ZONE -->|"15-30%"| GOLD_SCORE["Zone Score = 1.0 - |Δ - 22.5%| / 7.5%"]
        ZONE -->|"< 15%"| BELOW_SCORE["Zone Score = Δ / 15% × 0.5"]
        ZONE -->|"> 30%"| ABOVE_SCORE["Zone Score = max(0, 1 - (Δ-30%)/20%) × 0.5"]

        BOUNDARY["Boundary Crossed?"] -->|"Yes"| BONUS["+0.2 Bonus"]

        GOLD_SCORE --> FINAL
        BELOW_SCORE --> FINAL
        ABOVE_SCORE --> FINAL
        BONUS --> FINAL["<b>Final Goldilocks Score</b>"]
    end

    style FINAL fill:#69db7c,stroke:#2f9e44,stroke-width:2px
```

---

## bnAb Epitope Mapping

```mermaid
flowchart LR
    subgraph EPITOPES["bnAb Classes & Target Glycans"]
        subgraph V1V2_BNAB["V1/V2 Apex bnAbs"]
            PG9["PG9/PG16"]
            PGT145["PGT145"]
        end

        subgraph V3_BNAB["V3 Supersite bnAbs"]
            PGT121["PGT121"]
            PGT128["PGT128"]
        end

        subgraph CD4BS["CD4 Binding Site"]
            VRC01["VRC01-class"]
        end
    end

    N58 & N103 & N107 --> V1V2_BNAB
    N204 --> V3_BNAB

    style PG9 fill:#74c0fc,stroke:#1864ab
    style PGT121 fill:#b197fc,stroke:#7048e8
    style VRC01 fill:#ffd43b,stroke:#fab005
```

---

## Vaccine Immunogen Designs

```mermaid
flowchart TB
    subgraph DESIGNS["Recommended Immunogen Constructs"]
        D1["<b>Triple Sentinel</b><br/>N58Q + N103Q + N204Q<br/>Exposes V1/V2 + V3"]
        D2["<b>V1/V2 Focused</b><br/>N103Q + N107Q<br/>PG9/PG16 targets"]
        D3["<b>V3 Focused</b><br/>N204Q alone<br/>PGT121/128 targets"]
        D4["<b>All Goldilocks</b><br/>7-site removal<br/>Maximum exposure"]
    end

    BG505["BG505 SOSIP.664"] --> DESIGNS

    style D1 fill:#69db7c,stroke:#2f9e44,stroke-width:3px
    style BG505 fill:#74c0fc,stroke:#1864ab
```

### Design Priority

| Design | Sites | Target bnAbs | Complexity | Priority |
|:-------|:------|:-------------|:-----------|:---------|
| Triple Sentinel | N58Q, N103Q, N204Q | PG9, PGT121 | Low | **1** |
| V1/V2 Focused | N103Q, N107Q | PG9, PG16 | Low | **2** |
| V3 Focused | N204Q | PGT121, PGT128 | Very Low | **3** |
| All Goldilocks | 7 sites | Broad | High | 4 |

---

## AlphaFold3 Structural Validation

```mermaid
xychart-beta
    title "Goldilocks Score vs Structural Disorder (AF3)"
    x-axis ["N58", "N429", "N103", "N204", "N246", "N152"]
    y-axis "Disorder %" 0 --> 100
    bar [75, 100, 67, 68, 63, 61]
```

**Correlation:** r = -0.89 (Goldilocks score inversely correlates with structural stability)

---

## Distribution Summary

```mermaid
pie showData
    title "Glycan Site Distribution (n=24)"
    "Goldilocks Zone (15-30%)" : 7
    "Above Goldilocks (>30%)" : 17
    "Below Goldilocks (<15%)" : 0
```

---

## Key Insights

1. **7 sentinel glycans identified** in the Goldilocks Zone
2. **Top sites (N58, N429)** have near-perfect Goldilocks scores (1.19)
3. **V1/V2 and V3 regions** contain the most promising targets
4. **AlphaFold3 validates** structural sensitivity at predicted sites
5. **Multi-site removal** shows synergistic effects

---

## Related Documents

- [Elite Controllers](./02_ELITE_CONTROLLERS.md)
- [Therapeutic Applications](./04_THERAPEUTIC_APPLICATIONS.md)
- [Validation Results](./05_VALIDATION_RESULTS.md)
- [Glycan Shield Analysis](../glycan_shield/README.md)

---

**Navigation:** [← Elite Controllers](./02_ELITE_CONTROLLERS.md) | [Index](./README.md) | [Applications →](./04_THERAPEUTIC_APPLICATIONS.md)
