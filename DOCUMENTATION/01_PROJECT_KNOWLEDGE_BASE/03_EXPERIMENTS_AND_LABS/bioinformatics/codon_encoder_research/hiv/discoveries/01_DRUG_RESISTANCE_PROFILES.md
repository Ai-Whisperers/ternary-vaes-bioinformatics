# Drug Resistance Geometric Profiles

**Doc-Type:** Discovery Module | Version 1.0 | Updated 2025-12-24

---

## Overview

Each antiretroviral drug class has a characteristic p-adic distance profile reflecting evolutionary constraint on its target site. Drugs targeting conserved active sites force the virus to make larger geometric "jumps" to escape - jumps that carry significant fitness costs.

---

## Drug Class Distance Hierarchy

```mermaid
graph TB
    subgraph "P-Adic Distance by Drug Class"
        NRTI["<b>NRTI</b><br/>d = 6.05 ± 1.28<br/>RT Active Site"]
        NNRTI["<b>NNRTI</b><br/>d = 5.34 ± 1.40<br/>Allosteric Pocket"]
        INSTI["<b>INSTI</b><br/>d = 5.16 ± 1.45<br/>Integrase Active Site"]
        PI["<b>PI</b><br/>d = 3.60 ± 2.01<br/>Protease"]
    end

    HIGH["HIGH CONSTRAINT<br/>Active Sites"] --> NRTI
    HIGH --> INSTI

    LOW["LOW CONSTRAINT<br/>Flexible Regions"] --> NNRTI
    LOW --> PI

    style NRTI fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px,color:#fff
    style INSTI fill:#ff8787,stroke:#c92a2a,stroke-width:2px,color:#fff
    style NNRTI fill:#74c0fc,stroke:#1864ab,stroke-width:2px
    style PI fill:#69db7c,stroke:#2f9e44,stroke-width:2px
    style HIGH fill:#ffe066,stroke:#fab005,stroke-width:2px
    style LOW fill:#d0bfff,stroke:#7048e8,stroke-width:2px
```

---

## Target Site Constraint Map

```mermaid
quadrantChart
    title Drug Escape Distance vs Target Conservation
    x-axis Low Conservation --> High Conservation
    y-axis Low Escape Distance --> High Escape Distance
    quadrant-1 "High Barrier (Optimal Targets)"
    quadrant-2 "Unexpected Low Distance"
    quadrant-3 "Easy Escape (Avoid)"
    quadrant-4 "Moderate Barrier"

    NRTI: [0.85, 0.80]
    INSTI: [0.80, 0.68]
    NNRTI: [0.45, 0.70]
    PI: [0.35, 0.48]
```

---

## Detailed Mutation Analysis

### NRTI Mutations (Nucleoside RT Inhibitors)

```mermaid
flowchart LR
    subgraph NRTI["NRTI Resistance Mutations"]
        M184V["M184V<br/>d = 4.00<br/>3TC/FTC"]
        K65R["K65R<br/>d = 7.41<br/>TDF/ABC"]
        K70R["K70R<br/>d = 7.41<br/>AZT/D4T"]
        T215Y["T215Y<br/>d = 6.06<br/>TAM"]
        L74V["L74V<br/>d = 4.63<br/>ABC/DDI"]
    end

    RT["RT Active Site"] --> NRTI

    style K65R fill:#ff6b6b,stroke:#c92a2a,color:#fff
    style K70R fill:#ff6b6b,stroke:#c92a2a,color:#fff
    style T215Y fill:#ff8787,stroke:#c92a2a,color:#fff
```

| Mutation | Distance | Drugs | Fitness Cost |
|:---------|:---------|:------|:-------------|
| K65R | 7.41 | TDF, ABC | Moderate |
| K70R | 7.41 | AZT, D4T | Moderate |
| T215Y | 6.06 | AZT, D4T | Minimal |
| L74V | 4.63 | ABC, DDI | Moderate |
| M184V | 4.00 | 3TC, FTC | Moderate |

**Mean: 6.05 ± 1.28**

---

### INSTI Mutations (Integrase Inhibitors)

```mermaid
flowchart LR
    subgraph INSTI["INSTI Resistance Mutations"]
        R263K["R263K<br/>d = 7.41<br/>DTG"]
        Y143R["Y143R<br/>d = 5.72<br/>RAL"]
        N155H["N155H<br/>d = 4.19<br/>RAL/EVG"]
        Q148H["Q148H<br/>d = 4.27<br/>RAL/EVG/DTG"]
        E92Q["E92Q<br/>d = 4.19<br/>RAL/EVG"]
    end

    IN["Integrase Active Site<br/>DDE Catalytic Triad"] --> INSTI

    style R263K fill:#ff6b6b,stroke:#c92a2a,color:#fff
    style Y143R fill:#ff8787,stroke:#c92a2a,color:#fff
```

| Mutation | Distance | Drugs | Fitness Cost |
|:---------|:---------|:------|:-------------|
| R263K | 7.41 | DTG | High |
| Y143R | 5.72 | RAL | Moderate |
| Q148H | 4.27 | RAL, EVG, DTG | Moderate |
| N155H | 4.19 | RAL, EVG | Moderate |
| E92Q | 4.19 | RAL, EVG | Minimal |

**Mean: 5.16 ± 1.45**

---

### NNRTI Mutations (Non-Nucleoside RT Inhibitors)

```mermaid
flowchart LR
    subgraph NNRTI["NNRTI Resistance Mutations"]
        K103N["K103N<br/>d = 6.89<br/>EFV/NVP"]
        Y181C["Y181C<br/>d = 5.27<br/>NVP/EFV"]
        G190A["G190A<br/>d = 4.63<br/>NVP/EFV"]
        K101E["K101E<br/>d = 4.58<br/>NVP/EFV"]
    end

    POCKET["Allosteric Binding Pocket<br/>(Not Active Site)"] --> NNRTI

    style K103N fill:#ff8787,stroke:#c92a2a,color:#fff
```

| Mutation | Distance | Drugs | Fitness Cost |
|:---------|:---------|:------|:-------------|
| K103N | 6.89 | EFV, NVP | Minimal |
| Y181C | 5.27 | NVP, EFV | Minimal |
| G190A | 4.63 | NVP, EFV | Minimal |
| K101E | 4.58 | NVP, EFV | Minimal |

**Mean: 5.34 ± 1.40**

---

### PI Mutations (Protease Inhibitors)

```mermaid
flowchart LR
    subgraph PI["PI Resistance Mutations"]
        I84V["I84V<br/>d = 6.43<br/>DRV/ATV"]
        M46I["M46I<br/>d = 3.39<br/>IDV/NFV"]
        V82A["V82A<br/>d = 2.41<br/>IDV/RTV"]
        L90M["L90M<br/>d = 2.18<br/>SQV/NFV"]
    end

    PROT["Protease<br/>(More Flexible)"] --> PI

    style I84V fill:#ff8787,stroke:#c92a2a,color:#fff
    style L90M fill:#69db7c,stroke:#2f9e44
```

| Mutation | Distance | Drugs | Fitness Cost |
|:---------|:---------|:------|:-------------|
| I84V | 6.43 | DRV, ATV | Moderate |
| M46I | 3.39 | IDV, NFV | Minimal |
| V82A | 2.41 | IDV, RTV | Minimal |
| L90M | 2.18 | SQV, NFV | Minimal |

**Mean: 3.60 ± 2.01**

---

## Combination Therapy Implications

```mermaid
flowchart TB
    subgraph HIGH["HIGH BARRIER COMBINATIONS"]
        direction LR
        H1["INSTI + NRTI"] --> H1R["d ≈ 11.2"]
        H2["INSTI + NNRTI"] --> H2R["d ≈ 10.5"]
    end

    subgraph LOW["LOWER BARRIER COMBINATIONS"]
        direction LR
        L1["NNRTI + PI"] --> L1R["d ≈ 8.9"]
        L2["PI + PI"] --> L2R["d ≈ 7.2"]
    end

    REC["<b>RECOMMENDATION</b><br/>Use high-barrier combinations"] --> HIGH

    style HIGH fill:#d3f9d8,stroke:#2f9e44,stroke-width:2px
    style LOW fill:#ffe3e3,stroke:#c92a2a,stroke-width:2px
    style REC fill:#fff3bf,stroke:#fab005,stroke-width:2px
```

---

## Key Insights

1. **NRTIs have highest constraint** - RT active site is catalytically essential
2. **INSTIs target conserved DDE triad** - Metal coordination must be preserved
3. **NNRTIs escape more easily** - Allosteric pocket tolerates substitutions
4. **PIs show high variability** - Protease is structurally more flexible

---

## Related Documents

- [Main Discovery Report](./DISCOVERY_HIV_PADIC_RESISTANCE.md)
- [Elite Controller Mechanism](./02_ELITE_CONTROLLERS.md)
- [Therapeutic Applications](./04_THERAPEUTIC_APPLICATIONS.md)

---

**Navigation:** [← Back to Index](./README.md) | [Next: Elite Controllers →](./02_ELITE_CONTROLLERS.md)
