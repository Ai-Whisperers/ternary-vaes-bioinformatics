# Ternary VAE Bioinformatics Documentation

**Unified Documentation Hub**

This directory is divided into two primary sections:

## 🧠 01 Project Knowledge Base (`/01_PROJECT_KNOWLEDGE_BASE`)

_The "Static" Truth._

- **00 Strategy & Vision**: Pitch decks, long-term goals.
- **01 Presentation Suite**: Public-facing assets (Grants, Posters).
- **02 Theory**: Mathematical and Biological foundations.
- **03 Experiments**: Juypter notebooks and research logs.
- **04 Scientific History**: Archive of past discoveries.
- **05 Legal**: IP and Licensing.

## 🏗️ 02 Project Management (`/02_PROJECT_MANAGEMENT`)

_The "Active" Execution._

- **00 Tasks**: Actionable P0/P1 items.
- **01 Roadmaps & Plans**: Quarterly strategies and improvement plans.
- **02 Code Health Metrics**: Linting, auditing, and technical debt reports.
- **03 Archive**: Deprecated plans.

## 📐 Architecture Overview

<!-- embed: DOCUMENTATION/06_DIAGRAMS/01_ARCHITECTURE/models/ternary_vae_v5_composition.mmd -->
```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#2196f3', 'edgeLabelBackground':'#f9f9f9', 'tertiaryColor': '#e1e4e8'}}}%%
classDiagram
    classDef frozen fill:#e1e4e8,stroke:#333,stroke-dasharray: 5 5;
    classDef trainable fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px;
    classDef hyperbolic fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px;
    
    note "V5.11 Composition"

    class TernaryVAEV5_11 {
        +FrozenEncoder encoder_A
        +FrozenEncoder encoder_B
        +DualHyperbolicProjection projection
        +DifferentiableController controller
        +FrozenDecoder decoder_A
        +forward(x)
    }
    class FrozenEncoder:::frozen {
        <<Frozen>>
        +encode(x) -> (mu, logvar)
    }
    class FrozenDecoder:::frozen {
        <<Frozen>>
        +decode(z) -> logits
    }
    class DualHyperbolicProjection:::hyperbolic {
        +forward(z_A, z_B)
    }
    class DifferentiableController:::trainable {
        +forward(stats)
    }
    
    TernaryVAEV5_11 *-- FrozenEncoder
    TernaryVAEV5_11 *-- FrozenDecoder
    TernaryVAEV5_11 *-- DualHyperbolicProjection
    TernaryVAEV5_11 *-- DifferentiableController
```
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#2196f3', 'edgeLabelBackground':'#f9f9f9', 'tertiaryColor': '#e1e4e8'}}}%%
classDiagram
    classDef frozen fill:#e1e4e8,stroke:#333,stroke-dasharray: 5 5;
    classDef trainable fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px;
    classDef hyperbolic fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px;
    
    note "V5.11 Composition"

    class TernaryVAEV5_11 {
        +FrozenEncoder encoder_A
        +FrozenEncoder encoder_B
        +DualHyperbolicProjection projection
        +DifferentiableController controller
        +FrozenDecoder decoder_A
        +forward(x)
    }
    class FrozenEncoder:::frozen {
        <<Frozen>>
        +encode(x) -> (mu, logvar)
    }
    class FrozenDecoder:::frozen {
        <<Frozen>>
        +decode(z) -> logits
    }
    class DualHyperbolicProjection:::hyperbolic {
        +forward(z_A, z_B)
    }
    class DifferentiableController:::trainable {
        +forward(stats)
    }
    
    TernaryVAEV5_11 *-- FrozenEncoder
    TernaryVAEV5_11 *-- FrozenDecoder
    TernaryVAEV5_11 *-- DualHyperbolicProjection
    TernaryVAEV5_11 *-- DifferentiableController
```
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#2196f3', 'edgeLabelBackground':'#f9f9f9', 'tertiaryColor': '#e1e4e8'}}}%%
classDiagram
    classDef frozen fill:#e1e4e8,stroke:#333,stroke-dasharray: 5 5;
    classDef trainable fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px;
    classDef hyperbolic fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px;
    
    note "V5.11 Composition"

    class TernaryVAEV5_11 {
        +FrozenEncoder encoder_A
        +FrozenEncoder encoder_B
        +DualHyperbolicProjection projection
        +DifferentiableController controller
        +FrozenDecoder decoder_A
        +forward(x)
    }
    class FrozenEncoder:::frozen {
        <<Frozen>>
        +encode(x) -> (mu, logvar)
    }
    class FrozenDecoder:::frozen {
        <<Frozen>>
        +decode(z) -> logits
    }
    class DualHyperbolicProjection:::hyperbolic {
        +forward(z_A, z_B)
    }
    class DifferentiableController:::trainable {
        +forward(stats)
    }
    
    TernaryVAEV5_11 *-- FrozenEncoder
    TernaryVAEV5_11 *-- FrozenDecoder
    TernaryVAEV5_11 *-- DualHyperbolicProjection
    TernaryVAEV5_11 *-- DifferentiableController
```
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#2196f3', 'edgeLabelBackground':'#f9f9f9', 'tertiaryColor': '#e1e4e8'}} }%%
classDiagram
    classDef frozen fill:#e1e4e8,stroke:#333,stroke-dasharray: 5 5;
    classDef trainable fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px;
    classDef hyperbolic fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px;
    
    note "V5.11 Composition"

    class TernaryVAEV5_11 {
        +FrozenEncoder encoder_A
        +FrozenEncoder encoder_B
        +DualHyperbolicProjection projection
        +DifferentiableController controller
        +FrozenDecoder decoder_A
        +forward(x)
    }
    class FrozenEncoder:::frozen {
        <<Frozen>>
        +encode(x) -> (mu, logvar)
    }
    class FrozenDecoder:::frozen {
        <<Frozen>>
        +decode(z) -> logits
    }
    class DualHyperbolicProjection:::hyperbolic {
        +forward(z_A, z_B)
    }
    class DifferentiableController:::trainable {
        +forward(stats)
    }
    
    TernaryVAEV5_11 *-- FrozenEncoder
    TernaryVAEV5_11 *-- FrozenDecoder
    TernaryVAEV5_11 *-- DualHyperbolicProjection
    TernaryVAEV5_11 *-- DifferentiableController
```
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#2196f3', 'edgeLabelBackground':'#f9f9f9', 'tertiaryColor': '#e1e4e8'}} }%%
classDiagram
    classDef frozen fill:#e1e4e8,stroke:#333,stroke-dasharray: 5 5;
    classDef trainable fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px;
    classDef hyperbolic fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px;
    
    note "V5.11 Composition"

    class TernaryVAEV5_11 {
        +FrozenEncoder encoder_A
        +FrozenEncoder encoder_B
        +DualHyperbolicProjection projection
        +DifferentiableController controller
        +FrozenDecoder decoder_A
        +forward(x)
    }
    class FrozenEncoder:::frozen {
        <<Frozen>>
        +encode(x) -> (mu, logvar)
    }
    class FrozenDecoder:::frozen {
        <<Frozen>>
        +decode(z) -> logits
    }
    class DualHyperbolicProjection:::hyperbolic {
        +forward(z_A, z_B)
    }
    class DifferentiableController:::trainable {
        +forward(stats)
    }
    
    TernaryVAEV5_11 *-- FrozenEncoder
    TernaryVAEV5_11 *-- FrozenDecoder
    TernaryVAEV5_11 *-- DualHyperbolicProjection
    TernaryVAEV5_11 *-- DifferentiableController
```

---

_Generated by Antigravity - Reorganization v4_
