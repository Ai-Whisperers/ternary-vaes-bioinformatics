# Wet-Lab Validation Protocol: Project "Goldilocks"

**Objective**: Verify that "Hyperbolic Distance" predicts protein stability/fitness.
**Budget**: $10k - $25k (CRO Outsourced)

## Phase 1: The "Kill Sheet" Set (In Silico Selection)

We will select 100 variants of a target protein (e.g., GFP or HLA-B27):

1.  **20 Center Controls**: Wild Type & synonymous known-good (Low radius).
2.  **40 Radius High (Edge)**: Predicted unstable/dead (High radius).
3.  **40 Radius Mid (Goldilocks)**: Predicted stable but functionally distinct (Mid radius).

## Phase 2: Experimental Assay (In Vitro)

**Method**: Deep Mutational Scanning (DMS) or Fluorescence Sort.

- **Step 1**: Synthesize the 100 variants (Twist Bioscience).
- **Step 2**: Express in E. coli or Yeast.
- **Step 3**: Measure expression levels (Stability) and Binding Affinity (Function).

## Phase 3: Success Criteria

- **Hypothesis**: Stability should decay inversely with Hyperbolic Radius.
- **metric**: $R^2 > 0.6$ between Radius and Expression Level.
- **Win**: We demonstrate that "Geometry predicts Biology" without training on specific folding data.
