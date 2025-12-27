Codon Analysis Tutorial
=======================

This tutorial provides a deep dive into p-adic hyperbolic codon embeddings.


The Mathematical Foundation
---------------------------

Codons (nucleotide triplets) naturally map to a ternary number system:

- Each nucleotide position has 4 options (A, T, G, C)
- Each codon represents one of 64 amino acid assignments
- The genetic code has deep algebraic structure

P-adic numbers provide a natural metric for this structure.


P-adic Distance
---------------

The 3-adic valuation measures how divisible a number is by 3:

.. code-block:: python

    from src.geometry import padic_valuation, padic_distance

    # 3-adic valuation examples
    print(padic_valuation(9, 3))   # 2 (9 = 3^2)
    print(padic_valuation(12, 3))  # 1 (12 = 3 * 4)
    print(padic_valuation(5, 3))   # 0 (5 not divisible by 3)

    # Distance between codons
    d = padic_distance("ATG", "ATC", p=3)
    print(f"Distance: {d}")


Codon Indexing
--------------

Map codons to ternary indices:

.. code-block:: python

    from src.biology import CODON_TO_INDEX, INDEX_TO_CODON

    # Standard indexing
    print(CODON_TO_INDEX["ATG"])  # Start codon
    print(CODON_TO_INDEX["TAA"])  # Stop codon

    # Reverse lookup
    print(INDEX_TO_CODON[0])   # First codon
    print(INDEX_TO_CODON[63])  # Last codon


Hyperbolic Embedding
--------------------

The Poincaré ball model provides natural hierarchy:

- Center = "neutral" codons
- Boundary = "extreme" codons
- Distance grows exponentially near boundary

.. code-block:: python

    from src.geometry import poincare_distance, exp_map, log_map

    # Points in Poincaré ball (|x| < 1)
    x = torch.tensor([0.1, 0.2])
    y = torch.tensor([0.5, 0.6])

    # Hyperbolic distance
    d = poincare_distance(x, y)

    # Exponential map (tangent space -> manifold)
    v = torch.tensor([0.1, 0.0])  # Tangent vector at origin
    z = exp_map(v, origin=torch.zeros(2))


Amino Acid Properties
---------------------

Group codons by amino acid properties:

.. code-block:: python

    from src.biology import (
        GENETIC_CODE,
        HYDROPHOBIC_AA,
        CHARGED_AA,
        POLAR_AA
    )

    # Categorize by property
    hydrophobic_codons = [
        codon for codon, aa in GENETIC_CODE.items()
        if aa in HYDROPHOBIC_AA
    ]

    charged_codons = [
        codon for codon, aa in GENETIC_CODE.items()
        if aa in CHARGED_AA
    ]


Synonymous Mutations
--------------------

Analyze mutations that don't change amino acid:

.. code-block:: python

    from src.biology import get_synonymous_codons

    # All codons encoding Leucine
    leu_codons = get_synonymous_codons("L")
    print(f"Leucine codons: {leu_codons}")

    # Calculate p-adic distances within synonymous group
    for i, c1 in enumerate(leu_codons):
        for c2 in leu_codons[i+1:]:
            d = padic_distance(c1, c2, p=3)
            print(f"{c1} <-> {c2}: {d:.4f}")


Wobble Position Analysis
------------------------

The third codon position (wobble) often tolerates mutations:

.. code-block:: python

    from src.geometry import position_specific_distance

    # Compare mutations at each position
    codon1 = "ATG"

    for pos in range(3):
        for base in "ATGC":
            codon2 = list(codon1)
            codon2[pos] = base
            codon2 = "".join(codon2)

            if codon1 != codon2:
                d = padic_distance(codon1, codon2, p=3)
                aa1 = GENETIC_CODE[codon1]
                aa2 = GENETIC_CODE[codon2]
                print(f"Pos {pos}: {codon1}->{codon2} ({aa1}->{aa2}): d={d:.4f}")


Codon Usage Bias
----------------

Organisms prefer certain synonymous codons:

.. code-block:: python

    from collections import Counter

    def analyze_codon_usage(sequences):
        """Analyze codon usage bias in a set of sequences."""
        codon_counts = Counter()

        for seq in sequences:
            # Extract codons
            for i in range(0, len(seq) - 2, 3):
                codon = seq[i:i+3]
                if codon in GENETIC_CODE:
                    codon_counts[codon] += 1

        return codon_counts

    # Group by amino acid
    def codon_bias_by_aa(counts):
        """Calculate relative usage within synonymous groups."""
        from src.biology import AMINO_ACID_TO_CODONS

        bias = {}
        for aa, codons in AMINO_ACID_TO_CODONS.items():
            total = sum(counts[c] for c in codons)
            if total > 0:
                bias[aa] = {c: counts[c] / total for c in codons}
        return bias


Visualization
-------------

Visualize the codon space:

.. code-block:: python

    import matplotlib.pyplot as plt
    from src.visualization import plot_codon_embedding

    # Get embeddings
    encoder = CodonEncoder()
    embeddings = encoder.encode_all_codons()

    # Plot with amino acid coloring
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_codon_embedding(
        embeddings,
        color_by="amino_acid",
        ax=ax
    )
    plt.title("Codon Embedding Space")
    plt.show()


Key Insights
------------

1. **Synonymous codons cluster**: Codons for the same amino acid are
   geometrically close

2. **Wobble tolerance**: Third-position mutations have smaller distances

3. **Stop codon isolation**: Stop codons are geometrically distant from
   coding codons

4. **Property gradients**: Hydrophobic/hydrophilic amino acids show
   geometric separation
