HIV Drug Resistance Tutorial
============================

This tutorial demonstrates analyzing HIV drug resistance mutations using
hyperbolic codon embeddings.


Background
----------

HIV drug resistance occurs through mutations that reduce drug binding while
maintaining viral fitness. This tutorial shows how hyperbolic geometry
captures the geometric structure of these resistance patterns.


Dataset Overview
----------------

We use the Stanford HIV Drug Resistance Database:

- **PI** (Protease Inhibitors): 2,171 records
- **NRTI** (Nucleoside RT Inhibitors): 1,867 records
- **NNRTI** (Non-Nucleoside RT Inhibitors): 2,270 records
- **INI** (Integrase Inhibitors): 846 records


Loading the Data
----------------

.. code-block:: python

    from src.data.hiv import load_stanford_hivdb, parse_mutation_list

    # Load protease inhibitor data
    df = load_stanford_hivdb(drug_class="pi")
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")


Understanding Mutations
-----------------------

Mutations are encoded as position + amino acid change:

.. code-block:: python

    # Parse mutation strings
    sample_mutations = df.iloc[0]["CompMutList"]
    print(f"Raw: {sample_mutations}")

    mutations = parse_mutation_list(sample_mutations)
    for mut in mutations:
        print(f"  Position {mut['position']}: {mut['wild_type']} -> {mut['mutant']}")


Encoding Mutations as Codons
----------------------------

.. code-block:: python

    from src.encoders import CodonEncoder
    from src.biology import AMINO_ACID_TO_CODONS

    encoder = CodonEncoder()

    # Get codons for each mutation
    for mut in mutations:
        wild_codons = AMINO_ACID_TO_CODONS.get(mut['wild_type'], [])
        mutant_codons = AMINO_ACID_TO_CODONS.get(mut['mutant'], [])

        print(f"{mut['wild_type']} codons: {wild_codons}")
        print(f"{mut['mutant']} codons: {mutant_codons}")


Hyperbolic Embedding
--------------------

.. code-block:: python

    from src.geometry import padic_distance, hyperbolic_distance

    # Encode mutation to hyperbolic space
    embeddings = encoder.encode_sequence(df["SeqNA"].iloc[0])

    # Calculate geometric distances
    distances = hyperbolic_distance(embeddings)


Resistance Correlation
----------------------

Test the hypothesis that larger hyperbolic distances correlate with higher
drug resistance:

.. code-block:: python

    import pandas as pd
    import numpy as np

    # Extract fold-change values for each drug
    drug_columns = ['FPV', 'ATV', 'IDV', 'LPV', 'NFV', 'SQV', 'TPV', 'DRV']

    results = []
    for _, row in df.iterrows():
        mutations = parse_mutation_list(row["CompMutList"])

        for mut in mutations:
            total_distance = calculate_mutation_distance(mut)

            for drug in drug_columns:
                fold_change = row[drug]
                if pd.notna(fold_change):
                    results.append({
                        'mutation': f"{mut['wild_type']}{mut['position']}{mut['mutant']}",
                        'drug': drug,
                        'fold_change': fold_change,
                        'hyperbolic_distance': total_distance
                    })

    results_df = pd.DataFrame(results)


Visualization
-------------

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Scatter plot of distance vs fold-change
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=results_df,
        x='hyperbolic_distance',
        y='fold_change',
        hue='drug',
        alpha=0.6,
        ax=ax
    )

    ax.set_xlabel('Hyperbolic Distance from Wild-Type')
    ax.set_ylabel('Fold-Change Resistance')
    ax.set_yscale('log')
    ax.set_title('Drug Resistance vs Geometric Distance')

    plt.tight_layout()
    plt.show()


Primary vs Accessory Mutations
------------------------------

Classify mutations by their geometric properties:

.. code-block:: python

    from sklearn.cluster import KMeans

    # Cluster mutations by geometric features
    features = results_df[['hyperbolic_distance', 'radial_position']].values
    kmeans = KMeans(n_clusters=2, random_state=42)
    results_df['mutation_class'] = kmeans.fit_predict(features)

    # Primary mutations: higher distances, stronger resistance
    # Accessory mutations: smaller distances, compensatory role


Running Full Analysis
---------------------

Use the CLI for complete analysis:

.. code-block:: bash

    ternary-vae analyze stanford --drug-class pi --save-figures


Key Findings
------------

Expected results from this analysis:

1. **Distance-Resistance Correlation**: Higher hyperbolic distances correlate
   with stronger resistance (higher fold-change values)

2. **Mutation Classification**: Primary resistance mutations occupy
   geometrically distinct regions from accessory mutations

3. **Cross-Resistance**: Mutations conferring resistance to multiple drugs
   cluster in specific geometric regions


Next Steps
----------

- Analyze other drug classes (NRTI, NNRTI, INI)
- Integrate with CTL epitope data for trade-off analysis
- Build predictive models using geometric features
