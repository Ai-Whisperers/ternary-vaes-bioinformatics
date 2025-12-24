# p-ClustVal: A Novel p-adic Approach for Enhanced Clustering of High-Dimensional scRNASeq Data

**Author:** Becker, Sharma, Mishra, Kurban, Dalkilic
**Year:** 2024
**Link:** [BioRxiv](https://www.biorxiv.org/)
**Tags:** #padic #scRNA-seq #clustering #modern-application

## Abstract

A very recent application (2024) of p-adic theory to modern high-throughput biology. The authors introduce **p-ClustVal**, an algorithm that transforms scRNA-seq data into a p-adic metric space before clustering. They demonstrate that this transformation significantly improves cluster discernibility (separation) compared to standard Euclidean or cosine distance methods.

## Key Methodologies

### 1. p-ClustVal Algorithm

- **Input:** High-dimensional scRNA-seq count matrix.
- **Transformation:** Uses **p-adic valuation** ($v_p(x)$) to map gene expression counts into a p-adic metric space.
- **Clustering:** Applies standard clustering algorithms (K-Means, Louvain) _after_ the p-adic transformation.

### 2. The Heuristic

Does not require ground truth labels. Uses a data-centric heuristic to select the prime $p$ that maximizes the cluster separation index (Silhouette score, ARI).

## Key Findings

- **91% Improvement:** The method improved clustering performance in 91% of tested datasets compared to state-of-the-art Euclidean baselines.
- **Noise Reduction:** The ultrametric property of the p-adic space naturally filters out "noise" (small Euclidean variations that do not correspond to hierarchical differences), making it robust for dropout-heavy scRNA-seq data.

## Relevance to Project

**Proof of Concept:** This is the "smoking gun" that p-adic mathematics is not just theory but has practical utility in modern bioinformatics.

- **Action:** We should implement a `PAdicScaler` preprocessing step in our VAE pipeline, inspired by the p-ClustVal transformation, to see if it improves latent space separation of HIV/Host sequences.
