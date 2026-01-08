# ProteinGym Dataset (Optional)

**Doc-Type:** Dataset Documentation · Status: Not Downloaded · Version 1.0 · 2026-01-03

---

## Current Status: S669 is Sufficient

The project uses the **S669 dataset** for DDG (ΔΔG) prediction validation and it provides excellent results:

- **Location:** `deliverables/partners/jose_colbes/reproducibility/data/S669.zip`
- **Size:** 43 MB
- **Records:** 2,800 single-point mutations across 17 proteins
- **Performance:** TrainableCodonEncoder achieves LOO Spearman **0.61**

This performance is competitive with structure-based methods:
- Rosetta ddg_monomer: LOO Spearman 0.69 (structure-based, requires 3D)
- **TrainableCodonEncoder: LOO Spearman 0.61 (sequence-only, p-adic embeddings)**
- ELASPIC-2 (2024): LOO Spearman 0.50 (sequence-based)
- FoldX: LOO Spearman 0.48 (structure-based)

**Conclusion:** S669 provides sufficient validation for current research. ProteinGym is an optional future enhancement.

---

## ProteinGym (Optional Future Enhancement)

ProteinGym provides a much larger benchmark dataset for deep mutational scanning (DMS) analysis.

### Dataset Details

- **URL:** https://marks.hms.harvard.edu/proteingym/ProteinGym_substitutions.zip
- **Size:** ~500 MB compressed, ~2 GB uncompressed
- **Coverage:** 200,000+ protein variants with DMS data
- **Format:** Multiple CSV files, one per protein
- **Domains:** Covers diverse protein families

### When to Download

Consider downloading ProteinGym if you need to:

1. **Extend validation beyond S669's 17 proteins** - Test generalization across protein families
2. **Benchmark across broader coverage** - Compare against published deep learning models
3. **Mega-scale mutational analysis** - Analyze systematic mutagenesis data
4. **Protein family-specific validation** - Enzymes, receptors, transporters, etc.

### Use Cases

| Use Case | S669 | ProteinGym |
|----------|------|------------|
| DDG prediction validation | ✅ Sufficient | Optional |
| Codon encoder training | ✅ Sufficient | Optional |
| Protein family analysis | Limited (17 proteins) | ✅ Comprehensive |
| Benchmark publication | ✅ Standard | ✅ Extended |
| Mega-scale DMS studies | ❌ Too small | ✅ Ideal |

---

## How to Download (If Needed)

### Automated Download

Use the provided pipeline script:

```bash
cd research/codon-encoder/analysis
python proteingym_pipeline.py --download
```

This will:
1. Download ProteinGym_substitutions.zip (~500 MB)
2. Extract to `research/codon-encoder/data/proteingym/`
3. Parse CSV files into structured format
4. Compute embedding correlations (optional)

### Manual Download

1. Download from: https://marks.hms.harvard.edu/proteingym/ProteinGym_substitutions.zip
2. Extract to: `research/codon-encoder/data/proteingym/`
3. Verify: Should contain ~200 CSV files

---

## Pipeline Usage

Once downloaded, you can analyze embedding space correlations:

```bash
python research/codon-encoder/analysis/proteingym_pipeline.py --analyze
```

This will:
- Load all ProteinGym datasets
- Extract p-adic embeddings for variants
- Compute physical property correlations (mass, volume, hydrophobicity)
- Identify which embedding dimensions encode biological invariants
- Generate analysis report in `research/codon-encoder/results/`

---

## Storage Considerations

**Before downloading, ensure you have:**
- ~500 MB for compressed download
- ~2 GB for extracted CSVs
- ~1 GB for processed embeddings
- **Total: ~3.5 GB disk space**

**Alternative:** Process on-the-fly without saving all CSVs (streaming mode in pipeline)

---

## Current Recommendation

**For most research needs, stick with S669.**

Only download ProteinGym if you:
- Are publishing results requiring extended validation
- Need protein family-specific analysis
- Are comparing against published DMS benchmarks
- Have specific research questions requiring mega-scale data

The S669 dataset provides excellent validation for the p-adic codon encoder without the storage and processing overhead of ProteinGym.

---

## References

- **ProteinGym Paper:** Notin et al. (2023) - https://www.nature.com/articles/s41586-023-06328-6
- **S669 Dataset:** ProTherm database subset used for thermodynamic validation
- **Our Results:** See `research/codon-encoder/training/results/trained_codon_encoder.json`

---

**Last Updated:** 2026-01-03
**Maintainer:** AI Whisperers Research Team
**Status:** ProteinGym download is OPTIONAL - S669 is sufficient for current work
