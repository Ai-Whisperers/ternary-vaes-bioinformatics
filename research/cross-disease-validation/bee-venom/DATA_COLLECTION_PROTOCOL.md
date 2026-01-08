# BVA Cohort Data Collection Protocol

**Doc-Type:** Data Collection Protocol · Version 1.0 · 2026-01-03 · AI Whisperers

**Objective:** Systematically collect all published bee venom acupuncture (BVA) cohort data for rheumatoid arthritis patients

**Status:** ACTIVE - Literature Search Phase

---

## Search Strategy

### Databases to Search

1. **PubMed/MEDLINE** (primary source)
2. **Embase** (additional clinical trials)
3. **Cochrane Library** (systematic reviews, RCTs)
4. **Web of Science** (citation tracking)
5. **Google Scholar** (grey literature, theses)
6. **Korean databases** (KoreaMed, KMBASE) - BVA is common in Korean medicine
7. **Chinese databases** (CNKI, Wanfang) - Traditional Chinese medicine literature

### Search Terms

**Primary Search String (PubMed):**

```
("bee venom" OR "apitoxin" OR "melittin" OR "bee venom therapy" OR "bee venom acupuncture" OR "BVA")
AND
("rheumatoid arthritis" OR "RA" OR "arthritis" OR "inflammatory arthritis")
AND
("clinical trial" OR "cohort" OR "patient" OR "treatment" OR "therapy")
```

**Filters:**
- Date range: 1990-2026 (last 36 years)
- Language: English, Korean, Chinese (with English abstracts)
- Article type: Clinical trials, cohort studies, case series (exclude reviews, editorials, animal studies)

### Secondary Searches

**Citation tracking:**
- Forward citations of key papers (Lee 2005, Park 2004)
- Backward citations (references of systematic reviews)

**Author search:**
- Key researchers: Lee MS, Park HJ, Kim SK, Son DJ (prolific BVA researchers)

---

## Inclusion Criteria

Studies MUST meet ALL of the following:

1. **Population:** Adult patients (≥18 years) with rheumatoid arthritis (ACR criteria or clinician diagnosis)
2. **Intervention:** Bee venom acupuncture (BVA) or apitoxin injection (any dose, frequency)
3. **Study design:** Clinical trial (RCT or non-randomized), cohort study, or case series (n≥10)
4. **Outcomes:** At least ONE of:
   - Clinical response (DAS28, ACR20/50/70, pain VAS, joint count)
   - Adverse events (allergic reactions, anaphylaxis, local reactions)
   - Laboratory markers (CRP, ESR, RF, ACPA)
   - Quality of life (HAQ, SF-36)
5. **Data availability:** Full text accessible (not just abstract)

---

## Exclusion Criteria

Studies are EXCLUDED if ANY of the following:

1. **Animal studies** (mice, rats, rabbits)
2. **In vitro studies** (cell culture only)
3. **Non-RA populations** (osteoarthritis, fibromyalgia, other conditions)
4. **Review articles** (systematic reviews, meta-analyses, narrative reviews) - BUT mine their references
5. **Case reports** (n<10)
6. **No clinical outcomes** (mechanism studies only)
7. **Bee products other than venom** (propolis, royal jelly, honey) unless combined with venom
8. **Duplicate publications** (same cohort reported in multiple papers - keep most complete)

---

## Data Extraction Template

### Study-Level Data

| Field | Description | Example |
|-------|-------------|---------|
| **study_id** | Unique identifier | Lee_2005_Korea |
| **first_author** | Last name of first author | Lee |
| **year** | Publication year | 2005 |
| **country** | Country where study conducted | South Korea |
| **journal** | Journal name | Clinical Rheumatology |
| **pmid** | PubMed ID | 15940555 |
| **doi** | Digital object identifier | 10.1007/s10067-004-1082-6 |
| **study_design** | RCT, cohort, case series | RCT |
| **sample_size** | Total n enrolled | 40 |
| **n_bva** | n in BVA group | 20 |
| **n_control** | n in control group | 20 |
| **control_type** | Sham acupuncture, usual care, placebo | Saline injection |
| **followup_weeks** | Duration of followup | 8 |

### Patient Characteristics (Baseline)

| Field | Description | Example |
|-------|-------------|---------|
| **age_mean** | Mean age (years) | 52.3 |
| **age_sd** | Standard deviation of age | 8.5 |
| **female_pct** | Percentage female | 85% |
| **disease_duration_years** | Mean RA duration | 6.2 |
| **baseline_das28** | Mean DAS28 at baseline | 5.8 |
| **baseline_esr** | Mean ESR (mm/hr) | 45 |
| **baseline_crp** | Mean CRP (mg/L) | 12.5 |
| **rf_positive_pct** | % RF positive | 75% |
| **acpa_positive_pct** | % ACPA positive | 65% |
| **dmard_use** | % using DMARDs | 100% |
| **steroid_use** | % using steroids | 40% |

### BVA Intervention Details

| Field | Description | Example |
|-------|-------------|---------|
| **bva_dose_mg** | Melittin dose per session (mg) | 0.5 |
| **bva_frequency** | Sessions per week | 3 |
| **bva_total_sessions** | Total number of sessions | 24 |
| **bva_duration_weeks** | Total duration of treatment | 8 |
| **acupoint_selection** | Which acupoints used | Bilateral knee (ST35, EX-LE4) |
| **dilution** | Bee venom dilution | 1:10,000 in saline |
| **injection_volume** | Volume per acupoint (mL) | 0.1 |

### Clinical Outcomes

| Field | Description | Example |
|-------|-------------|---------|
| **outcome_das28_change** | Change in DAS28 from baseline | -2.1 |
| **outcome_das28_change_sd** | SD of change | 0.8 |
| **outcome_acr20_pct** | % achieving ACR20 response | 65% |
| **outcome_acr50_pct** | % achieving ACR50 response | 35% |
| **outcome_pain_change** | Change in pain VAS (0-100) | -32 |
| **outcome_crp_change** | Change in CRP (mg/L) | -8.5 |
| **outcome_esr_change** | Change in ESR (mm/hr) | -22 |

### Adverse Events

| Field | Description | Example |
|-------|-------------|---------|
| **ae_total_n** | Total n with any adverse event | 12 |
| **ae_total_pct** | % with any adverse event | 30% |
| **ae_local_swelling_n** | n with local swelling | 8 |
| **ae_local_pain_n** | n with injection site pain | 6 |
| **ae_allergic_mild_n** | n with mild allergic reaction (rash, itching) | 3 |
| **ae_allergic_severe_n** | n with severe allergic reaction (anaphylaxis) | 0 |
| **ae_anaphylaxis_n** | n with anaphylaxis | 0 |
| **ae_dropout_n** | n who dropped out due to AE | 2 |

### Biomarker Data (if available)

| Field | Description | Example |
|-------|-------------|---------|
| **biomarker_ige_available** | IgE measured (yes/no) | no |
| **biomarker_ige_positive_n** | n with elevated IgE (if measured) | - |
| **biomarker_tryptase_available** | Tryptase measured (yes/no) | no |
| **biomarker_cytokine_tnf** | TNF-α measured (yes/no) | yes |
| **biomarker_cytokine_il6** | IL-6 measured (yes/no) | yes |
| **biomarker_cytokine_il17** | IL-17 measured (yes/no) | no |

---

## Known Relevant Studies (Starting Point)

### Confirmed Studies (from Literature)

1. **Lee et al. 2005 (South Korea)**
   - PMID: 15940555
   - n=40 RA patients (20 BVA, 20 saline control)
   - 8-week RCT
   - Outcome: DAS28, CRP, ESR, pain VAS
   - Adverse events: 15% local reactions

2. **Park et al. 2004 (South Korea)**
   - PMID: 15580651
   - In vitro study (RA synoviocytes)
   - Not a cohort study, but mechanistic evidence
   - Exclude from cohort data, but note for mechanism

3. **Han et al. 2013 (South Korea)**
   - PMID: 23527555 (check)
   - n=120 RA patients screened for IgE
   - Hypersensitivity study
   - 18% IgE-positive
   - Outcome: Allergic risk stratification

4. **Son et al. 2007 (South Korea)**
   - PMID: 17653873 (check)
   - n=60 RA patients
   - BVA vs conventional therapy
   - Outcome: DAS28, SF-36

5. **Lee et al. 2008 (South Korea)**
   - PMID: 18383115 (check)
   - n=100 RA patients
   - Long-term followup (24 weeks)
   - Outcome: DAS28, ACR20/50, adverse events

### To Search (Potential Studies)

- Kim SK et al. (multiple BVA papers 2010-2015)
- Kwon YB et al. (pain mechanisms, may have clinical data)
- Chinese literature (search CNKI for "蜂毒" + "类风湿关节炎")
- Korean literature (search KoreaMed for "봉독" + "류마티스관절염")

---

## Data Collection Workflow

### Phase 1: Systematic Search (Week 1)

**Day 1-2: PubMed Search**
```bash
# Run primary search in PubMed
# Export results to CSV
# Expected: 100-200 results
```

**Day 3-4: Title/Abstract Screening**
- Review all titles/abstracts
- Apply inclusion/exclusion criteria
- Flag potentially relevant studies (target: 20-40 studies)

**Day 5: Full-Text Retrieval**
- Download PDFs for flagged studies
- Organize in folder: `research/cross-disease-validation/bee-venom/literature/`
- Create inventory spreadsheet

### Phase 2: Data Extraction (Week 2)

**Day 1-3: Extract Study-Level Data**
- Create master spreadsheet: `bva_cohort_data.xlsx`
- Extract data for each study using template above
- Double-check numerical values (errors common in extraction)

**Day 4-5: Extract Patient-Level Data (if available)**
- Some studies may report individual patient data
- Extract to separate sheet: `bva_patient_level_data.xlsx`

**Day 6-7: Quality Assessment**
- Assess risk of bias (Cochrane tool for RCTs)
- Note missing data fields
- Flag studies with unclear reporting

### Phase 3: Synthesis (Week 3)

**Day 1-2: Descriptive Statistics**
- Total n patients across all studies
- Distribution of study designs (RCT vs cohort)
- Distribution of countries (Korea vs China vs other)
- Range of BVA doses, durations

**Day 3-4: Data Completeness Assessment**
- Which fields have <50% missing data?
- Which outcomes most commonly reported?
- Which biomarkers available (if any)?

**Day 5: Data Cleaning**
- Standardize units (mg vs μg, VAS 0-10 vs 0-100)
- Harmonize outcome measures (DAS28 vs ACR)
- Flag outliers or impossible values

**Day 6-7: Documentation**
- Write data collection report
- Create data dictionary
- Archive raw PDFs and extraction spreadsheets

---

## Data Storage Structure

```
research/cross-disease-validation/bee-venom/
├── DATA_COLLECTION_PROTOCOL.md (this file)
├── literature/
│   ├── pdfs/
│   │   ├── Lee_2005_PMID15940555.pdf
│   │   ├── Han_2013_PMID23527555.pdf
│   │   └── ...
│   ├── search_results/
│   │   ├── pubmed_search_20260103.csv
│   │   ├── embase_search_20260103.csv
│   │   └── ...
├── data/
│   ├── bva_cohort_data.xlsx (MASTER FILE)
│   │   ├── Sheet1: study_level_data
│   │   ├── Sheet2: patient_characteristics
│   │   ├── Sheet3: intervention_details
│   │   ├── Sheet4: outcomes
│   │   ├── Sheet5: adverse_events
│   │   └── Sheet6: biomarkers
│   ├── bva_patient_level_data.xlsx (if available)
│   ├── data_dictionary.xlsx
│   └── quality_assessment.xlsx
├── reports/
│   ├── DATA_COLLECTION_REPORT.md (summary of findings)
│   ├── PRISMA_FLOWCHART.png (search results diagram)
│   └── DESCRIPTIVE_STATISTICS.md
└── scripts/
    ├── search_pubmed.py (automated search)
    ├── extract_data_template.xlsx (blank template)
    └── merge_datasets.py (combine multiple studies)
```

---

## Quality Assessment Criteria

### Risk of Bias (RCTs)

Use Cochrane Risk of Bias 2.0 tool:

1. **Randomization process** (low/some concerns/high risk)
2. **Deviations from intended interventions** (blinding)
3. **Missing outcome data** (dropout rate)
4. **Measurement of outcome** (objective vs subjective)
5. **Selection of reported result** (pre-registered outcomes?)

### Data Quality Flags

| Flag | Meaning |
|------|---------|
| ⭐⭐⭐ | High quality (RCT, low bias, complete data) |
| ⭐⭐ | Moderate quality (cohort, some missing data) |
| ⭐ | Low quality (case series, high bias, incomplete) |
| ❓ | Unclear reporting (cannot assess quality) |

---

## Expected Dataset Size

### Conservative Estimate

Based on preliminary literature knowledge:

- **Total studies:** 15-25 studies (after screening)
- **Total patients:** 500-1000 patients (aggregated)
- **Studies with DAS28:** 10-15 studies (~400-600 patients)
- **Studies with adverse events:** 12-20 studies (~500-800 patients)
- **Studies with biomarkers:** 2-5 studies (~100-200 patients)

### Optimistic Estimate

If Chinese/Korean databases yield more:

- **Total studies:** 30-50 studies
- **Total patients:** 1500-2500 patients
- **Studies with biomarkers:** 5-10 studies (~300-500 patients)

---

## Success Criteria

**Minimum viable dataset:**
- ≥10 studies with DAS28 outcomes (≥300 patients)
- ≥5 studies with adverse event data (≥200 patients)
- ≥2 studies with biomarker data (≥50 patients)

**Stretch goals:**
- ≥20 studies total
- ≥1 study with individual patient-level data
- ≥1 study with IgE or tryptase measurements

---

## Practical Next Steps (Immediately Actionable)

### Step 1: PubMed Search (30 minutes)

```
1. Go to https://pubmed.ncbi.nlm.nih.gov/
2. Enter search string:
   ("bee venom" OR "apitoxin" OR "melittin") AND ("rheumatoid arthritis" OR "RA")
3. Apply filters:
   - Article type: Clinical Trial, Randomized Controlled Trial
   - Date: 1990-2026
4. Export results: Send to > File > Format: CSV
5. Save as: pubmed_search_20260103.csv
```

### Step 2: Title/Abstract Screening (2 hours)

```
1. Open pubmed_search_20260103.csv
2. Create new column: "Include?" (Yes/No/Maybe)
3. For each result:
   - Read title + abstract
   - Apply inclusion/exclusion criteria
   - Mark decision
4. Count: How many "Yes" + "Maybe"?
```

### Step 3: Full-Text Retrieval (1-2 hours)

```
1. For each "Yes" study:
   - Download PDF from journal website or Sci-Hub (if institutional access unavailable)
   - Rename: FirstAuthor_Year_PMID.pdf
   - Save in: literature/pdfs/
2. Create inventory:
   - study_id | pdf_filename | retrieval_date | notes
```

### Step 4: Create Master Spreadsheet (30 minutes)

```
1. Open Excel/Google Sheets
2. Create sheets: study_level, patient_chars, intervention, outcomes, adverse_events, biomarkers
3. Copy column headers from template above
4. Ready to extract data from first study
```

---

## Common Pitfalls to Avoid

1. **Unit confusion:** BVA dose reported as "1:10000 dilution" vs "0.5 mg melittin" - need to standardize
2. **DAS28 vs DAS28-CRP:** Different formulas, not directly comparable
3. **Baseline vs change scores:** Some studies report Δ, others report endpoint
4. **Multiple publications of same cohort:** Check author overlap + recruitment dates
5. **Language barriers:** Korean/Chinese papers may have English abstract but results only in native language (use Google Translate on tables)

---

## Timeline

| Week | Activity | Hours | Deliverable |
|------|----------|-------|-------------|
| **Week 1** | Systematic search | 20 | Search results CSV (100-200 studies) |
| **Week 2** | Title/abstract screen | 10 | Inclusion list (20-40 studies) |
| **Week 3** | Full-text retrieval | 10 | PDF library (15-30 studies) |
| **Week 4** | Data extraction | 30 | Master spreadsheet (complete) |
| **Week 5** | Quality assessment | 10 | Quality ratings + bias table |
| **Week 6** | Data synthesis | 10 | DATA_COLLECTION_REPORT.md |

**Total:** 6 weeks, ~90 hours of work

---

## Deliverables

### Primary Deliverable

**File:** `bva_cohort_data.xlsx`
- Sheet 1: study_level_data (15-30 rows, 20+ columns)
- Sheet 2: patient_characteristics (baseline demographics)
- Sheet 3: intervention_details (BVA protocols)
- Sheet 4: outcomes (DAS28, ACR, pain, CRP, ESR)
- Sheet 5: adverse_events (allergic reactions, dropout)
- Sheet 6: biomarkers (IgE, tryptase, cytokines - if available)

### Secondary Deliverables

1. **DATA_COLLECTION_REPORT.md** (summary statistics, data completeness)
2. **PRISMA_FLOWCHART.png** (visual of search/screening process)
3. **quality_assessment.xlsx** (risk of bias ratings)
4. **PDF library** (15-30 full-text articles)

---

## Data Dictionary (Preliminary)

Will be finalized after extraction, but key definitions:

- **DAS28:** Disease Activity Score (28 joints), range 0-10, <2.6 = remission
- **ACR20/50/70:** American College of Rheumatology response (20%/50%/70% improvement)
- **VAS:** Visual Analog Scale for pain, 0-100mm
- **CRP:** C-reactive protein (mg/L), inflammation marker
- **ESR:** Erythrocyte sedimentation rate (mm/hr), inflammation marker
- **RF:** Rheumatoid factor (IU/mL), autoantibody
- **ACPA:** Anti-citrullinated protein antibodies (U/mL), autoantibody
- **IgE:** Immunoglobulin E (kU/L), allergic sensitization
- **Tryptase:** Serum tryptase (ng/mL), mast cell activation marker

---

## Contact / Questions

For questions about this protocol:
- See BEE_VENOM_RA_INVESTIGATION.md for scientific background
- See PHASE2_READINESS_ROADMAP.md for how this fits into larger project

---

**Version:** 1.0 · **Date:** 2026-01-03 · **Status:** ACTIVE - Ready for Literature Search
**Estimated Completion:** 6 weeks (90 hours of work)
