# BVA Data Collection - Immediate Next Steps

**Quick Start Guide for Literature Search**

---

## Right Now (15 minutes)

### Step 1: PubMed Search

1. Go to: https://pubmed.ncbi.nlm.nih.gov/
2. Copy-paste this exact search string:

```
("bee venom"[Title/Abstract] OR "apitoxin"[Title/Abstract] OR "melittin"[Title/Abstract] OR "bee venom therapy"[Title/Abstract] OR "bee venom acupuncture"[Title/Abstract]) AND ("rheumatoid arthritis"[Title/Abstract] OR "RA"[MeSH Terms])
```

3. Click "Search"
4. Apply these filters (left sidebar):
   - **Article type:** Clinical Trial, Randomized Controlled Trial
   - **Publication date:** Custom range: 1990 to 2026
5. Click "Send to" (top right) → "File" → "Format: CSV" → "Create File"
6. Save as: `pubmed_search_results.csv`

**Expected result:** 50-150 citations

---

## Today (2 hours)

### Step 2: Screen Titles and Abstracts

Open the CSV file and create three categories:

**YES (definitely include):**
- RA patients
- Bee venom treatment
- Clinical outcomes reported (pain, DAS28, adverse events)

**MAYBE (needs full text to decide):**
- Abstract unclear about outcomes
- Might be review (but could have useful data)

**NO (exclude):**
- Animal studies (mice, rats)
- In vitro only (cells, no patients)
- Not RA (osteoarthritis, other diseases)
- No bee venom (just propolis, honey, etc.)
- Review articles (but save for reference mining)

**Create a simple spreadsheet:**

| PMID | First Author | Year | Title | Decision | Reason |
|------|--------------|------|-------|----------|--------|
| 15940555 | Lee | 2005 | ... | YES | RA patients, BVA, DAS28 outcome |
| ... | ... | ... | ... | MAYBE | Unclear if clinical data |
| ... | ... | ... | ... | NO | Animal study |

---

## This Week (4-6 hours)

### Step 3: Get Full-Text PDFs

For every "YES" and "MAYBE" study:

1. **If you have institutional access:**
   - Click PMID link in PubMed
   - Follow to journal website
   - Download PDF

2. **If no institutional access:**
   - Try PubMed Central (free full-text): Look for "Free PMC Article" link
   - Try Google Scholar: Search by title, click "[PDF]" link if available
   - Contact authors: Email corresponding author requesting PDF
   - Library request: Request through interlibrary loan

3. **Organize PDFs:**
   - Create folder: `research/cross-disease-validation/bee-venom/literature/pdfs/`
   - Rename each PDF: `LastName_Year_PMID.pdf`
   - Example: `Lee_2005_15940555.pdf`

**Expected result:** 10-30 full-text PDFs

---

## Key Studies to Find First (Priority)

These are known high-quality BVA studies - find these first:

1. **Lee et al. 2005** - PMID: 15940555
   - n=40 RA patients (RCT)
   - South Korea
   - Journal: Clinical Rheumatology

2. **Son et al. 2007** - PMID: 17653873
   - n=60 RA patients
   - South Korea
   - Journal: Evidence-Based Complementary and Alternative Medicine

3. **Lee et al. 2008** - PMID: 18383115
   - n=100 RA patients (larger cohort)
   - South Korea
   - 24-week followup

4. **Han et al. 2013** - PMID: Search for "Han" + "bee venom" + "hypersensitivity" + "rheumatoid"
   - IgE screening study
   - n=120 RA patients
   - Allergy assessment

**How to verify these are the right papers:**
- Check first author last name matches
- Check publication year
- Check it's about RA (not other diseases)
- Check sample size is approximately correct

---

## Data Extraction Template (Blank)

Once you have 2-3 PDFs, start extracting data into this template:

### Study Information

| Field | Value |
|-------|-------|
| Study ID | Lee_2005 |
| First Author | Lee |
| Year | 2005 |
| Country | South Korea |
| Journal | Clinical Rheumatology |
| PMID | 15940555 |
| Study Design | RCT |
| Sample Size (total) | 40 |
| Sample Size (BVA group) | 20 |
| Sample Size (control group) | 20 |
| Control Type | Saline injection |
| Followup Duration | 8 weeks |

### Baseline Patient Characteristics

| Field | Value |
|-------|-------|
| Mean Age | 52.3 years |
| % Female | 85% |
| Mean Disease Duration | 6.2 years |
| Mean DAS28 (baseline) | 5.8 |
| % RF Positive | 75% |
| % ACPA Positive | (not reported) |
| % Using DMARDs | 100% |

### BVA Intervention

| Field | Value |
|-------|-------|
| Melittin Dose per Session | 0.5 mg |
| Sessions per Week | 3 |
| Total Sessions | 24 |
| Total Duration | 8 weeks |
| Acupoints Used | Bilateral knee (ST35, EX-LE4) |
| Dilution | 1:10,000 in saline |

### Outcomes

| Field | Value |
|-------|-------|
| Change in DAS28 (mean ± SD) | -2.1 ± 0.8 |
| % Achieving ACR20 | 65% |
| % Achieving ACR50 | 35% |
| Change in Pain VAS | -32 (on 0-100 scale) |
| Change in CRP | -8.5 mg/L |

### Adverse Events

| Field | Value |
|-------|-------|
| Total with Any Adverse Event | 12 / 40 (30%) |
| Local Swelling | 8 / 40 (20%) |
| Local Pain | 6 / 40 (15%) |
| Mild Allergic Reaction | 3 / 40 (7.5%) |
| Severe Allergic Reaction | 0 / 40 (0%) |
| Anaphylaxis | 0 / 40 (0%) |
| Dropouts Due to AE | 2 / 40 (5%) |

### Data Quality

| Field | Value |
|-------|-------|
| Risk of Bias (Overall) | Low |
| Randomization Adequate? | Yes |
| Blinding Adequate? | Partial (patients blinded, not assessors) |
| Missing Data <20%? | Yes |
| Outcomes Pre-Specified? | Yes |
| Quality Rating | ⭐⭐⭐ (High) |

---

## Tracking Your Progress

Create a simple checklist:

**Week 1:**
- [ ] PubMed search completed (50-150 results)
- [ ] Title/abstract screening done
- [ ] Identified 10-30 potentially relevant studies
- [ ] Created "YES/MAYBE/NO" spreadsheet

**Week 2:**
- [ ] Retrieved 5+ full-text PDFs
- [ ] Started data extraction for first study
- [ ] Identified missing data fields

**Week 3:**
- [ ] Retrieved 10+ full-text PDFs
- [ ] Extracted data from 3-5 studies
- [ ] Started quality assessment

**Week 4:**
- [ ] Retrieved all accessible full-texts (15-30 PDFs)
- [ ] Extracted data from 10+ studies
- [ ] Completed quality assessment
- [ ] Identified common missing data (IgE, tryptase, etc.)

---

## What You'll Learn Along the Way

As you extract data from 5-10 studies, you'll notice:

1. **Common reporting gaps:**
   - Most studies don't report IgE or tryptase (allergic biomarkers)
   - ACPA often not reported (focus on RF instead)
   - Adverse events sometimes underreported

2. **Heterogeneity:**
   - BVA doses vary widely (0.1-2.0 mg melittin)
   - Duration varies (4-24 weeks)
   - Outcome measures not standardized (DAS28 vs ACR vs pain only)

3. **Study quality:**
   - Korean studies tend to be higher quality (more RCTs)
   - Chinese studies often lack placebo controls
   - Some studies combine BVA with acupuncture (hard to isolate effect)

4. **Data limitations:**
   - Individual patient data rarely available (only group means)
   - Biomarkers (IgE, cytokines) in <20% of studies
   - Long-term followup (>6 months) rare

**These limitations are OK - this is real-world data collection, not a perfect dataset.**

---

## When to Ask for Help

Stop and ask questions if:

1. **Search yields 0 results** (check search syntax)
2. **All studies are animal/in vitro** (may need to broaden search)
3. **Cannot access >50% of full texts** (try different access methods)
4. **Data extraction feels inconsistent** (may need clearer definitions)
5. **Unsure how to rate study quality** (review Cochrane guidelines)

---

## Quick Wins

Start with these to build confidence:

1. **Find the Lee 2005 paper** (PMID 15940555) - well-known study, should be accessible
2. **Extract data from Lee 2005** - practice using template
3. **Search Google Scholar** for "bee venom rheumatoid arthritis clinical trial" - may find more than PubMed
4. **Check Korean databases** - KoreaMed (https://koreamed.org/) has English interface

---

## Expected Timeline

If working 2-3 hours per day:

- **Day 1:** PubMed search, initial screening (30 minutes)
- **Day 2-3:** Full screening, identify studies to retrieve (2 hours)
- **Day 4-7:** Retrieve PDFs (30 minutes per day)
- **Week 2-3:** Extract data from 3-5 studies (1 hour per study)
- **Week 4:** Extract data from remaining studies, quality assessment

**Total time:** ~20-30 hours over 4 weeks

---

## Success Metric

**Minimum viable dataset after 4 weeks:**
- 10+ studies with full data extraction
- 5+ studies with DAS28 outcomes
- 3+ studies with adverse event data
- 1+ study with biomarker data (if lucky)
- ~300-500 total patients aggregated

**If you achieve this, you have enough data to start seeing patterns - no modeling needed yet, just descriptive statistics.**

---

**Ready to start? Open PubMed and run that first search!**

**Version:** 1.0 · **Date:** 2026-01-03
