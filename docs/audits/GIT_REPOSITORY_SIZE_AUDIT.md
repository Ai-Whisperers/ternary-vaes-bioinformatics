# Git Repository Size Audit - Deep Analysis

**Doc-Type:** Technical Audit · Version 1.0 · Updated 2026-01-10 · AI Whisperers

---

## Executive Summary

**Current Status:** Repository size reduced from 6.4GB → 3.5GB (45% reduction), but still contains significant optimization opportunities.

**Breakdown:**
- **Git Objects Pack:** 2.0GB (single pack file with massive 1.4GB blob)
- **Git LFS Objects:** 1.5GB (288 tracked files, many 30-89MB each)
- **Other:** 0.5GB (logs, index, refs, hooks)

**Primary Issue:** Swiss Prot CIF file (1.4GB) remains in git history despite removal from working tree.

---

## Current State Analysis

### 1. Git Objects Pack (2.0GB) - CRITICAL

#### Largest Objects in History
| Size | Object ID | Type | Notes |
|------|-----------|------|-------|
| **1.4GB** | 742aa93d4b | blob | Swiss Prot CIF file (removed but in history) |
| 985MB | 654bb261a0 | blob | Orphaned LFS object |
| 919MB | 16131cd8ff | blob | Orphaned LFS object |
| 871MB | a8d33c1b6f | blob | Orphaned LFS object |
| 171MB | cac8e13737 | blob | Large dataset file |

#### Status
- Single pack file contains 16,541 objects
- **1.4GB Swiss Prot CIF blob** dominates the pack
- Previous filter-branch only removed smaller files

### 2. Git LFS Objects (1.5GB) - HIGH

#### Distribution Summary
| Category | Files | Size Range | Total Est. |
|----------|-------|------------|-------------|
| AlphaFold HIV predictions | 250+ | 1-89MB | ~1.2GB |
| Research proteome data | 20+ | 30-89MB | ~300MB |
| HuggingFace datasets | 2 | Unknown | Unknown |

#### Largest LFS Objects
| Size | Category | File Pattern |
|------|----------|--------------|
| 89MB | Proteome data | `geometric_features_summary.csv` |
| 66-60MB | Proteome data | Various parquet/JSON files |
| 30-58MB | AlphaFold MSA | `*.a3m` files (Multiple Sequence Alignments) |

#### AlphaFold Variants (22 variants × 11 files each)
- `bg505_cmp_*` (5 variants)
- `bg505_deglyc_*` (17 variants)
- Each variant contains: 5 full_data + 5 confidence + 1 request + MSA files

---

## Optimization Checklist

### Phase 1: Git History Cleanup (CRITICAL - 1.4GB recovery)

#### ☐ 1.1 Remove Swiss Prot CIF from History
```bash
# WARNING: This rewrites git history - coordinate with team
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch research/big_data/swissprot_cif_v6.tar' \
  --prune-empty --tag-name-filter cat -- --all
```

**Expected Recovery:** ~1.4GB
**Risk Level:** HIGH (rewrites history)
**Team Approval:** ✅ Obtained

#### ☐ 1.2 Cleanup Orphaned Objects
```bash
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

**Expected Recovery:** Additional 200-500MB
**Risk Level:** LOW

#### ☐ 1.3 Force Push Updated History
```bash
git push --force-with-lease origin main
```

**Risk Level:** HIGH (affects collaborators)

### Phase 2: LFS Optimization (MEDIUM - ~500MB recovery)

#### ☐ 2.1 Archive Old AlphaFold Predictions
**Target:** `outputs/results/validation/alphafold_predictions/hiv/`

**Options:**
- [ ] **Option A:** Move to external storage (Google Drive/AWS S3)
- [ ] **Option B:** Keep only recent/validated predictions
- [ ] **Option C:** Compress MSA files (`.a3m` → `.tar.gz`)

**Recovery Potential:** 800MB-1GB

#### ☐ 2.2 Research Data Cleanup
**Target:** `research/bioinformatics/codon_encoder_research/rheumatoid_arthritis/`

**Actions:**
- [ ] Verify which files are regenerable vs unique results
- [ ] Move large intermediate files to local-only storage
- [ ] Compress parquet files if not actively used

**Recovery Potential:** 300-400MB

#### ☐ 2.3 Evaluate HuggingFace Datasets
**Target:** `data/external/huggingface/`

**Actions:**
- [ ] Check if datasets can be downloaded on-demand
- [ ] Move to `.gitignore` with download scripts
- [ ] Document dataset versions for reproducibility

**Recovery Potential:** Unknown (check file sizes)

### Phase 3: Ongoing Maintenance (LOW - Prevention)

#### ☐ 3.1 Enhanced .gitignore Patterns
```bash
# Add comprehensive large file patterns
**/*.cif     # Structure files
**/*.tar     # Archives
**/*.zip     # Compressed archives
**/msas/     # MSA directories
**/*proteome*.* # Proteome datasets
```

#### ☐ 3.2 Pre-commit Hooks
- [ ] Set up file size limits (e.g., 100MB max)
- [ ] Automated LFS detection for large files
- [ ] Alert for potential large file commits

#### ☐ 3.3 Regular Audits
- [ ] Monthly repository size monitoring
- [ ] Quarterly LFS object review
- [ ] Annual history cleanup assessment

---

## Implementation Priority

### Immediate (High Impact, Low Risk)
1. **Enhanced .gitignore** - Prevent future large files
2. **Research data assessment** - Identify regenerable files
3. **LFS object inventory** - Detailed catalog with usage status

### Coordinated (High Impact, High Risk)
1. **Swiss Prot CIF removal** - Requires team coordination
2. **AlphaFold archive strategy** - Requires research team input
3. **Force push history** - Requires all collaborators to re-clone

### Ongoing (Medium Impact, Low Risk)
1. **Pre-commit hooks** - Long-term prevention
2. **Automated monitoring** - Early warning system
3. **Documentation updates** - Preserve institutional knowledge

---

## Risk Assessment

### High Risk Actions
- **Filter-branch on Swiss Prot file** - Rewrites entire history
- **Force pushing** - Affects all collaborators
- **Large file deletion** - Potential data loss

### Mitigation Strategies
1. **Full backup verification** - Confirm Google Drive backup complete
2. **Team notification** - 48-hour advance notice before history rewrite
3. **Staged rollout** - Test on branch before main
4. **Rollback plan** - Document exact commands to revert

### Success Criteria
- Repository size < 2GB total
- All research data preserved locally
- No disruption to active research
- Improved clone/push performance

---

## Expected Outcomes

### Optimistic Scenario (Full cleanup)
- **Size reduction:** 3.5GB → 1.5GB (57% reduction)
- **Primary savings:** 1.4GB (Swiss Prot) + 500MB (LFS optimization)
- **GitHub compatibility:** Full push/clone capability restored

### Conservative Scenario (History cleanup only)
- **Size reduction:** 3.5GB → 2.1GB (40% reduction)
- **Primary savings:** 1.4GB (Swiss Prot removal only)
- **Risk level:** Low (no research data affected)

### Minimal Scenario (Prevention only)
- **Size reduction:** Minimal immediate impact
- **Long-term benefit:** Prevents further growth
- **Risk level:** None (only adds protections)

---

## File Categories for Decision Making

### Definitely Keep (Core Research)
- Model checkpoints (`checkpoints/`)
- Partner deliverables (`deliverables/partners/`)
- Source code (`src/`, `src/configs/`, `src/scripts/`)
- Documentation (`docs/`)

### Probably Archive (Large Intermediate)
- AlphaFold predictions (`outputs/results/validation/alphafold_predictions/`)
- Large proteome datasets (`research/.../proteome_wide/`)
- MSA files (`**/msas/`)

### Evaluate Case-by-Case
- HuggingFace datasets - check if downloadable
- Large parquet files - check if regenerable
- Compressed archives - check contents

### Definitely Remove from History
- Swiss Prot CIF file (already removed from working tree)
- Orphaned LFS objects (no longer referenced)
- Large temporary files that entered history by mistake

---

## Next Steps

1. **Immediate:** Team review of this audit (approval for high-risk actions)
2. **Phase 1:** Execute Swiss Prot removal (coordinate with all team members)
3. **Phase 2:** Assess and archive AlphaFold data (research team input)
4. **Phase 3:** Implement prevention measures (ongoing)

**Estimated Timeline:** 1-2 weeks for full optimization, depending on team coordination requirements.

---

**Status:** AUDIT COMPLETE - Awaiting implementation decisions
**Contact:** AI Whisperers Development Team
**Last Updated:** 2026-01-10

---

## Appendix: Technical Commands

### Size Analysis Commands
```bash
# Repository breakdown
du -sh .git/*

# LFS file listing
git lfs ls-files

# Large object identification
git verify-pack -v .git/objects/pack/*.pack | sort -k 3 -nr | head -20

# Object count and size
git count-objects -vH
```

### Cleanup Commands (USE WITH CAUTION)
```bash
# Filter-branch (DESTRUCTIVE)
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch <large-file>' \
  --prune-empty --tag-name-filter cat -- --all

# Aggressive cleanup
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push (TEAM COORDINATION REQUIRED)
git push --force-with-lease origin main
```