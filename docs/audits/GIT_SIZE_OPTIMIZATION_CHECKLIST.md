# Git Repository Size Optimization Checklist

**Doc-Type:** Implementation Checklist · Version 1.0 · Updated 2026-01-10

**Current Status:** 3.5GB (down from 6.4GB)
**Target:** <2GB
**Primary Issue:** 1.4GB Swiss Prot file in git history

---

## Quick Action Checklist

### 🔴 CRITICAL (1.4GB Recovery)
- [ ] **Team coordination meeting** - Notify all collaborators before history rewrite
- [ ] **Backup verification** - Confirm Google Drive backup complete
- [ ] **Filter-branch Swiss Prot file** - Remove from entire git history
  ```bash
  git filter-branch --force --index-filter \
    'git rm --cached --ignore-unmatch research/big_data/swissprot_cif_v6.tar' \
    --prune-empty --tag-name-filter cat -- --all
  ```
- [ ] **Aggressive garbage collection**
  ```bash
  git reflog expire --expire=now --all
  git gc --prune=now --aggressive
  ```
- [ ] **Force push coordinated** - All team members ready to re-clone
- [ ] **Verify size reduction** - Check final repository size

### 🟡 HIGH (500MB Recovery)
- [ ] **AlphaFold data assessment** - Review 22 variants × 11 files each
- [ ] **Research data inventory** - Catalog proteome analysis files (300MB)
- [ ] **Archive strategy decision** - Local storage vs cloud vs regeneration
- [ ] **MSA file compression** - Consider `.a3m` → `.tar.gz` for inactive files
- [ ] **HuggingFace dataset evaluation** - Check if downloadable on-demand

### 🟢 PREVENTIVE
- [ ] **Enhanced .gitignore** - Add patterns for large files
  ```
  **/*.cif
  **/*.tar
  **/msas/
  **/*proteome*.*
  ```
- [ ] **Pre-commit hooks** - File size limits (100MB)
- [ ] **Monthly monitoring** - Repository size tracking
- [ ] **Documentation updates** - Large file handling procedures

---

## Decision Points

| Issue | Options | Recommendation | Risk |
|-------|---------|----------------|------|
| Swiss Prot CIF (1.4GB) | Remove from history | ✅ DO IT | High |
| AlphaFold predictions (800MB) | Archive/Compress/Keep | 🤔 EVALUATE | Medium |
| Proteome research (300MB) | Local storage | ✅ MOVE | Low |
| HuggingFace data (?MB) | Download scripts | 🤔 CHECK SIZE | Low |

---

## Progress Tracking

### Completed ✅
- [x] Initial cleanup (6.4GB → 3.5GB)
- [x] Orphaned LFS object removal (2.8GB)
- [x] Comprehensive size audit
- [x] Risk assessment and planning

### In Progress 🚧
- [ ] Team coordination for history rewrite
- [ ] Research data evaluation

### Pending ⏳
- [ ] Swiss Prot history removal
- [ ] AlphaFold archival strategy
- [ ] Prevention measures implementation

---

## Success Metrics

**Size Targets:**
- [ ] **<3GB** (Swiss Prot removed)
- [ ] **<2GB** (Full optimization)
- [ ] **<1.5GB** (Aggressive archival)

**Performance Targets:**
- [ ] Clone time <5 minutes
- [ ] Push/pull <30 seconds
- [ ] No GitHub size warnings

**Team Targets:**
- [ ] Zero data loss
- [ ] Minimal workflow disruption
- [ ] Documented procedures

---

## Emergency Rollback

If something goes wrong:

```bash
# 1. Reset to backup
git clone <backup-url> repository-backup

# 2. Force restore main
git push --force origin backup-main:main

# 3. Notify team immediately
```

**Backup Locations:**
- Google Drive: ✅ Verified
- Local backup: [ ] Create before history rewrite

---

**Next Action:** Schedule team meeting for Swiss Prot removal coordination
**Owner:** AI Whisperers Development Team
**Deadline:** 2026-01-17 (1 week)