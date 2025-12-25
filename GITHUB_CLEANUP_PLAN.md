# GitHub Repository Cleanup & Configuration Plan

**Generated:** 2025-12-25
**Repository:** Ai-Whisperers/ternary-vaes-bioinformatics
**Current Branch:** feature/research-prioritization

---

## Branch Analysis

### Local Branches Status

| Branch | Status | Commits Ahead | Commits Behind | Last Commit | Recommendation |
|--------|--------|---------------|----------------|-------------|----------------|
| `main` | Base | - | - | 11h ago | Keep |
| `develop` | **Merged** | 0 | 1 | 11h ago | **DELETE** |
| `feature/critical-bug-fixes` | **Merged** | 0 | 1 | 11h ago | **DELETE** |
| `feature/refactoring-tests` | **Merged** | 0 | 4 | 11h ago | **DELETE** |
| `feature/research-implementation` | **Merged** | 0 | 6 | 13h ago | **DELETE** |
| `feature/documentation-cleanup` | Active | 5 | 0 | 8h ago | Keep (has open PR #1) |
| `feature/research-prioritization` | **Active** | 29 | 0 | now | Keep (current branch) |
| `feature/visualization-refactor` | Active | 2 | 0 | 8h ago | Keep (unmerged work) |

### Remote-Only Branches

| Branch | Status | Recommendation |
|--------|--------|----------------|
| `refactor/srp-implementation` | Remote only (orphaned) | **DELETE** |

---

## Branches to Delete

### Commands to Delete Merged Branches

```bash
# Delete local merged branches
git branch -d develop
git branch -d feature/critical-bug-fixes
git branch -d feature/refactoring-tests
git branch -d feature/research-implementation

# Delete remote merged branches
git push origin --delete feature/critical-bug-fixes
git push origin --delete feature/refactoring-tests
git push origin --delete refactor/srp-implementation
```

### Branches to Keep

- `main` - Default branch
- `feature/documentation-cleanup` - Has open PR #1
- `feature/research-prioritization` - Current active work (29 commits ahead)
- `feature/visualization-refactor` - Unmerged work in progress

---

## GitHub Configuration Audit

### Repository Info

| Property | Value | Assessment |
|----------|-------|------------|
| **Visibility** | Private | Good for proprietary work |
| **Default Branch** | `main` | Standard |
| **Issues** | Enabled | Good |
| **Projects** | Enabled | Good |
| **Wiki** | Disabled | Consider enabling for docs |
| **License** | PolyForm Noncommercial | Custom, appropriate |
| **Branch Protection** | **NONE** | **CRITICAL ISSUE** |

---

## Critical Issues

### 1. No Branch Protection on `main`

**Current state:** Anyone can push directly to main, force-push, or delete it.

**Recommended settings (GitHub → Settings → Branches → Add rule):**

```
Branch name pattern: main

[x] Require a pull request before merging
    [x] Require approvals: 1
    [x] Dismiss stale pull request approvals when new commits are pushed

[x] Require status checks to pass before merging
    [x] Require branches to be up to date before merging
    Required checks:
      - lint
      - test
      - compliance

[x] Require conversation resolution before merging

[ ] Require signed commits (optional, but recommended)

[x] Do not allow bypassing the above settings

[ ] Allow force pushes: DISABLED
[ ] Allow deletions: DISABLED
```

### 2. Duplicate CI Workflows

**Problem:** Two overlapping test workflows exist:

| File | Purpose | Python Versions | Issues |
|------|---------|-----------------|--------|
| `ci.yml` | Comprehensive | 3.11 | Current, good |
| `test.yml` | Basic | 3.8, 3.10 | Outdated, redundant |

**Action:** Delete `.github/workflows/test.yml`

### 3. Empty FUNDING.yml

**Current state:**
```yaml
github: # Replace with your GitHub Sponsors username
custom: # ['support@aiwhisperers.com']
```

**Action:** Either configure properly or delete to avoid empty "Sponsor" button.

---

## Missing GitHub Features

### 1. CODEOWNERS File (Missing)

Create `.github/CODEOWNERS`:

```
# Global owners - all changes require review
* @IvanWeissVanDerPol

# Critical paths require owner review
/src/models/ @IvanWeissVanDerPol
/src/losses/ @IvanWeissVanDerPol
/src/training/ @IvanWeissVanDerPol
/.github/ @IvanWeissVanDerPol
/LICENSE @IvanWeissVanDerPol
/CLA.md @IvanWeissVanDerPol
```

### 2. Dependabot Configuration (Missing)

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
    labels:
      - "dependencies"
    commit-message:
      prefix: "deps"

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "ci"
```

### 3. CodeQL Security Scanning (Missing)

Create `.github/workflows/codeql.yml`:

```yaml
name: CodeQL

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday

jobs:
  analyze:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    steps:
      - uses: actions/checkout@v4
      - uses: github/codeql-action/init@v3
        with:
          languages: python
      - uses: github/codeql-action/analyze@v3
```

### 4. CI Workflow Improvements

Add to `.github/workflows/ci.yml`:

```yaml
# Add at the top level (after 'on:')
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

### 5. Repository Topics (Missing)

Add these topics in GitHub → Settings → General → Topics:

- `bioinformatics`
- `variational-autoencoder`
- `hyperbolic-geometry`
- `pytorch`
- `machine-learning`
- `p-adic`
- `codon-optimization`

---

## GitHub Features Available But Not Used

| Feature | Status | Benefit |
|---------|--------|---------|
| **Branch Protection** | Not configured | Prevent accidental pushes to main |
| **Required Reviews** | Not configured | Code quality control |
| **Status Checks** | Not required | Ensure CI passes before merge |
| **Dependabot** | Not configured | Automated security updates |
| **Code Scanning** | Not configured | Find vulnerabilities |
| **Secret Scanning** | Unknown | Prevent credential leaks |
| **Releases** | No releases | Version tracking |
| **Projects v2** | Enabled, unused | Sprint/task planning |
| **Wiki** | Disabled | Documentation |
| **Discussions** | Unknown | Community Q&A |

---

## Action Plan

### Immediate (Today)

- [ ] Enable branch protection on `main`
- [ ] Delete merged branches (see commands above)
- [ ] Delete duplicate `test.yml` workflow

### Short-term (This Week)

- [ ] Add `CODEOWNERS` file
- [ ] Add Dependabot configuration
- [ ] Fix or remove empty `FUNDING.yml`
- [ ] Add concurrency control to CI
- [ ] Enable wiki for documentation

### Medium-term (This Month)

- [ ] Add CodeQL security scanning
- [ ] Create first GitHub Release (v5.11.0)
- [ ] Add repository topics
- [ ] Set up GitHub Projects for tracking
- [ ] Enable Discussions for community

---

## Verification Commands

After cleanup, verify with:

```bash
# Check remaining branches
git branch -a

# Verify remote is clean
git remote prune origin
git fetch --prune

# Check branch protection (requires gh CLI)
gh api repos/Ai-Whisperers/ternary-vaes-bioinformatics/branches/main/protection
```

---

## Notes

- **Open PR #1:** `feature/documentation-cleanup` - Review and merge or close
- **Current work:** `feature/research-prioritization` has 29 commits to merge
- **Orphaned remote:** `refactor/srp-implementation` has no local branch

---

*This document was auto-generated by Claude Code during repository analysis.*
