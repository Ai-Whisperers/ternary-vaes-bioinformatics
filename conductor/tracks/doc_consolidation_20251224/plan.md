# Track Plan: Documentation Consolidation

## Phase 1: Audit and Restructuring
- [ ] Task: Audit existing documentation files to create a complete inventory of content.
    - [ ] Subtask: Scan root directory and subfolders for all `.md` and `.txt` files.
    - [ ] Subtask: Categorize each file into one of the target tiers (Executive, Scientific, Developer).
- [ ] Task: Create the new directory structure within `DOCUMENTATION/`.
    - [ ] Subtask: Create folders: `00_INDEX`, `01_STAKEHOLDER_RESOURCES`, `02_THEORY_AND_RESEARCH`, `03_DEVELOPER_GUIDE`, `04_PROJECT_MANAGEMENT`.
- [ ] Task: Move and rename files into the new structure.
    - [ ] Subtask: Move pitch/vision docs to `01_STAKEHOLDER_RESOURCES`.
    - [ ] Subtask: Move research/math docs to `02_THEORY_AND_RESEARCH`.
    - [ ] Subtask: Move setup/api docs to `03_DEVELOPER_GUIDE`.
    - [ ] Subtask: Update file headers to reflect their new location.
- [ ] Task: Conductor - User Manual Verification 'Audit and Restructuring' (Protocol in workflow.md)

## Phase 2: Content Standardization and Gap Filling
- [ ] Task: Standardize the `03_DEVELOPER_GUIDE`.
    - [ ] Subtask: Verify `setup.md` instructions work on a clean environment.
    - [ ] Subtask: Ensure API documentation matches the current codebase (v5.11).
- [ ] Task: Standardize the `02_THEORY_AND_RESEARCH`.
    - [ ] Subtask: Consolidate dispersed math notes into a coherent `Mathematical_Foundations.md`.
    - [ ] Subtask: Ensure bio-informatics context is clearly linked to the math.
- [ ] Task: Create the Master Index.
    - [ ] Subtask: Write `DOCUMENTATION/00_INDEX.md` with links to all major sections.
    - [ ] Subtask: Check for and fix broken relative links between files.
- [ ] Task: Conductor - User Manual Verification 'Content Standardization' (Protocol in workflow.md)

## Phase 3: Tiered Presentation Creation
- [ ] Task: Create Tier 1 (Executive) Presentation.
    - [ ] Subtask: Write `DOCUMENTATION/01_STAKEHOLDER_RESOURCES/EXECUTIVE_SUMMARY.md` focusing on vision and impact.
- [ ] Task: Create Tier 2 (Scientific) Presentation.
    - [ ] Subtask: Write `DOCUMENTATION/01_STAKEHOLDER_RESOURCES/SCIENTIFIC_DEEP_DIVE.md` focusing on 3-adic/hyperbolic novelty.
- [ ] Task: Create Tier 3 (Technical) Presentation.
    - [ ] Subtask: Write `DOCUMENTATION/01_STAKEHOLDER_RESOURCES/TECHNICAL_OVERVIEW.md` focusing on architecture and implementation.
- [ ] Task: Update Root README.
    - [ ] Subtask: Rewrite `README.md` to be a lightweight landing page pointing to the new docs.
- [ ] Task: Conductor - User Manual Verification 'Tiered Presentation Creation' (Protocol in workflow.md)
