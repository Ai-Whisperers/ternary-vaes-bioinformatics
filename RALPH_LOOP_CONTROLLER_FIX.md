# Ralph Loop: Controller Metrics Fix (Priority 7)

**Doc-Type:** Focused Ralph Loop Strategy · Version 1.0 · Created 2026-01-08 · AI Whisperers

---

## CRITICAL ISSUE: Controller Receives Placeholder Hierarchy Values

### The Problem

**Location:** `src/models/ternary_vae.py` lines 375-376

```python
batch_stats = torch.stack([
    radius_A,
    radius_B,
    torch.tensor(1.0, device=x.device),  # H_A placeholder ❌
    torch.tensor(1.0, device=x.device),  # H_B placeholder ❌
    kl_A,
    kl_B,
    geo_loss_placeholder,
    rad_loss_placeholder,
])
```

**Impact:**
- DifferentiableController cannot learn hierarchy-aware loss weighting
- Controller receives constant H_A=1.0, H_B=1.0 instead of actual Spearman correlations
- Homeostatic training capabilities severely limited
- Controller's 8→32→32→6 MLP learns on meaningless hierarchy signals

### The Solution

Replace placeholder values with **real hierarchy calculations** using Spearman correlation between 3-adic valuations and hyperbolic radii, following the pattern from `src/core/metrics.py:222-223`.

---

## RALPH LOOP CONFIGURATION

```bash
/ralph-loop "Fix the DifferentiableController placeholder metrics issue in src/models/ternary_vae.py. The controller currently receives H_A=1.0, H_B=1.0 placeholders instead of real hierarchy values. Replace lines 375-376 with actual Spearman correlation calculations between 3-adic valuations and radii. Follow the pattern from src/core/metrics.py lines 222-223. Validate the fix with training runs and ensure controller can now learn hierarchy-aware loss weighting. Test that all existing functionality is preserved. Output <promise>CONTROLLER_HIERARCHY_FIXED</promise> when the controller receives real hierarchy values and training validation confirms improved hierarchy learning." --max-iterations 10 --completion-promise "CONTROLLER_HIERARCHY_FIXED"
```

---

## TECHNICAL IMPLEMENTATION DETAILS

### 1. Understanding the Current Hierarchy Calculation

**Reference Implementation** (`src/core/metrics.py:216-223`):
```python
valuations = TERNARY.valuation(indices).numpy()
# ...
hierarchy_A = spearmanr(valuations, all_radii_A)[0]
hierarchy_B = spearmanr(valuations, all_radii_B)[0]
```

### 2. Required Imports

Add to `src/models/ternary_vae.py`:
```python
from scipy.stats import spearmanr
from src.core import TERNARY
```

### 3. Fix Implementation Template

**Replace lines 375-376 with:**
```python
# Calculate actual hierarchy values (not placeholders)
# Get indices for current batch to calculate valuations
batch_size = x.shape[0]
batch_indices = torch.arange(batch_size, device=x.device)  # Or actual operation indices if available

# Calculate 3-adic valuations for current batch
batch_valuations = TERNARY.valuation(batch_indices).cpu().numpy()

# Calculate hierarchy as Spearman correlation
radii_A_cpu = radius_A.detach().cpu().numpy()
radii_B_cpu = radius_B.detach().cpu().numpy()

# Handle edge cases for small batches
if len(batch_valuations) > 1 and len(np.unique(batch_valuations)) > 1:
    hierarchy_A = spearmanr(batch_valuations, radii_A_cpu)[0]
    hierarchy_B = spearmanr(batch_valuations, radii_B_cpu)[0]

    # Handle NaN values from spearmanr
    if np.isnan(hierarchy_A):
        hierarchy_A = 0.0
    if np.isnan(hierarchy_B):
        hierarchy_B = 0.0
else:
    # Fallback for small batches or uniform valuations
    hierarchy_A = 0.0
    hierarchy_B = 0.0

torch.tensor(hierarchy_A, device=x.device),  # Real H_A
torch.tensor(hierarchy_B, device=x.device),  # Real H_B
```

### 4. Key Technical Considerations

**Batch vs Global Hierarchy:**
- Current implementation needs **batch-level** hierarchy calculation
- Full evaluation in `compute_all_metrics()` uses **global** hierarchy across all operations
- Controller needs **local** batch hierarchy for real-time learning

**Edge Cases to Handle:**
- Small batch sizes (< 2 samples)
- Uniform valuations in batch (no correlation possible)
- NaN values from `spearmanr`
- Device compatibility (CPU/GPU)

**Performance Considerations:**
- `spearmanr` requires CPU arrays
- Add `.detach().cpu().numpy()` conversions
- Consider caching valuations if indices are predictable

---

## ITERATION PLAN

### Iteration 1-2: Investigation & Setup
- **Goal:** Understand current code flow and identify exact fix location
- **Actions:**
  - Examine `src/models/ternary_vae.py` controller integration
  - Trace how `batch_stats` is constructed and used
  - Review `src/core/metrics.py` hierarchy calculation reference
  - Understand difference between batch vs global hierarchy
- **Validation:** Clear understanding of the fix scope

### Iteration 3-4: Core Fix Implementation
- **Goal:** Replace placeholder values with real hierarchy calculation
- **Actions:**
  - Add required imports (`scipy.stats.spearmanr`, `src.core.TERNARY`)
  - Implement hierarchy calculation for current batch
  - Handle edge cases (small batch, uniform valuations, NaN values)
  - Ensure device compatibility
- **Validation:** Code compiles without errors

### Iteration 5-6: Basic Functionality Testing
- **Goal:** Ensure fix doesn't break existing functionality
- **Actions:**
  - Run model forward pass with controller enabled
  - Verify `batch_stats` tensor has correct shape and values
  - Test with different batch sizes
  - Check that non-hierarchy metrics remain unchanged
- **Validation:** Model runs without errors, other metrics preserved

### Iteration 7-8: Training Validation
- **Goal:** Confirm controller now learns from real hierarchy signals
- **Actions:**
  - Run short training with controller enabled
  - Monitor controller outputs for changing loss weights
  - Compare hierarchy learning before/after fix
  - Verify improved convergence or stability
- **Validation:** Controller demonstrates hierarchy-aware learning

### Iteration 9-10: Comprehensive Testing & Documentation
- **Goal:** Ensure robustness and document the improvement
- **Actions:**
  - Test edge cases (small batches, extreme hierarchy values)
  - Run full training validation
  - Compare training metrics with v5.12.4 baseline
  - Document the fix and its impact
- **Validation:** Training performance improved or maintained

---

## VALIDATION CRITERIA

### ✅ Functional Validation
1. **Code Compiles:** No import or syntax errors
2. **Model Runs:** Forward pass completes without errors
3. **Tensor Shapes:** `batch_stats` maintains correct 8-element structure
4. **Device Compatibility:** Works on both CPU and GPU

### ✅ Scientific Validation
1. **Real Hierarchy Values:** Controller receives actual Spearman correlations, not 1.0
2. **Dynamic Learning:** Controller outputs change based on real hierarchy signals
3. **Hierarchy Range:** H_A and H_B values span expected range (-1.0 to +1.0)
4. **Edge Case Handling:** Graceful handling of small batches and uniform valuations

### ✅ Training Validation
1. **Controller Learning:** Loss weights adapt to hierarchy signals during training
2. **Convergence Stability:** Training remains stable or improves
3. **Hierarchy Improvement:** VAE-B hierarchy metric improves or maintains level
4. **Other Metrics Preserved:** Coverage, richness, and other metrics unchanged

### ✅ Performance Validation
1. **No Significant Slowdown:** Training time increase <10%
2. **Memory Usage:** No significant memory overhead
3. **Batch Size Scaling:** Works efficiently across different batch sizes
4. **GPU Compatibility:** CUDA operations remain efficient

---

## EXPECTED OUTCOMES

### Immediate (After Fix)
- ✅ Controller receives real hierarchy values instead of placeholders
- ✅ H_A and H_B values span meaningful range based on actual data
- ✅ Controller can learn hierarchy-aware loss weighting

### Short-term (Training Validation)
- ✅ Controller demonstrates adaptive behavior based on hierarchy signals
- ✅ Training stability maintained or improved
- ✅ Homeostatic capabilities enhanced

### Medium-term (Scientific Impact)
- ✅ Potential improvement in VAE-B hierarchy learning (-0.82 → closer to -0.8321)
- ✅ Better balance between coverage, hierarchy, and richness
- ✅ Enhanced controller-guided training capabilities

---

## RISK MITIGATION

### Low-Risk Areas ✅
- **Isolated Change:** Fix is contained to controller input preparation
- **Preserves Architecture:** No changes to model structure or training loop
- **Reference Implementation:** Following proven pattern from `metrics.py`
- **Clear Rollback:** Easy to revert to placeholder values if needed

### Potential Issues & Solutions

**Issue:** Batch hierarchy differs significantly from global hierarchy
**Solution:** Monitor both batch and global hierarchy; adjust controller interpretation if needed

**Issue:** Performance overhead from `spearmanr` calculation
**Solution:** Profile training time; consider approximation or caching if significant impact

**Issue:** Edge cases with small batches or uniform valuations
**Solution:** Implement robust fallbacks; test with various batch configurations

**Issue:** Controller learning instability with real hierarchy signals
**Solution:** Monitor training curves; adjust controller architecture if needed

---

## SUCCESS METRICS

### Code Quality
- [ ] **Zero compilation errors** after fix implementation
- [ ] **All existing tests pass** (no regression)
- [ ] **Clean code structure** with proper error handling
- [ ] **Comprehensive documentation** of the fix

### Functional Quality
- [ ] **Real hierarchy values** H_A, H_B computed correctly
- [ ] **Dynamic controller behavior** based on actual data
- [ ] **Edge case robustness** for various batch configurations
- [ ] **Device agnostic operation** (CPU/GPU compatibility)

### Scientific Quality
- [ ] **Controller learning demonstrated** through changing loss weights
- [ ] **Training stability preserved** or improved
- [ ] **Hierarchy metrics maintained** or improved
- [ ] **Homeostatic capabilities enhanced** for advanced training strategies

---

## POST-COMPLETION ACTIONS

### 1. Documentation Updates
- Update `docs/audits/V5_12_5_IMPLEMENTATION_PLAN.md` with completion status
- Add fix details to `.claude/CLAUDE.md` project context
- Document the improvement in training capabilities

### 2. Testing Integration
- Add specific test for controller hierarchy calculation
- Include edge case testing for small batches
- Validate across different device configurations

### 3. Training Validation
- Run extended training with fixed controller
- Compare training curves with v5.12.4 baseline
- Document any improvements in convergence or stability

### 4. Future Enhancements
- Consider controller architecture improvements now that real signals are available
- Explore advanced homeostatic training strategies
- Investigate hierarchy-guided loss scheduling

---

## READY TO EXECUTE

This Ralph Loop is **immediately actionable** with:
- ✅ **Clear problem definition** with exact file locations
- ✅ **Technical implementation details** with reference patterns
- ✅ **Comprehensive validation plan** for each iteration
- ✅ **Risk mitigation strategies** for potential issues
- ✅ **Success criteria** for completion

**Estimated Completion:** 4-6 hours across 6-10 iterations
**Risk Level:** LOW (isolated change with clear reference implementation)
**Scientific Impact:** HIGH (enables proper controller learning for advanced training)