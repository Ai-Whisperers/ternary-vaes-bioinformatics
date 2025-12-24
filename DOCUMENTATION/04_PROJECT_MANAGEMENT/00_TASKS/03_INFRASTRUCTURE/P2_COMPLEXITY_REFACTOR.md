# Task: Cyclomatic Complexity Refactoring

**Objective**: Refactor components with critical Cyclomatic Complexity (Rank F, E, D) to improve maintainability and testability.

## 1. Critical Refactoring (Rank F - Score > 50)

These functions are unmaintainable monster functions and must be broken down immediately.

- [ ] `tau_phospho_sweep.py`: `main` (Score: 55). Break into `setup_config`, `run_sweep`, `save_results`.
- [ ] `tau_mtbr_interface.py`: `main` (Score: 57).
- [ ] `cross_validation.py`: `main` (Score: 56). This file appears to be entirely one giant function.

## 2. High Priority (Rank E - Score 31-50)

- [ ] `tau_combinatorial.py`: `main` (Score: 33).
- [ ] `citrullination_analysis.py`: `create_visualization` (Score: 33). Visualization logic is too complex. Move plotting to `visualizations/utils/plotting.py`.
- [ ] `autoantigen_epitope_analysis.py`: `main` (Score: 36).
- [ ] `immunogenicity_analysis_augmented.py`: `main` (Score: 38).
- [ ] `fast_reverse_search.py`: `main` (Score: 36).
- [ ] `semantic_amplification_benchmark.py`: `run_benchmark` (Score: 33).

## 3. Moderate Priority (Rank D - Score 21-30)

- [ ] `tau_vae_trajectory.py`: `main` (Score: 29).
- [ ] `hla_expanded_analysis.py`: `create_expanded_visualization` (Score: 30).
- [ ] `citrullination_analysis.py`: `main` (Score: 26).
- [ ] `codon_optimizer.py`: `CodonOptimizer.optimize` (Score: 26). Core algorithm logic needs simplification (extract methods).
- [ ] `citrullination_shift_analysis.py`: `main` (Score: 30).
- [ ] `deep_structural_analysis.py`: `compute_spatial_clustering` (Score: 25).
- [ ] `handshake_interface_analysis.py`: `main` (Score: 21).
- [ ] `deep_handshake_sweep.py`: `main` (Score: 30).
- [ ] `genetic_code_padic.py`: `phase_1_2_padic_balls` (Score: 21).
- [ ] `variational_orthogonality_test.py`: `run_test` (Score: 21).
- [ ] `padic_biology_validation.py`: `run_validation_suite` (Score: 24).

## Refactoring Strategy

1.  **Extract Method**: Identify distinct logical blocks (e.g., data loading, processing, plotting) and move them to helper functions.
2.  **Move Class**: If `main` is doing heavy lifting, create a `Runner` or `Analysis` class to encapsulate state.
3.  **Simplify Control Flow**: Reduce nested `if/for` loops. Use guard clauses (`return` early).
