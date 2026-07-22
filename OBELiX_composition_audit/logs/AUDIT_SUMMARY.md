# OBELiX production-structure composition audit

Generated: 2026-07-21T20:27:09.068128+00:00
Option-A spectra audited: 311

## Audit status counts
- PASS: 169
- PASS_INTEGERIZED_COMPOSITION: 74
- WARN_SHORT_DISTANCE: 21
- WARN_INTEGERIZATION_APPROXIMATION: 20
- FAIL_GENERAL_STOICHIOMETRY: 11
- FAIL_ELEMENT_SET: 11
- FAIL_LI_STOICHIOMETRY: 4
- FAIL_SEVERE_OVERLAP: 1

## Interpretation rules
- PASS means exact scale-proportional agreement with the official OBELiX composition.
- PASS_INTEGERIZED_COMPOSITION means the finite ordered cell differs only by nearest-integer atom-count rounding or <=2% relative residuals, with <=2% total-variation composition error.
- WARN_INTEGERIZATION_APPROXIMATION means the ordered finite cell requires up to a one-atom adjustment for at least one species but remains within the configured 6% total-variation limit.
- FAIL_LI_STOICHIOMETRY means the non-Li framework is acceptable but Li is not, consistent with vacancy over/under-filling.
- source_to_production_ordering_audit.csv compares ordered_adaptive_<ID>.cif with relaxed_<ID>.cif and therefore tests whether relaxation changed atom counts.
- Negative plotted-DOS grid endpoints are not used as stability evidence.
- True instability diagnostics come from phonon_eigen_data.npz mesh frequencies.
- The oxidation-state result is only a feasibility heuristic, not a chemical validation.

## Primary files
- tables/production_structure_composition_audit.csv
- tables/source_to_production_ordering_audit.csv
- tables/spectrum_grid_qc.csv
- tables/eigenfrequency_stability_qc.csv
- tables/regeneration_priority.csv