
OBELiX reviewer-response output package
=======================================

Automatically generated from the raw Drive spectra and official OBELiX split.

Core outputs that run without additional inputs:
  tables/scalar_baseline_performance.csv
  tables/scalar_vs_full_distribution_paired_bootstrap.csv
  tables/frequency_warp_sensitivity.csv
  tables/Li_PDOS_imaginary_mode_QC.csv
  tables/imaginary_mode_exclusion_sensitivity.csv
  tables/censoring_counts_by_split_and_family.csv
  tables/formal_censored_kernel_regression_performance.csv
  tables/fusion_and_stacking_ablation_performance.csv
  tables/per_family_scalar_dependence.csv
  tables/per_family_model_ablation_metrics.csv
  tables/per_family_incremental_Li_information.csv

Optional analyses:
  1. Put convergence_manifest.csv at CONVERGENCE_MANIFEST to run representative
     2x2x2-versus-3x3x3, displacement, q-mesh, and optional NAC tests.
  2. Put reference_spectra_manifest.csv at REFERENCE_MANIFEST to compare additional
     exact-phase DFT spectra or properly preprocessed experimental spectra.
  3. NAC requires an NPZ file containing born, dielectric, and factor arrays. Born
     tensors must follow the Phonopy primitive-cell atom order.
  4. Experimental INS should be compared only after neutron weighting and
     instrument-resolution broadening; the code does not infer these corrections.

All reviewer figures are saved as vector PDF and 600-dpi PNG. CSV files retain
material identifiers and model predictions for exact reproduction.
