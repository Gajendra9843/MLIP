import os
import sys
import gc
import math
import random
import warnings
from pathlib import Path
from fractions import Fraction

# Hardened memory allocation flags for large cluster nodes
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
import matplotlib
# Force Matplotlib to non-interactive backend for headless cluster jobs
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import torch

warnings.filterwarnings("ignore")

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor

from ase import Atoms
from ase.io import write
from ase.optimize import FIRE, BFGS
from ase.filters import ExpCellFilter

from mattersim.forcefield import MatterSimCalculator
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


def get_integer_scale_factor_from_composition(structure, max_denominator=100):
    scale = 1
    for amount in structure.composition.values():
        frac = Fraction(float(amount)).limit_denominator(max_denominator)
        scale = lcm(scale, frac.denominator)
    return scale


def choose_integer_supercell_matrix(scale_factor):
    best = None
    best_score = 1e99
    for a in range(1, scale_factor + 1):
        if scale_factor % a != 0:
            continue
        rem1 = scale_factor // a
        for b in range(1, rem1 + 1):
            if rem1 % b != 0:
                continue
            c = rem1 // b
            dims = sorted([a, b, c])
            score = (dims[2] - dims[0]) + 0.1 * (dims[1] - dims[0])
            if score < best_score:
                best_score = score
                best = [a, b, c]
    return best


def composition_distance_score(structure):
    if len(structure) < 2:
        return 0.0
    dist_mat = structure.distance_matrix
    nonzero = dist_mat[dist_mat > 1e-6]
    if len(nonzero) == 0:
        return 0.0
    min_dist = np.min(nonzero)
    mean_near = np.mean(np.sort(nonzero)[: min(50, len(nonzero))])
    return min_dist + 0.1 * mean_near


def make_adaptive_ordered_structure(structure, seed=101, n_trials=80, small_occupancy_threshold=0.10, max_denominator=100):
    print("Creating adaptive ordered structure...")
    
    # Check if the structure is already ordered and has whole numbers
    has_decimals = any(not float(amount).is_integer() for amount in structure.composition.values())
    if structure.is_ordered and not has_decimals:
        print("✓ Structure is already completely ordered with integer coefficients. Skipping processing loop.")
        return structure.copy()

    element_amounts = {
        el.symbol: float(amount)
        for el, amount in structure.composition.items()
    }
    print("Original composition amounts:", element_amounts)

    smallest_nonzero = min(amount for amount in element_amounts.values() if amount > 0)
    print(f"Smallest nonzero element amount: {smallest_nonzero:.6f}")

    use_exact_scaling = smallest_nonzero < small_occupancy_threshold
    structure = structure.copy()

    if use_exact_scaling:
        print(f"Small occupancy below {small_occupancy_threshold} detected. Using integer-composition scaling.")
        scale_factor = get_integer_scale_factor_from_composition(structure, max_denominator=max_denominator)
        print(f"Required composition scale factor: {scale_factor}")

        if scale_factor > 1:
            supercell_shape = choose_integer_supercell_matrix(scale_factor)
            print(f"Making supercell: {supercell_shape}")
            structure.make_supercell(supercell_shape)

        target_counts = {
            el.symbol: int(round(amount))
            for el, amount in structure.composition.items()
        }
    else:
        print("No very small occupancy detected. Using nearest-integer rounding.")
        target_counts = {
            el.symbol: int(round(amount))
            for el, amount in structure.composition.items()
        }

        # Safety: if an element exists but would round to zero, keep at least one.
        for el, amount in element_amounts.items():
            if amount > 0 and target_counts.get(el, 0) == 0:
                target_counts[el] = 1

    print("Target integer composition:", target_counts)
    print("Target atom count:", sum(target_counts.values()))

    best_structure = None
    best_score = -1e99

    for trial in range(n_trials):
        rng = random.Random(seed + trial)
        assigned_counts = {el: 0 for el in target_counts}
        used_indices = set()
        final_species = []
        final_coords = []

        # Pass 1: Keep fully occupied sites first
        for idx, site in enumerate(structure):
            items = list(site.species.items())
            if len(items) == 1:
                sp, occ = items[0]
                el = sp.symbol
                if el in target_counts and assigned_counts[el] < target_counts[el]:
                    final_species.append(el)
                    final_coords.append(site.frac_coords)
                    assigned_counts[el] += 1
                    used_indices.add(idx)

        # Pass 2: Fill partially occupied/disordered sites
        element_order = list(target_counts.keys())
        rng.shuffle(element_order)

        for el in element_order:
            missing = target_counts[el] - assigned_counts[el]
            if missing <= 0:
                continue

            candidates = []
            for idx, site in enumerate(structure):
                if idx in used_indices:
                    continue
                for sp, occ in site.species.items():
                    if sp.symbol == el and float(occ) > 0:
                        candidates.append((idx, float(occ)))
                        break

            # Fallback Shield: If constraints are too aggressive, search all partial sites containing element
            if len(candidates) < missing:
                candidates = []
                for idx, site in enumerate(structure):
                    for sp, occ in site.species.items():
                        if sp.symbol == el:
                            candidates.append((idx, float(occ) if float(occ) > 0 else 1.0))
                            break

            # Critical Fail-safe: Handle edge case missing arrays
            if len(candidates) == 0:
                candidates = [(idx, 1.0) for idx in range(len(structure)) if idx not in used_indices]
                if not candidates:
                    candidates = [(idx, 1.0) for idx in range(len(structure))]

            pool = candidates[:]
            for _ in range(min(missing, len(pool))):
                weights = np.array([max(x[1], 1e-4) for x in pool], dtype=float)
                weights = weights / weights.sum()
                selected_pos = rng.choices(range(len(pool)), weights=weights, k=1)[0]
                chosen_idx, _ = pool.pop(selected_pos)

                site = structure[chosen_idx]
                final_species.append(el)
                final_coords.append(site.frac_coords)
                assigned_counts[el] += 1
                used_indices.add(chosen_idx)

        trial_structure = Structure(
            lattice=structure.lattice,
            species=final_species,
            coords=final_coords,
            coords_are_cartesian=False,
        )

        score = composition_distance_score(trial_structure)
        if score > best_score:
            best_score = score
            best_structure = trial_structure

    print("Final ordered composition:", best_structure.composition)
    print("Final atom count:", len(best_structure))
    print(f"Ordering score: {best_score:.4f}")
    return best_structure


def main():
    # Parse inputs from Slurm Array wrapper
    if len(sys.argv) < 3:
        print("❌ Error: Script requires two arguments passed from Slurm script.")
        print("Usage: python pdos.py <cif_filename> <job_identifier>")
        sys.exit(1)

    cif_filename = sys.argv[1].strip()
    job_number = str(sys.argv[2]).strip()

    # Dynamic workspace directory routing
    cif_dir = Path("cif_files").resolve()
    cif_path = cif_dir / cif_filename
    
    if not cif_path.is_file():
        raise FileNotFoundError(f"❌ CIF file not found at path: {cif_path}")

    material_name = cif_path.stem  
    save_dir = Path("output_work2") / f"{material_name}_{job_number}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # USER SETTINGS CONFIGURABLES RE-MAPPED TO LOCAL VARIABLES
    RANDOM_SEED = 101
    N_ORDERING_TRIALS = 80
    SMALL_OCCUPANCY_THRESHOLD = 0.10
    MAX_DENOMINATOR = 100
    RATTLE_STDEV = 0.03
    FIRE_FMAX = 1e-3
    BFGS_FMAX = 1e-5
    FIRE_STEPS = 500
    BFGS_STEPS = 1500
    DISPLACEMENT_DISTANCE = 0.03
    DOS_MESH = [20, 20, 20]
    PHONON_SUPERCELL_MATRIX = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

    print("=======================================================")
    print(f"Processing material: {material_name}")
    print(f"Slurm Track Token:   {job_number}")
    print(f"Output folder:       {save_dir}")
    print("=======================================================")

    # ==========================================================
    # LOAD AND GENERATE ADAPTIVE STRUCTURE
    # ==========================================================
    structure = Structure.from_file(str(cif_path))
    print("Original composition:", structure.composition)
    print("Original number of crystallographic sites:", len(structure))

    try:
        sga = SpacegroupAnalyzer(structure, symprec=1e-3)
        print("Detected crystal system:", sga.get_crystal_system())
    except Exception as e:
        print(f"Symmetry analysis skipped: {e}")

    ordered_structure = make_adaptive_ordered_structure(
        structure,
        seed=RANDOM_SEED,
        n_trials=N_ORDERING_TRIALS,
        small_occupancy_threshold=SMALL_OCCUPANCY_THRESHOLD,
        max_denominator=MAX_DENOMINATOR
    )

    ordered_cif = save_dir / f"ordered_adaptive_{material_name}.cif"
    ordered_structure.to(filename=str(ordered_cif))
    print(f"Ordered CIF saved: {ordered_cif}")

    # ==========================================================
    # MATTERSIM RELAXATION
    # ==========================================================
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available on this compute node.")

    print("\nInitializing MatterSim on GPU...")
    shared_calc = MatterSimCalculator(device="cuda")

    atoms = AseAtomsAdaptor.get_atoms(ordered_structure)
    atoms.calc = shared_calc

    print(f"Applying rattling perturbation: {RATTLE_STDEV} A")
    atoms.rattle(stdev=RATTLE_STDEV, seed=RANDOM_SEED)

    print("Starting FIRE pre-relaxation...")
    pre_filter = ExpCellFilter(atoms)
    pre_opt = FIRE(pre_filter, logfile=None)
    pre_opt.run(fmax=FIRE_FMAX, steps=FIRE_STEPS)

    print("Starting BFGS relaxation...")
    filtered_atoms = ExpCellFilter(atoms)
    optimizer = BFGS(filtered_atoms, logfile=None)

    optimizer.attach(clear_memory, interval=25)
    optimizer.run(fmax=BFGS_FMAX, steps=BFGS_STEPS)

    relaxed_cif = save_dir / f"relaxed_{material_name}.cif"
    write(str(relaxed_cif), atoms)
    print(f"Relaxed CIF saved: {relaxed_cif}")

    del pre_filter, pre_opt, filtered_atoms, optimizer, atoms, ordered_structure
    clear_memory()

    # ==========================================================
    # PHONOPY SETUP
    # ==========================================================
    print("\nPreparing Phonopy calculation...")
    relaxed_structure = Structure.from_file(str(relaxed_cif))
    num_atoms = len(relaxed_structure)

    print("Relaxed atom count:", num_atoms)
    print("Using Phonopy supercell matrix:", PHONON_SUPERCELL_MATRIX)

    supercell_multiplier = int(round(np.linalg.det(np.array(PHONON_SUPERCELL_MATRIX))))
    print(f"Atoms per displaced phonon supercell: {num_atoms * supercell_multiplier}")

    atoms_unitcell = AseAtomsAdaptor.get_atoms(relaxed_structure)
    phonopy_atoms = PhonopyAtoms(
        symbols=atoms_unitcell.get_chemical_symbols(),
        cell=atoms_unitcell.cell.array,
        scaled_positions=atoms_unitcell.get_scaled_positions(),
    )

    phonon = Phonopy(phonopy_atoms, PHONON_SUPERCELL_MATRIX, symprec=1e-3)
    phonon.generate_displacements(distance=DISPLACEMENT_DISTANCE)
    supercells = phonon.supercells_with_displacements
    print("Number of displaced supercells:", len(supercells))

    del atoms_unitcell, relaxed_structure
    clear_memory()

    # ==========================================================
    # FORCE CALCULATIONS
    # ==========================================================
    print("\nCalculating forces for displaced supercells...")
    force_sets = []

    for i, sc in enumerate(supercells):
        atoms_sc = Atoms(symbols=sc.symbols, positions=sc.positions, cell=sc.cell, pbc=True)
        atoms_sc.calc = shared_calc
        raw_forces = atoms_sc.get_forces()

        force_sets.append(np.array(raw_forces, dtype=np.float32))

        atoms_sc.calc = None
        del atoms_sc, raw_forces

        if (i + 1) % 5 == 0 or (i + 1) == len(supercells):
            print(f"Force progress: {i + 1}/{len(supercells)}")
            clear_memory()

    del supercells, shared_calc
    clear_memory()

    # ==========================================================
    # FORCE CONSTANTS
    # ==========================================================
    print("\nProducing force constants...")
    phonon.produce_force_constants(force_sets)
    del force_sets
    clear_memory()

    print("Symmetrizing force constants...")
    phonon.symmetrize_force_constants()

    # ==========================================================
    # PHONON BAND STRUCTURE PNG
    # ==========================================================
    band_png = save_dir / f"phonon_band_{material_name}.png"
    try:
        print("\nGenerating phonon band structure...")
        phonon.auto_band_structure()
        phonon.plot_band_structure()
        plt.savefig(str(band_png), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Phonon band PNG saved: {band_png}")
    except Exception as e:
        print(f"Band structure skipped: {e}")

    # ==========================================================
    # TOTAL DOS
    # ==========================================================
    print("\nRunning mesh for DOS...")
    phonon.run_mesh(DOS_MESH, with_eigenvectors=True, is_mesh_symmetry=False)

    print("Calculating total DOS...")
    phonon.run_total_dos()

    dos = phonon.get_total_dos_dict()
    frequency = dos["frequency_points"]
    total_dos = dos["total_dos"]

    total_dos_csv = save_dir / f"Total_DOS_{material_name}.csv"
    total_dos_png = save_dir / f"Total_DOS_{material_name}.png"

    pd.DataFrame({"Frequency_THz": frequency, "Total_DOS": total_dos}).to_csv(str(total_dos_csv), index=False)

    plt.figure(figsize=(6.8, 4.8))
    plt.plot(frequency, total_dos, color="black", linewidth=2.2, label="Total DOS")
    plt.title(f"Total Phonon DOS - {material_name}", fontsize=12)
    plt.xlabel("Frequency (THz)", fontsize=11)
    plt.ylabel("Density of States", fontsize=11)
    plt.xlim(min(frequency), max(frequency))
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(total_dos_png), dpi=300)
    plt.close()

    print(f"Total DOS CSV saved: {total_dos_csv}")
    print(f"Total DOS PNG saved: {total_dos_png}")

    # ==========================================================
    # LI-PROJECTED DOS
    # ==========================================================
    print("\nCalculating Li-projected DOS...")
    phonon.run_projected_dos()

    pdos = phonon.get_projected_dos_dict()
    projected = pdos["projected_dos"]
    primitive_symbols = phonon.primitive.symbols

    li_indices = [
        i for i, symbol in enumerate(primitive_symbols)
        if str(symbol).strip().capitalize() == "Li"
    ]

    if li_indices:
        print(f"Li atoms found in primitive cell: {len(li_indices)}")
        li_pdos = np.sum(projected[li_indices, :], axis=0)
    else:
        print("No Li found. Saving Li PDOS as zeros.")
        li_pdos = np.zeros_like(total_dos)

    li_pdos_csv = save_dir / f"Lithium_PDOS_{material_name}.csv"
    li_pdos_png = save_dir / f"Lithium_PDOS_{material_name}.png"

    pd.DataFrame({"Frequency_THz": frequency, "Lithium_PDOS": li_pdos}).to_csv(str(li_pdos_csv), index=False)

    plt.figure(figsize=(6.8, 4.8))
    plt.plot(frequency, li_pdos, color="crimson", linewidth=2.2, label="Li PDOS")
    plt.title(f"Lithium Projected Phonon DOS - {material_name}", fontsize=12)
    plt.xlabel("Frequency (THz)", fontsize=11)
    plt.ylabel("Density of States", fontsize=11)
    plt.xlim(min(frequency), max(frequency))
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(li_pdos_png), dpi=300)
    plt.close()

    print(f"Li PDOS CSV saved: {li_pdos_csv}")
    print(f"Li PDOS PNG saved: {li_pdos_png}")

    # ==========================================================
    # COMBINED DOS PNG
    # ==========================================================
    combined_png = save_dir / f"Combined_Total_Li_DOS_{material_name}.png"

    plt.figure(figsize=(7.2, 5.0))
    plt.plot(frequency, total_dos, color="black", linewidth=2.2, linestyle="--", alpha=0.6, label="Total DOS")
    plt.plot(frequency, li_pdos, color="crimson", linewidth=2.2, label="Li PDOS")
    plt.title(f"Total and Li-Projected Phonon DOS - {material_name}", fontsize=12)
    plt.xlabel("Frequency (THz)", fontsize=11)
    plt.ylabel("Density of States", fontsize=11)
    plt.xlim(min(frequency), max(frequency))
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(combined_png), dpi=300)
    plt.close()

    print(f"Combined DOS PNG saved: {combined_png}")

    print("\n=======================================================")
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print(f"Lowest sampled frequency: {np.min(frequency):.4f} THz")
    print("=======================================================")


if __name__ == "__main__":
    main()
