# OBELiX x NIMS PhononDB reference-spectrum preparation -- SINGLE COLAB CELL
# Purpose
# -------
# 1. Mount Google Drive and locate the Option-A MatterSim spectra used in the
#    manuscript (Frequency_THz, Total_DOS, Li_PDOS from the same PDOS run).
# 2. Download the shortlisted NIMS PhononDB archives.
# 3. Extract and load phonopy_params.yaml.xz.
# 4. Regenerate NIMS total DOS and Li-PDOS on a common 18x18x18 mesh.
# 5. Verify composition and structural equivalence with the OBELiX/production
#    structure using pymatgen StructureMatcher.
# 6. Save standardized model/reference spectra, QC tables, preview figures,
#    and the executable reference_spectra_manifest.csv expected by the master
#    reviewer-response notebook.
#
# IMPORTANT
# ---------
# - Formula + space group is only a screening criterion. By default, a row is
#   included in the final manifest only after a strict StructureMatcher fit.
# - Edit MANUAL_ACCEPT_IDS only after inspecting the saved structural QC.
# - NIMS spectra are generated with NAC parameters as stored in the archive.
#   Change REFERENCE_NAC_MODE to "disabled" for a no-NAC sensitivity dataset.
# ============================================================================

# ------------------------------- USER SETTINGS -------------------------------
PHONOPY_VERSION = "4.2.2"
PYMATGEN_SPEC = "pymatgen>=2025.6.14"

REFERENCE_MESH = (18, 18, 18)
REFERENCE_SIGMA_THz = 0.10
REFERENCE_NAC_MODE = "as_stored"       # "as_stored" or "disabled"

STRICT_MATCHER = dict(ltol=0.20, stol=0.30, angle_tol=5.0)
RELAXED_MATCHER = dict(ltol=0.30, stol=0.50, angle_tol=10.0)
REQUIRE_STRICT_STRUCTURE_MATCH = True
MANUAL_ACCEPT_IDS = set()               # e.g. {"9lo"}; use only after inspection
MANUAL_REJECT_IDS = set()

FORCE_REDOWNLOAD = False
KEEP_NIMS_ZIP_ARCHIVES = True
SAVE_PREVIEW_FIGURES = True

# Optional direct root overrides. Leave blank to use the same shared-folder IDs
# as the DOS_PDOS_Association notebook.
TRAIN_PDOS_ROOT_OVERRIDE = ""
TEST_PDOS_ROOT_OVERRIDE = ""

DRIVE_INPUT_ROOT = "/content/drive/MyDrive/OBELiX_reviewer_inputs"
PRIORITY_CSV = f"{DRIVE_INPUT_ROOT}/OBELiX_PhononDB_priority_validation_set.csv"
OUTPUT_ROOT = f"{DRIVE_INPUT_ROOT}/reference_spectra_preparation"
FINAL_MANIFEST = f"{DRIVE_INPUT_ROOT}/reference_spectra_manifest.csv"

# Shared Drive folder IDs from the previous DOS_PDOS_Association workflow.
PARENT_FOLDER_ID = "1DjYaQ_hrY0kk_XPL7NMYO_wrYqOPoAxv"
TRAIN_PDOS_ID = "1a_L9USkXLE886DepKrVLk3INozgrHgZ7"
TEST_PDOS_ID = "1OTNY9LTNcAZjDLPOCLgloUu8r3sYy1ji"

# --------------------------- INSTALL AND IMPORT ------------------------------
import os, sys, re, io, gc, json, math, time, shutil, hashlib, zipfile, warnings
from pathlib import Path
from collections import defaultdict

import subprocess
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    f"phonopy=={PHONOPY_VERSION}", PYMATGEN_SPEC,
    "requests>=2.31", "pandas>=2.1", "numpy>=1.26", "matplotlib>=3.8",
])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import phonopy
from scipy.integrate import cumulative_trapezoid
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

warnings.filterwarnings("ignore", category=RuntimeWarning)

from google.colab import drive, files
MOUNT_POINT = "/content/drive"
drive.mount(MOUNT_POINT, force_remount=False)

MYDRIVE_ROOT = Path(MOUNT_POINT) / "MyDrive"
SHORTCUT_ROOT = Path(MOUNT_POINT) / ".shortcut-targets-by-id"
INPUT_ROOT = Path(DRIVE_INPUT_ROOT)
OUT = Path(OUTPUT_ROOT)
RAW_DIR = OUT / "nims_archives"
PARAM_DIR = OUT / "nims_phonopy_params"
REF_DIR = OUT / "reference_csv"
MODEL_DIR = OUT / "model_csv"
STRUCT_DIR = OUT / "structures"
FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"
LOG_DIR = OUT / "logs"
for p in [INPUT_ROOT, OUT, RAW_DIR, PARAM_DIR, REF_DIR, MODEL_DIR,
          STRUCT_DIR, FIG_DIR, TABLE_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})

# ----------------------- EMBEDDED PRIORITY TABLE FALLBACK --------------------
# This permits the cell to run even if the previously generated priority CSV
# has not yet been copied to Drive.
EMBEDDED_PRIORITY = [
    dict(split="train", obelix_id="9lo", obelix_formula="Li3PO4", obelix_space_group="Pnma", space_group_number=62, family="LISICON", mp_id="mp-2878", phonondb_formula="Li3PO4", phonondb_space_group="Pnma (62)", nims_dataset_url="https://mdr.nims.go.jp/concern/datasets/kd17cz12z", nims_download_all_url="https://mdr.nims.go.jp/download_all/kd17cz12z.zip", priority_tier="Tier 1"),
    dict(split="train", obelix_id="w7d", obelix_formula="Li3PO4", obelix_space_group="Pnma", space_group_number=62, family="LISICON", mp_id="mp-2878", phonondb_formula="Li3PO4", phonondb_space_group="Pnma (62)", nims_dataset_url="https://mdr.nims.go.jp/concern/datasets/kd17cz12z", nims_download_all_url="https://mdr.nims.go.jp/download_all/kd17cz12z.zip", priority_tier="Tier 1"),
    dict(split="train", obelix_id="qno", obelix_formula="LiTi2(PO4)3", obelix_space_group="R-3c", space_group_number=167, family="NASICON", mp_id="mp-18640", phonondb_formula="LiTi2(PO4)3", phonondb_space_group="R-3c (167)", nims_dataset_url="https://mdr.nims.go.jp/concern/datasets/8336h6993", nims_download_all_url="https://mdr.nims.go.jp/download_all/8336h6993.zip", priority_tier="Tier 1"),
    dict(split="train", obelix_id="rr5", obelix_formula="LiTi2(PO4)3", obelix_space_group="R-3c", space_group_number=167, family="NASICON", mp_id="mp-18640", phonondb_formula="LiTi2(PO4)3", phonondb_space_group="R-3c (167)", nims_dataset_url="https://mdr.nims.go.jp/concern/datasets/8336h6993", nims_download_all_url="https://mdr.nims.go.jp/download_all/8336h6993.zip", priority_tier="Tier 1"),
    dict(split="test", obelix_id="4p7", obelix_formula="Li3ClO", obelix_space_group="Pm-3m", space_group_number=221, family="antiperovskite", mp_id="mp-985585", phonondb_formula="Li3ClO", phonondb_space_group="Pm-3m (221)", nims_dataset_url="https://mdr.nims.go.jp/concern/datasets/qn59q7766", nims_download_all_url="https://mdr.nims.go.jp/download_all/qn59q7766.zip", priority_tier="Tier 1"),
    dict(split="train", obelix_id="qbb", obelix_formula="Li3OBr", obelix_space_group="Pm-3m", space_group_number=221, family="antiperovskite", mp_id="mp-28593", phonondb_formula="Li3BrO", phonondb_space_group="Pm-3m (221)", nims_dataset_url="https://mdr.nims.go.jp/concern/datasets/4m90f035n", nims_download_all_url="https://mdr.nims.go.jp/download_all/4m90f035n.zip", priority_tier="Tier 1"),
    dict(split="train", obelix_id="5wz", obelix_formula="Li6PS5I", obelix_space_group="F-43m", space_group_number=216, family="argyrodites", mp_id="mp-985582", phonondb_formula="Li6PS5I", phonondb_space_group="F-43m (216)", nims_dataset_url="https://mdr.nims.go.jp/concern/datasets/d217qw467", nims_download_all_url="https://mdr.nims.go.jp/download_all/d217qw467.zip", priority_tier="Tier 1"),
    dict(split="train", obelix_id="p9u", obelix_formula="Li6PS5I", obelix_space_group="F-43m", space_group_number=216, family="argyrodites", mp_id="mp-985582", phonondb_formula="Li6PS5I", phonondb_space_group="F-43m (216)", nims_dataset_url="https://mdr.nims.go.jp/concern/datasets/d217qw467", nims_download_all_url="https://mdr.nims.go.jp/download_all/d217qw467.zip", priority_tier="Tier 1"),
    dict(split="train", obelix_id="ywe", obelix_formula="Li6PS5I", obelix_space_group="F-43m", space_group_number=216, family="argyrodites", mp_id="mp-985582", phonondb_formula="Li6PS5I", phonondb_space_group="F-43m (216)", nims_dataset_url="https://mdr.nims.go.jp/concern/datasets/d217qw467", nims_download_all_url="https://mdr.nims.go.jp/download_all/d217qw467.zip", priority_tier="Tier 1"),
    dict(split="train", obelix_id="l9v", obelix_formula="Li3AlF6", obelix_space_group="C2/c", space_group_number=15, family="fluorides", mp_id="mp-15254", phonondb_formula="Li3AlF6", phonondb_space_group="C2/c (15)", nims_dataset_url="https://mdr.nims.go.jp/concern/datasets/zs25xc85p", nims_download_all_url="https://mdr.nims.go.jp/download_all/zs25xc85p.zip", priority_tier="Tier 1"),
    dict(split="test", obelix_id="p81", obelix_formula="LiAlCl4", obelix_space_group="P21/c", space_group_number=14, family="halides", mp_id="mp-22983", phonondb_formula="LiAlCl4", phonondb_space_group="P2_1/c (14)", nims_dataset_url="https://mdr.nims.go.jp/concern/datasets/c821gq85b", nims_download_all_url="https://mdr.nims.go.jp/download_all/c821gq85b.zip", priority_tier="Tier 1"),
    dict(split="train", obelix_id="gvd", obelix_formula="Li3N", obelix_space_group="P6/mmm", space_group_number=191, family="nitrides", mp_id="mp-2251", phonondb_formula="Li3N", phonondb_space_group="P6/mmm (191)", nims_dataset_url="https://mdr.nims.go.jp/concern/datasets/m900nz63k", nims_download_all_url="https://mdr.nims.go.jp/download_all/m900nz63k.zip", priority_tier="Tier 1"),
    dict(split="test", obelix_id="i47", obelix_formula="LiZnPS4", obelix_space_group="I-4", space_group_number=82, family="sulfides", mp_id="mp-11175", phonondb_formula="LiZnPS4", phonondb_space_group="I-4 (82)", nims_dataset_url="https://mdr.nims.go.jp/concern/datasets/1544bv007", nims_download_all_url="https://mdr.nims.go.jp/download_all/1544bv007.zip", priority_tier="Tier 1"),
    dict(split="train", obelix_id="nhq", obelix_formula="Li3PS4", obelix_space_group="Pnma", space_group_number=62, family="thio-LISICON", mp_id="mp-985583", phonondb_formula="Li3PS4", phonondb_space_group="Pnma (62)", nims_dataset_url="https://mdr.nims.go.jp/concern/datasets/9306t291b", nims_download_all_url="https://mdr.nims.go.jp/download_all/9306t291b.zip", priority_tier="Tier 1"),
    dict(split="train", obelix_id="mq1", obelix_formula="Li3SbS4", obelix_space_group="Pmn21", space_group_number=31, family="thio-LISICON", mp_id="mp-756316", phonondb_formula="Li3SbS4", phonondb_space_group="Pmn2_1 (31)", nims_dataset_url="https://mdr.nims.go.jp/concern/datasets/zg64tr70g", nims_download_all_url="https://mdr.nims.go.jp/download_all/zg64tr70g.zip", priority_tier="Tier 1"),
    dict(split="train", obelix_id="e6b", obelix_formula="Li4GeS4", obelix_space_group="Pnma", space_group_number=62, family="thio-LISICON", mp_id="mp-30249", phonondb_formula="Li4GeS4", phonondb_space_group="Pnma (62)", nims_dataset_url="https://mdr.nims.go.jp/concern/datasets/cj82kc69q", nims_download_all_url="https://mdr.nims.go.jp/download_all/cj82kc69q.zip", priority_tier="Tier 1"),
    dict(split="train", obelix_id="4ba", obelix_formula="LiGe2(PO4)3", obelix_space_group="R-3c", space_group_number=167, family="NASICON", mp_id="mp-541272", phonondb_formula="LiGe2(PO4)3", phonondb_space_group="R-3c (167)", nims_dataset_url="https://mdr.nims.go.jp/concern/datasets/bz60d2656", nims_download_all_url="https://mdr.nims.go.jp/download_all/bz60d2656.zip", priority_tier="Tier 2"),
    dict(split="train", obelix_id="ldx", obelix_formula="LiGe2(PO4)3", obelix_space_group="R-3c", space_group_number=167, family="NASICON", mp_id="mp-541272", phonondb_formula="LiGe2(PO4)3", phonondb_space_group="R-3c (167)", nims_dataset_url="https://mdr.nims.go.jp/concern/datasets/bz60d2656", nims_download_all_url="https://mdr.nims.go.jp/download_all/bz60d2656.zip", priority_tier="Tier 2"),
    dict(split="train", obelix_id="ur9", obelix_formula="LiZr2(PO4)3", obelix_space_group="R-3c", space_group_number=167, family="NASICON", mp_id="mp-541661", phonondb_formula="LiZr2(PO4)3", phonondb_space_group="R-3c (167)", nims_dataset_url="https://mdr.nims.go.jp/concern/datasets/rf55zc484", nims_download_all_url="https://mdr.nims.go.jp/download_all/rf55zc484.zip", priority_tier="Tier 2"),
    dict(split="train", obelix_id="be9", obelix_formula="LiGaBr4", obelix_space_group="P21/c", space_group_number=14, family="halides", mp_id="mp-28326", phonondb_formula="LiGaBr4", phonondb_space_group="P2_1/c (14)", nims_dataset_url="https://mdr.nims.go.jp/concern/datasets/9c67ws95f", nims_download_all_url="https://mdr.nims.go.jp/download_all/9c67ws95f.zip", priority_tier="Tier 2"),
]

if Path(PRIORITY_CSV).exists():
    priority = pd.read_csv(PRIORITY_CSV, dtype={"obelix_id": str, "mp_id": str})
    print(f"Loaded priority table: {PRIORITY_CSV}")
else:
    priority = pd.DataFrame(EMBEDDED_PRIORITY)
    priority.to_csv(PRIORITY_CSV, index=False)
    print(f"Created embedded priority table at: {PRIORITY_CSV}")

required_priority_cols = {
    "split", "obelix_id", "obelix_formula", "space_group_number", "family",
    "mp_id", "phonondb_formula", "nims_dataset_url", "nims_download_all_url",
}
missing = required_priority_cols - set(priority.columns)
if missing:
    raise ValueError(f"Priority CSV is missing columns: {sorted(missing)}")
priority["obelix_id"] = priority["obelix_id"].astype(str).str.strip().str.lower()
priority["split"] = priority["split"].astype(str).str.strip().str.lower()
priority["mp_id"] = priority["mp_id"].astype(str).str.strip()
priority = priority.drop_duplicates(subset=["split", "obelix_id", "mp_id"]).reset_index(drop=True)
priority.to_csv(TABLE_DIR / "priority_input_used.csv", index=False)
KNOWN_IDS = set(priority["obelix_id"])

# ------------------------ RESOLVE OPTION-A MODEL ROOTS ------------------------
# Your current Drive folders. Each material subfolder, e.g. 122/, aab/, 6ji/,
# contains both:
#   Lithium_PDOS_<material_id>.csv
#   Total_DOS_<material_id>.csv
# The code below pairs those two files and creates a temporary combined model CSV
# with columns: Frequency_THz, Total_DOS, Li_PDOS.

TRAIN_ROOT = Path("/content/drive/.shortcut-targets-by-id/1_rZkiXnggqDBzgsITf1d7V4K_4aJ00gF")
TEST_ROOT  = Path("/content/drive/.shortcut-targets-by-id/1uDZz5V21SkVn1nYYRn2QD8YrBPfGKoF1")

# If direct folder-ID paths do not work in your Colab session, comment the two
# lines above and uncomment these two MyDrive paths instead:
# TRAIN_ROOT = Path("/content/drive/MyDrive/Train_cif_dos_code-2")
# TEST_ROOT  = Path("/content/drive/MyDrive/Test_cif_dos_code-2")

COMBINED_MODEL_INPUT_DIR = MODEL_DIR / "combined_from_separate_csv"
COMBINED_MODEL_INPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SOURCE_DIRS = {}


def usable_directory(path):
    path = Path(path)
    try:
        return path.exists() and path.is_dir() and os.access(path, os.R_OK)
    except Exception:
        return False


def show_directory(path, max_entries=100):
    path = Path(path)
    print(f"\nContents of: {path}")
    if not usable_directory(path):
        print("  [not accessible]")
        return
    try:
        entries = sorted(path.iterdir(), key=lambda p: p.name.lower())
        for p in entries[:max_entries]:
            kind = "DIR " if p.is_dir() else "FILE"
            print(f"  {kind}  {repr(p.name)}")
        if len(entries) > max_entries:
            print(f"  ... and {len(entries) - max_entries} more entries")
    except Exception as exc:
        print(f"  Could not list directory: {exc}")


def check_model_root(root, label):
    root = Path(root)
    if not usable_directory(root):
        print("\nTop-level MyDrive entries:")
        show_directory(MYDRIVE_ROOT, max_entries=100)
        print("\nVisible shortcut-target IDs:")
        show_directory(SHORTCUT_ROOT, max_entries=100)
        raise FileNotFoundError(
            f"{label} folder not found or not readable:\n{root}\n\n"
            "If this is a shared folder, add a shortcut to My Drive or run Colab "
            "with the Google account that has access."
        )
    li_files = sorted(root.rglob("Lithium_PDOS_*.csv"))
    total_files = sorted(root.rglob("Total_DOS_*.csv"))
    print(f"{label}: {root}")
    print(f"  Li PDOS files: {len(li_files)}")
    print(f"  Total DOS files: {len(total_files)}")
    if not li_files:
        raise RuntimeError(f"No Lithium_PDOS_*.csv files found in {root}")
    if not total_files:
        raise RuntimeError(f"No Total_DOS_*.csv files found in {root}")


check_model_root(TRAIN_ROOT, "TRAIN")
check_model_root(TEST_ROOT, "TEST")
TRAIN_PDOS_ROOT = TRAIN_ROOT
TEST_PDOS_ROOT = TEST_ROOT
print(f"TRAIN model root: {TRAIN_PDOS_ROOT}")
print(f"TEST  model root: {TEST_PDOS_ROOT}")


# --------------------------- INDEX MODEL SPECTRA -----------------------------
def extract_obelix_id(path, collection_root, known_ids):
    try:
        parts = path.relative_to(collection_root).parts
    except Exception:
        parts = path.parts
    for part in reversed(parts):
        stem = Path(part).stem.lower()
        tokens = re.findall(r"(?<![a-z0-9])([a-z0-9]{3})(?![a-z0-9])", stem)
        for token in tokens:
            if token in known_ids:
                return token
        if stem in known_ids:
            return stem
    for part in reversed(parts):
        stem = Path(part).stem.lower()
        for token in re.split(r"[^a-z0-9]+", stem):
            if token in known_ids:
                return token
        for oid in known_ids:
            if stem.startswith((oid + "_", oid + "-")) or stem.endswith(("_" + oid, "-" + oid)):
                return oid
    return None


def _normalized_column_map(df):
    return {re.sub(r"[^a-z0-9]+", "", str(c).lower()): c for c in df.columns}


def _pick_column(df, possible_keys, label, path):
    cmap = _normalized_column_map(df)
    for key in possible_keys:
        if key in cmap:
            return cmap[key]
    raise KeyError(f"Could not find {label} column in {path}. Columns: {list(df.columns)}")


def _read_total_csv(path):
    df = pd.read_csv(path)
    fcol = _pick_column(df, ["frequencythz", "frequency", "freqthz", "freq"], "frequency", path)
    ycol = _pick_column(df, ["totaldos", "dos", "densityofstates"], "Total_DOS", path)
    out = pd.DataFrame({
        "Frequency_THz": pd.to_numeric(df[fcol], errors="coerce"),
        "Total_DOS": pd.to_numeric(df[ycol], errors="coerce"),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    return out.sort_values("Frequency_THz").groupby("Frequency_THz", as_index=False).mean(numeric_only=True)


def _read_li_csv(path):
    df = pd.read_csv(path)
    fcol = _pick_column(df, ["frequencythz", "frequency", "freqthz", "freq"], "frequency", path)
    ycol = _pick_column(df, ["lithium_pdos", "lipdos", "lithiumpdos", "lithiumdos", "lidos"], "Li_PDOS", path)
    out = pd.DataFrame({
        "Frequency_THz": pd.to_numeric(df[fcol], errors="coerce"),
        "Li_PDOS": pd.to_numeric(df[ycol], errors="coerce"),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    return out.sort_values("Frequency_THz").groupby("Frequency_THz", as_index=False).mean(numeric_only=True)


def combine_total_and_li_csv(total_path, li_path, out_path):
    total = _read_total_csv(total_path)
    li = _read_li_csv(li_path)
    freq = total["Frequency_THz"].to_numpy(float)
    li_freq = li["Frequency_THz"].to_numpy(float)
    li_y = li["Li_PDOS"].to_numpy(float)
    total_y = total["Total_DOS"].to_numpy(float)

    if len(freq) == len(li_freq) and np.allclose(freq, li_freq, rtol=1e-8, atol=1e-10):
        li_on_total = li_y
    else:
        li_on_total = np.interp(freq, li_freq, li_y, left=0.0, right=0.0)

    combined = pd.DataFrame({
        "Frequency_THz": freq,
        "Total_DOS": np.clip(total_y, 0, None),
        "Li_PDOS": np.clip(li_on_total, 0, None),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    return out_path


def index_option_a_files(root, known_ids, split):
    root = Path(root)
    rows = []
    selected = {}

    candidate_dirs = sorted({p.parent for p in root.rglob("*.csv")})
    for folder in candidate_dirs:
        oid = extract_obelix_id(folder, root, known_ids)
        if oid is None:
            for p in folder.glob("*.csv"):
                oid = extract_obelix_id(p, root, known_ids)
                if oid is not None:
                    break
        if oid is None:
            continue

        li_files = sorted(folder.glob("Lithium_PDOS_*.csv"))
        total_files = sorted(folder.glob("Total_DOS_*.csv"))
        if not li_files or not total_files:
            continue

        li_path = next((p for p in li_files if oid in p.name.lower()), li_files[0])
        total_path = next((p for p in total_files if oid in p.name.lower()), total_files[0])

        combined_path = COMBINED_MODEL_INPUT_DIR / split / oid / f"{oid}_combined_Total_DOS_Li_PDOS.csv"
        try:
            combine_total_and_li_csv(total_path, li_path, combined_path)
            selected[oid] = combined_path
            MODEL_SOURCE_DIRS[str(combined_path.resolve())] = folder
            rows.append({
                "ID": oid,
                "selected": True,
                "rank": 1,
                "score": 1000,
                "file": str(combined_path),
                "source_total_dos": str(total_path),
                "source_li_pdos": str(li_path),
                "source_folder": str(folder),
            })
        except Exception as exc:
            rows.append({
                "ID": oid,
                "selected": False,
                "rank": 1,
                "score": -1,
                "file": "",
                "source_total_dos": str(total_path),
                "source_li_pdos": str(li_path),
                "source_folder": str(folder),
                "error": str(exc),
            })

    return selected, pd.DataFrame(rows)


train_model_index, train_index_df = index_option_a_files(TRAIN_PDOS_ROOT, KNOWN_IDS, "train")
test_model_index, test_index_df = index_option_a_files(TEST_PDOS_ROOT, KNOWN_IDS, "test")
model_index = {("train", k): v for k, v in train_model_index.items()}
model_index.update({("test", k): v for k, v in test_model_index.items()})

pd.concat([
    train_index_df.assign(split="train"), test_index_df.assign(split="test")
], ignore_index=True).to_csv(TABLE_DIR / "optionA_model_file_index.csv", index=False)

print(f"Indexed train model spectra: {len(train_model_index)}")
print(f"Indexed test model spectra: {len(test_model_index)}")

# --------------------------- GENERAL UTILITIES -------------------------------
def sha256_file(path, block=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def safe_token(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")

def download_stream(url, destination, retries=4):
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 1000 and not FORCE_REDOWNLOAD:
        try:
            with zipfile.ZipFile(destination) as zf:
                if zf.testzip() is None:
                    return destination, "cached"
        except Exception:
            destination.unlink(missing_ok=True)
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0 OBELiX-phonon-validation/1.0"}
    last_error = None
    for attempt in range(1, retries + 1):
        tmp = destination.with_suffix(destination.suffix + ".part")
        tmp.unlink(missing_ok=True)
        try:
            with session.get(url, headers=headers, stream=True, timeout=(30, 600), allow_redirects=True) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            with zipfile.ZipFile(tmp) as zf:
                bad = zf.testzip()
                if bad is not None:
                    raise zipfile.BadZipFile(f"Corrupt member: {bad}")
            tmp.replace(destination)
            return destination, "downloaded"
        except Exception as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            time.sleep(3 * attempt)
    raise RuntimeError(f"Failed to download {url}: {last_error}")

def extract_best_phonopy_params(zip_path, mp_id):
    target_dir = PARAM_DIR / safe_token(mp_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "phonopy_params.yaml.xz"
    if target.exists() and target.stat().st_size > 100 and not FORCE_REDOWNLOAD:
        return target, "cached"
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir() and Path(m.filename).name.lower() == "phonopy_params.yaml.xz"]
        if not members:
            # Less preferred fallbacks still accepted if they contain force constants.
            members = [m for m in zf.infolist() if not m.is_dir() and Path(m.filename).name.lower() in {
                "phonopy.yaml.xz", "phonopy_params.yaml", "phonopy.yaml"
            }]
        if not members:
            names = [m.filename for m in zf.infolist()[:100]]
            raise FileNotFoundError(f"No phonopy parameter YAML found in {zip_path}. First members: {names}")
        def member_score(m):
            n = m.filename.lower()
            s = 0
            if Path(n).name == "phonopy_params.yaml.xz": s += 1000
            if "phonon" in n: s += 20
            if "band" in n or "mesh" in n: s -= 30
            return (s, m.file_size)
        member = sorted(members, key=member_score, reverse=True)[0]
        data = zf.read(member)
        target.write_bytes(data)
        (target_dir / "archive_member.txt").write_text(member.filename)
    return target, member.filename

def phonopy_atoms_to_structure(atoms):
    return Structure(
        lattice=np.asarray(atoms.cell, float),
        species=list(atoms.symbols),
        coords=np.asarray(atoms.scaled_positions, float),
        coords_are_cartesian=False,
    )

def robust_spacegroup(structure, symprecs=(0.01, 0.03, 0.05, 0.10)):
    result = {}
    for s in symprecs:
        try:
            a = SpacegroupAnalyzer(structure, symprec=s, angle_tolerance=5)
            result[f"sg_symbol_symprec_{s:g}"] = a.get_space_group_symbol()
            result[f"sg_number_symprec_{s:g}"] = int(a.get_space_group_number())
        except Exception:
            result[f"sg_symbol_symprec_{s:g}"] = ""
            result[f"sg_number_symprec_{s:g}"] = np.nan
    return result

def normalized_composition_signature(structure):
    comp = structure.composition.fractional_composition
    return tuple(sorted((el.symbol, round(float(v), 8)) for el, v in comp.items()))

def locate_model_cif(split, oid, model_csv):
    split_folder = "Train_cif" if split == "train" else "Test_cif"
    source_dir = MODEL_SOURCE_DIRS.get(str(Path(model_csv).resolve()))
    roots = [MYDRIVE_ROOT / split_folder / oid, Path(model_csv).parent]
    if source_dir is not None:
        roots.insert(0, Path(source_dir))
    direct_names = [
        f"relaxed_{oid}.cif", f"{oid}.cif", "ultra_relaxed_generated.cif",
        f"relaxed_{Path(model_csv).parent.name}.cif",
    ]
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for name in direct_names:
            p = root / name
            if p.exists():
                candidates.append((1000, p))
        for p in root.rglob("*.cif"):
            n = p.name.lower()
            score = 0
            if "ultra_relaxed" in n: score += 500
            if "relaxed" in n: score += 400
            if oid in n or oid in str(p.parent).lower(): score += 300
            if score > 0:
                candidates.append((score, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], -len(str(x[1]))), reverse=True)
    return candidates[0][1]

def normalize_model_columns(df):
    aliases = {}
    for c in df.columns:
        key = re.sub(r"[^a-z0-9]+", "", str(c).lower())
        aliases[key] = c
    need = {"frequencythz": "Frequency_THz", "totaldos": "Total_DOS", "lipdos": "Li_PDOS"}
    missing = [k for k in need if k not in aliases]
    if missing:
        raise KeyError(f"Option-A columns not found; missing normalized columns {missing}; got {list(df.columns)}")
    out = pd.DataFrame({new: pd.to_numeric(df[aliases[key]], errors="coerce") for key, new in need.items()})
    out = out.replace([np.inf, -np.inf], np.nan).dropna().sort_values("Frequency_THz")
    out = out.groupby("Frequency_THz", as_index=False).mean(numeric_only=True)
    out[["Total_DOS", "Li_PDOS"]] = out[["Total_DOS", "Li_PDOS"]].clip(lower=0)
    return out

def load_phonopy_object(path):
    try:
        return phonopy.load(str(path))
    except TypeError:
        return phonopy.load(phonopy_yaml=str(path))

def generate_reference_csv(params_path, row):
    phonon = load_phonopy_object(params_path)
    nac_present = phonon.nac_params is not None
    if REFERENCE_NAC_MODE == "disabled":
        phonon.nac_params = None
    elif REFERENCE_NAC_MODE != "as_stored":
        raise ValueError("REFERENCE_NAC_MODE must be 'as_stored' or 'disabled'.")

    primitive = phonon.primitive
    unitcell = phonon.unitcell
    if primitive is None or unitcell is None:
        raise RuntimeError("Loaded Phonopy object has no primitive or unit cell.")

    phonon.run_mesh(list(REFERENCE_MESH), with_eigenvectors=True, is_mesh_symmetry=False)
    phonon.run_projected_dos(sigma=REFERENCE_SIGMA_THz)
    d = phonon.get_projected_dos_dict()
    freq = np.asarray(d["frequency_points"], dtype=float)
    pdos = np.asarray(d["projected_dos"], dtype=float)
    if pdos.ndim != 2:
        raise ValueError(f"Unexpected projected DOS shape: {pdos.shape}")
    if pdos.shape[1] != len(freq) and pdos.shape[0] == len(freq):
        pdos = pdos.T
    symbols = list(primitive.symbols)
    if pdos.shape[0] != len(symbols):
        raise ValueError(
            f"Projected DOS has {pdos.shape[0]} rows but primitive has {len(symbols)} atoms."
        )
    li_idx = [i for i, s in enumerate(symbols) if str(s) == "Li"]
    if not li_idx:
        raise ValueError("No Li atoms were found in the NIMS primitive cell.")
    total = np.sum(pdos, axis=0)
    li = np.sum(pdos[li_idx, :], axis=0)
    df = pd.DataFrame({"Frequency_THz": freq, "Total_DOS": total, "Li_PDOS": li})
    df = df.replace([np.inf, -np.inf], np.nan).dropna().sort_values("Frequency_THz")
    df[["Total_DOS", "Li_PDOS"]] = df[["Total_DOS", "Li_PDOS"]].clip(lower=0)

    ref_name = f"{safe_token(row.mp_id)}_{safe_token(row.phonondb_formula)}_NIMS_{REFERENCE_NAC_MODE}.csv"
    ref_path = REF_DIR / ref_name
    df.to_csv(ref_path, index=False)

    ref_unit = phonopy_atoms_to_structure(unitcell)
    ref_prim = phonopy_atoms_to_structure(primitive)
    ref_unit.to(filename=str(STRUCT_DIR / f"{safe_token(row.mp_id)}_NIMS_unitcell.cif"))
    ref_prim.to(filename=str(STRUCT_DIR / f"{safe_token(row.mp_id)}_NIMS_primitive.cif"))

    info = {
        "nac_present_in_archive": bool(nac_present),
        "nac_mode_used": REFERENCE_NAC_MODE,
        "reference_mesh": "x".join(map(str, REFERENCE_MESH)),
        "reference_sigma_THz": REFERENCE_SIGMA_THz,
        "reference_primitive_natoms": len(ref_prim),
        "reference_unitcell_natoms": len(ref_unit),
        "reference_primitive_formula": ref_prim.composition.reduced_formula,
        "reference_unitcell_formula": ref_unit.composition.reduced_formula,
        "reference_primitive_volume_per_atom_A3": ref_prim.volume / len(ref_prim),
        "reference_unitcell_volume_per_atom_A3": ref_unit.volume / len(ref_unit),
    }
    info.update({f"reference_{k}": v for k, v in robust_spacegroup(ref_unit).items()})
    return ref_path, df, ref_unit, ref_prim, info

def structure_match_qc(model_structure, ref_structure):
    strict = StructureMatcher(
        primitive_cell=True, scale=True, attempt_supercell=True,
        allow_subset=False, **STRICT_MATCHER
    )
    relaxed = StructureMatcher(
        primitive_cell=True, scale=True, attempt_supercell=True,
        allow_subset=False, **RELAXED_MATCHER
    )
    strict_fit = bool(strict.fit(model_structure, ref_structure))
    relaxed_fit = bool(relaxed.fit(model_structure, ref_structure))
    rms, max_dist = np.nan, np.nan
    if strict_fit:
        try:
            vals = strict.get_rms_dist(model_structure, ref_structure)
            if vals is not None:
                rms, max_dist = map(float, vals)
        except Exception:
            pass
    elif relaxed_fit:
        try:
            vals = relaxed.get_rms_dist(model_structure, ref_structure)
            if vals is not None:
                rms, max_dist = map(float, vals)
        except Exception:
            pass
    return {
        "strict_structure_match": strict_fit,
        "relaxed_structure_match": relaxed_fit,
        "matcher_rms_fractional_free_length": rms,
        "matcher_max_fractional_free_length": max_dist,
        "composition_fraction_match": normalized_composition_signature(model_structure) == normalized_composition_signature(ref_structure),
        "model_natoms": len(model_structure),
        "reference_match_cell_natoms": len(ref_structure),
        "model_formula": model_structure.composition.reduced_formula,
        "reference_match_cell_formula": ref_structure.composition.reduced_formula,
        "model_volume_per_atom_A3": model_structure.volume / len(model_structure),
        "reference_match_cell_volume_per_atom_A3": ref_structure.volume / len(ref_structure),
        "volume_per_atom_relative_difference": (
            model_structure.volume / len(model_structure) - ref_structure.volume / len(ref_structure)
        ) / (ref_structure.volume / len(ref_structure)),
    }

def unit_area_curve(df, intensity_col, lo=0.0, hi=100.0, n=4001):
    x = df["Frequency_THz"].to_numpy(float)
    y = np.clip(df[intensity_col].to_numpy(float), 0, None)
    grid = np.linspace(lo, hi, n)
    yy = np.interp(grid, x, y, left=0, right=0)
    area = np.trapezoid(yy, grid)
    if area <= 0:
        raise ValueError(f"Zero positive area for {intensity_col}")
    return grid, yy / area

def spectral_metrics(model_df, ref_df, col):
    hi = min(100.0, float(model_df.Frequency_THz.max()), float(ref_df.Frequency_THz.max()))
    if hi <= 0:
        raise ValueError("No common positive-frequency interval")
    grid, m = unit_area_curve(model_df, col, 0, hi)
    _, r = unit_area_curve(ref_df, col, 0, hi)
    cm = np.concatenate([[0], cumulative_trapezoid(m, grid)])
    cr = np.concatenate([[0], cumulative_trapezoid(r, grid)])
    w1 = float(np.trapezoid(np.abs(cm - cr), grid))
    overlap = float(np.trapezoid(np.minimum(m, r), grid))
    centroid_m = float(np.trapezoid(grid * m, grid))
    centroid_r = float(np.trapezoid(grid * r, grid))
    mask5 = grid <= 5
    f5m = float(np.trapezoid(m[mask5], grid[mask5]))
    f5r = float(np.trapezoid(r[mask5], grid[mask5]))
    q05m = float(np.interp(0.05, cm, grid))
    q05r = float(np.interp(0.05, cr, grid))
    return {
        "W1_THz": w1, "overlap": overlap,
        "model_centroid_THz": centroid_m, "reference_centroid_THz": centroid_r,
        "centroid_error_THz": centroid_m - centroid_r,
        "model_fraction_below_5THz": f5m, "reference_fraction_below_5THz": f5r,
        "delta_fraction_below_5THz": f5m - f5r,
        "model_q05_THz": q05m, "reference_q05_THz": q05r,
        "delta_q05_THz": q05m - q05r,
    }, grid, m, r

def save_overlay(case_id, spectrum_label, grid, model, reference):
    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    ax.plot(grid, model, lw=1.8, label="MatterSim-Phonopy")
    ax.plot(grid, reference, lw=1.8, label="NIMS DFT-Phonopy")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Unit-area spectral density")
    ax.set_title(f"{case_id}: {spectrum_label}")
    ax.legend(frameon=False)
    fig.tight_layout()
    stem = FIG_DIR / f"{safe_token(case_id)}_{safe_token(spectrum_label)}"
    fig.savefig(str(stem) + ".pdf")
    fig.savefig(str(stem) + ".png", dpi=600)
    plt.close(fig)

# -------------------- DOWNLOAD/PROCESS UNIQUE NIMS PHASES --------------------
reference_cache = {}
download_rows = []
unique_refs = priority.drop_duplicates(subset=["mp_id"]).copy()
print(f"Preparing {len(unique_refs)} unique NIMS reference phases...")
for row in unique_refs.itertuples(index=False):
    mp_safe = safe_token(row.mp_id)
    zip_path = RAW_DIR / f"{mp_safe}_NIMS_download_all.zip"
    try:
        zip_path, download_status = download_stream(row.nims_download_all_url, zip_path)
        params_path, member = extract_best_phonopy_params(zip_path, row.mp_id)
        ref_path, ref_df, ref_unit, ref_prim, ref_info = generate_reference_csv(params_path, row)
        reference_cache[row.mp_id] = dict(
            ref_path=ref_path, ref_df=ref_df, ref_unit=ref_unit, ref_prim=ref_prim,
            params_path=params_path, info=ref_info,
        )
        download_rows.append({
            "mp_id": row.mp_id, "status": "ok", "download_status": download_status,
            "archive": str(zip_path), "archive_bytes": zip_path.stat().st_size,
            "archive_sha256": sha256_file(zip_path), "params_file": str(params_path),
            "archive_member": member, "reference_csv": str(ref_path),
            "nims_dataset_url": row.nims_dataset_url,
            "nims_download_all_url": row.nims_download_all_url, "error": "",
            **ref_info,
        })
        print(f"  OK {row.mp_id}: {row.phonondb_formula}")
        if not KEEP_NIMS_ZIP_ARCHIVES:
            zip_path.unlink(missing_ok=True)
    except Exception as exc:
        download_rows.append({
            "mp_id": row.mp_id, "status": "failed", "download_status": "",
            "archive": str(zip_path), "params_file": "", "reference_csv": "",
            "nims_dataset_url": row.nims_dataset_url,
            "nims_download_all_url": row.nims_download_all_url,
            "error": repr(exc),
        })
        print(f"  FAILED {row.mp_id}: {exc}")
    gc.collect()

pd.DataFrame(download_rows).to_csv(TABLE_DIR / "nims_download_and_generation_log.csv", index=False)

# -------------------- MATCH EACH OBELIX RECORD AND BUILD MANIFEST ------------
manifest_rows = []
qc_rows = []
metric_rows = []
errors = []

for row in priority.itertuples(index=False):
    oid, split, mp_id = row.obelix_id, row.split, row.mp_id
    case_prefix = f"{oid}_{safe_token(row.obelix_formula)}_{safe_token(mp_id)}"
    try:
        model_source = model_index.get((split, oid))
        if model_source is None:
            raise FileNotFoundError(f"No Option-A model spectrum indexed for {split}/{oid}")
        model_df = normalize_model_columns(pd.read_csv(model_source))
        model_std = MODEL_DIR / f"{oid}_{safe_token(row.obelix_formula)}_MatterSim_OptionA.csv"
        model_df.to_csv(model_std, index=False)

        if mp_id not in reference_cache:
            raise FileNotFoundError(f"NIMS reference generation failed for {mp_id}")
        ref = reference_cache[mp_id]
        ref_df, ref_path = ref["ref_df"], ref["ref_path"]

        model_cif = locate_model_cif(split, oid, model_source)
        if model_cif is None:
            raise FileNotFoundError(f"No production/relaxed CIF found for {split}/{oid}")
        model_structure = Structure.from_file(model_cif)
        model_structure.to(filename=str(STRUCT_DIR / f"{oid}_model_structure_used.cif"))

        # Prefer the NIMS unit cell for matching. StructureMatcher internally
        # reduces primitive cells and can handle axis permutations/supercells.
        match_info = structure_match_qc(model_structure, ref["ref_unit"])
        match_info.update({f"model_{k}": v for k, v in robust_spacegroup(model_structure).items()})

        strict_fit = bool(match_info["strict_structure_match"])
        relaxed_fit = bool(match_info["relaxed_structure_match"])
        manual_accept = oid in {x.lower() for x in MANUAL_ACCEPT_IDS}
        manual_reject = oid in {x.lower() for x in MANUAL_REJECT_IDS}
        accepted = (strict_fit if REQUIRE_STRICT_STRUCTURE_MATCH else relaxed_fit)
        if manual_accept:
            accepted = True
        if manual_reject:
            accepted = False
        acceptance_reason = (
            "manual_reject" if manual_reject else
            "manual_accept" if manual_accept else
            "strict_structure_match" if strict_fit else
            "relaxed_only_not_accepted" if relaxed_fit and REQUIRE_STRICT_STRUCTURE_MATCH else
            "relaxed_structure_match" if relaxed_fit else
            "structure_mismatch"
        )

        qc_row = {
            "split": split, "obelix_id": oid, "obelix_formula": row.obelix_formula,
            "family": row.family, "candidate_space_group_number": row.space_group_number,
            "mp_id": mp_id, "phonondb_formula": row.phonondb_formula,
            "priority_tier": getattr(row, "priority_tier", ""),
            "model_source_csv": str(model_source), "model_standardized_csv": str(model_std),
            "model_cif": str(model_cif), "reference_csv": str(ref_path),
            "phonopy_params": str(ref["params_path"]),
            "accepted_for_manifest": accepted, "acceptance_reason": acceptance_reason,
            **match_info, **ref["info"],
        }
        qc_rows.append(qc_row)

        for intensity_col, label in [("Total_DOS", "Total DOS"), ("Li_PDOS", "Li-PDOS")]:
            met, grid, m, r = spectral_metrics(model_df, ref_df, intensity_col)
            case_id = f"{case_prefix}_{'total' if intensity_col == 'Total_DOS' else 'LiPDOS'}"
            metric_rows.append({
                "case_id": case_id, "split": split, "obelix_id": oid,
                "formula": row.obelix_formula, "family": row.family,
                "mp_id": mp_id, "spectrum_label": label,
                "accepted_for_manifest": accepted, **met,
            })
            if SAVE_PREVIEW_FIGURES:
                save_overlay(case_id, label, grid, m, r)
            if accepted:
                manifest_rows.append({
                    "case_id": case_id,
                    "reference_type": f"NIMS_DFT_Phonopy_{REFERENCE_NAC_MODE}",
                    "model_file": str(model_std),
                    "reference_file": str(ref_path),
                    "model_frequency_column": "Frequency_THz",
                    "model_intensity_column": intensity_col,
                    "reference_frequency_column": "Frequency_THz",
                    "reference_intensity_column": intensity_col,
                    "spectrum_label": label,
                    "notes": (
                        f"Structurally screened OBELiX {split} ID {oid}; {row.obelix_formula}; "
                        f"NIMS {mp_id}; mesh={REFERENCE_MESH}; sigma={REFERENCE_SIGMA_THz} THz; "
                        f"NAC={REFERENCE_NAC_MODE}; acceptance={acceptance_reason}."
                    ),
                })
        print(f"  {oid}/{mp_id}: accepted={accepted} ({acceptance_reason})")
    except Exception as exc:
        errors.append({
            "split": split, "obelix_id": oid, "formula": row.obelix_formula,
            "mp_id": mp_id, "error": repr(exc),
        })
        print(f"  FAILED {oid}/{mp_id}: {exc}")
    gc.collect()

qc_df = pd.DataFrame(qc_rows)
metrics_df = pd.DataFrame(metric_rows)
errors_df = pd.DataFrame(errors)
manifest_df = pd.DataFrame(manifest_rows)

qc_df.to_csv(TABLE_DIR / "structure_match_qc_all_candidates.csv", index=False)
metrics_df.to_csv(TABLE_DIR / "reference_preview_spectral_metrics.csv", index=False)
errors_df.to_csv(TABLE_DIR / "reference_preparation_errors.csv", index=False)

if not qc_df.empty:
    qc_df[qc_df["accepted_for_manifest"]].to_csv(TABLE_DIR / "accepted_structure_matches.csv", index=False)
    qc_df[~qc_df["accepted_for_manifest"]].to_csv(TABLE_DIR / "rejected_or_manual_review_structure_matches.csv", index=False)

manifest_columns = [
    "case_id", "reference_type", "model_file", "reference_file",
    "model_frequency_column", "model_intensity_column",
    "reference_frequency_column", "reference_intensity_column",
    "spectrum_label", "notes",
]
if manifest_df.empty:
    manifest_df = pd.DataFrame(columns=manifest_columns)
else:
    manifest_df = manifest_df[manifest_columns].sort_values(["case_id", "spectrum_label"])
manifest_df.to_csv(FINAL_MANIFEST, index=False)
manifest_df.to_csv(OUT / "reference_spectra_manifest.csv", index=False)

# Save machine-readable run metadata.
run_metadata = {
    "phonopy_version": phonopy.__version__,
    "pymatgen_version": __import__("importlib.metadata").metadata.version("pymatgen"),
    "reference_mesh": REFERENCE_MESH,
    "reference_sigma_THz": REFERENCE_SIGMA_THz,
    "reference_nac_mode": REFERENCE_NAC_MODE,
    "strict_matcher": STRICT_MATCHER,
    "relaxed_matcher": RELAXED_MATCHER,
    "require_strict_structure_match": REQUIRE_STRICT_STRUCTURE_MATCH,
    "manual_accept_ids": sorted(MANUAL_ACCEPT_IDS),
    "manual_reject_ids": sorted(MANUAL_REJECT_IDS),
    "priority_records": int(len(priority)),
    "unique_nims_phases": int(priority["mp_id"].nunique()),
    "accepted_obelix_records": int(qc_df["accepted_for_manifest"].sum()) if not qc_df.empty else 0,
    "manifest_rows": int(len(manifest_df)),
    "final_manifest": FINAL_MANIFEST,
}
(Path(LOG_DIR) / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2, default=str))
subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=open(LOG_DIR / "pip_freeze.txt", "w"), check=False)

# Compact summary and archive.
summary_rows = []
if not qc_df.empty:
    summary_rows = (
        qc_df.groupby(["family"], dropna=False)
        .agg(candidate_records=("obelix_id", "size"), accepted_records=("accepted_for_manifest", "sum"))
        .reset_index()
    )
    summary_rows.to_csv(TABLE_DIR / "acceptance_summary_by_family.csv", index=False)

zip_base = str(OUT.parent / "reference_spectra_preparation_outputs")
zip_path = shutil.make_archive(zip_base, "zip", root_dir=OUT)

print("\n" + "=" * 88)
print("REFERENCE-SPECTRUM PREPARATION COMPLETE")
print("=" * 88)
print(f"Candidate OBELiX records:       {len(priority)}")
print(f"Unique NIMS phases:             {priority['mp_id'].nunique()}")
print(f"Successful structural QC rows:  {len(qc_df)}")
print(f"Accepted OBELiX records:        {int(qc_df['accepted_for_manifest'].sum()) if not qc_df.empty else 0}")
print(f"Final manifest rows:            {len(manifest_df)} (2 per accepted record)")
print(f"Preparation errors:             {len(errors_df)}")
print(f"\nFinal manifest:\n  {FINAL_MANIFEST}")
print(f"Output archive:\n  {zip_path}")
print("\nInspect these before using the manifest:")
print(f"  {TABLE_DIR / 'structure_match_qc_all_candidates.csv'}")
print(f"  {TABLE_DIR / 'rejected_or_manual_review_structure_matches.csv'}")
print(f"  {TABLE_DIR / 'reference_preview_spectral_metrics.csv'}")

try:
    from IPython.display import display
    if not qc_df.empty:
        display(qc_df[[
            "obelix_id", "obelix_formula", "family", "mp_id",
            "strict_structure_match", "relaxed_structure_match",
            "accepted_for_manifest", "acceptance_reason",
            "volume_per_atom_relative_difference",
        ]].sort_values(["accepted_for_manifest", "family"], ascending=[False, True]))
    display(manifest_df.head(10))
except Exception:
    pass
