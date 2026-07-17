# OBELiX DOS / Li-PDOS publication-level kernel workflow
# INPUT: the ZIP produced by the preceding DOS/PDOS association cell:
#        OBELiX_DOS_PDOS_association_outputs.zip
#
# PRIMARY COHORT: structures having matched total DOS, Li-PDOS, static
# descriptors, and ionic conductivity in the official OBELiX split.
# All model comparisons therefore use identical samples.
# ------------------------- USER CONFIGURATION -------------------------
# This kernel workflow does not read raw Train/Test DOS folders directly.
# It reads the feature tables produced by the preceding association workflow,
# which should be run first using your Drive folders:
#   /content/drive/.shortcut-targets-by-id/1_rZkiXnggqDBzgsITf1d7V4K_4aJ00gF
#   /content/drive/.shortcut-targets-by-id/1uDZz5V21SkVn1nYYRn2QD8YrBPfGKoF1
#
# Preferred input: the live results folder from that preceding workflow.
INPUT_RESULTS_DIR = "/content/obelix_dos_pdos_association/results"

# Fallback input: a zipped copy of the preceding workflow outputs, if you saved it.
INPUT_ZIP = "/content/drive/MyDrive/OBELiX_DOS_PDOS_association_outputs.zip"

WORKDIR = "/content/obelix_kernel_publication"
AUTO_UPLOAD_IF_MISSING = True
AUTO_DOWNLOAD_ZIP = True

# Where the final kernel output ZIP will be copied in Google Drive.
FINAL_OUTPUT_DRIVE_DIR = "/content/drive/MyDrive/OBELiX_convergence_outputs"

PRIMARY_BIN_WIDTH_THz = 1.0
FREQ_MIN_THz = 0.0
FREQ_MAX_THz = 100.0
PRIMARY_CENSOR_POLICY = "limit"     # limit, half_limit, exclude
RANDOM_SEED = 20260624

# Publication defaults. Reduce only for debugging.
OUTER_FOLDS = 5
INNER_FOLDS = 4
ALPHA_LOG10_MIN = -5
ALPHA_LOG10_MAX = 3
N_ALPHA_VALUES = 7
SINGLE_KERNEL_SCALE_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
ADDITIVE_KERNEL_SCALE_GRID = (0.5, 1.0, 2.0)
N_TWO_KERNEL_WEIGHTS = 6
THREE_KERNEL_WEIGHT_STEP = 0.25
WASSERSTEIN_QUANTILES = 256
CLR_PSEUDOCOUNT = 1e-8

N_BOOTSTRAP = 3000
N_HSIC_PERMUTATIONS = 5000
HSIC_BATCH_SIZE = 64
MIN_FAMILY_SIZE_LOFO = 8
RUN_LOFO = True
RUN_CENSOR_SENSITIVITY = True
SAVE_KERNEL_MATRICES = True

# Models to run. Additive kernels use a common relative bandwidth multiplier
# for their component kernels, while component weights are selected by CV.
MODEL_SPECS = {
    "static_rbf": {
        "label": "Static RBF",
        "components": ("static_rbf",),
        "additive": False,
    },
    "li_linear": {
        "label": "Li-PDOS linear",
        "components": ("li_linear",),
        "additive": False,
    },
    "li_rbf": {
        "label": "Li-PDOS RBF",
        "components": ("li_rbf",),
        "additive": False,
    },
    "li_clr_rbf": {
        "label": "Li-PDOS CLR–RBF",
        "components": ("li_clr_rbf",),
        "additive": False,
    },
    "li_cdf_rbf": {
        "label": "Li-PDOS CDF–RBF",
        "components": ("li_cdf_rbf",),
        "additive": False,
    },
    "li_wasserstein": {
        "label": "Li-PDOS Wasserstein",
        "components": ("li_wasserstein",),
        "additive": False,
    },
    "total_rbf": {
        "label": "Total-DOS RBF",
        "components": ("total_rbf",),
        "additive": False,
    },
    "total_cdf_rbf": {
        "label": "Total-DOS CDF–RBF",
        "components": ("total_cdf_rbf",),
        "additive": False,
    },
    "total_wasserstein": {
        "label": "Total-DOS Wasserstein",
        "components": ("total_wasserstein",),
        "additive": False,
    },
    "static_plus_li_wasserstein": {
        "label": "Static + Li Wasserstein",
        "components": ("static_rbf", "li_wasserstein"),
        "additive": True,
    },
    "static_plus_total_wasserstein": {
        "label": "Static + total Wasserstein",
        "components": ("static_rbf", "total_wasserstein"),
        "additive": True,
    },
    "static_plus_li_plus_total": {
        "label": "Static + Li + total Wasserstein",
        "components": ("static_rbf", "li_wasserstein", "total_wasserstein"),
        "additive": True,
    },
}

# ------------------------- INSTALL / IMPORTS --------------------------
import sys, subprocess, os, re, json, math, shutil, zipfile, warnings, itertools, time, ast, zlib
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import importlib.util
package_checks = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
}
missing_packages = [pkg for module, pkg in package_checks.items() if importlib.util.find_spec(module) is None]
if missing_packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing_packages])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Re-create grids after numpy import.
ALPHA_GRID = np.logspace(ALPHA_LOG10_MIN, ALPHA_LOG10_MAX, N_ALPHA_VALUES)
TWO_KERNEL_WEIGHT_GRID = tuple(np.linspace(0.0, 1.0, N_TWO_KERNEL_WEIGHTS))

try:
    import torch
except Exception as exc:
    raise RuntimeError("PyTorch is required in Colab for this GPU workflow.") from exc

GPU_AVAILABLE = bool(torch.cuda.is_available())
DEVICE = torch.device("cuda" if GPU_AVAILABLE else "cpu")
DEVICE_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "CPU fallback"
DTYPE = torch.float64

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if GPU_AVAILABLE:
    torch.cuda.manual_seed_all(RANDOM_SEED)

print(f"Compute device: {DEVICE_NAME}")
if not GPU_AVAILABLE:
    print("WARNING: CUDA was not detected. Select a GPU runtime for the intended workflow.")
else:
    print("CUDA will be used for pairwise distances, eigensolves, KRR prediction, and HSIC permutations.")
if not GPU_AVAILABLE:
    torch.set_num_threads(min(8, os.cpu_count() or 1))

# ----------------------------- DIRECTORIES ----------------------------
ROOT = Path(WORKDIR)
INPUT_ROOT = ROOT / "input_results"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
KERNELS_DIR = RESULTS / "kernels"
PRED_DIR = RESULTS / "predictions"
for p in [ROOT, INPUT_ROOT, RESULTS, TABLES, FIGURES, KERNELS_DIR, PRED_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# -------------------------- GOOGLE DRIVE MOUNT -------------------------
# Mount Drive so INPUT_ZIP and FINAL_OUTPUT_DRIVE_DIR can be accessed.
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except Exception as exc:
    print(f"Google Drive mount skipped or unavailable: {exc}")

# -------------------------- INPUT ZIP HANDLING -------------------------
def locate_or_upload_input_zip():
    candidate = Path(INPUT_ZIP)
    if candidate.exists():
        return candidate
    alternatives = sorted(Path("/content").glob("*DOS*PDOS*association*outputs*.zip"))
    drive_alternatives = []
    search_bases = [Path("/content/drive/MyDrive")]
    if "FINAL_OUTPUT_DRIVE_DIR" in globals():
        search_bases.append(Path(FINAL_OUTPUT_DRIVE_DIR))
    for base in search_bases:
        if base.exists():
            drive_alternatives.extend(sorted(base.rglob("*DOS*PDOS*association*outputs*.zip")))
    alternatives.extend(drive_alternatives)
    if alternatives:
        return alternatives[0]
    if not AUTO_UPLOAD_IF_MISSING:
        raise FileNotFoundError(f"Input ZIP not found: {candidate}")
    try:
        from google.colab import files
        print("Upload OBELiX_DOS_PDOS_association_outputs.zip")
        uploaded = files.upload()
        zips = [Path("/content") / name for name in uploaded if str(name).lower().endswith(".zip")]
        if not zips:
            raise FileNotFoundError("No ZIP file was uploaded.")
        return zips[0]
    except Exception as exc:
        raise FileNotFoundError(
            "The previous workflow output ZIP was not found. Set INPUT_ZIP or upload it when prompted."
        ) from exc


def find_one(root: Path, filename: str) -> Path:
    hits = list(root.rglob(filename))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected exactly one {filename}; found {len(hits)} under {root}")
    return hits[0]


if INPUT_RESULTS_DIR is not None and Path(INPUT_RESULTS_DIR).exists():
    DATA_ROOT = Path(INPUT_RESULTS_DIR)
    print(f"Using association results directory: {DATA_ROOT}")
else:
    zip_input = locate_or_upload_input_zip()
    if INPUT_ROOT.exists():
        shutil.rmtree(INPUT_ROOT)
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_input, "r") as zf:
        zf.extractall(INPUT_ROOT)
    DATA_ROOT = INPUT_ROOT
    print(f"Extracted kernel input: {zip_input}")

# ----------------------------- LOAD TABLES -----------------------------
def load_feature(filename):
    p = find_one(DATA_ROOT, filename)
    df = pd.read_csv(p, dtype={"ID": str})
    if "ID" not in df.columns:
        raise ValueError(f"ID column missing from {p}")
    df["ID"] = df["ID"].astype(str).str.strip().str.lower()
    return df.set_index("ID")


width_tag = f"{PRIMARY_BIN_WIDTH_THz:g}THz"
train_li = load_feature(f"train_li_shape_{width_tag}.csv")
test_li = load_feature(f"test_li_shape_{width_tag}.csv")
train_total = load_feature(f"train_total_shape_{width_tag}.csv")
test_total = load_feature(f"test_total_shape_{width_tag}.csv")
static_all = load_feature("composition_and_static_confounders.csv")

TRAIN_CSV = "https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/train.csv"
TEST_CSV = "https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/test.csv"


def load_obelix_metadata(url, split):
    try:
        df = pd.read_csv(url, dtype={"ID": str})
    except Exception:
        try:
            from google.colab import files
            print(f"Could not download official {split}.csv. Upload it now.")
            uploaded = files.upload()
            names = [n for n in uploaded if n.lower().endswith((".csv", ".xlsx"))]
            if not names:
                raise FileNotFoundError(f"No metadata table uploaded for {split}.")
            name = names[0]
            df = pd.read_excel(name, dtype={"ID": str}) if name.lower().endswith(".xlsx") else pd.read_csv(name, dtype={"ID": str})
        except Exception as exc:
            raise RuntimeError(f"Could not load OBELiX {split} metadata.") from exc
    df["ID"] = df["ID"].astype(str).str.strip().str.lower()
    df["split"] = split
    return df.set_index("ID", drop=False)


train_meta_raw = load_obelix_metadata(TRAIN_CSV, "train")
test_meta_raw = load_obelix_metadata(TEST_CSV, "test")


def parse_conductivity(value):
    if pd.isna(value):
        return np.nan, False, ""
    s = str(value).strip().replace("−", "-").replace("×", "x")
    comparator = "<" if "<" in s else (">" if ">" in s else "")
    s = re.sub(r"([0-9.]+)\s*[xX]\s*10\s*\^?\s*([-+]?\d+)", r"\1e\2", s)
    m = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", s)
    if not m:
        return np.nan, False, comparator
    val = float(m.group(0))
    return val, comparator == "<", comparator


def prepare_meta(meta):
    out = meta.copy()
    parsed = out["Ionic conductivity (S cm-1)"].map(parse_conductivity)
    out["ic_reported"] = [x[0] for x in parsed]
    out["ic_is_upper_limit"] = [x[1] for x in parsed]
    out.loc[out["ic_reported"] <= 0, "ic_reported"] = np.nan
    out["ic_limit"] = out["ic_reported"]
    out["ic_half_limit"] = np.where(out["ic_is_upper_limit"], 0.5 * out["ic_reported"], out["ic_reported"])
    out["ic_exclude"] = np.where(out["ic_is_upper_limit"], np.nan, out["ic_reported"])
    for policy in ["limit", "half_limit", "exclude"]:
        out[f"log10_ic_{policy}"] = np.log10(pd.to_numeric(out[f"ic_{policy}"], errors="coerce"))
    if "Family" not in out.columns:
        out["Family"] = "Unknown"
    out["Family"] = out["Family"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    if "DOI" not in out.columns:
        out["DOI"] = ""
    return out


train_meta = prepare_meta(train_meta_raw)
test_meta = prepare_meta(test_meta_raw)

# --------------------------- DATA PREPARATION --------------------------
def common_numeric_columns(train_df, test_df):
    cols = train_df.columns.intersection(test_df.columns)
    valid = []
    for c in cols:
        a = pd.to_numeric(train_df[c], errors="coerce")
        b = pd.to_numeric(test_df[c], errors="coerce")
        if a.notna().any() or b.notna().any():
            valid.append(c)
    return valid


li_cols = common_numeric_columns(train_li, test_li)
total_cols = common_numeric_columns(train_total, test_total)
static_cols = common_numeric_columns(static_all.loc[static_all.index.intersection(train_meta.index)], static_all.loc[static_all.index.intersection(test_meta.index)])


def row_normalize(X):
    X = np.asarray(X, dtype=float)
    X = np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    sums = X.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("At least one normalized spectrum has zero area.")
    return X / sums


def parse_bin_centers(columns):
    centers = []
    for c in columns:
        m = re.search(r"_(\d+\.\d+)_(\d+\.\d+)THz$", str(c))
        if not m:
            raise ValueError(f"Could not parse frequency interval from {c}")
        centers.append(0.5 * (float(m.group(1)) + float(m.group(2))))
    return np.asarray(centers, float)


li_centers = parse_bin_centers(li_cols)
total_centers = parse_bin_centers(total_cols)
if len(li_centers) != len(total_centers) or not np.allclose(li_centers, total_centers):
    raise ValueError("Li-PDOS and total-DOS frequency grids do not match.")
FREQ_CENTERS = li_centers
BIN_WIDTH = float(np.median(np.diff(FREQ_CENTERS))) if len(FREQ_CENTERS) > 1 else PRIMARY_BIN_WIDTH_THz


def clr_transform(P, eps=CLR_PSEUDOCOUNT):
    P = np.asarray(P, float)
    Q = P + eps
    Q /= Q.sum(axis=1, keepdims=True)
    L = np.log(Q)
    return L - L.mean(axis=1, keepdims=True)


def quantile_embedding(P, centers, n_quantiles=WASSERSTEIN_QUANTILES):
    u = (np.arange(n_quantiles, dtype=float) + 0.5) / n_quantiles
    out = np.empty((P.shape[0], n_quantiles), dtype=float)
    left_edge = max(FREQ_MIN_THz, float(centers[0] - 0.5 * BIN_WIDTH))
    for i, row in enumerate(P):
        cdf = np.cumsum(row)
        cdf[-1] = 1.0
        # Discrete inverse CDF. searchsorted remains well-defined when the CDF
        # has plateaus caused by zero-weight frequency bins.
        indices = np.searchsorted(cdf, u, side="left")
        indices = np.clip(indices, 0, len(centers) - 1)
        out[i] = centers[indices]
    return out


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def leakage_groups(meta_subset):
    """Connected components linking entries that share composition or DOI."""
    n = len(meta_subset)
    uf = UnionFind(n)
    for col in ["Reduced Composition", "DOI"]:
        if col not in meta_subset.columns:
            continue
        buckets = defaultdict(list)
        for i, val in enumerate(meta_subset[col].fillna("").astype(str).str.strip()):
            if val and val.lower() not in {"nan", "none"}:
                buckets[val.lower()].append(i)
        for inds in buckets.values():
            for j in inds[1:]:
                uf.union(inds[0], j)
    roots = [uf.find(i) for i in range(n)]
    mapping = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.asarray([mapping[r] for r in roots], dtype=int)


@dataclass
class KernelContext:
    policy: str
    train_ids: np.ndarray
    test_ids: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    groups_train: np.ndarray
    family_train: np.ndarray
    family_test: np.ndarray
    meta_train: pd.DataFrame
    meta_test: pd.DataFrame
    train_arrays: dict
    test_arrays: dict


def make_context(policy):
    target = f"log10_ic_{policy}"
    train_ids = train_li.index.intersection(train_total.index).intersection(static_all.index).intersection(train_meta.index)
    test_ids = test_li.index.intersection(test_total.index).intersection(static_all.index).intersection(test_meta.index)
    train_ids = train_ids[np.isfinite(train_meta.loc[train_ids, target].to_numpy(float))]
    test_ids = test_ids[np.isfinite(test_meta.loc[test_ids, target].to_numpy(float))]

    li_tr = row_normalize(train_li.loc[train_ids, li_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    li_te = row_normalize(test_li.loc[test_ids, li_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    to_tr = row_normalize(train_total.loc[train_ids, total_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    to_te = row_normalize(test_total.loc[test_ids, total_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    st_tr = static_all.loc[train_ids, static_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    st_te = static_all.loc[test_ids, static_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    train_arrays = {
        "li_raw": li_tr,
        "li_clr": clr_transform(li_tr),
        "li_cdf": np.cumsum(li_tr, axis=1) * BIN_WIDTH,
        "li_w2": quantile_embedding(li_tr, FREQ_CENTERS),
        "total_raw": to_tr,
        "total_cdf": np.cumsum(to_tr, axis=1) * BIN_WIDTH,
        "total_w2": quantile_embedding(to_tr, FREQ_CENTERS),
        "static": st_tr,
    }
    test_arrays = {
        "li_raw": li_te,
        "li_clr": clr_transform(li_te),
        "li_cdf": np.cumsum(li_te, axis=1) * BIN_WIDTH,
        "li_w2": quantile_embedding(li_te, FREQ_CENTERS),
        "total_raw": to_te,
        "total_cdf": np.cumsum(to_te, axis=1) * BIN_WIDTH,
        "total_w2": quantile_embedding(to_te, FREQ_CENTERS),
        "static": st_te,
    }
    mt = train_meta.loc[train_ids].copy()
    me = test_meta.loc[test_ids].copy()
    return KernelContext(
        policy=policy,
        train_ids=np.asarray(train_ids), test_ids=np.asarray(test_ids),
        y_train=mt[target].to_numpy(float), y_test=me[target].to_numpy(float),
        groups_train=leakage_groups(mt),
        family_train=mt["Family"].to_numpy(str), family_test=me["Family"].to_numpy(str),
        meta_train=mt, meta_test=me,
        train_arrays=train_arrays, test_arrays=test_arrays,
    )


CTX = make_context(PRIMARY_CENSOR_POLICY)
print(f"Paired kernel cohort: train={len(CTX.train_ids)}, test={len(CTX.test_ids)}")
print(f"Frequency grid: {len(FREQ_CENTERS)} bins, {FREQ_CENTERS[0]:g}–{FREQ_CENTERS[-1]:g} THz centers")

cohort_table = pd.concat([
    CTX.meta_train.assign(kernel_split="train"),
    CTX.meta_test.assign(kernel_split="test")
], axis=0)
cohort_table.to_csv(TABLES / "kernel_cohort_metadata.csv", index=False)

# ------------------------- GPU KERNEL ENGINE ---------------------------
def to_tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=DTYPE, device=DEVICE)


def squared_distances(A, B):
    At, Bt = to_tensor(A), to_tensor(B)
    return torch.cdist(At, Bt, p=2.0).pow(2)


def median_positive_distance(D2):
    n = D2.shape[0]
    vals = D2[torch.triu_indices(n, n, offset=1, device=DEVICE).unbind()]
    vals = vals[vals > 1e-24]
    if vals.numel() == 0:
        return 1.0
    return float(torch.sqrt(torch.median(vals)).item())


def normalize_kernel(Ktr, Kcross):
    scale = torch.mean(torch.diagonal(Ktr))
    scale = torch.clamp(scale, min=1e-12)
    return Ktr / scale, Kcross / scale


def center_kernel(Ktr, Kcross):
    col_mean = Ktr.mean(dim=0, keepdim=True)
    row_mean = Ktr.mean(dim=1, keepdim=True)
    grand = Ktr.mean()
    Ktrc = Ktr - row_mean - col_mean + grand
    Kxc = Kcross - Kcross.mean(dim=1, keepdim=True) - col_mean + grand
    return Ktrc, Kxc


COMPONENT_MAP = {
    "li_linear": ("li_raw", "linear"),
    "li_rbf": ("li_raw", "rbf"),
    "li_clr_rbf": ("li_clr", "rbf"),
    "li_cdf_rbf": ("li_cdf", "rbf"),
    "li_wasserstein": ("li_w2", "rbf"),
    "total_rbf": ("total_raw", "rbf"),
    "total_cdf_rbf": ("total_cdf", "rbf"),
    "total_wasserstein": ("total_w2", "rbf"),
    "static_rbf": ("static", "rbf"),
}


class KernelEngine:
    def __init__(self, context):
        self.ctx = context
        self.cache = {}

    def _prepare_arrays(self, component, fit_idx, eval_idx, eval_split):
        source, kind = COMPONENT_MAP[component]
        Xfit = self.ctx.train_arrays[source][fit_idx]
        Xeval = self.ctx.train_arrays[source][eval_idx] if eval_split == "train" else self.ctx.test_arrays[source][eval_idx]
        if source == "static":
            imp = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            Xfit = scaler.fit_transform(imp.fit_transform(Xfit))
            Xeval = scaler.transform(imp.transform(Xeval))
        elif kind == "linear":
            scaler = StandardScaler()
            Xfit = scaler.fit_transform(Xfit)
            Xeval = scaler.transform(Xeval)
        return Xfit, Xeval, kind

    def component_kernel(self, component, fit_idx, eval_idx, eval_split="train", scale_mult=1.0):
        key = (component, tuple(np.asarray(fit_idx, int)), tuple(np.asarray(eval_idx, int)), eval_split)
        if key not in self.cache:
            Xfit, Xeval, kind = self._prepare_arrays(component, np.asarray(fit_idx), np.asarray(eval_idx), eval_split)
            if kind == "linear":
                A, B = to_tensor(Xfit), to_tensor(Xeval)
                Ktr = (A @ A.T) / max(A.shape[1], 1)
                Kx = (B @ A.T) / max(A.shape[1], 1)
                self.cache[key] = (kind, Ktr, Kx, 1.0)
            else:
                Dtr = squared_distances(Xfit, Xfit)
                Dx = squared_distances(Xeval, Xfit)
                median = median_positive_distance(Dtr)
                self.cache[key] = (kind, Dtr, Dx, median)
        kind, A, B, median = self.cache[key]
        if kind == "linear":
            return normalize_kernel(A, B)
        ell = max(float(scale_mult) * median, 1e-12)
        Ktr = torch.exp(-A / (2.0 * ell * ell))
        Kx = torch.exp(-B / (2.0 * ell * ell))
        return normalize_kernel(Ktr, Kx)

    def combined_kernel(self, spec, fit_idx, eval_idx, eval_split, scale_mult, weights):
        kernels = [self.component_kernel(c, fit_idx, eval_idx, eval_split, scale_mult) for c in spec["components"]]
        Ktr = torch.zeros_like(kernels[0][0])
        Kx = torch.zeros_like(kernels[0][1])
        for w, (kt, kx) in zip(weights, kernels):
            Ktr = Ktr + float(w) * kt
            Kx = Kx + float(w) * kx
        return Ktr, Kx


ENGINE = KernelEngine(CTX)

# ------------------------- CV / KRR FUNCTIONS -------------------------
def make_group_splits(indices, groups, n_splits, seed):
    indices = np.asarray(indices, int)
    local_groups = np.asarray(groups)[indices]
    unique = np.unique(local_groups)
    if len(unique) >= 3:
        splitter = GroupKFold(n_splits=min(n_splits, len(unique)))
        return [(indices[a], indices[b]) for a, b in splitter.split(indices, groups=local_groups)]
    splitter = KFold(n_splits=min(n_splits, max(2, len(indices) // 10)), shuffle=True, random_state=seed)
    return [(indices[a], indices[b]) for a, b in splitter.split(indices)]


def eig_predictions(Ktr, Kcross, y_train, alphas):
    Kc, Kxc = center_kernel(Ktr, Kcross)
    y = to_tensor(y_train)
    mean = y.mean()
    yc = y - mean
    eigvals, eigvecs = torch.linalg.eigh(Kc)
    eigvals = torch.clamp(eigvals, min=0.0)
    proj = eigvecs.T @ yc
    aa = to_tensor(np.asarray(alphas, float))
    coef = proj[:, None] / (eigvals[:, None] + aa[None, :])
    preds = Kxc @ eigvecs @ coef + mean
    return preds.detach().cpu().numpy()


def simplex_weights(n_components, step):
    if n_components == 1:
        return [(1.0,)]
    if n_components == 2:
        return [(1.0 - w, w) for w in TWO_KERNEL_WEIGHT_GRID]
    units = int(round(1.0 / step))
    out = []
    for parts in itertools.product(range(units + 1), repeat=n_components):
        if sum(parts) == units:
            out.append(tuple(p / units for p in parts))
    return out


def parameter_grid(spec):
    components = spec["components"]
    weights = simplex_weights(len(components), THREE_KERNEL_WEIGHT_STEP)
    if len(components) == 1 and COMPONENT_MAP[components[0]][1] == "linear":
        scales = (1.0,)
    elif spec.get("additive", False):
        scales = ADDITIVE_KERNEL_SCALE_GRID
    else:
        scales = SINGLE_KERNEL_SCALE_GRID
    return scales, weights


def tune_model(context, engine, spec, subset_idx, seed):
    subset_idx = np.asarray(subset_idx, int)
    splits = make_group_splits(subset_idx, context.groups_train, INNER_FOLDS, seed)
    scales, weights_grid = parameter_grid(spec)
    records = []
    best = None

    for scale in scales:
        for weights in weights_grid:
            fold_mae = np.zeros((len(splits), len(ALPHA_GRID)), dtype=float)
            for f, (tr_idx, va_idx) in enumerate(splits):
                Ktr, Kva = engine.combined_kernel(spec, tr_idx, va_idx, "train", scale, weights)
                pred = eig_predictions(Ktr, Kva, context.y_train[tr_idx], ALPHA_GRID)
                fold_mae[f] = np.mean(np.abs(context.y_train[va_idx, None] - pred), axis=0)
            mean_mae = fold_mae.mean(axis=0)
            sd_mae = fold_mae.std(axis=0, ddof=1) if len(splits) > 1 else np.zeros_like(mean_mae)
            for i, alpha in enumerate(ALPHA_GRID):
                row = {
                    "scale_mult": float(scale), "weights": tuple(float(x) for x in weights),
                    "alpha": float(alpha), "cv_MAE": float(mean_mae[i]), "cv_MAE_sd": float(sd_mae[i]),
                }
                records.append(row)
                key = (row["cv_MAE"], row["cv_MAE_sd"], -row["alpha"])
                if best is None or key < best[0]:
                    best = (key, row)
    return best[1], pd.DataFrame(records)


def predict_with_params(context, engine, spec, fit_idx, eval_idx, eval_split, params):
    Ktr, Kx = engine.combined_kernel(
        spec, np.asarray(fit_idx, int), np.asarray(eval_idx, int), eval_split,
        params["scale_mult"], params["weights"]
    )
    pred = eig_predictions(Ktr, Kx, context.y_train[np.asarray(fit_idx, int)], [params["alpha"]])[:, 0]
    return pred


def metrics(y, pred):
    rho = stats.spearmanr(y, pred).statistic if len(y) > 2 else np.nan
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)),
        "Spearman_rho": float(rho),
    }


def run_nested_model(context, engine, model_name, spec):
    n = len(context.y_train)
    all_idx = np.arange(n)
    outer = make_group_splits(all_idx, context.groups_train, OUTER_FOLDS, RANDOM_SEED + 17)
    oof = np.full(n, np.nan)
    outer_rows = []
    search_frames = []

    print(f"\n[{model_name}] nested grouped CV")
    for fold, (tr_idx, va_idx) in enumerate(outer, start=1):
        best, search = tune_model(context, engine, spec, tr_idx, RANDOM_SEED + 1000 * fold)
        oof[va_idx] = predict_with_params(context, engine, spec, tr_idx, va_idx, "train", best)
        row = {"model": model_name, "outer_fold": fold, **best, **metrics(context.y_train[va_idx], oof[va_idx])}
        outer_rows.append(row)
        search["model"] = model_name
        search["search_scope"] = f"outer_fold_{fold}"
        search_frames.append(search)
        print(f"  fold {fold}: MAE={row['MAE']:.3f}; params={best}")

    best_full, search_full = tune_model(context, engine, spec, all_idx, RANDOM_SEED + 99999)
    test_idx = np.arange(len(context.y_test))
    test_pred = predict_with_params(context, engine, spec, all_idx, test_idx, "test", best_full)
    search_full["model"] = model_name
    search_full["search_scope"] = "full_train"
    search_frames.append(search_full)

    result = {
        "model": model_name, "label": spec["label"],
        "oof_pred": oof, "test_pred": test_pred,
        "oof_metrics": metrics(context.y_train, oof),
        "test_metrics": metrics(context.y_test, test_pred),
        "best_full": best_full,
        "outer_rows": pd.DataFrame(outer_rows),
        "search": pd.concat(search_frames, ignore_index=True),
    }
    print(f"  held-out test: {result['test_metrics']}")
    return result


# ----------------------------- RUN MODELS ------------------------------
model_results = {}
outer_frames = []
search_frames = []
for model_name, spec in MODEL_SPECS.items():
    result = run_nested_model(CTX, ENGINE, model_name, spec)
    model_results[model_name] = result
    outer_frames.append(result["outer_rows"])
    search_frames.append(result["search"])

outer_df = pd.concat(outer_frames, ignore_index=True)
search_df = pd.concat(search_frames, ignore_index=True)
outer_df.to_csv(TABLES / "nested_grouped_outer_fold_results.csv", index=False)
search_df.to_csv(TABLES / "nested_hyperparameter_search.csv", index=False)

# Predictions and metrics
prediction_rows = []
metric_rows = []
for name, res in model_results.items():
    for oid, y, p, fam in zip(CTX.train_ids, CTX.y_train, res["oof_pred"], CTX.family_train):
        prediction_rows.append({"ID": oid, "split": "train_OOF", "Family": fam, "model": name, "observed": y, "predicted": p})
    for oid, y, p, fam in zip(CTX.test_ids, CTX.y_test, res["test_pred"], CTX.family_test):
        prediction_rows.append({"ID": oid, "split": "test", "Family": fam, "model": name, "observed": y, "predicted": p})
    metric_rows.append({"model": name, "label": res["label"], "evaluation": "nested_train_OOF", "n": len(CTX.y_train), **res["oof_metrics"]})
    metric_rows.append({"model": name, "label": res["label"], "evaluation": "official_test", "n": len(CTX.y_test), **res["test_metrics"]})

pred_df = pd.DataFrame(prediction_rows)
metrics_df = pd.DataFrame(metric_rows)
pred_df.to_csv(PRED_DIR / "all_kernel_predictions.csv", index=False)
metrics_df.to_csv(TABLES / "kernel_performance_summary.csv", index=False)

# ---------------------- BOOTSTRAP UNCERTAINTY -------------------------
def safe_metric(metric_name, y, p):
    if metric_name == "MAE": return mean_absolute_error(y, p)
    if metric_name == "RMSE": return math.sqrt(mean_squared_error(y, p))
    if metric_name == "R2": return r2_score(y, p)
    if metric_name == "Spearman_rho": return stats.spearmanr(y, p).statistic
    raise ValueError(metric_name)


rng = np.random.default_rng(RANDOM_SEED + 123)
boot_indices = rng.integers(0, len(CTX.y_test), size=(N_BOOTSTRAP, len(CTX.y_test)))
metric_names = ["MAE", "RMSE", "R2", "Spearman_rho"]
bootstrap_rows = []
boot_store = {}
for name, res in model_results.items():
    for metric_name in metric_names:
        vals = np.empty(N_BOOTSTRAP, float)
        for b, idx in enumerate(boot_indices):
            vals[b] = safe_metric(metric_name, CTX.y_test[idx], res["test_pred"][idx])
        boot_store[(name, metric_name)] = vals
        bootstrap_rows.append({
            "model": name, "metric": metric_name,
            "point": safe_metric(metric_name, CTX.y_test, res["test_pred"]),
            "ci95_low": float(np.nanpercentile(vals, 2.5)),
            "ci95_high": float(np.nanpercentile(vals, 97.5)),
            "n_bootstrap": N_BOOTSTRAP,
        })
bootstrap_df = pd.DataFrame(bootstrap_rows)
bootstrap_df.to_csv(TABLES / "heldout_bootstrap_metric_intervals.csv", index=False)

baseline_name = "static_rbf"
paired_rows = []
for name in model_results:
    if name == baseline_name:
        continue
    for metric_name in metric_names:
        if metric_name in {"MAE", "RMSE"}:
            diff = boot_store[(name, metric_name)] - boot_store[(baseline_name, metric_name)]
            point = safe_metric(metric_name, CTX.y_test, model_results[name]["test_pred"]) - safe_metric(metric_name, CTX.y_test, model_results[baseline_name]["test_pred"])
            better_direction = "negative"
        else:
            diff = boot_store[(name, metric_name)] - boot_store[(baseline_name, metric_name)]
            point = safe_metric(metric_name, CTX.y_test, model_results[name]["test_pred"]) - safe_metric(metric_name, CTX.y_test, model_results[baseline_name]["test_pred"])
            better_direction = "positive"
        p_two = min(1.0, 2.0 * min(np.mean(diff <= 0), np.mean(diff >= 0)))
        paired_rows.append({
            "model": name, "baseline": baseline_name, "metric": metric_name,
            "difference_model_minus_baseline": float(point),
            "ci95_low": float(np.nanpercentile(diff, 2.5)),
            "ci95_high": float(np.nanpercentile(diff, 97.5)),
            "bootstrap_p_two_sided": float(p_two),
            "better_direction": better_direction,
        })
paired_df = pd.DataFrame(paired_rows)
paired_df.to_csv(TABLES / "paired_bootstrap_differences_vs_static.csv", index=False)

# ----------------------------- HSIC TESTS ------------------------------
def center_square_kernel(K):
    return K - K.mean(0, keepdim=True) - K.mean(1, keepdim=True) + K.mean()


def target_rbf(y):
    yy = np.asarray(y, float)[:, None]
    D2 = squared_distances(yy, yy)
    med = median_positive_distance(D2)
    return torch.exp(-D2 / (2.0 * max(med, 1e-12) ** 2))


def normalized_hsic(K, L):
    Kc, Lc = center_square_kernel(K), center_square_kernel(L)
    num = torch.sum(Kc * Lc)
    den = torch.sqrt(torch.sum(Kc * Kc) * torch.sum(Lc * Lc)).clamp(min=1e-15)
    return float((num / den).item()), Kc, Lc


def block_permutation_indices(labels, batch_size, rng):
    labels = np.asarray(labels)
    n = len(labels)
    out = np.tile(np.arange(n), (batch_size, 1))
    for b in range(batch_size):
        for lab in np.unique(labels):
            loc = np.where(labels == lab)[0]
            if len(loc) > 1:
                out[b, loc] = rng.permutation(loc)
    return out


def hsic_permutation_test(K, y, n_perm, within_labels=None, seed=0):
    L = target_rbf(y)
    observed, Kc, Lc = normalized_hsic(K, L)
    denom = torch.sqrt(torch.sum(Kc * Kc) * torch.sum(Lc * Lc)).clamp(min=1e-15)
    rng = np.random.default_rng(seed)
    count = 0
    null_sum = 0.0
    null_sq = 0.0
    done = 0
    n = len(y)
    while done < n_perm:
        b = min(HSIC_BATCH_SIZE, n_perm - done)
        if within_labels is None:
            perm = np.argsort(rng.random((b, n)), axis=1)
        else:
            perm = block_permutation_indices(within_labels, b, rng)
        pt = torch.as_tensor(perm, dtype=torch.long, device=DEVICE)
        Lp = Lc[pt[:, :, None], pt[:, None, :]]
        vals = torch.einsum("ij,bij->b", Kc, Lp) / denom
        arr = vals.detach().cpu().numpy()
        count += int(np.sum(arr >= observed))
        null_sum += float(arr.sum())
        null_sq += float(np.square(arr).sum())
        done += b
    mean = null_sum / n_perm
    sd = math.sqrt(max(null_sq / n_perm - mean * mean, 0.0))
    return {
        "normalized_HSIC": observed,
        "permutation_p": (count + 1) / (n_perm + 1),
        "null_mean": mean,
        "null_sd": sd,
        "n_permutations": n_perm,
    }


def full_train_component_kernel(component, scale):
    idx = np.arange(len(CTX.y_train))
    K, _ = ENGINE.component_kernel(component, idx, idx, "train", scale)
    return K


# HSIC bandwidths use the unsupervised median heuristic (scale=1), not
# supervised KRR-selected bandwidths. This avoids selection-induced inflation
# of permutation significance.
hsic_components = [
    "static_rbf", "li_rbf", "li_clr_rbf", "li_cdf_rbf",
    "li_wasserstein", "total_rbf", "total_cdf_rbf", "total_wasserstein"
]
hsic_rows = []
full_kernels = {}
for component in hsic_components:
    K = full_train_component_kernel(component, 1.0)
    full_kernels[component] = K
    global_test = hsic_permutation_test(K, CTX.y_train, N_HSIC_PERMUTATIONS, None, RANDOM_SEED + len(hsic_rows))
    family_test = hsic_permutation_test(K, CTX.y_train, N_HSIC_PERMUTATIONS, CTX.family_train, RANDOM_SEED + 100 + len(hsic_rows))
    hsic_rows.append({"component": component, "permutation_scheme": "global", **global_test})
    hsic_rows.append({"component": component, "permutation_scheme": "within_Family", **family_test})

# Cross-fitted static residual HSIC.
static_oof = model_results["static_rbf"]["oof_pred"]
residual_y = CTX.y_train - static_oof
for component in ["li_wasserstein", "total_wasserstein", "li_cdf_rbf", "total_cdf_rbf"]:
    K = full_kernels[component]
    global_test = hsic_permutation_test(K, residual_y, N_HSIC_PERMUTATIONS, None, RANDOM_SEED + 500 + len(hsic_rows))
    family_test = hsic_permutation_test(K, residual_y, N_HSIC_PERMUTATIONS, CTX.family_train, RANDOM_SEED + 700 + len(hsic_rows))
    hsic_rows.append({"component": component + "__static_OOF_residual", "permutation_scheme": "global", **global_test})
    hsic_rows.append({"component": component + "__static_OOF_residual", "permutation_scheme": "within_Family", **family_test})

hsic_df = pd.DataFrame(hsic_rows)

def bh_adjust(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out

for scheme, inds in hsic_df.groupby("permutation_scheme").groups.items():
    hsic_df.loc[inds, "permutation_p_BH_FDR"] = bh_adjust(hsic_df.loc[inds, "permutation_p"].to_numpy(float))
hsic_df.to_csv(TABLES / "HSIC_dependence_tests.csv", index=False)

# Direct paired comparison: Li versus total Wasserstein HSIC using identical permutations.
def hsic_difference_test(K1, K2, y, n_perm, seed):
    L = target_rbf(y)
    s1, K1c, Lc = normalized_hsic(K1, L)
    s2, K2c, _ = normalized_hsic(K2, L)
    d_obs = s1 - s2
    den1 = torch.sqrt(torch.sum(K1c*K1c)*torch.sum(Lc*Lc)).clamp(min=1e-15)
    den2 = torch.sqrt(torch.sum(K2c*K2c)*torch.sum(Lc*Lc)).clamp(min=1e-15)
    rng = np.random.default_rng(seed)
    ge = 0; done = 0; null = []
    n = len(y)
    while done < n_perm:
        b = min(HSIC_BATCH_SIZE, n_perm-done)
        perm = np.argsort(rng.random((b,n)), axis=1)
        pt = torch.as_tensor(perm, dtype=torch.long, device=DEVICE)
        Lp = Lc[pt[:,:,None], pt[:,None,:]]
        v1 = torch.einsum("ij,bij->b", K1c, Lp)/den1
        v2 = torch.einsum("ij,bij->b", K2c, Lp)/den2
        d = (v1-v2).detach().cpu().numpy()
        ge += int(np.sum(d >= d_obs)); null.extend(d.tolist()); done += b
    return {
        "HSIC_Li": s1, "HSIC_total": s2, "difference_Li_minus_total": d_obs,
        "one_sided_permutation_p": (ge+1)/(n_perm+1),
        "null_ci95_low": float(np.percentile(null,2.5)),
        "null_ci95_high": float(np.percentile(null,97.5)),
    }

hsic_diff = hsic_difference_test(full_kernels["li_wasserstein"], full_kernels["total_wasserstein"], CTX.y_train, N_HSIC_PERMUTATIONS, RANDOM_SEED+900)
pd.DataFrame([hsic_diff]).to_csv(TABLES / "HSIC_Li_vs_total_Wasserstein.csv", index=False)

# ----------------------- LEAVE-ONE-FAMILY-OUT -------------------------
lofo_rows = []
if RUN_LOFO:
    lofo_models = ["static_rbf", "li_wasserstein", "static_plus_li_wasserstein"]
    families, counts = np.unique(CTX.family_train, return_counts=True)
    eligible = [f for f, c in zip(families, counts) if c >= MIN_FAMILY_SIZE_LOFO and f != "Unknown"]
    print(f"\nLOFO families (n>={MIN_FAMILY_SIZE_LOFO}): {eligible}")
    for family in eligible:
        hold = np.where(CTX.family_train == family)[0]
        fit = np.where(CTX.family_train != family)[0]
        for model_name in lofo_models:
            spec = MODEL_SPECS[model_name]
            best, _ = tune_model(CTX, ENGINE, spec, fit, RANDOM_SEED + zlib.crc32(f"{family}|{model_name}".encode()) % 100000)
            pred = predict_with_params(CTX, ENGINE, spec, fit, hold, "train", best)
            for oid, y, p in zip(CTX.train_ids[hold], CTX.y_train[hold], pred):
                lofo_rows.append({
                    "Family": family, "ID": oid, "model": model_name,
                    "observed": y, "predicted": p, **best,
                })
    lofo_pred = pd.DataFrame(lofo_rows)
    lofo_pred.to_csv(PRED_DIR / "leave_one_family_out_predictions.csv", index=False)
    lofo_metric_rows = []
    if not lofo_pred.empty:
        for (family, model), grp in lofo_pred.groupby(["Family", "model"]):
            lofo_metric_rows.append({"Family": family, "model": model, "n": len(grp), **metrics(grp["observed"], grp["predicted"])})
        for model, grp in lofo_pred.groupby("model"):
            lofo_metric_rows.append({"Family": "ALL_ELIGIBLE_FAMILIES", "model": model, "n": len(grp), **metrics(grp["observed"], grp["predicted"])})
    pd.DataFrame(lofo_metric_rows).to_csv(TABLES / "leave_one_family_out_performance.csv", index=False)

# ---------------------- CENSORING SENSITIVITY -------------------------
sensitivity_rows = []
if RUN_CENSOR_SENSITIVITY:
    sensitivity_models = ["static_rbf", "li_wasserstein", "static_plus_li_wasserstein"]
    for policy in ["limit", "half_limit", "exclude"]:
        cctx = CTX if policy == PRIMARY_CENSOR_POLICY else make_context(policy)
        cengine = ENGINE if policy == PRIMARY_CENSOR_POLICY else KernelEngine(cctx)
        all_idx = np.arange(len(cctx.y_train))
        test_idx = np.arange(len(cctx.y_test))
        for model_name in sensitivity_models:
            spec = MODEL_SPECS[model_name]
            if policy == PRIMARY_CENSOR_POLICY:
                best = model_results[model_name]["best_full"]
                pred = model_results[model_name]["test_pred"]
            else:
                best, _ = tune_model(cctx, cengine, spec, all_idx, RANDOM_SEED + zlib.crc32(f"{policy}|{model_name}".encode()) % 100000)
                pred = predict_with_params(cctx, cengine, spec, all_idx, test_idx, "test", best)
            sensitivity_rows.append({
                "censor_policy": policy, "model": model_name,
                "n_train": len(cctx.y_train), "n_test": len(cctx.y_test),
                **best, **metrics(cctx.y_test, pred)
            })
pd.DataFrame(sensitivity_rows).to_csv(TABLES / "censoring_policy_kernel_sensitivity.csv", index=False)

# ----------------------- SAVE KERNEL MATRICES -------------------------
if SAVE_KERNEL_MATRICES:
    order = np.argsort(CTX.y_train)
    for component in ["static_rbf", "li_wasserstein", "total_wasserstein", "li_cdf_rbf", "total_cdf_rbf"]:
        K = full_kernels[component].detach().cpu().numpy()
        pd.DataFrame(K, index=CTX.train_ids, columns=CTX.train_ids).to_csv(KERNELS_DIR / f"train_kernel_{component}.csv.gz", compression="gzip")
        plt.figure(figsize=(6.5, 5.6))
        plt.imshow(K[np.ix_(order, order)], aspect="auto", origin="lower")
        plt.xlabel("Materials ordered by conductivity")
        plt.ylabel("Materials ordered by conductivity")
        plt.title(component.replace("_", " "))
        plt.colorbar(label="Kernel similarity")
        plt.tight_layout()
        plt.savefig(FIGURES / f"kernel_heatmap_{component}.png", dpi=300)
        plt.close()

# ------------------------------- FIGURES -------------------------------
# Held-out MAE with bootstrap intervals.
mae_boot = bootstrap_df[bootstrap_df["metric"] == "MAE"].copy()
mae_boot["label"] = mae_boot["model"].map({k:v["label"] for k,v in MODEL_SPECS.items()})
mae_boot = mae_boot.sort_values("point")
plt.figure(figsize=(9.5, max(5.0, 0.42 * len(mae_boot))))
ypos = np.arange(len(mae_boot))
err = np.vstack([mae_boot["point"]-mae_boot["ci95_low"], mae_boot["ci95_high"]-mae_boot["point"]])
plt.errorbar(mae_boot["point"], ypos, xerr=err, fmt="o", capsize=3)
plt.yticks(ypos, mae_boot["label"])
plt.xlabel("Official-test MAE in log$_{10}$(S cm$^{-1}$)")
plt.title("Kernel model performance with paired bootstrap intervals")
plt.tight_layout()
plt.savefig(FIGURES / "01_kernel_test_MAE_bootstrap.png", dpi=300)
plt.close()

# Parity plots for three principal models, each as a separate figure.
for num, name in enumerate(["static_rbf", "li_wasserstein", "static_plus_li_wasserstein"], start=2):
    pred = model_results[name]["test_pred"]
    lo = min(np.min(CTX.y_test), np.min(pred)); hi = max(np.max(CTX.y_test), np.max(pred))
    plt.figure(figsize=(5.8, 5.4))
    plt.scatter(CTX.y_test, pred, s=38, alpha=0.8)
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    m = model_results[name]["test_metrics"]
    plt.xlabel("Observed log$_{10}$ conductivity")
    plt.ylabel("Predicted log$_{10}$ conductivity")
    plt.title(f"{MODEL_SPECS[name]['label']}\nMAE={m['MAE']:.2f}, $R^2$={m['R2']:.2f}, $\\rho$={m['Spearman_rho']:.2f}")
    plt.tight_layout()
    plt.savefig(FIGURES / f"{num:02d}_parity_{name}.png", dpi=300)
    plt.close()

# HSIC summary.
hplot = hsic_df[(hsic_df["permutation_scheme"] == "global") & (~hsic_df["component"].str.contains("residual"))].copy()
hplot = hplot.sort_values("normalized_HSIC")
plt.figure(figsize=(8.2, 5.8))
plt.barh(hplot["component"], hplot["normalized_HSIC"])
for i, (_, row) in enumerate(hplot.iterrows()):
    plt.text(row["normalized_HSIC"], i, f" p={row['permutation_p']:.3g}", va="center", fontsize=8)
plt.xlabel("Normalized HSIC")
plt.title("Whole-spectrum nonlinear dependence with conductivity")
plt.tight_layout()
plt.savefig(FIGURES / "05_HSIC_global_dependence.png", dpi=300)
plt.close()

# Selected additive weights across outer folds.
weight_rows = outer_df[outer_df["model"].isin(["static_plus_li_wasserstein", "static_plus_total_wasserstein", "static_plus_li_plus_total"])].copy()
if not weight_rows.empty:
    expanded = []
    for _, row in weight_rows.iterrows():
        w = row["weights"] if isinstance(row["weights"], tuple) else ast.literal_eval(str(row["weights"]))
        for comp, value in zip(MODEL_SPECS[row["model"]]["components"], w):
            expanded.append({"model": row["model"], "outer_fold": row["outer_fold"], "component": comp, "weight": value})
    wdf = pd.DataFrame(expanded)
    wdf.to_csv(TABLES / "outer_fold_selected_kernel_weights.csv", index=False)
    summary_w = wdf.groupby(["model", "component"])["weight"].agg(["mean", "std"]).reset_index()
    labels = summary_w["model"] + " | " + summary_w["component"]
    plt.figure(figsize=(9.2, max(4.8, 0.42 * len(summary_w))))
    y = np.arange(len(summary_w))
    plt.errorbar(summary_w["mean"], y, xerr=summary_w["std"].fillna(0), fmt="o", capsize=3)
    plt.yticks(y, labels)
    plt.xlim(-0.05, 1.05)
    plt.xlabel("Kernel weight selected in outer folds")
    plt.title("Multiple-kernel weight stability")
    plt.tight_layout()
    plt.savefig(FIGURES / "06_additive_kernel_weight_stability.png", dpi=300)
    plt.close()

# LOFO aggregate model comparison.
lofo_perf_path = TABLES / "leave_one_family_out_performance.csv"
if lofo_perf_path.exists():
    ldf = pd.read_csv(lofo_perf_path)
    agg = ldf[ldf["Family"] == "ALL_ELIGIBLE_FAMILIES"]
    if not agg.empty:
        plt.figure(figsize=(7.2, 4.8))
        plt.bar(agg["model"], agg["MAE"])
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("LOFO MAE")
        plt.title("Cross-family transfer on eligible OBELiX families")
        plt.tight_layout()
        plt.savefig(FIGURES / "07_leave_one_family_out_MAE.png", dpi=300)
        plt.close()

# ------------------------- SUMMARY / CONFIG ----------------------------
best_test = metrics_df[metrics_df["evaluation"] == "official_test"].sort_values("MAE")
summary_lines = [
    "OBELiX PUBLICATION-LEVEL KERNEL ANALYSIS",
    "=" * 55,
    f"Device: {DEVICE_NAME}",
    f"Primary frequency representation: normalized 1-THz bins over 0 < nu <= 100 THz",
    f"Primary censoring policy: {PRIMARY_CENSOR_POLICY}",
    f"Paired cohort: train={len(CTX.y_train)}, official test={len(CTX.y_test)}",
    f"Leakage groups: connected components sharing reduced composition or DOI",
    f"Nested CV: {OUTER_FOLDS} outer x {INNER_FOLDS} inner grouped folds",
    f"Bootstrap replicates: {N_BOOTSTRAP}; HSIC permutations: {N_HSIC_PERMUTATIONS}",
    "",
    "OFFICIAL TEST PERFORMANCE",
    best_test[["label", "n", "MAE", "RMSE", "R2", "Spearman_rho"]].to_string(index=False),
    "",
    "HSIC TESTS",
    hsic_df.to_string(index=False),
    "",
    "LI VS TOTAL WASSERSTEIN HSIC",
    pd.DataFrame([hsic_diff]).to_string(index=False),
]
if not paired_df.empty:
    summary_lines += ["", "PAIRED BOOTSTRAP DIFFERENCES VS STATIC", paired_df.to_string(index=False)]
summary = "\n".join(summary_lines)
(RESULTS / "SUMMARY.txt").write_text(summary)
print("\n" + summary)

best_params_json = {
    name: {
        "label": MODEL_SPECS[name]["label"],
        "components": MODEL_SPECS[name]["components"],
        "best_full": {
            **{k:v for k,v in res["best_full"].items() if k != "weights"},
            "weights": list(res["best_full"]["weights"]),
        }
    }
    for name, res in model_results.items()
}
(RESULTS / "selected_hyperparameters.json").write_text(json.dumps(best_params_json, indent=2))

config = {
    "input_zip": str(INPUT_ZIP),
    "device": DEVICE_NAME,
    "primary_bin_width_THz": PRIMARY_BIN_WIDTH_THz,
    "frequency_range_THz": [FREQ_MIN_THz, FREQ_MAX_THz],
    "primary_censor_policy": PRIMARY_CENSOR_POLICY,
    "random_seed": RANDOM_SEED,
    "outer_folds": OUTER_FOLDS,
    "inner_folds": INNER_FOLDS,
    "alpha_grid": ALPHA_GRID.tolist(),
    "single_kernel_scale_grid": list(SINGLE_KERNEL_SCALE_GRID),
    "additive_kernel_scale_grid": list(ADDITIVE_KERNEL_SCALE_GRID),
    "wasserstein_quantiles": WASSERSTEIN_QUANTILES,
    "clr_pseudocount": CLR_PSEUDOCOUNT,
    "n_bootstrap": N_BOOTSTRAP,
    "n_hsic_permutations": N_HSIC_PERMUTATIONS,
    "models": MODEL_SPECS,
}
(RESULTS / "run_config.json").write_text(json.dumps(config, indent=2, default=list))

# ------------------------------- ARCHIVE -------------------------------
output_zip = Path("/content/OBELiX_kernel_publication_outputs.zip")
if output_zip.exists():
    output_zip.unlink()
shutil.make_archive(str(output_zip.with_suffix("")), "zip", root_dir=RESULTS)
print(f"\nFinished. Output archive: {output_zip}")

try:
    drive_out = Path(FINAL_OUTPUT_DRIVE_DIR)
    drive_out.mkdir(parents=True, exist_ok=True)
    drive_zip = drive_out / output_zip.name
    shutil.copy2(output_zip, drive_zip)
    print(f"Copied output archive to Drive: {drive_zip}")
except Exception as exc:
    print(f"Could not copy output archive to Drive: {exc}")

if AUTO_DOWNLOAD_ZIP:
    try:
        from google.colab import files
        files.download(str(output_zip))
    except Exception as exc:
        print(f"Automatic download was not started: {exc}")
