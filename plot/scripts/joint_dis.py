#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.stats as stats

# ======================= Base paths =======================
BASE = Path("/data/gulab/yzdai/data4/phase_identification")

# ======================= System configuration =======================
# drop_ranges: list of (start, end) columns to drop, [start, end) with end exclusive; end=None means to the last column
SYSTEMS = {
	"psmdopochl": {
		"phase": {
			"HMM":     BASE / "plot/input/HMM/psmdopochl/train7-psmdopochl300k-rawdata.xvg",
			"density": BASE / "phase_out/psmdopochl/19000-20000/psmdopochl-rawdata.xvg",
		},
		"scd":  BASE / "plot/input/last1us_gap5_scd_area/psmdopochl_scd.xvg",
		"area": BASE / "plot/input/last1us_gap5_scd_area/psmdopochl_area.xvg",
		"drop_ranges_phase": [(324, 476)],
		"drop_ranges_scd": [(324, 476)],
		"drop_ranges_area": [(324, 476)],
	},

	"dpdochl290k": {
		"phase": {
			"HMM":     BASE / "plot/input/HMM/dpdochl290k/train1-dpdochl290k-rawdata.xvg",
			"density": BASE / "phase_out/dpdochl290k/8000-9000/dpdochl290k-rawdata.xvg",
		},
		"scd":  BASE / "plot/input/last1us_gap5_scd_area/dpdochl290k_scd.xvg",
		"area": BASE / "plot/input/last1us_gap5_scd_area/dpdochl290k_area.xvg",
		"drop_ranges_phase": [(404, 576), (980, 1152)],
		"drop_ranges_scd": [],
		"drop_ranges_area": [(404, 576), (980, 1152)],
	},

	"dpdochl280k": {
		"phase": {
			"HMM":     BASE / "plot/input/HMM/dpdochl280k/train1-dpdochl280k-rawdata.xvg",
			"density": BASE / "phase_out/dpdochl280k/7000-8000/dpdochl280k-rawdata.xvg",
		},
		"scd":  BASE / "plot/input/last1us_gap5_scd_area/dpdochl280k_scd.xvg",
		"area": BASE / "plot/input/last1us_gap5_scd_area/dpdochl280k_area.xvg",
		"drop_ranges_phase": [(404, 576), (980, 1152)],
		"drop_ranges_scd": [],
		"drop_ranges_area": [(404, 576), (980, 1152)],
	},

	"dpdo280k": {
		"phase": {
			"HMM":     BASE / "plot/input/HMM/dpdo280k/train5-dpdo280k-rawdata.xvg",
			"density": BASE / "phase_out/dpdo280k/9000-10000/dpdo280k-rawdata.xvg",
		},
		"scd":  BASE / "plot/input/last1us_gap5_scd_area/dpdo280k_scd.xvg",
		"area": BASE / "plot/input/last1us_gap5_scd_area/dpdo280k_area.xvg",
		"drop_ranges_phase": [],
		"drop_ranges_scd": [],
		"drop_ranges_area": [],
	},

	"dpdo290k": {
		"phase": {
			"HMM":     BASE / "plot/input/HMM/dpdo290k/train0-dpdo290k-rawdata.xvg",
			"density": BASE / "phase_out/dpdo290k/9000-10000/dpdo290k-rawdata.xvg",
		},
		"scd":  BASE / "plot/input/last1us_gap5_scd_area/dpdo290k_scd.xvg",
		"area": BASE / "plot/input/last1us_gap5_scd_area/dpdo290k_area.xvg",
		"drop_ranges_phase": [],
		"drop_ranges_scd": [],
		"drop_ranges_area": [],
	},
}

# ======================= Axis presets (Å²) =======================
AXIS_PRESETS = {
    # dpdo*
    "dpdo280k": {
        "xlim":  (-0.1, 0.51),
        "ylim":  (30, 85),
        "xticks": np.arange(0.0, 0.5, 0.2),
        "yticks": np.arange(35, 86, 15),
    },
    "dpdo290k": {
        "xlim":  (-0.1, 0.51),
        "ylim":  (30, 85),
        "xticks": np.arange(0.0, 0.5, 0.2),
        "yticks": np.arange(35, 86, 15),
    },
    # dpdochl*
    "dpdochl280k": {
        "xlim":  (-0.02, 0.5),
        "ylim":  (35, 75),
        "xticks": np.arange(0.0, 0.5, 0.2),
        "yticks": np.arange(45, 76, 10),
    },
    "dpdochl290k": {
        "xlim":  (-0.02, 0.5),
        "ylim":  (35, 75),
        "xticks": np.arange(0.0, 0.5, 0.2),
        "yticks": np.arange(45, 76, 10),
    },
    # psmdopochl
    "psmdopochl": {
        "xlim":  (0.03, 0.5),
        "ylim":  (37, 73),
        "xticks": np.arange(0.0, 0.5, 0.2),
        "yticks": np.arange(45, 71, 10),
    },
}

# ======================= Utilities =======================
def load_xvg(path: Path) -> np.ndarray:
    """Load .xvg, ignoring lines starting with @ or #. Always return 2D ndarray."""
    arr = np.loadtxt(path, comments='@#')
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr

def drop_time_if_present(arr: np.ndarray, reference_ncols_without_time: Optional[int]) -> np.ndarray:
    """If reference_ncols_without_time is provided and arr has exactly one more column, drop the first column."""
    if reference_ncols_without_time is not None and arr.shape[1] == reference_ncols_without_time + 1:
        return arr[:, 1:]
    return arr

def flip_hmm_if_needed(phase_no_time: np.ndarray, is_hmm: bool) -> np.ndarray:
    """Flip 0/1 labels only for HMM source if count(0) > count(1) in the first feature column."""
    if not is_hmm:
        return phase_no_time
    col0 = phase_no_time[:, 0]
    zeros = np.sum(col0 == 0)
    ones  = np.sum(col0 == 1)
    if zeros > ones:
        phase_no_time = 1 - phase_no_time
    return phase_no_time

def drop_column_ranges(arr: np.ndarray, ranges: List[Tuple[int, Optional[int]]]) -> np.ndarray:
    """Drop column ranges: [(start, end), ...], where end is exclusive; end=None means end of array."""
    if not ranges:
        return arr
    ncols = arr.shape[1]
    keep = np.ones(ncols, dtype=bool)
    for start, end in ranges:
        s = int(start)
        e = ncols if end is None else int(end)
        if not (0 <= s < ncols) or not (0 < e <= ncols) or not (s < e):
            raise ValueError(f"Invalid range [{s}:{e}] for ncols={ncols}")
        keep[s:e] = False
    return arr[:, keep]

def check_same_ncols(*arrays: np.ndarray, names=None) -> None:
    """Ensure arrays have the same number of columns."""
    cols = [a.shape[1] for a in arrays]
    if len(set(cols)) != 1:
        nms = names or [f"arr{i}" for i in range(len(arrays))]
        detail = ", ".join(f"{n}:{c}" for n, c in zip(nms, cols))
        raise ValueError(f"Inconsistent column counts: {detail}")

# ======================= Core pipeline =======================
def run(sys_key: str, source: str):
    if sys_key not in SYSTEMS:
        raise KeyError(f"Unknown system '{sys_key}'. Available: {', '.join(SYSTEMS.keys())}")
    if source not in ("HMM", "density"):
        raise ValueError("source must be 'HMM' or 'density'")

    cfg = SYSTEMS[sys_key]
    phase_path = cfg["phase"][source]
    scd_path   = cfg["scd"]
    area_path  = cfg["area"]

    print(f"== System: {sys_key} (source={source}) ==")
    print("Files:")
    print("  phase:", phase_path)
    print("  scd  :", scd_path)
    print("  area :", area_path)

    # 1) Load raw (possibly with time column)
    phase_raw = load_xvg(phase_path)
    scd_raw   = load_xvg(scd_path)
    area_raw  = load_xvg(area_path)

    print("Raw shapes (may include time column):")
    print("  phase_raw:", phase_raw.shape)
    print("  scd_raw  :", scd_raw.shape)
    print("  area_raw :", area_raw.shape)

    # 2) Remove time column: for phase always drop col 0; for scd/area infer by matching phase width
    phase = phase_raw[:, 1:]
    scd   = drop_time_if_present(scd_raw,  reference_ncols_without_time=phase.shape[1])
    area  = drop_time_if_present(area_raw, reference_ncols_without_time=phase.shape[1])

    print("After dropping time column (if present):")
    print("  phase:", phase.shape)
    print("  scd  :", scd.shape)
    print("  area :", area.shape)

    # 3) HMM label flip if needed
    phase = flip_hmm_if_needed(phase, is_hmm=(source == "HMM"))

    # 4) Drop per-file chol ranges
    phase_drop = cfg.get("drop_ranges_phase", [])
    scd_drop   = cfg.get("drop_ranges_scd", [])
    area_drop  = cfg.get("drop_ranges_area", [])
    if phase_drop:
        phase = drop_column_ranges(phase, phase_drop)
    if scd_drop:
        scd   = drop_column_ranges(scd, scd_drop)
    if area_drop:
        area  = drop_column_ranges(area, area_drop)

    # 5) Ensure aligned column counts
    check_same_ncols(phase, scd, area, names=["phase", "scd", "area"])

    print("After dropping chol ranges:")
    print("  phase:", phase.shape)
    print("  scd  :", scd.shape)
    print("  area :", area.shape)

    return phase, scd, area

# ======================= CLI =======================
def parse_args():
    p = argparse.ArgumentParser(description="Load and preprocess phase/scd/area (select system and source).")
    p.add_argument("--sys",    default="psmdopochl", help=f"System key. Options: {', '.join(SYSTEMS.keys())}")
    p.add_argument("--source", default="HMM", choices=["HMM", "density"], help="Phase source.")
    return p.parse_args()

# ======================= Main =======================
if __name__ == "__main__":
    args = parse_args()
    phase, scd, area = run(args.sys, args.source)

    # Build a long-form DataFrame for plotting/statistics (area in Å²)
    data_list = [
        {
            "time": t,
            "lipid": l,
            "phase": phase[t, l],
            "scd": scd[t, l],
            "area": area[t, l],  # Å²
        }
        for t in range(phase.shape[0])
        for l in range(phase.shape[1])
    ]
    df = pd.DataFrame(data_list)
    df_ld = df[df["phase"] == 0]  # ld
    df_lo = df[df["phase"] == 1]  # lo

    print("DataFrame (head):")
    print(df.head())
    print("\nld (head):")
    print(df_ld.head())
    print("\nlo (head):")
    print(df_lo.head())

    # KDE estimation (scipy) on a grid using ld ranges (assumes both classes present)
    fig = plt.figure(figsize=(3, 3))
    grid = plt.GridSpec(1, 1, hspace=0.0, wspace=0.0)
    ax_joint = plt.subplot(grid[0, 0])

    kde_ld = stats.gaussian_kde([df_ld['scd'], df_ld['area']])
    kde_lo = stats.gaussian_kde([df_lo['scd'], df_lo['area']])

    xgrid = np.linspace(df_ld['scd'].min(), df_ld['scd'].max(), 100)
    ygrid = np.linspace(df_ld['area'].min(), df_ld['area'].max(), 100)
    X, Y = np.meshgrid(xgrid, ygrid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    density_ld = kde_ld(positions).reshape(X.shape)
    density_lo = kde_lo(positions).reshape(X.shape)

    centroid_ld = np.array([np.sum(X * density_ld) / np.sum(density_ld),
                            np.sum(Y * density_ld) / np.sum(density_ld)])
    centroid_lo = np.array([np.sum(X * density_lo) / np.sum(density_lo),
                            np.sum(Y * density_lo) / np.sum(density_lo)])

    distance = np.linalg.norm(centroid_ld - centroid_lo)

    # Seaborn KDE contours
    sns.kdeplot(data=df_ld, x="scd", y="area", cmap="Reds",  fill=False, ax=ax_joint, alpha=1)
    sns.kdeplot(data=df_lo, x="scd", y="area", cmap="Blues", fill=False, ax=ax_joint, alpha=1)

    # Apply axis presets by system (Å²)
    preset = AXIS_PRESETS.get(args.sys)
    if preset is not None:
        ax_joint.set_xlim(*preset["xlim"])
        ax_joint.set_ylim(*preset["ylim"])
        ax_joint.set_xticks(preset["xticks"])
        ax_joint.set_yticks(preset["yticks"])

    # Distance annotation
    ax_joint.text(
        sum(ax_joint.get_xlim()) / 2,
        ax_joint.get_ylim()[0] + (ax_joint.get_ylim()[1] - ax_joint.get_ylim()[0]) / 10,
        f"Distance: {distance:.2f}",
        fontsize=18, color="black", ha="center"
    )

    # Two colorbars as insets for aesthetics
    axins1 = inset_axes(ax_joint, width="60%", height="15%",
                        bbox_to_anchor=(0.98, 1.1 - 0.05, 0.4, 0.1), bbox_transform=ax_joint.transAxes)
    axins2 = inset_axes(ax_joint, width="60%", height="15%",
                        bbox_to_anchor=(0.98, 1.06 - 0.05, 0.4, 0.1), bbox_transform=ax_joint.transAxes)

    norm1 = plt.Normalize(vmin=0, vmax=1)
    sm1 = plt.cm.ScalarMappable(cmap="Reds", norm=norm1); sm1.set_array([])
    norm2 = plt.Normalize(vmin=0, vmax=1)
    sm2 = plt.cm.ScalarMappable(cmap="Blues", norm=norm2); sm2.set_array([])

    cbar1 = plt.colorbar(sm1, cax=axins1, orientation="horizontal")
    cbar2 = plt.colorbar(sm2, cax=axins2, orientation="horizontal")

    cbar1.ax.tick_params(labelbottom=False, labeltop=False, bottom=False, top=False)
    cbar2.ax.tick_params(labelsize=18)

    # Ticks and spines
    ax_joint.xaxis.set_ticks_position("bottom")
    ax_joint.yaxis.set_ticks_position("left")
    ax_joint.tick_params(axis="both", which="major", width=2, size=6, labelsize=18, direction="in")

    for side in ("top", "bottom", "left", "right"):
        ax_joint.spines[side].set_linewidth(2)

    ax_joint.set_xlabel("-Scd", fontsize=18)
    ax_joint.set_ylabel(r"APL ($\AA^{2}$)", fontsize=18)  # Å²

    # Title and save
    title = f"{'HMM ' if args.source == 'HMM' else ''}{args.sys}"
    plt.suptitle(title, fontsize=20, fontweight="bold", y=1)

    out_dir = BASE / "plot/output/scd_area_joint_distri"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = (f"HMM_{args.sys}-joint.png" if args.source == "HMM" else f"{args.sys}-joint.png")
    out_path = out_dir / out_name

    plt.savefig(
        out_path,
        dpi=350,
        format="png",
        bbox_inches="tight",
        pad_inches=0.0,
        transparent=True,
    )
    print(f"Saved figure to: {out_path}")