from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_financial_exposure(exposure_df: pd.DataFrame, output_path: Path, top_n: int = 10) -> Path:
    """
    Create a bar chart of financial exposure by parent company.
    """
    top = exposure_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["Parent"], top["exposure_usd"] / 1e9, color="#2f5caa")
    ax.set_xlabel("Exposure (USD billions)")
    ax.set_ylabel("Parent Company")
    ax.set_title(f"Top {top_n} Financial Exposures at Carbon Price ${int(exposure_df.attrs.get('price', 0))}/tCO2")
    ax.invert_yaxis()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_emissions_gap(gap_df: pd.DataFrame, output_path: Path) -> Path:
    """
    Plot expected emissions vs Paris-aligned target and the gap.
    gap_df expects columns: year, expected_emissions_mt, paris_target_mt, gap_from_target.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gap_df["year"], gap_df["expected_emissions_mt"], marker="o", color="#2f5caa", label="Expected emissions")
    ax.plot(gap_df["year"], gap_df["paris_target_mt"], marker="s", linestyle="--", color="#e74c3c", label="Paris-aligned target")
    ax.fill_between(
        gap_df["year"],
        gap_df["expected_emissions_mt"],
        gap_df["paris_target_mt"],
        color="#e74c3c",
        alpha=0.15,
        label="Gap vs target",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Emissions (MtCO2)")
    ax.set_title("Emissions vs Paris-aligned target (risk-adjusted, 2024-2030)")
    ax.legend(loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
