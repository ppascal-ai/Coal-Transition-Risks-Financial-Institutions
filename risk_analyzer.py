from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from data_loader import get_clean_data
from emissions_engine import compute_annual_emissions, forecast_emissions
from ml_models import score_current_fleet, train_early_closure_model

CARBON_PRICES = [50, 100, 150]
FORECAST_YEAR = 2030
ANNUAL_REDUCTION = 0.042
BASELINE_YEAR = 2024
ASSET_VALUE_PER_MW = 1_000_000  # USD proxy per MW for Climate VaR denominator


@dataclass
class ExposureResult:
    price: float
    exposures: pd.DataFrame


def risk_adjust_forecast(forecast_df: pd.DataFrame, scored_risk: pd.DataFrame) -> pd.DataFrame:
    """
    Merge forecasted emissions with early-closure risk and scale expected emissions accordingly.
    Expected emissions = annual emissions * (1 - early_closure_risk) when risk is known.
    """
    risk_cols = ["GEM unit/phase ID", "early_closure_risk"]
    risk_lookup = scored_risk[["GEM unit/phase ID", "early_closure_risk"]].drop_duplicates()
    merged = forecast_df.merge(risk_lookup, on="GEM unit/phase ID", how="left")
    merged["early_closure_risk"] = merged["early_closure_risk"].fillna(0)
    merged["expected_emissions_mt"] = merged["annual_emissions_mt"] * (1 - merged["early_closure_risk"])
    return merged


def build_paris_gap_timeseries(
    risk_adjusted_forecast: pd.DataFrame,
    annual_reduction: float = ANNUAL_REDUCTION,
    base_year: int = BASELINE_YEAR,
    end_year: int = FORECAST_YEAR,
) -> pd.DataFrame:
    """
    Aggregate expected emissions by year and compute a Paris-aligned target path:
    target_year = baseline_2024 * (1 - annual_reduction) ** (year - base_year).
    """
    by_year = (
        risk_adjusted_forecast.groupby("year", as_index=False)["expected_emissions_mt"]
        .sum()
        .sort_values("year")
    )
    baseline_row = by_year.loc[by_year["year"] == base_year, "expected_emissions_mt"]
    baseline_value = baseline_row.iloc[0] if not baseline_row.empty else None
    if baseline_value is None:
        by_year["paris_target_mt"] = pd.NA
        by_year["gap_from_target"] = pd.NA
        return by_year

    by_year["paris_target_mt"] = baseline_value * (
        (1 - annual_reduction) ** (by_year["year"] - base_year)
    )
    by_year["gap_from_target"] = by_year["expected_emissions_mt"] - by_year["paris_target_mt"]
    return by_year


def company_gap_status(
    risk_adjusted_forecast: pd.DataFrame,
    clean_units: pd.DataFrame,
    annual_reduction: float = ANNUAL_REDUCTION,
    base_year: int = BASELINE_YEAR,
    target_year: int = FORECAST_YEAR,
) -> pd.DataFrame:
    """
    Compute Paris-aligned gap per Parent for the target year.
    Gap = projected expected emissions (target_year) - Paris target (4.2% annual reduction from baseline year).
    """
    parent_lookup = clean_units[["GEM unit/phase ID", "Parent"]].drop_duplicates()
    merged = risk_adjusted_forecast.merge(parent_lookup, on="GEM unit/phase ID", how="left")
    merged["Parent"] = merged["Parent"].fillna("Unknown")

    baseline = (
        merged[merged["year"] == base_year]
        .groupby("Parent", as_index=False)["expected_emissions_mt"]
        .sum()
        .rename(columns={"expected_emissions_mt": "baseline_2024_mt"})
    )

    projected = (
        merged[merged["year"] == target_year]
        .groupby("Parent", as_index=False)["expected_emissions_mt"]
        .sum()
        .rename(columns={"expected_emissions_mt": "projected_target_year_mt"})
    )

    combined = baseline.merge(projected, on="Parent", how="outer").fillna(0)
    reduction_factor = (1 - annual_reduction) ** (target_year - base_year)
    combined["paris_target_mt"] = combined["baseline_2024_mt"] * reduction_factor
    combined["gap_mt"] = combined["projected_target_year_mt"] - combined["paris_target_mt"]
    combined["status"] = combined["gap_mt"].apply(lambda g: "On-track" if g <= 0 else "Off-track")
    combined = combined.sort_values("gap_mt", ascending=False)
    return combined


def compute_financial_exposure(
    forecast_df: pd.DataFrame,
    clean_units: pd.DataFrame,
    scored_risk: pd.DataFrame,
    price: float,
    year: int = FORECAST_YEAR,
) -> pd.DataFrame:
    """
    Compute financial exposure per Parent company for a given carbon price.
    """
    year_slice = forecast_df[forecast_df["year"] == year]
    if year_slice.empty:
        return pd.DataFrame(columns=["Parent", "expected_emissions_mt", "exposure_usd"])

    risk_adjusted = risk_adjust_forecast(year_slice, scored_risk)
    parent_lookup = clean_units[
        ["GEM unit/phase ID", "Parent", "Capacity (MW)"]
    ].drop_duplicates()
    parent_lookup["Capacity (MW)"] = pd.to_numeric(
        parent_lookup["Capacity (MW)"], errors="coerce"
    ).fillna(0)

    combined = risk_adjusted.merge(parent_lookup, on="GEM unit/phase ID", how="left")
    combined["Parent"] = combined["Parent"].fillna("Unknown")

    aggregated = (
        combined.groupby("Parent", as_index=False)
        .agg(
            expected_emissions_mt=("expected_emissions_mt", "sum"),
            capacity_mw=("Capacity (MW)", "sum"),
        )
        .sort_values("expected_emissions_mt", ascending=False)
    )
    aggregated["exposure_usd"] = aggregated["expected_emissions_mt"] * price * 1e6
    # Climate VaR proxy: exposure over proxy enterprise value (capacity * $1M/MW)
    proxy_value_usd = aggregated["capacity_mw"] * ASSET_VALUE_PER_MW
    aggregated["climate_var_pct"] = (aggregated["exposure_usd"] / proxy_value_usd) * 100
    aggregated.loc[proxy_value_usd <= 0, "climate_var_pct"] = pd.NA
    return aggregated


def run_scenarios() -> Dict[float, ExposureResult]:
    """
    Train model, score risk, forecast emissions, and compute exposures for all scenarios.
    """
    clean_df = get_clean_data()
    model_artifacts = train_early_closure_model(clean_df)
    scored = score_current_fleet(clean_df, model_artifacts.pipeline)
    emissions_units = compute_annual_emissions(clean_df)
    forecast = forecast_emissions(emissions_units)

    results: Dict[float, ExposureResult] = {}
    for price in CARBON_PRICES:
        exposure = compute_financial_exposure(forecast, clean_df, scored, price=price)
        results[price] = ExposureResult(price=price, exposures=exposure)
    return results


if __name__ == "__main__":
    scenarios = run_scenarios()
    exposure_150 = scenarios[150].exposures
    top10 = exposure_150.head(10)
    print("Top 10 Parent Companies by financial exposure under $150/tCO2 (USD):")
    print(top10.to_string(index=False, formatters={"exposure_usd": "{:,.0f}".format}))
