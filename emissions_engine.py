from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from data_loader import get_clean_data, load_co2_parameters

CAPACITY_FACTOR = 0.55
HOURS_PER_YEAR = 8760
BTU_TO_TJ = 1055.05585 / 1e12  # Convert BTU to terajoules.
DEFAULT_LIFETIME_YEARS = 35
FORECAST_START_YEAR = 2024
FORECAST_END_YEAR = 2030


def _build_emission_factor_map() -> Dict[str, float]:
    params = load_co2_parameters()
    return dict(
        zip(params["coal_type"].astype(str).str.lower(), params["emission_factor_kg_per_TJ"])
    )


def _assign_emission_factor(units_df: pd.DataFrame, factors: Dict[str, float]) -> pd.Series:
    coal_key = units_df["Coal type"].astype(str).str.strip().str.lower()
    explicit = pd.to_numeric(units_df.get("Emission factor (kg of CO2 per TJ)"), errors="coerce")
    mapped = coal_key.map(factors)
    if "unknown" in factors:
        mapped = mapped.fillna(factors["unknown"])
    return explicit.fillna(mapped)


def compute_annual_emissions(units_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annual CO2 emissions (million tonnes) for each plant using
    capacity, heat rate, capacity factor, and coal-type emission factors.
    """
    factors = _build_emission_factor_map()
    df = units_df.copy()
    df["emission_factor_kg_per_TJ"] = _assign_emission_factor(df, factors)
    df["Heat rate (Btu per kWh)"] = pd.to_numeric(df["Heat rate (Btu per kWh)"], errors="coerce")
    df["Capacity (MW)"] = pd.to_numeric(df["Capacity (MW)"], errors="coerce")
    df["Start year"] = pd.to_numeric(df["Start year"], errors="coerce")

    needed_cols = ["Capacity (MW)", "Heat rate (Btu per kWh)", "emission_factor_kg_per_TJ", "Start year"]
    df = df.dropna(subset=needed_cols)

    generation_kwh = df["Capacity (MW)"] * 1000 * HOURS_PER_YEAR * CAPACITY_FACTOR
    total_btu = generation_kwh * df["Heat rate (Btu per kWh)"]
    total_tj = total_btu * BTU_TO_TJ
    emissions_kg = total_tj * df["emission_factor_kg_per_TJ"]
    df["annual_emissions_mt"] = emissions_kg / 1e9  # million tonnes

    return df


def forecast_emissions(
    units_df: pd.DataFrame,
    start_year: int = FORECAST_START_YEAR,
    end_year: int = FORECAST_END_YEAR,
    lifetime_years: int = DEFAULT_LIFETIME_YEARS,
) -> pd.DataFrame:
    """
    Build a year-by-year emissions projection per unit respecting a fixed lifetime.
    """
    rows: List[Dict[str, object]] = []
    for _, row in units_df.iterrows():
        start = int(row["Start year"])
        last_operating_year = start + lifetime_years - 1
        annual_emission = float(row["annual_emissions_mt"])
        for year in range(start_year, end_year + 1):
            if start <= year <= last_operating_year:
                rows.append(
                    {
                        "GEM unit/phase ID": row["GEM unit/phase ID"],
                        "Country/Area": row["Country/Area"],
                        "year": year,
                        "annual_emissions_mt": annual_emission,
                    }
                )
    return pd.DataFrame(rows)


def projected_emissions_by_country(forecast_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Aggregate forecast emissions by country for a given year.
    """
    return (
        forecast_df[forecast_df["year"] == year]
        .groupby("Country/Area", as_index=False)["annual_emissions_mt"]
        .sum()
        .sort_values("annual_emissions_mt", ascending=False)
    )


if __name__ == "__main__":
    units = compute_annual_emissions(get_clean_data())
    forecast = forecast_emissions(units)
    top_countries_2030 = projected_emissions_by_country(forecast, 2030).head(5)
    print("Top 5 countries by projected emissions in 2030 (MtCO2):")
    print(top_countries_2030)
