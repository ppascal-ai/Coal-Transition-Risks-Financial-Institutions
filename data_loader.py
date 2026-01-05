from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# Source data directory
DATA_DIR = Path(__file__).resolve().parent / "data"
REFERENCE_YEAR = 2025  # Dataset snapshot year used to infer missing start years.

# Map known status spellings to a normalized, lower-case value.
STATUS_NORMALIZATION = {
    "announced": "announced",
    "cancelled": "cancelled",
    "pre-permit": "pre-permit",
    "pre permit": "pre-permit",
    "permitted": "permitted",
    "shelved": "shelved",
    "construction": "construction",
    "mothballed": "mothballed",
    "operating": "operating",
    "retired": "retired",
}


def load_units(path: Path | str = DATA_DIR / "units.csv") -> pd.DataFrame:
    """
    Read the units dataset with correct delimiters and decimal handling.
    """
    return pd.read_csv(path, sep=";", decimal=",", low_memory=False)


def load_co2_parameters(path: Path | str = DATA_DIR / "co2_parameters.csv") -> pd.DataFrame:
    """
    Read CO2 emission factors keyed by coal type.
    """
    df = pd.read_csv(
        path,
        sep=";",
        skiprows=3,  # Skip documentation rows at the top of the file.
        header=0,
        usecols=[0, 1],
        names=["coal_type", "emission_factor_kg_per_TJ"],
    )
    df["coal_type"] = df["coal_type"].astype(str).str.strip().str.lower()
    df["emission_factor_kg_per_TJ"] = (
        df["emission_factor_kg_per_TJ"]
        .astype(str)
        .str.replace(" ", "", regex=False)
    )
    df["emission_factor_kg_per_TJ"] = pd.to_numeric(
        df["emission_factor_kg_per_TJ"], errors="coerce"
    )
    return df.dropna(subset=["coal_type", "emission_factor_kg_per_TJ"])


def load_about_metadata(path: Path | str = DATA_DIR / "about.csv") -> pd.DataFrame:
    """
    Load project metadata supplied with the dataset.
    """
    return pd.read_csv(path, sep=";", header=None)


def _standardize_status(status_series: pd.Series) -> pd.Series:
    """
    Normalize status values: trim, lowercase, and map common variants.
    """
    cleaned = status_series.fillna("").astype(str).str.strip().str.lower()
    return cleaned.map(lambda s: STATUS_NORMALIZATION.get(s, s))


def _impute_start_year(df: pd.DataFrame) -> pd.Series:
    """
    Fill missing start years using plant age and the dataset reference year.
    """
    start_year = pd.to_numeric(df.get("Start year"), errors="coerce")
    plant_age = pd.to_numeric(df.get("Plant age (years)"), errors="coerce")
    inferred_start = (REFERENCE_YEAR - plant_age).where(
        plant_age.notna(), pd.NA
    )
    start_year = start_year.fillna(inferred_start)
    return start_year.round().astype("Int64")


def clean_units(units_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to the units dataset.
    """
    df = units_df.copy()
    df["Status"] = _standardize_status(df["Status"])
    df["Start year"] = _impute_start_year(df)
    return df


def get_clean_data() -> pd.DataFrame:
    """
    Load all source files and return the cleaned units DataFrame.
    """
    units_df = load_units()
    # Load auxiliary sources for completeness; they are returned by dedicated helpers.
    _ = load_co2_parameters()
    _ = load_about_metadata()
    return clean_units(units_df)


def load_all_sources() -> Dict[str, pd.DataFrame]:
    """
    Convenience helper to load all raw datasets at once.
    """
    return {
        "units": load_units(),
        "co2_parameters": load_co2_parameters(),
        "about": load_about_metadata(),
    }
