from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
import xlsxwriter  # type: ignore

from data_loader import get_clean_data
from emissions_engine import compute_annual_emissions, forecast_emissions
from ml_models import (
    train_early_closure_model,
    score_current_fleet,
    top_at_risk_plants,
)
from risk_analyzer import (
    compute_financial_exposure,
    risk_adjust_forecast,
    build_paris_gap_timeseries,
    company_gap_status,
)
from visuals import plot_financial_exposure, plot_emissions_gap

CARBON_PRICES = [50, 100, 150]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
REPORT_PATH = OUTPUT_DIR / "Transition_Risk_Report_2030.xlsx"
FINANCIAL_NOTE = (
    "Data gap: Climate VaR here reflects carbon-cost exposure only. "
    "Next step for a financial institution is to pull actual revenue/EBITDA via a market data API "
    "(e.g., Bloomberg/Reuters) to convert exposure into Margin at Risk."
)


def _clean_parent_names(df: pd.DataFrame, column: str = "Parent") -> pd.DataFrame:
    """
    Standardize parent names for reporting (e.g., replace 'to be determined' with 'Unidentified Owners').
    """
    cleaned = df.copy()
    cleaned[column] = cleaned[column].apply(
        lambda x: "Unidentified Owners" if isinstance(x, str) and "to be determined" in x.lower() else x
    )
    return cleaned


def run_pipeline():
    clean_df = get_clean_data()

    # Train predictive model and score fleet.
    artifacts = train_early_closure_model(clean_df)
    scored = score_current_fleet(clean_df, artifacts.pipeline)
    at_risk = top_at_risk_plants(scored, top_n=5)

    # Emissions forecast and risk adjustment.
    emissions_units = compute_annual_emissions(clean_df)
    forecast = forecast_emissions(emissions_units)
    risk_adjusted = risk_adjust_forecast(forecast, scored)

    # Financial exposures for each price scenario.
    exposures = {}
    for price in CARBON_PRICES:
        exposure_df = compute_financial_exposure(forecast, clean_df, scored, price=price)
        exposure_df.attrs["price"] = price
        exposures[price] = _clean_parent_names(exposure_df)

    # Build emissions gap vs Paris-aligned target.
    gap_df = build_paris_gap_timeseries(risk_adjusted)
    company_gaps = _clean_parent_names(company_gap_status(risk_adjusted, clean_df))

    # Export Excel report with two tabs.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exposure_150_sorted = exposures[150].sort_values("exposure_usd", ascending=False)
    alignment_sorted = company_gaps.sort_values("gap_mt", ascending=False)

    report_path = REPORT_PATH
    try:
        if report_path.exists():
            report_path.unlink()
        with pd.ExcelWriter(report_path, engine="xlsxwriter") as writer:
            # Financial exposure sheet with note on data gap.
            exposure_sheet = "Financial Exposure"
            exposure_150_sorted.to_excel(writer, sheet_name=exposure_sheet, index=False, startrow=2)
            alignment_sorted.to_excel(writer, sheet_name="Paris Alignment", index=False)

            # Conditional formatting for alignment sheet.
            workbook = writer.book
            fx_sheet = writer.sheets[exposure_sheet]
            fx_sheet.write(0, 0, FINANCIAL_NOTE, workbook.add_format({"text_wrap": True}))

            align_sheet = writer.sheets["Paris Alignment"]
            off_format = workbook.add_format({"font_color": "red"})
            on_format = workbook.add_format({"font_color": "green"})
            status_col = alignment_sorted.columns.get_loc("status")
            n_rows = len(alignment_sorted) + 1  # include header row
            status_col_letter = xlsxwriter.utility.xl_col_to_name(status_col)
            status_range = f"{status_col_letter}2:{status_col_letter}{n_rows}"
            align_sheet.conditional_format(
                status_range,
                {"type": "text", "criteria": "containing", "value": "Off-track", "format": off_format},
            )
            align_sheet.conditional_format(
                status_range,
                {"type": "text", "criteria": "containing", "value": "On-track", "format": on_format},
            )
    except PermissionError:
        # If file is locked (e.g., opened in Excel), write to a timestamped fallback.
        fallback = OUTPUT_DIR / f"Transition_Risk_Report_2030_{int(time.time())}.xlsx"
        with pd.ExcelWriter(fallback, engine="xlsxwriter") as writer:
            exposure_150_sorted.to_excel(writer, sheet_name="Financial Exposure", index=False)
            alignment_sorted.to_excel(writer, sheet_name="Paris Alignment", index=False)
        report_path = fallback

    # Create visuals for the highest price scenario.
    exposure_chart = plot_financial_exposure(
        exposures[150],
        OUTPUT_DIR / "financial_exposure_150.png",
        top_n=10,
    )
    gap_chart = plot_emissions_gap(gap_df, OUTPUT_DIR / "emissions_gap_2024_2030.png")

    return {
        "accuracy": artifacts.accuracy,
        "roc_auc": artifacts.roc_auc,
        "feature_importances": artifacts.feature_importances,
        "at_risk": at_risk,
        "exposures": exposures,
        "gap": gap_df,
        "company_gaps": company_gaps,
        "charts": {"exposure": exposure_chart, "gap": gap_chart},
        "company_alignment_report": report_path,
    }


if __name__ == "__main__":
    results = run_pipeline()
    print(f"Model accuracy: {results['accuracy']:.3f}")
    if results["roc_auc"] is not None:
        print(f"Model ROC-AUC: {results['roc_auc']:.3f}")
    else:
        print("Model ROC-AUC: not available.")
    if results["feature_importances"] is not None:
        print("Top 5 feature importances:")
        print(
            results["feature_importances"]
            .head(5)
            .to_string(index=False, formatters={"importance": "{:.4f}".format})
        )
    else:
        print("Feature importances unavailable.")
    print("Top 5 at-risk plants (probability of early closure):")
    print(results["at_risk"].to_string(index=False))
    print("\nTop 10 exposures at $150/tCO2 (USD):")
    top10 = results["exposures"][150].head(10)
    print(top10.to_string(index=False, formatters={"exposure_usd": "{:,.0f}".format}))
    print("\nTop 5 Climate VaR (proxy, % = cost/carbon ÷ capacity ×100) at $150/tCO2:")
    var_top = (
        results["exposures"][150]
        .sort_values("climate_var_pct", ascending=False)
        .head(5)
    )
    print(
        var_top[["Parent", "climate_var_pct", "capacity_mw", "exposure_usd"]]
        .to_string(
            index=False,
            formatters={
                "climate_var_pct": "{:,.2f}".format,
                "capacity_mw": "{:,.0f}".format,
                "exposure_usd": "{:,.0f}".format,
            },
        )
    )
    print(f"\nCharts saved to: {results['charts']['exposure']} and {results['charts']['gap']}")
    print(f"Excel report saved to: {results['company_alignment_report']}")
    print("\nTop 5 off-track vs Paris target (2030 gap):")
    off_track = results["company_gaps"][results["company_gaps"]["status"] == "Off-track"].head(5)
    if off_track.empty:
        print("All companies on-track for Paris target.")
    else:
        print(
            off_track[["Parent", "gap_mt", "paris_target_mt", "projected_target_year_mt"]]
            .to_string(index=False, formatters={"gap_mt": "{:,.2f}".format, "paris_target_mt": "{:,.2f}".format, "projected_target_year_mt": "{:,.2f}".format})
        )
    print("\nTop 5 on-track vs Paris target (most over-performing):")
    on_track = (
        results["company_gaps"][results["company_gaps"]["status"] == "On-track"]
        .sort_values("gap_mt")  # most negative first
        .head(5)
    )
    if on_track.empty:
        print("No companies are on-track for the Paris target.")
    else:
        print(
            on_track[["Parent", "gap_mt", "paris_target_mt", "projected_target_year_mt"]]
            .to_string(index=False, formatters={"gap_mt": "{:,.2f}".format, "paris_target_mt": "{:,.2f}".format, "projected_target_year_mt": "{:,.2f}".format})
        )
