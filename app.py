from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pycountry

# Avoid silent downcasting warnings in pandas future versions
pd.set_option("future.no_silent_downcasting", True)

from data_loader import get_clean_data
from emissions_engine import compute_annual_emissions, forecast_emissions
from ml_models import score_current_fleet, train_early_closure_model
from risk_analyzer import (
    build_paris_gap_timeseries,
    company_gap_status,
    compute_financial_exposure,
    risk_adjust_forecast,
)

CARBON_PRICES = [50, 100, 150]


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return get_clean_data()


@st.cache_resource(show_spinner=False)
def load_models(clean_df: pd.DataFrame):
    artifacts = train_early_closure_model(clean_df)
    scored = score_current_fleet(clean_df, artifacts.pipeline)
    emissions_units = compute_annual_emissions(clean_df)
    forecast = forecast_emissions(emissions_units)
    risk_adjusted = risk_adjust_forecast(forecast, scored)
    return {
        "artifacts": artifacts,
        "scored": scored,
        "forecast": forecast,
        "risk_adjusted": risk_adjusted,
    }


def filter_data(clean_df: pd.DataFrame, forecast: pd.DataFrame, region_sel, country_sel):
    mask_units = pd.Series(True, index=clean_df.index)
    if region_sel:
        mask_units &= clean_df["Region"].isin(region_sel)
    if country_sel:
        mask_units &= clean_df["Country/Area"].isin(country_sel)
    filtered_units = clean_df[mask_units]

    mask_forecast = pd.Series(True, index=forecast.index)
    if region_sel:
        mask_forecast &= forecast["GEM unit/phase ID"].isin(filtered_units["GEM unit/phase ID"])
    if country_sel:
        mask_forecast &= forecast["Country/Area"].isin(country_sel)
    filtered_forecast = forecast[mask_forecast]
    return filtered_units, filtered_forecast


def country_to_iso3(name: str) -> str | None:
    if not isinstance(name, str) or not name:
        return None
    try:
        return pycountry.countries.search_fuzzy(name)[0].alpha_3
    except Exception:
        return None


def kpi_cards(col_container, plants_count, emissions_2030, avg_var_pct, ontrack_pct):
    col1, col2, col3, col4 = col_container
    col1.metric("Plants analyzed", f"{plants_count:,}")
    col2.metric("Projected emissions 2030 (MtCO2)", f"{emissions_2030:,.1f}")
    col3.metric("Avg Climate VaR (%)", f"{avg_var_pct:,.2f}%")
    col4.metric("On-track share (%)", f"{ontrack_pct:,.1f}")


def main():
    st.set_page_config(
        page_title="Coal Transition Risk Assessment",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Coal Transition Risk Assessment — Investor Dashboard")
    st.caption("Modern, investor-ready view with carbon pricing, risk, and Paris alignment insights.")

    clean_df = load_data()
    data_bundle = load_models(clean_df)
    scored = data_bundle["scored"]
    forecast = data_bundle["forecast"]
    risk_adjusted = data_bundle["risk_adjusted"]
    artifacts = data_bundle["artifacts"]

    # Sidebar filters
    # Sidebar branding (text fallback to avoid broken images)
    st.sidebar.markdown("### Coal Transition")
    st.sidebar.header("Filters")
    price = st.sidebar.selectbox("Carbon price ($/tCO2)", CARBON_PRICES, index=2)
    regions = sorted(clean_df["Region"].dropna().unique())
    region_sel = st.sidebar.multiselect("Region", regions, default=None)
    countries = sorted(clean_df["Country/Area"].dropna().unique())
    country_sel = st.sidebar.multiselect("Country/Area", countries, default=None)

    # Apply filters
    filtered_units, filtered_forecast = filter_data(clean_df, forecast, region_sel, country_sel)
    risk_adjusted_filtered = risk_adjusted[
        risk_adjusted["GEM unit/phase ID"].isin(filtered_units["GEM unit/phase ID"])
    ]

    # Financial exposure for selected scenario
    exposure_df = compute_financial_exposure(
        filtered_forecast, filtered_units, scored, price=price
    )

    # Paris alignment
    gap_df = build_paris_gap_timeseries(risk_adjusted_filtered)
    company_gaps = company_gap_status(risk_adjusted_filtered, filtered_units)

    # KPIs
    emissions_2030 = gap_df.loc[gap_df["year"] == 2030, "expected_emissions_mt"].sum()
    avg_var_pct = exposure_df["climate_var_pct"].dropna().mean() if "climate_var_pct" in exposure_df else 0
    if len(company_gaps) > 0:
        ontrack_pct = 100 * (company_gaps["status"] == "On-track").mean()
    else:
        ontrack_pct = 0.0

    st.subheader("Overview")
    kpi_cards(
        st.columns(4),
        plants_count=len(filtered_units),
        emissions_2030=emissions_2030,
        avg_var_pct=avg_var_pct if pd.notna(avg_var_pct) else 0.0,
        ontrack_pct=ontrack_pct,
    )

    st.caption(
        "KPIs reflect current filters and selected carbon price scenario. "
        "Climate VaR uses capacity (MW) multiplied by a $1M/MW asset-value proxy due to missing financials."
    )

    # Predictive Analytics
    st.subheader("Predictive Analytics — Early Closure Risk")
    col_perf, col_plot = st.columns([1, 2])
    with col_perf:
        st.markdown("**Model performance**")
        st.metric("Accuracy", f"{artifacts.accuracy:.3f}")
        roc = artifacts.roc_auc
        st.metric("ROC-AUC", f"{roc:.3f}" if roc is not None else "n/a")
    with col_plot:
        st.markdown("**Top 5 feature importances (Random Forest)**")
        fi = artifacts.feature_importances
        if fi is not None and not fi.empty:
            fi_top = fi.head(5)
            fig_fi = px.bar(
                fi_top.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Blues",
            )
            fig_fi.update_layout(
                xaxis_title="Importance",
                yaxis_title="Feature",
                showlegend=False,
                margin=dict(l=0, r=10, t=10, b=0),
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importances unavailable.")

    st.markdown("**Top 10 highest early-closure risk units**")
    high_risk = scored.sort_values("early_closure_risk", ascending=False).head(10)
    cols_to_show = [
        "Plant name",
        "Unit name",
        "Country/Area",
        "early_closure_risk",
        "natural_retirement_year",
    ]
    st.dataframe(
        high_risk[cols_to_show].rename(
            columns={
                "Plant name": "Plant",
                "Unit name": "Unit",
                "Country/Area": "Country",
                "early_closure_risk": "Risk",
                "natural_retirement_year": "Natural retirement",
            }
        ),
        width="stretch",
        height=300,
    )

    # Financial Exposure
    st.subheader("Financial Exposure & Climate VaR")
    col_bubble, col_rank = st.columns([2, 1])
    with col_bubble:
        st.markdown("**Exposure bubble chart (by company)**")
        if not exposure_df.empty and "capacity_mw" in exposure_df and "climate_var_pct" in exposure_df:
            top_n = st.slider("Top N companies", min_value=5, max_value=50, value=20, step=5)
            ranked = exposure_df.sort_values("exposure_usd", ascending=False).head(top_n)
            fig_bubble = px.scatter(
                ranked,
                x="capacity_mw",
                y="climate_var_pct",
                size="expected_emissions_mt",
                color="expected_emissions_mt",
                hover_data=["Parent", "exposure_usd", "expected_emissions_mt", "capacity_mw"],
                labels={
                    "capacity_mw": "Capacity (MW)",
                    "climate_var_pct": "Climate VaR (%)",
                    "expected_emissions_mt": "Emissions (MtCO2)",
                },
                color_continuous_scale="Blues",
            )
            fig_bubble.update_layout(
                xaxis_title="Capacity (MW)",
                yaxis_title="Climate VaR (%)",
                legend_title="Emissions (MtCO2)",
                margin=dict(l=0, r=0, t=10, b=10),
            )
            st.plotly_chart(fig_bubble, use_container_width=True)
        else:
            st.info("No exposure data for the selected filters.")

    with col_rank:
        st.markdown("**Top financial exposures**")
        table_cols = ["Parent", "exposure_usd", "expected_emissions_mt", "capacity_mw", "climate_var_pct"]
        if not exposure_df.empty:
            top_expo = exposure_df.sort_values("exposure_usd", ascending=False).head(10)
            st.dataframe(
                top_expo[[c for c in table_cols if c in top_expo.columns]].rename(
                    columns={
                        "Parent": "Company",
                        "exposure_usd": "Exposure (USD)",
                        "expected_emissions_mt": "Emissions (MtCO2)",
                        "capacity_mw": "Capacity (MW)",
                        "climate_var_pct": "Climate VaR (%)",
                    }
                ),
                width="stretch",
                height=360,
            )
        else:
            st.info("No exposure records for the current filter.")
    st.caption(
        "Climate VaR (%) = (carbon cost / capacity) × 100, using capacity as a proxy for enterprise value due to financial data gaps."
    )

    # ESG & Paris Alignment
    st.subheader("ESG & Paris Alignment")
    col_gap, col_onoff = st.columns([2, 1])
    with col_gap:
        st.markdown("**Emissions trajectory vs Paris target (−4.2%/yr)**")
        if not gap_df.empty and gap_df["paris_target_mt"].notna().any():
            fig_gap = go.Figure()
            fig_gap.add_trace(
                go.Scatter(
                    x=gap_df["year"],
                    y=gap_df["expected_emissions_mt"],
                    mode="lines+markers",
                    name="Expected",
                    line=dict(color="#2f5caa"),
                )
            )
            fig_gap.add_trace(
                go.Scatter(
                    x=gap_df["year"],
                    y=gap_df["paris_target_mt"],
                    mode="lines+markers",
                    name="Paris-aligned target",
                    line=dict(color="#e74c3c", dash="dash"),
                )
            )
            fig_gap.update_layout(
                xaxis_title="Year",
                yaxis_title="Emissions (MtCO2)",
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_gap, use_container_width=True)
        else:
            st.info("No emissions trajectory available for the current filter.")

    with col_onoff:
        st.markdown("**On/Off-track status (2030 gap)**")
        if not company_gaps.empty:
            top_on = (
                company_gaps[company_gaps["status"] == "On-track"]
                .sort_values("gap_mt")
                .head(3)
            )
            top_off = (
                company_gaps[company_gaps["status"] == "Off-track"]
                .sort_values("gap_mt", ascending=False)
                .head(3)
            )
            st.write("On-track (best performers)")
            st.dataframe(
                top_on[["Parent", "gap_mt", "paris_target_mt", "projected_target_year_mt"]],
                width="stretch",
                height=150,
            )
            st.write("Off-track (worst performers)")
            st.dataframe(
                top_off[["Parent", "gap_mt", "paris_target_mt", "projected_target_year_mt"]],
                width="stretch",
                height=150,
            )
        else:
            st.info("No companies available for status display.")

    st.markdown("**Regional concentration of Off-track companies**")
    offtrack = company_gaps[company_gaps["status"] == "Off-track"]
    if not offtrack.empty:
        # Aggregate by country for the map and convert to ISO-3 to avoid future deprecation
        country_gaps = (
            offtrack.merge(
                filtered_units[["Parent", "Country/Area"]].drop_duplicates(),
                on="Parent",
                how="left",
            )
            .groupby("Country/Area", as_index=False)["gap_mt"]
            .sum()
        )
        country_gaps["iso3"] = country_gaps["Country/Area"].apply(country_to_iso3)
        country_gaps = country_gaps.dropna(subset=["iso3"])
        if not country_gaps.empty:
            fig_map = px.choropleth(
                country_gaps,
                locations="iso3",
                locationmode="ISO-3",
                color="gap_mt",
                color_continuous_scale="Reds",
                title="Off-track gap by country",
            )
            fig_map.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No country data available for off-track companies.")
    else:
        st.info("No off-track companies to map.")

    # Methodology & Uncertainty
    st.markdown("---")
    st.subheader("Methodology & Uncertainty")
    st.markdown(
        """
        - Capacity factor fixed at 55%; lifetime 35y (BAU) for emissions forecast.
        - Early-closure probabilities from Random Forest (accuracy ~0.97, ROC-AUC ~0.994).
        - Paris alignment path: -4.2%/yr from 2024 (SBTi 1.5°C proxy).
        - Climate VaR proxy: carbon cost ÷ capacity (MW) due to missing revenue/EBITDA; integrating market data (Bloomberg/Reuters) would yield true Margin at Risk.
        - Key uncertainties: utilization rates, heat rates, emission factors, policy shocks (carbon prices), financing/ESG pressures.
        """
    )

    # Data Explorer
    st.subheader("Data Explorer")
    search_query = st.text_input("Search plant or company", "")
    # Prepare merged view with risk and alignment status
    merged_view = scored.merge(
        company_gaps[["Parent", "status"]] if "status" in company_gaps else company_gaps,
        on="Parent",
        how="left",
        suffixes=("", "_status"),
    )
    rename_map = {
        "status": "Paris alignment",
        "early_closure_risk": "Risk probability",
        "climate_var_pct": "Climate VaR (%)",
    }
    merged_view = merged_view.rename(columns={k: v for k, v in rename_map.items() if k in merged_view.columns})
    if "Climate VaR (%)" not in merged_view.columns:
        merged_view["Climate VaR (%)"] = pd.NA
    # Add Climate VaR from exposure if available
    if not exposure_df.empty and "climate_var_pct" in exposure_df:
        var_lookup = exposure_df[["Parent", "climate_var_pct"]].drop_duplicates()
        merged_view = merged_view.merge(var_lookup, on="Parent", how="left", suffixes=("", "_va"))
        merged_view["Climate VaR (%)"] = merged_view["Climate VaR (%)"].fillna(merged_view.get("climate_var_pct"))
        merged_view = merged_view.drop(columns=[c for c in ["climate_var_pct"] if c in merged_view.columns])

    if search_query:
        mask = (
            merged_view["Plant name"].str.contains(search_query, case=False, na=False)
            | merged_view["Parent"].str.contains(search_query, case=False, na=False)
        )
        merged_view = merged_view[mask]

    critical_cols = [
        "Plant name",
        "Unit name",
        "Country/Area",
        "Parent",
        "Risk probability",
        "natural_retirement_year",
        "Capacity (MW)",
        "Climate VaR (%)",
        "Paris alignment",
    ]
    # Bring critical columns to front if present
    ordered_cols = [c for c in critical_cols if c in merged_view.columns] + [
        c for c in merged_view.columns if c not in critical_cols
    ]
    merged_view = merged_view[ordered_cols]

    st.dataframe(merged_view, width="stretch", height=420)

    csv_data = merged_view.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Data as CSV",
        data=csv_data,
        file_name="data_explorer.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
