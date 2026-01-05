# Coal Transition Risk Assessment Tool

Modular Python toolkit for assessing coal transition risks for financial institutions. It loads GEM unit-level data, cleans it, models early-closure risk, forecasts emissions, and quantifies financial exposure under carbon price scenarios. Plots highlight exposure concentration and the emissions gap through 2030.

## Project Structure
- `data/` — Input CSVs (`units.csv`, `co2_parameters.csv`, `about.csv`).
- `data_loader.py` — Data ingestion/cleaning (units, CO2 parameters, metadata).
- `emissions_engine.py` — Annual emission calculation and BAU forecast to 2030.
- `ml_models.py` — Random Forest classifier for early-closure risk; scoring current fleet.
- `risk_analyzer.py` — Risk-adjusted emissions, Paris-aligned gap, and carbon price exposure by parent company.
- `visuals.py` — Charts for exposure and emissions gap vs Paris target.
- `main.py` — End-to-end orchestration and reporting (CLI).
- `app.py` — Streamlit investor dashboard (interactive UI).
- `outputs/` — Generated figures and Excel report (git-ignored).

## Key Findings (current run)
- Model performance: accuracy 0.969, ROC-AUC 0.994.
- Highest exposure at $150/tCO2: Korea Electric Power Corp (~$9.53B), NTPC Ltd (~$7.29B), Power Finance Corp (~$5.09B).
- Emissions gap: chart `outputs/emissions_gap_2024_2030.png` compares expected, risk-adjusted emissions with the Paris-aligned path (4.2% annual decline from 2024).
- Exposure concentration: chart `outputs/financial_exposure_150.png` shows top 10 parent companies under the $150 scenario.
- Excel report: `outputs/Transition_Risk_Report_2030.xlsx` (a timestamped fallback is created if the file is locked by Excel).

## Model Performance
- Accuracy: 96.9%
- ROC-AUC: 0.994
- Top 5 feature importances (Random Forest):
  - Start year — 44.3%
  - Subregion_Eastern Asia — 11.8%
  - Country/Area_China — 9.4%
  - Capacity — 8.1%
  - Region_Asia — 7.8%
- Note: The dominance of Start year confirms the age effect (early units retire sooner) and suggests no obvious data leakage (most weight on time-in-service rather than any target-like field).

## Climate VaR (proxy)
- Climate VaR (%) is computed as (Carbon cost / total capacity) × 100, using capacity (MW) as a proxy for enterprise value because EBITDA/revenue are not available in the dataset.
- This proxy satisfies the need for a VaR-like metric while documenting the financial data gap; integrating actual financials via a market data API (e.g., Bloomberg/Reuters) would upgrade this to a true Margin at Risk.

## Requirements
- Python 3.10+
- pip packages: `pandas`, `scikit-learn`, `matplotlib`, `xlsxwriter`, `streamlit`, `plotly`, `pycountry`

## How to Run
Install dependencies:
```bash
pip install pandas scikit-learn matplotlib xlsxwriter streamlit plotly pycountry
```

Pipeline (CLI):
```bash
python main.py
```
Outputs:
- Console summary of model metrics, at-risk units, Paris alignment (On/Off-track), exposures, and Climate VaR proxy.
- Charts in `outputs/financial_exposure_150.png` and `outputs/emissions_gap_2024_2030.png`.
- Excel report in `outputs/Transition_Risk_Report_2030.xlsx` (close it before re-running to avoid a timestamped fallback file).

Interactive dashboard (Streamlit):
```bash
streamlit run app.py
```
- Sidebar: carbon price scenario, region/country filters.
- Overview KPIs, Predictive Analytics (performance, feature importances, high-risk units), Financial Exposure & Climate VaR (bubble + ranking), ESG & Paris Alignment (trajectory, On/Off-track, map), Data Explorer with search and CSV download.

## Assumptions
- Capacity factor fixed at 55%; fixed 35-year lifetime for BAU forecasting.
- Early-closure probability scales expected emissions (emissions × (1 - risk)).
- Missing start years inferred from plant age with a 2025 reference year.
- Paris alignment: 4.2% annual reduction from 2024 (SBTi 1.5°C proxy) to compute gaps and On/Off-track status.

## Data Gap (Climate VaR)
- Current tool reports Climate VaR as carbon-cost exposure. Financial linkage (Margin at Risk) requires actual revenue/EBITDA per parent.
- Next step: pull financials via a market data API (e.g., Bloomberg/Reuters) and convert exposure into Margin at Risk. The Excel report documents this gap in the `Financial Exposure` tab header.
