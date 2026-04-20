# Generosity Intelligence Dashboard

**Role:** Fundraising Strategists for a National Charity
**Goal:** Identify ZIP codes where residents give the highest share of income to charity — finding *genuinely generous* communities, not just wealthy ones.

**Core Metrics (calculated from IRS SOI data):**
- **Generosity Index** = `A19700 ÷ A00100` (Charitable Contributions / Adjusted Gross Income (AGI))
- **Participation Rate** = `N19700 ÷ N1` (Itemizing Donors / Total Returns)

---
## App Deployment URL
[Open the Streamlit App](https://philanthropy-advisors-project.streamlit.app/)

## Local Setup Instructions
```bash
git clone https://github.com/evelyn-hosana/team7-philanthropy-advisor-project.git
cd team7-philanthropy-advisor-project
uv sync
uv run streamlit run app.py
````

## Setup

This project uses [`uv`](https://astral.sh/uv), which manages dependencies and creates a `.venv` virtual environment automatically.

### 1. Install `uv`

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Create virtual environment and install dependencies
```bash
uv sync
```
This creates `.venv/` and installs all locked dependencies. All subsequent commands run inside this venv via `uv run`.

---

## Data

If `data/zpallagi_cleaned.csv` does not exist, you must download the original IRS datasets and generate it.

### Download original data

Download the **ZIP Code Data (with AGI)** files from the IRS SOI page:
https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi

**CSV years (2011–2022):** Place files directly in `data/original/` using the naming convention `YYzpallagi.csv` (e.g., `22zpallagi.csv`).

**Legacy XLS years (2007, 2009, 2010):** Place per-state XLS files in:
```
data/original/convertXLSData/2007zipcode/
data/original/convertXLSData/2009zipcode/
data/original/convertXLSData/2010zipcode/
```
Then run the conversion script to produce CSVs in `data/original/`:
```bash
uv run python data_conversion.py
```

### Generate cleaned dataset
```bash
uv run python data_processing.py
```
Outputs `data/zpallagi_cleaned.csv`.

---

## Run the Dashboard
```bash
uv run streamlit run app.py
```


---

## Dashboard Features

- **Global filters:** Year(s) and state(s) with TCJA cross-era warning (2017–2018 boundary)
- **Top N ZIP codes** ranked by Generosity Index or Participation Rate
- **Scatter plot** with quadrant overlays and user-defined thresholds
- **US state map** colored by generosity or participation
- **Heatmap** of AGI vs. charitable contributions distribution
- **Trend charts** by year and state

**Data caveats shown in-app:** Itemization bias (only itemizers report contributions), the 2017 TCJA structural break, and nominal (non-inflation-adjusted) dollar values.
