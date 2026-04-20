# Generosity Intelligence Dashboard

## Project Description

A Streamlit dashboard built for a national charity planning a fundraising gala. It analyzes IRS ZIP code tax data to identify communities where residents give the highest share of their income to charity — the goal being to find genuinely generous ZIP codes, not just wealthy ones.

The two core metrics are:
- Generosity Index = `A19700 / A00100` (Charitable Contributions / AGI)
- Participation Rate = `N19700 / N1` (Itemizing Donors / Total Returns)

## App Deployment URL

https://philanthropy-advisors-project.streamlit.app/

## Local Setup Instructions

```bash
git clone https://github.com/evelyn-hosana/team7-philanthropy-advisor-project.git
cd team7-philanthropy-advisor-project
uv sync
uv run streamlit run app.py
```

## Setup

This project uses [`uv`](https://astral.sh/uv), which manages dependencies and creates a `.venv` virtual environment automatically.

### 1. Install `uv`

macOS/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Windows:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Create virtual environment and install dependencies
```bash
uv sync
```
This creates `.venv/` and installs all locked dependencies. All subsequent commands run inside this venv via `uv run`.

## Data

If `data/zpallagi_cleaned.csv` does not exist, you must download the original IRS datasets (from years 2007, 2009-2011, 2013-2014, 2016-2020, and 2022) and generate it.

### Download original data (optional, unless cleaned dataset doesn't exist)

Download the ZIP Code Data (with AGI) files from the IRS SOI page:
https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi

CSV years (2011–2022): Place files directly in `data/original/` using the naming convention `YYzpallagi.csv` (e.g., `22zpallagi.csv`).

Legacy XLS years (2007, 2009, 2010): Place per-state XLS files in:
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

## Run the Dashboard
```bash
uv run streamlit run app.py
```

## Dashboard Features

- Global filters: year(s) and state(s), with a warning when the selected range crosses the 2017 TCJA boundary
- Top N ZIP codes ranked by Generosity Index or Participation Rate
- Scatter plot with quadrant overlays and user-defined thresholds
- US state map colored by generosity or participation
- Trend charts by year and state
- AI advisor chat and per-ZIP fundraising brief generator

Note: the app surfaces data caveats inline — itemization bias, the 2017 TCJA structural break, and the fact that dollar values are nominal.
