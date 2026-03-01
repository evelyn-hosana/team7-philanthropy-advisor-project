# Generosity Intelligence Dashboard

## The Philanthropy Advisors
**Role:** Fundraising Strategists for a National Charity.

**The Business Problem:** We are planning a major fundraising gala. We need to invite people who are *actually* generous, rather than those who are just rich. 

**The Mandate:** Analyze giving patterns across the United States. Identify which ZIP codes have the highest **"Generosity Index"**, where residents naturally give the largest portion of their income to charity.

## Why This Dataset?
To fulfill the mandate, we chose the **IRS Statistics of Income (SOI) ZIP Code Data (with AGI)** for the 2022 tax year. 

**Why AGI?** 
If we used non AGI datasets (like simple population or wealth estimates), we could only find out *who has the most money* or *who gives the highest raw dollar amount*. This would simply lead us to the wealthiest ZIP codes (like Beverly Hills or Manhattan). 

However, by utilizing the dataset that includes **Adjusted Gross Income (AGI)**, we gain access to two critical variables:
1. `A00100`: Total Adjusted Gross Income
2. `A19700`: Total Charitable Contributions

We mathematically calculate the two core metrics using the following formulas:

* **Generosity Index**: The proportion of a ZIP code's total income that is donated to charity.
  * Formula: `Total Charitable Contributions (A19700) ÷ Total Adjusted Gross Income (A00100)`
* **Participation Rate**: The proportion of tax returns in a ZIP code that include charitable contributions.
  * Formula: `Number of Returns with Charitable Contributions (N19700) ÷ Total Number of Returns (N1)`

This shows us not just how much a community gives, but how widespread that culture of giving is.

***

## Usage & Installation

This project utilizes [uv](https://astral.sh/uv), an extremely fast Python package manager, to ensure identical environments across all team members' machines.

### 1. Install `uv`
If you do not already have `uv` installed on your machine, you must install it first:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Sync Dependencies
To ensure your local environment perfectly matches the exact library versions locked by the team, navigate to the project root and run:
```bash
uv sync
```

### 3. Generate the Cleaned Dataset

The original dataset must be downloaded at https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-2022-zip-code-data-soi (specifically the AGI version) and placed in the `original\` folder. Git cannot handle such a large dataset, so it has been added to `.gitignore`, but the cleaned dataset is included in this repository.

First, ensure you have the original IRS data in `data/original/22zpallagi.csv`, then run the data processing pipeline locally:
```bash
uv run python data_processing.py
```
This script will filter out anomalous ZIP codes, remove placeholders like `00000` and `99999`, aggregate the necessary columns, mathematically calculate the Generosity Index and Participation Rate, and output the final `data/22zpallagi_cleaned.csv` file.

### 4. Run the Dashboard
Once your data is cleaned, launch the interactive Streamlit dashboard:
```bash
uv run streamlit run app.py
```
This will start a local server and open the streamlit app in your default web browser.

***

## Dashboard Features & UI Configuration

This project takes full advantage of Streamlit's native components for an optimized, dependency-free UI experience.

* **Native Dark Mode:** Fully configured via `.streamlit/config.toml` to provide a visually relaxing, true-blue dark theme (`#1a1d24` background). No custom CSS injection or external text toggles are required; the styling is built directly into the engine.
* **Interactive Sidebar Controls:** 
  * Filter states globally via multi-select dropdowns.
  * Adjust chart granularity (bins and Top N) using native `st.slider`.
  * Seamlessly swap metrics using `st.segmented_control` buttons instead of plain text, optimizing for the new color palette.
* **Symmetrically Scaled Color Palettes:** All Data visualizations (Altair maps, heatmaps, and scatter plots) utilize a mathematical `symlog` color scale to dynamically curve the visual gradient and prevent extreme outlier clustering.
* **Data Transparency:** Each chart includes a collapsible expander outlining the specific calculation formulas and definitions for the metrics presented.

***

## Future Ideas & Expansion Opportunities

While this dashboard successfully handles the 2022 dataset, there are several exciting paths for future analysis:

* **Historical & Macroeconomic Comparisons**: The IRS provides this data spanning back to 1998. We could fetch data from major historical inflection points (e.g., 2001 following 9/11, the 2008 Great Recession, or the 2020 pandemic anomalies) to examine how economic stress or national tragedies impact America's Generosity Index. Do people give more when times are tough?
* **Cost of Living Adjustments**: $100,000 in AGI goes much further in rural Ohio than in San Francisco. Cross referencing this dataset with regional Cost of Living Indexes might reveal an even deeper "True Generosity" metric.
* **Bracket Level Analysis**: Currently, we aggregate all 6 AGI income brackets together to get a ZIP code level average. A future dashboard tab could allow users to drill down and see if the *middle class* in a specific ZIP code is more generous than the *upper class* within that same neighborhood.