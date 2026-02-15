# Team 7: The Philanthropy Advisors
ITCS 5122 - Visual Analytics

## What This Project Is About
For this project, team 7 is playing the role of fundraising strategists for a national charity that's planning a gala. The problem is we don't want to invite wealthy people overall but instead people who are actually generous. The main goal is to figure out which ZIP codes have the highest "Generosity Index" which will demonstrate which residents give the largest share of their income to charity.

## The Dataset
We're using IRS ZIP Code tax data from 2022, pulled from the official IRS SOI (Statistics of Income) page:

https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi

The download from the IRS site comes as a ZIP file that includes:
- Individual files for each state
- An all-states file **with** AGI breakdowns
- An all-states file **without** AGI breakdowns
- A Users Guide and Record Layouts doc

Our chosen dataset from the selection above is the all-states file **without** AGI breakdowns (`data/22zpallnoagi.csv`). This one made the most sense to use because it provides pre-aggregated totals for every ZIP code, providing national-level analysis. Since our Generosity Index is calculated from the summary totals (total charity donations vs. total income per ZIP), we don't need the per-income-bracket breakdowns that the AGI version offers. The noagi file is leaner and more efficient for our use case.

## Program Usage
1. Ensure you have Python installed with the following packages: `pandas`, `matplotlib`, `seaborn`
2. Open `gala_data_exploration.ipynb` in Jupyter Notebook (or any compatible environment like VS Code)
3. Run all cells from top to bottom

The notebook will:
- Load the IRS ZIP code tax data from `data/original/22zpallnoagi.csv`
- Clean and filter the data to neighborhood-level summary rows
- Calculate a **Generosity Index** and **Participation Rate** for each ZIP code
- Export the filtered results to `data/updated_gala_list.csv`
- Generate two visualizations saved to the `images/` folder:
  - `generosity_rankings.png` — Top 10 most generous ZIP codes
  - `hidden_gems_map.png` — Scatter plot of generosity vs. income to find high-generosity in moderate-income neighborhoods