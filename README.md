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

Our chosen dataset from the selection above is the all-states file that includes AGI breakdowns (`data/22zpallagi.csv`). This one made the most sense to use because we need income bracket information to calculate generosity as a proportion of income. We think that knowing how generous someone is also requires knowing how much they made.