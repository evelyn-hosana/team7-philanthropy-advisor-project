import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =================================================================
# MODULE 2: VISUALIZATION SUITE
# TASK: Strategic Insights & Stakeholder Visuals
# =================================================================

# We use the processed list from Module 1
df = pd.read_csv('../data/updated_gala_list.csv')

# Configure the look and feel
sns.set_theme(style="whitegrid")

# --- CHART 1: THE GENEROSITY LEADERS (Bar Chart) ---
# Purpose: To show the "Whales"—the absolute best ZIPs to target for the Gala.
top_10 = df.sort_values(by='generosity_index', ascending=False).head(10)
top_10['Label'] = top_10['STATE'] + " " + top_10['ZIPCODE'].astype(str)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_10, x='generosity_index', y='Label', palette='viridis')
plt.title('Top 10 Generous Donors in the US', fontsize=14)
plt.xlabel('Generosity Index (Donations as % of Income)', fontsize=12)
plt.tight_layout()
plt.savefig('../images/generosity_rankings.png') # Progress evidence for your slide

# --- CHART 2: THE MARKET MAP (Scatter Plot) ---
# Purpose: To find "Hidden Gems" (High generosity, Moderate income)
plt.figure(figsize=(10, 6))
# avg_income_k = Total Income (A00100) / Total Households (N1)
df['avg_income_k'] = df['A00100'] / df['N1']

# We zoom in on areas earning < $500k to see the main cluster clearly
sns.scatterplot(data=df[df['avg_income_k'] < 500], 
                x='avg_income_k', y='generosity_index', 
                alpha=0.4, color='teal')

plt.title('"Hidden Gem" Donors', fontsize=14)
plt.xlabel('Average Household Income ($1,000s)', fontsize=12)
plt.ylabel('Generosity Index', fontsize=12)
plt.tight_layout()
plt.savefig('../images/hidden_gems_map.png') # Progress evidence for your slide

print("Visual saved: 'images/generosity_rankings.png'.")
print("Visual saved: 'images/hidden_gems_map.png'.")