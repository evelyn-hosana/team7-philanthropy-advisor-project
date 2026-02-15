import pandas as pd

# =================================================================
# MODULE 1: STRATEGY ENGINE
# TASK: Data Processing & Strategic Filtering
# =================================================================

def process_philanthropy_data(file_path):
    # 1. LOAD THE MASTER DATA
    # We use the 'noagi' file because it contains pre-calculated 
    # totals for every ZIP code, making it the fastest for national analysis.
    df = pd.read_csv(file_path)

    # 2. DATA SANITIZATION (The 'Cleaning' Step)
    # ZIP codes are often read as numbers, deleting leading zeros (e.g., 08701 -> 8701).
    # .zfill(5) ensures we have a valid 5-digit string for all geographic regions.
    df['ZIPCODE'] = df['ZIPCODE'].astype(str).str.zfill(5)

    # agi_stub 0 represents the SUMMARY row for the entire ZIP code.
    # We remove '00000' (State totals) and '99999' (Other) to focus on neighborhoods.
    neighborhoods = df[(df['agi_stub'] == 0) & (~df['ZIPCODE'].isin(['00000', '99999']))].copy()

    # 3. CALCULATE PROPRIETARY METRICS (The 'Intelligence' Step)
    # Generosity Index (GI): A19700 (Charity $) / A00100 (Total Income $)
    # This measures the neighborhood's "sacrifice ratio."
    neighborhoods['generosity_index'] = neighborhoods['A19700'] / neighborhoods['A00100']

    # Participation Rate (PR): N19700 (Count of Donors) / N1 (Total Households)
    # This measures how ingrained giving is in the community culture.
    neighborhoods['participation_rate'] = neighborhoods['N19700'] / neighborhoods['N1']

    # 4. THE RELIABILITY FILTER (The 'Noise' Step)
    # We ignore ZIP codes with fewer than 500 households.
    # This ensures our strategy is based on community behavior, not one outlier.
    filtered_targets = neighborhoods[neighborhoods['N1'] >= 500].copy()

    # 5. SORT & EXPORT
    # We sort by GI to prioritize the neighborhoods with the "biggest hearts."
    final_list = filtered_targets.sort_values(by='generosity_index', ascending=False)
    
    # Export for use in email marketing or mailing software
    final_list.to_csv('../data/updated_gala_list.csv', index=False)

    print("SUCCESS: Updated list generated as 'updated_gala_list.csv'.")
    return final_list

if __name__ == "__main__":
    # Ensure the file name matches your local data file
    data = process_philanthropy_data('../data/original/22zpallnoagi.csv')