"""
Anomaly Detection Module for Philanthropy Intelligence Dashboard

Detects irregular giving patterns across ZIP codes:
- Income-Generosity Mismatches (e.g., low income, high generosity from religious communities)
- Wealth Concentration Anomalies (high income but low participation or giving)
- Donor Fatigue Patterns (declining despite rising wealth)
- Hidden Gems (modest income, exceptional generosity)
- Underperforming Affluent Areas (wealth opportunity zones)
- Emerging Generosity Hotspots (rapid growth in giving culture)
- Philanthropic Hubs (natural concentrations: universities, medical centers)

Uses Isolation Forest, statistical analysis, and domain-specific heuristics.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def detect_income_generosity_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect ZIP codes where generosity doesn't match expected income level.
    
    Real-world examples:
    - Utah religious communities: low AGI, high generosity (churches, synagogues)
    - New York philanthropic hubs: high engagement despite varied incomes
    - Southern hospitality ZIP codes: strong giving culture
    """
    df_anom = df.copy()
    df_anom = df_anom.dropna(subset=['A00100', 'generosity_index', 'participation_rate', 'N1'])
    
    if len(df_anom) < 10:
        df_anom['is_anomaly'] = False
        df_anom['anomaly_type'] = ''
        df_anom['anomaly_score'] = 0.0
        df_anom['anomaly_category'] = 'Insufficient Data'
        return df_anom
    
    # Calculate expected generosity based on AGI
    X = df_anom[['A00100']].values
    y = df_anom['generosity_index'].values
    
    reg = LinearRegression()
    reg.fit(X, y)
    expected_generosity = reg.predict(X)
    
    residuals = y - expected_generosity
    residual_std = np.std(residuals)
    
    if residual_std < 1e-6:
        residual_std = 1e-6
    
    standardized_residuals = residuals / residual_std
    df_anom['generosity_zscore'] = standardized_residuals
    df_anom['expected_generosity'] = expected_generosity
    df_anom['generosity_residual'] = residuals
    
    # Percentiles for context
    agi_pct = df_anom['A00100'].quantile([0.25, 0.5, 0.75])
    gi_median = df_anom['generosity_index'].median()
    pr_median = df_anom['participation_rate'].median()
    
    # Mark anomalies with more granular logic
    is_high_giver_anomaly = (standardized_residuals > 1.5) & (df_anom['generosity_index'] > 0.03)
    is_low_giver_anomaly = (standardized_residuals < -1.5) & (df_anom['A00100'] > agi_pct[0.75])
    is_anomaly = is_high_giver_anomaly | is_low_giver_anomaly
    
    df_anom['is_anomaly'] = is_anomaly
    df_anom['anomaly_score'] = np.abs(standardized_residuals)
    
    # Rich categorization with real-world reasoning
    def categorize_income_anomaly(row):
        if pd.isna(row['is_anomaly']) or not row['is_anomaly']:
            return ''
        
        gi = row['generosity_index']
        pr = row['participation_rate']
        agi = row['A00100']
        residual = row['generosity_zscore']
        
        # High generosity, low income -> community driven
        if residual > 1.5 and agi < agi_pct[0.5]:
            if pr > pr_median:
                return 'Community Driven'  # Strong broad participation
            else:
                return ''  # Concentrated Faith Giving removed
        
        # High generosity, moderate income -> hidden gem
        elif residual > 1.5 and (agi_pct[0.25] <= agi <= agi_pct[0.75]):
            return 'Hidden Gem Community'
        
        # High generosity, high income -> philanthropic core
        elif residual > 1.5 and agi >= agi_pct[0.75]:
            if pr > pr_median:
                return 'Philanthropic Powerhouse'
            else:
                return ''  # Ultra-High-Net-Worth Hub removed
        
        # High income but low generosity -> opportunity
        elif residual < -1.5 and agi >= agi_pct[0.75]:
            if pr < (pr_median * 0.7):
                return 'Affluent Underperformers'  # Major opportunity
            else:
                return 'Wealth Concentration Zone'  # Many people, modest giving
        
        else:
            return 'Income-Giving Mismatch'
    
    df_anom['anomaly_category'] = df_anom.apply(categorize_income_anomaly, axis=1)
    df_anom['anomaly_type'] = df_anom.apply(
        lambda row: 'High Generosity' if row['generosity_zscore'] > 1.5 else 
                   ('Low Generosity' if row['generosity_zscore'] < -1.5 else 'Normal'),
        axis=1
    )
    
    return df_anom


def detect_wealth_participation_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify wealth concentration patterns - high income but low participation.
    
    Real-world: indicates either:
    - Few ultra-wealthy individuals (donor concentration risk)
    - Non-itemizers dominating population (standard deduction effect)
    - Transient/commuter population (weekday workers)
    """
    df_wealth = df.copy()
    df_wealth = df_wealth.dropna(subset=['A00100', 'participation_rate', 'N1', 'A19700', 'N19700'])
    
    if len(df_wealth) < 10:
        df_wealth['wealth_anomaly'] = False
        df_wealth['wealth_category'] = ''
        return df_wealth
    
    # Flag: High aggregate giving but low participation
    agi_high = df_wealth['A00100'] > df_wealth['A00100'].quantile(0.75)
    pr_low = df_wealth['participation_rate'] < df_wealth['participation_rate'].quantile(0.40)
    avg_gift = df_wealth['A19700'] / df_wealth['N19700'].clip(lower=1)
    avg_gift_high = avg_gift > df_wealth['A19700'].sum() / df_wealth['N19700'].sum()
    
    df_wealth['wealth_anomaly'] = agi_high & pr_low & avg_gift_high
    
    def categorize_wealth(row):
        if not row['wealth_anomaly']:
            return ''
        
        n_donors = row['N19700']
        total_giving = row['A19700']
        n_filers = row['N1']
        
        if n_donors < (n_filers * 0.05):  # <5% participation
            return 'Concentration Risk - Few Ultra-Wealthy'
        else:
            return 'Non-Itemizer Dominated'
    
    df_wealth['wealth_category'] = df_wealth.apply(categorize_wealth, axis=1)
    return df_wealth


def detect_donor_lifecycle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify lifecycle patterns across ZIP codes.
    
    Real-world patterns:
    - College towns: young professionals, lower giving now but high future potential
    - Retirement communities: stable high giving
    - Emerging tech hubs: new wealth, growing generosity
    - Post-industrial decline: falling participation
    """
    df_lifecycle = df.copy()
    df_lifecycle['donor_lifecycle_type'] = ''
    df_lifecycle['lifecycle_score'] = 0.0
    
    # Use N1 (total filers) as proxy for population/economic activity
    # Use participation_rate change as indicator of health
    
    pr_median = df_lifecycle['participation_rate'].median()
    n1_median = df_lifecycle['N1'].median()
    gi_median = df_lifecycle['generosity_index'].median()
    
    # Classify lifecycle segments
    def classify_lifecycle(row):
        pr = row['participation_rate']
        n1 = row['N1']
        gi = row['generosity_index']
        
        # Emerging growth
        if n1 > n1_median and pr > pr_median and gi > gi_median:
            return 'Emerging Prosperity Zone'
        
        # Stagnation
        elif n1 < n1_median * 0.8 and pr < pr_median * 0.8:
            return 'Declining Community'
        
        # Youth-focused (high filers, lower giving)
        elif n1 > n1_median and pr < pr_median * 0.7 and gi < gi_median * 0.6:
            return 'Young Professional Zone'
        
        # Mature giving
        elif n1 < n1_median and pr > pr_median * 1.2 and gi > gi_median:
            return 'Mature Philanthropic Core'
        
        return ''
    
    df_lifecycle['donor_lifecycle_type'] = df_lifecycle.apply(classify_lifecycle, axis=1)
    return df_lifecycle


def detect_peer_group_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect ZIP codes that significantly outperform/underperform their peer group.
    
    Peer groups defined by: AGI band + State + Participation rate band
    """
    df_peers = df.copy()
    df_peers = df_peers.dropna(subset=['A00100', 'generosity_index', 'participation_rate', 'STATE'])
    
    # Create peer groups
    agi_bands = pd.qcut(df_peers['A00100'], q=4, labels=['Low', 'Mid-Low', 'Mid-High', 'High'], duplicates='drop')
    pr_bands = pd.qcut(df_peers['participation_rate'], q=3, labels=['Low', 'Mid', 'High'], duplicates='drop')
    
    df_peers['agi_band'] = agi_bands
    df_peers['pr_band'] = pr_bands
    df_peers['peer_group'] = df_peers['STATE'].astype(str) + '_' + df_peers['agi_band'].astype(str) + '_' + df_peers['pr_band'].astype(str)
    
    # Compute peer stats
    peer_stats = df_peers.groupby('peer_group', as_index=False).agg(
        peer_gi_mean=('generosity_index', 'mean'),
        peer_gi_std=('generosity_index', 'std'),
        peer_count=('zipcode', 'count')
    )
    
    df_peers = df_peers.merge(peer_stats, on='peer_group', how='left')
    
    # Flag significant deviations from peer group (>1.5 std from mean)
    df_peers['peer_gi_zscore'] = (df_peers['generosity_index'] - df_peers['peer_gi_mean']) / df_peers['peer_gi_std'].clip(lower=1e-6)
    df_peers['peer_outlier'] = (df_peers['peer_gi_zscore'].abs() > 1.5) & (df_peers['peer_count'] > 3)
    
    def categorize_peer_anomaly(row):
        if not row['peer_outlier']:
            return ''
        
        if row['peer_gi_zscore'] > 1.5:
            return 'Peer Group Star'
        else:
            return 'Peer Group Laggard'
    
    df_peers['peer_anomaly_category'] = df_peers.apply(categorize_peer_anomaly, axis=1)
    return df_peers


def detect_isolation_forest_anomalies(df: pd.DataFrame, contamination: float = 0.08) -> pd.DataFrame:
    """
    Multivariate outlier detection in 5D space.
    
    Features: Generosity Index, Participation Rate, AGI, N19700 (donors), N1 (filers)
    Detects combinations of metrics that are statistically rare.
    """
    df_iso = df.copy()
    df_iso = df_iso.dropna(subset=['generosity_index', 'participation_rate', 'A00100', 'N19700', 'N1'])
    
    if len(df_iso) < 10:
        df_iso['isolation_score'] = 0.0
        df_iso['is_isolation_anomaly'] = False
        return df_iso
    
    features = df_iso[['generosity_index', 'participation_rate', 'A00100', 'N19700', 'N1']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    df_iso['isolation_anomaly_pred'] = iso_forest.fit_predict(features_scaled)
    df_iso['isolation_score'] = iso_forest.score_samples(features_scaled)
    df_iso['is_isolation_anomaly'] = df_iso['isolation_anomaly_pred'] == -1
    
    return df_iso


def detect_trend_anomalies(df: pd.DataFrame, min_years: int = 3) -> pd.DataFrame:
    """
    Detect sudden changes in generosity trends within ZIPs over time.
    
    Example: ZIP code had stable 5% generosity for 3 years, then jumped to 12%.
    
    Returns DataFrame with trend anomaly flags and momentum indicators.
    """
    df_trend = df.copy()
    
    # Group by ZIP
    df_trend['trend_anomaly'] = False
    df_trend['momentum_change'] = 0.0
    df_trend['is_rising_star'] = False
    df_trend['is_declining'] = False
    
    for zipcode in df_trend['zipcode'].unique():
        zip_data = df_trend[df_trend['zipcode'] == zipcode].sort_values('year')
        
        if len(zip_data) < min_years:
            continue
        
        # Calculate year-over-year changes
        zip_data_local = zip_data.copy()
        zip_data_local['gi_change'] = zip_data_local['generosity_index'].diff()
        zip_data_local['gi_pct_change'] = zip_data_local['generosity_index'].pct_change()
        
        # Find sudden jumps (>50% change or >0.05 absolute change in one year)
        sudden_changes = (
            (zip_data_local['gi_pct_change'].abs() > 0.5) |
            (zip_data_local['gi_change'].abs() > 0.05)
        )
        
        if sudden_changes.any():
            df_trend.loc[df_trend['zipcode'] == zipcode, 'trend_anomaly'] = True
            idx = sudden_changes[sudden_changes].index
            if len(idx) > 0:
                max_change_idx = zip_data_local.loc[idx, 'gi_change'].abs().idxmax()
                df_trend.loc[max_change_idx, 'momentum_change'] = zip_data_local.loc[max_change_idx, 'gi_change']
        
        # Identify rising stars (consistent growth)
        if len(zip_data) >= 3:
            recent_years = zip_data_local.tail(3)
            if (recent_years['gi_change'].dropna() > 0).sum() >= 2:
                df_trend.loc[zip_data.index, 'is_rising_star'] = True
        
        # Identify declining (consistent decline)
        if len(zip_data) >= 3:
            recent_years = zip_data_local.tail(3)
            if (recent_years['gi_change'].dropna() < 0).sum() >= 2:
                df_trend.loc[zip_data.index, 'is_declining'] = True
    
    return df_trend


def compute_anomaly_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive metadata for anomalies with real-world interpretations.
    """
    df_meta = df.copy()
    
    # Enhanced context hints based on multiple factors
    def get_comprehensive_context(row):
        hints = []
        
        category = row.get('anomaly_category', '')
        lifecycle = row.get('donor_lifecycle_type', '')
        peer_cat = row.get('peer_anomaly_category', '')
        wealth_cat = row.get('wealth_category', '')
        
        # Primary category interpretation
        if 'Community Driven' in category:
            hints.append('🏘️ Strong community-driven giving culture')
            hints.append('High participation + generosity = engaged community')
            hints.append('Strategy: Broaden donor base, community partnership programs')
        
        elif 'Hidden Gem Community' in category:
            hints.append('💎 Exceptional generosity despite modest income')
            hints.append('Strong giving culture, untapped potential')
            hints.append('Strategy: Increase visibility, showcase impact stories')
        
        elif 'Philanthropic Powerhouse' in category:
            hints.append('🚀 Highest-tier prospect ZIP code')
            hints.append('Wealthy + generous + broad participation')
            hints.append('Strategy: Premium gala targets, major gift focus')
        
        elif 'Affluent Underperformers' in category:
            hints.append('🎯 MAJOR OPPORTUNITY: Wealthy but low giving')
            hints.append('Strong cultivation potential, capacity gap')
            hints.append('Strategy: Focus on wealth + engagement initiatives')
        
        elif 'Wealth Concentration Zone' in category:
            hints.append('📊 Many residents but modest giving patterns')
            hints.append('Possible non-itemizer effect or transient population')
            hints.append('Strategy: Community events, grassroots engagement')
        
        # Lifecycle indicators
        if 'Emerging Prosperity Zone' in lifecycle:
            hints.append('📈 Growing wealth + increasing generosity = rising star')
        
        elif 'Young Professional Zone' in lifecycle:
            hints.append('👨‍💼 Entry-level earners, future major donor pipeline')
            hints.append('Build loyalty early, expect growing capacity')
        
        elif 'Mature Philanthropic Core' in lifecycle:
            hints.append('🏆 Established donors, stable giving')
            hints.append('Sustainer focus, legacy planning opportunities')
        
        elif 'Declining Community' in lifecycle:
            hints.append('⚠️ Economic/demographic headwinds')
            hints.append('Track causes: job loss, aging out, migration')
        
        # Peer performance
        if 'Peer Group Star' in peer_cat:
            hints.append('⭐ Outperforms similar communities significantly')
        
        elif 'Peer Group Laggard' in peer_cat:
            hints.append('❌ Underperforms similar communities')
        
        # Wealth structure
        if 'Concentration Risk' in wealth_cat:
            hints.append('⚠️ Few ultra-wealthy, donor diversification needed')
        
        elif 'Non-Itemizer Dominated' in wealth_cat:
            hints.append('📋 Few itemizers, many standard deduction filers')
        
        return ' | '.join(hints[:4]) if hints else 'Real-world pattern pending analysis'
    
    df_meta['context_hint'] = df_meta.apply(get_comprehensive_context, axis=1)
    
    # Add fundraising strategy score (0-100)
    def calculate_fundraising_score(row):
        score = 50  # baseline
        
        gi = row.get('generosity_index', 0)
        pr = row.get('participation_rate', 0)
        agi = row.get('A00100', 0)
        
        # Generosity rewards
        if gi > 0.05:
            score += 20
        elif gi > 0.03:
            score += 10
        
        # Participation rewards
        if pr > 0.6:
            score += 15
        elif pr > 0.4:
            score += 8
        
        # Wealth capacity
        if agi > agi.quantile(0.75) if hasattr(agi, 'quantile') else agi > 1500:
            score += 15
        
        # Anomaly bonuses
        if row.get('is_anomaly'):
            score += 10
        
        if row.get('is_rising_star'):
            score += 15
        
        if row.get('peer_outlier'):
            score += 5
        
        return min(score, 100)
    
    df_meta['fundraising_priority_score'] = df_meta.apply(calculate_fundraising_score, axis=1)
    
    # Percentile ranks for benchmarking
    df_meta['gi_percentile_state'] = df_meta.groupby('STATE')['generosity_index'].rank(pct=True) * 100
    df_meta['pr_percentile_state'] = df_meta.groupby('STATE')['participation_rate'].rank(pct=True) * 100
    df_meta['agi_percentile_state'] = df_meta.groupby('STATE')['A00100'].rank(pct=True) * 100
    
    return df_meta


def detect_all_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point: Run ALL anomaly detection methods and integrate results.
    
    Returns enriched DataFrame with comprehensive anomaly indicators and actionable insights.
    """
    # Income-generosity anomalies
    df = detect_income_generosity_anomalies(df)
    
    # Wealth & participation structure
    df = detect_wealth_participation_anomalies(df)
    
    # Donor lifecycle patterns
    df = detect_donor_lifecycle_patterns(df)
    
    # Peer group performance
    df = detect_peer_group_anomalies(df)
    
    # Multivariate outliers
    df = detect_isolation_forest_anomalies(df, contamination=0.08)
    
    # Trend-based anomalies
    df = detect_trend_anomalies(df)
    
    # Comprehensive metadata and context
    df = compute_anomaly_metadata(df)
    
    # Combined anomaly flags
    df['is_any_anomaly'] = (
        df.get('is_anomaly', False) | 
        df.get('is_isolation_anomaly', False) | 
        df.get('trend_anomaly', False) |
        df.get('is_rising_star', False) |
        df.get('is_declining', False) |
        (df.get('wealth_anomaly', False)) |
        (df.get('peer_outlier', False))
    )
    
    return df


def filter_anomalies(df: pd.DataFrame, anomaly_type: str = 'all') -> pd.DataFrame:
    """
    Filter DataFrame to show specific anomaly types.
    
    Args:
        df: DataFrame with anomaly columns
        anomaly_type: 'all', 'income_mismatch', 'multivariate', 'trend', 
                     'rising_stars', 'declining', 'wealth_concentration', 
                     'peer_outliers', 'lifestyle_changing'
    
    Returns:
        Filtered DataFrame
    """
    if anomaly_type == 'all':
        return df[df['is_any_anomaly']].copy()
    elif anomaly_type == 'income_mismatch':
        return df[df['is_anomaly']].copy()
    elif anomaly_type == 'multivariate':
        return df[df['is_isolation_anomaly']].copy()
    elif anomaly_type == 'trend':
        return df[df['trend_anomaly']].copy()
    elif anomaly_type == 'rising_stars':
        return df[df['is_rising_star']].copy()
    elif anomaly_type == 'declining':
        return df[df['is_declining']].copy()
    elif anomaly_type == 'wealth_concentration':
        return df[df['wealth_anomaly']].copy()
    elif anomaly_type == 'peer_outliers':
        return df[df['peer_outlier']].copy()
    elif anomaly_type == 'lifestyle_changing':
        # Emerging prosperity or young professional zones
        return df[df['donor_lifecycle_type'].isin(['Emerging Prosperity Zone', 'Young Professional Zone'])].copy()
    else:
        return df.copy()


def get_anomaly_detail(df: pd.DataFrame, zipcode: str, state: str, full_df: pd.DataFrame = None) -> dict:
    """
    Generate comprehensive anomaly analysis for a specific ZIP code.
    
    Returns dictionary with:
    - Why it's anomalous
    - Real-world interpretation
    - Fundraising strategy
    - Comparable ZIP codes
    """
    target = df[(df['zipcode'] == zipcode) & (df['STATE'] == state)]
    
    if target.empty:
        return {'error': 'ZIP code not found'}
    
    row = target.iloc[0]
    analysis = {}
    
    # What type of anomaly
    anomaly_types = []
    if row.get('is_anomaly'):
        anomaly_types.append('Income-Generosity Mismatch')
    if row.get('is_isolation_anomaly'):
        anomaly_types.append('Multivariate Outlier')
    if row.get('is_rising_star'):
        anomaly_types.append('Rising Star Trend')
    if row.get('is_declining'):
        anomaly_types.append('Declining Trend')
    if row.get('wealth_anomaly'):
        anomaly_types.append('Wealth Concentration')
    if row.get('peer_outlier'):
        anomaly_types.append('Peer Group Outlier')
    
    analysis['anomaly_types'] = anomaly_types
    analysis['category'] = row.get('anomaly_category', 'Unclassified')
    analysis['lifecycle'] = row.get('donor_lifecycle_type', 'Unknown')
    
    # Why it's interesting
    analysis['why_anomalous'] = {
        'generosity_index': f"{row['generosity_index']:.2%}",
        'participation_rate': f"{row['participation_rate']:.2%}",
        'total_agi': f"${row['A00100']:,.0f}k",
        'itemizing_donors': f"{int(row['N19700']):,}",
        'total_filers': f"{int(row['N1']):,}",
        'generosity_zscore': f"{row.get('generosity_zscore', 0):.2f} std devs from mean",
        'anomaly_score': f"{row.get('anomaly_score', 0):.3f}"
    }
    
    # State percentile rankings
    analysis['state_ranking'] = {
        'generosity_percentile': f"{row.get('gi_percentile_state', 0):.0f}th percentile",
        'participation_percentile': f"{row.get('pr_percentile_state', 0):.0f}th percentile",
        'wealth_percentile': f"{row.get('agi_percentile_state', 0):.0f}th percentile"
    }
    
    # Real-world interpretation
    analysis['real_world_reason'] = row.get('context_hint', 'Complex pattern requiring investigation')
    analysis['fundraising_priority'] = row.get('fundraising_priority_score', 0)
    
    # Strategy recommendations
    strategy = []
    category = row.get('anomaly_category', '')
    
    if 'Philanthropic Powerhouse' in category:
        strategy.append('🎯 Primary strategy: Premium gala VIP targeting')
        strategy.append('💰 Approach: Major gift proposal with customized impact')
        strategy.append('🤝 Tactic: C-suite relationship building')
    
    elif 'Affluent Underperformers' in category:
        strategy.append('🎯 Primary strategy: Wealth cultivation initiative')
        strategy.append('💰 Approach: Demonstrate impact to build engagement')
        strategy.append('🤝 Tactic: Small group briefings with board members')
    
    elif 'Community Driven' in category:
        strategy.append('🎯 Primary strategy: Community partnership programs')
        strategy.append('💰 Approach: Align with local community values & missions')
        strategy.append('🤝 Tactic: Grassroots engagement, community events')
    
    elif 'Hidden Gem Community' in category:
        strategy.append('🎯 Primary strategy: Visibility & engagement campaigns')
        strategy.append('💰 Approach: Showcase impact stories from similar communities')
        strategy.append('🤝 Tactic: Social media spotlights, volunteer opportunities')
    
    elif 'Young Professional Zone' in analysis['lifecycle']:
        strategy.append('🎯 Primary strategy: Build loyalty early')
        strategy.append('💰 Approach: Entry-level giving programs, future stewardship')
        strategy.append('🤝 Tactic: Young professional networks, corporate partnerships')
    
    elif 'Declining Community' in analysis['lifecycle']:
        strategy.append('⚠️ Primary strategy: Investigate root cause')
        strategy.append('💰 Approach: Address economic headwinds')
        strategy.append('🤝 Tactic: Community economic development alignment')
    
    analysis['strategy'] = strategy
    
    return analysis


def get_similar_zips(df: pd.DataFrame, target_zipcode: str, state: str, n: int = 5) -> pd.DataFrame:
    """
    Find similar ZIP codes for comparison context.
    
    Similarity based on: Generosity Index, Participation Rate, and AGI.
    
    Returns top N most similar ZIPs with same characteristics.
    """
    target_row = df[(df['zipcode'] == target_zipcode) & (df['STATE'] == state)]
    
    if target_row.empty:
        return pd.DataFrame()
    
    target_row = target_row.iloc[0]
    
    # Euclidean distance in normalized space
    features = df[['generosity_index', 'participation_rate', 'A00100']].copy()
    features = features.fillna(features.mean())
    
    # Normalize
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    target_normalized = scaler.transform([[
        target_row['generosity_index'],
        target_row['participation_rate'],
        target_row['A00100']
    ]])[0]
    
    # Calculate distances
    distances = np.linalg.norm(features_normalized - target_normalized, axis=1)
    df['distance_to_target'] = distances
    
    # Return n nearest neighbors (excluding target itself)
    similar = df[
        (df['zipcode'] != target_zipcode) |
        (df['STATE'] != state)
    ].nsmallest(n, 'distance_to_target')[['zipcode', 'STATE', 'generosity_index', 'participation_rate', 'A00100', 'distance_to_target']]
    
    return similar
