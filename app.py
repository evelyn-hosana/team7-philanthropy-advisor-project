import streamlit as st
import pandas as pd
import altair as alt

# set layout
st.set_page_config(page_title="Generosity Intelligence", layout="wide")

# load data
@st.cache_data
def load_data(filepath='data/zpallagi_cleaned.csv'):
    df = pd.read_csv(filepath, dtype={'zipcode': str})
    df['year'] = df['year'].astype(int)
    return df

# setup titles
st.title("Philanthropy Advisor Project")
st.markdown("An analysis of charitable giving patterns across US ZIP codes. Metrics are based on IRS data: **Generosity Index** (Total Charitable Contributions / Total AGI) and **Participation Rate** (Returns with Charitable Contributions / Total Returns).")

# data disclaimer
with st.expander("Data Limitations & Key Tax Insights - Read Before Interpreting Results"):
    st.markdown(
        "**1. The Itemization Floor:** This analysis relies strictly on IRS Schedule A charitable deductions. "
        "Because only taxpayers who itemize their deductions report charitable contributions to the IRS, "
        "standard deduction filers are not captured. This means results reflect generosity specifically "
        "among itemizing filers and may underrepresent middle-income communities.\n\n"
        "**2. The TCJA Data Gap (2017-2018):** The 2017 Tax Cuts and Jobs Act (TCJA) roughly tripled the "
        "standard deduction, dropping the national share of itemizing filers from ~30% to ~10%. "
        "**Data before and after 2017 is not directly comparable.** A sharp decline in giving metrics "
        "crossing that threshold essentially reflects this reporting floor change, not a literal collapse in charitable behavior.\n\n"
        "**3. Outlier Exclusion:** ZIP codes where Generosity Index exceeds 100% (A19700 / A00100 > 1) are excluded. "
        "These extreme values typically result from large non-cash donations or donor-advised fund activity concentrated in very small ZIP codes "
        "and would produce misleading rankings.\n\n"
        "**4. Nominal Dollar Values:** A00100 (Total AGI) and A19700 (Total Charitable Contributions) are reported in nominal thousands of dollars "
        "and are not inflation-adjusted. Raw dollar figures are not directly comparable across years. "
        "The Generosity Index ratio (A19700 / A00100) partially mitigates this since both values inflate together, "
        "but absolute dollar charts and tooltips should be interpreted as nominal."
    )

# try loading data
try:
    full_df = load_data()
except FileNotFoundError:
    st.error("Cleaned data not found. Please run data_processing.py first.")
    st.stop()

# global filters section
st.sidebar.header("Global Filters")

# setup themes
alt.theme.enable('dark')
alt.data_transformers.enable('vegafusion')
THEME_PALETTE = alt.Scale(range=['#38bdf8', '#1e3a8a'])
chart_bg = '#1a1d24'
axis_color = '#fafafa'

# year filter - above state so it scopes available states correctly
all_years = sorted(full_df['year'].unique())
year_options = ["All Available Years"] + list(all_years)

st.sidebar.markdown("**Select Year:**")
st.sidebar.markdown("<p style='font-size: 0.8rem; color: gray; margin-top: -10px; margin-bottom: 5px;'>Includes range bounds and datapoint display</p>", unsafe_allow_html=True)

def handle_year_selection():
    opts = st.session_state['year_filter']
    if "All Available Years" in opts and len(opts) > 1:
        st.session_state['year_filter'] = ["All Available Years"]

if 'year_filter' not in st.session_state:
    st.session_state['year_filter'] = [max(all_years)]

selected_option = st.sidebar.multiselect(
    "Select Year:",
    options=year_options,
    key='year_filter',
    on_change=handle_year_selection,
    label_visibility="collapsed"
)

if not selected_option or "All Available Years" in selected_option:
    selected_years = all_years
else:
    selected_years = [y for y in selected_option if y != "All Available Years"]

# calc full range between min and max selected years
min_year = min(selected_years)
max_year = max(selected_years)
selected_years_range = [y for y in all_years if min_year <= y <= max_year]

df = full_df[full_df['year'].isin(selected_years_range)]

# setup state filter
all_states = sorted(df['STATE'].unique())
selected_states = st.sidebar.multiselect("Select State(s):", options=all_states, default=[])

# apply filter
if selected_states:
    df = df[df['STATE'].isin(selected_states)]

if df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# cross-era warning when selected range spans the TCJA boundary
pre_tcja_years = [y for y in selected_years_range if y <= 2016]
post_tcja_years = [y for y in selected_years_range if y >= 2018]
spans_tcja = min_year <= 2017 and max_year >= 2018
if pre_tcja_years and post_tcja_years:
    st.error(
        "**Cross-era comparison active.** Your selected year range includes both pre-TCJA years "
        f"({', '.join(str(y) for y in pre_tcja_years)}) and post-TCJA years "
        f"({', '.join(str(y) for y in post_tcja_years)}). "
        "These represent different filer populations (~30% itemizing pre-2017 vs. ~10% post-2017). "
        "Metrics are not directly comparable across this boundary. Aggregate values and state rankings "
        "will reflect the mix of both regimes. Use a single-regime year range for valid trend comparisons."
    )

# chart settings
st.sidebar.header("Chart Customizations")

# metric toggle
with st.sidebar:
    selected_metric_label = st.segmented_control(
        "Rank Top N ZIP Codes By:",
        options=["Generosity", "Participation"],
        default="Generosity"
    )
    selected_metric_label = selected_metric_label or "Generosity"
metric_map = {"Generosity": "generosity_index", "Participation": "participation_rate"}
selected_metric = metric_map[selected_metric_label]

# config sliders
top_n = st.sidebar.slider("Top N ZIP Codes to Display:", min_value=5, max_value=100, value=20, step=5)
heatmap_bins = st.sidebar.slider("Heatmap Detail Level (Bins):", min_value=10, max_value=100, value=40, step=10)
hist_bins = st.sidebar.slider("Histogram Detail Level (Bins):", min_value=10, max_value=100, value=50, step=10)

# map filter
with st.sidebar:
    map_metric_label = st.segmented_control(
        "Color State Map By:",
        options=["Generosity", "Participation"],
        default="Generosity"
    )
    map_metric_label = map_metric_label or "Generosity"
map_metric_map = {"Generosity": "Generosity Index", "Participation": "Participation Rate"}
map_metric = map_metric_map[map_metric_label]

# calc averages
avg_gen = df['generosity_index'].mean()
med_gen = df['generosity_index'].median()
avg_part = df['participation_rate'].mean()
total_zips = len(df)

# display metrics
col1, col2, col3 = st.columns(3)
col1.metric("Average Generosity Index", f"{avg_gen:.2%}")
col2.metric("Average Participation", f"{avg_part:.2%}")
col3.metric("Total ZIP Codes Analyzed", f"{total_zips:,}")

with st.expander("How are these metrics calculated?"):
    st.markdown(
        "**Average Generosity Index** = mean(A19700 / A00100) across all ZIP codes in the selected filter\n\n"
        "- A19700 = Total Charitable Contributions (sum of itemized Schedule A deductions, in $thousands)\n"
        "- A00100 = Total AGI - Adjusted Gross Income (sum of all filers' AGI, in $thousands)\n\n"
        "**Average Participation** = mean(N19700 / N1) across all ZIP codes in the selected filter\n\n"
        "- N19700 = Itemizing Donors - count of returns reporting a charitable deduction\n"
        "- N1 = Total Returns - total number of tax returns filed"
    )

# divider
st.divider()
st.subheader("Top N ZIP Codes")

# expander info
with st.expander("How is this calculated?"):
    st.markdown(
        f"ZIP codes ranked in descending order by **{'Generosity Index' if selected_metric_label == 'Generosity' else 'Participation Rate'}**.\n\n"
        "- **Generosity Index** = A19700 / A00100 (Total Charitable Contributions / Total AGI)\n"
        "  - A19700 = Total Charitable Contributions (sum of itemized Schedule A deductions, in $thousands)\n"
        "  - A00100 = Total AGI - Adjusted Gross Income (sum of all filers' AGI, in $thousands)\n"
        "- **Participation Rate** = N19700 / N1 (Returns with Contributions / Total Returns)\n"
        "  - N19700 = Itemizing Donors - count of returns reporting a charitable deduction\n"
        "  - N1 = Total Returns - total number of tax returns filed"
    )

# get top n metrics
top_n_df = df.nlargest(top_n, selected_metric)
min_top_n = float(top_n_df[selected_metric].min()) if not top_n_df.empty else 0
max_top_n = float(top_n_df[selected_metric].max()) if not top_n_df.empty else 0

# parse tooltips
def create_tooltip(fields):
    return [alt.Tooltip(f, title=t, format=fmt) if fmt else alt.Tooltip(f, title=t) for f, t, fmt in fields]

# set shared tooltips
tooltip_common = create_tooltip([
    ('zipcode:N', 'ZIP Code', None),
    ('STATE:N', 'State', None),
    ('year:O', 'Year', None),
    ('A00100:Q', 'Total AGI ($ thousands)', ','),
    ('A19700:Q', 'Total Charitable Contributions ($ thousands)', ','),
    ('generosity_index:Q', 'Generosity Index', '.2%'),
    ('participation_rate:Q', 'Participation Rate', '.2%')
])

# render bar chart
chart1 = alt.Chart(top_n_df).mark_bar().encode(
    y=alt.Y('zipcode:N', sort='-x', title='ZIP Code', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color)),
    x=alt.X(f'{selected_metric}:Q', title=f'{selected_metric_label} (%)', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.0f')")),
    color=alt.Color(f'{selected_metric}:Q', scale=THEME_PALETTE, legend=None),
    tooltip=tooltip_common
).properties(height=500).configure(background=chart_bg)

# embed chart
st.altair_chart(chart1, width='stretch', theme=None)

# gala target table
st.markdown("##### Gala Target List")
table_df = top_n_df[['zipcode', 'STATE', 'year', 'generosity_index', 'participation_rate', 'N19700', 'A19700', 'N1']].copy()
table_df['generosity_index']   = (table_df['generosity_index']   * 100).round(2)
table_df['participation_rate'] = (table_df['participation_rate'] * 100).round(2)
st.dataframe(
    table_df,
    width='stretch',
    hide_index=True,
    column_config={
        'zipcode':            st.column_config.TextColumn('ZIP Code'),
        'STATE':              st.column_config.TextColumn('State'),
        'year':               st.column_config.NumberColumn('Year', format='%d'),
        'generosity_index':   st.column_config.NumberColumn('Generosity Index % (A19700 / A00100)', format='%.2f%%'),
        'participation_rate': st.column_config.NumberColumn('Participation Rate % (N19700 / N1)',   format='%.2f%%'),
        'N19700':             st.column_config.NumberColumn('Itemizing Donors (N19700)',            format='%d'),
        'A19700':             st.column_config.NumberColumn('Total Giving in $k (A19700)',          format='$%,.0f'),
        'N1':                 st.column_config.NumberColumn('Total Filers (N1)',                    format='%d'),
    }
)

# divider
st.divider()
st.subheader("Participation vs. Generosity")
with st.expander("How is this calculated?"):
    st.markdown(
        "Scatter plot comparing **Generosity Index** (vertical axis, depth) against **Participation Rate** (horizontal axis, reach) per ZIP code bin. "
        "Bubble size encodes the sum of N1 (Total Returns) within each bin.\n\n"
        "- **Generosity Index** = A19700 / A00100 (Total Charitable Contributions / Total AGI)\n"
        "  - A19700 = Total Charitable Contributions (sum of itemized Schedule A deductions, in $thousands)\n"
        "  - A00100 = Total AGI - Adjusted Gross Income (sum of all filers' AGI, in $thousands)\n"
        "- **Participation Rate** = N19700 / N1 (Returns with Contributions / Total Returns)\n"
        "  - N19700 = Itemizing Donors - count of returns reporting a charitable deduction\n"
        "Dashed red lines mark the dataset averages for each axis. White dashed lines trace the user-set target thresholds. Quadrant shading reflects combinations of above/below target reach and depth:\n"
        "- **Top Left (Blue)**: Low Reach, High Depth\n"
        "- **Top Right (Green)**: High Reach, High Depth\n"
        "- **Bottom Left (Red)**: Low Reach, Low Depth\n"
        "- **Bottom Right (Yellow)**: High Reach, Low Depth"
    )

st.markdown("#### Set Target Thresholds")
col_t1, col_t2, col_t3 = st.columns(3)
min_income_k = col_t1.slider("Min Avg Household Income ($k)", min_value=0, max_value=500, value=100, step=10, format="$%dk") * 1000
min_gen    = col_t2.slider("Min Generosity Index (%)", min_value=0.0, max_value=20.0, value=float(avg_gen * 100), step=0.1, format="%.1f%%") / 100
min_part   = col_t3.slider("Min Participation Rate (%)", min_value=0.0, max_value=50.0, value=float(avg_part * 100), step=1.0, format="%.1f%%") / 100

st.markdown("##### Target Statistics")
df['avg_hh_income'] = (df['A00100'] * 1000) / df['N1']
target_df = df[
    (df['avg_hh_income'] >= min_income_k) &
    (df['generosity_index'] >= min_gen) &
    (df['participation_rate'] >= min_part)
].copy()

target_count = len(target_df)
target_avg_gen = target_df['generosity_index'].mean() if target_count > 0 else 0
target_avg_part = target_df['participation_rate'].mean() if target_count > 0 else 0
give_score = target_avg_gen / target_avg_part if target_avg_part > 0 else 0
target_avg_agi = ((target_df['A00100'].sum() * 1000) / target_df['N1'].sum()) if target_count > 0 and target_df['N1'].sum() > 0 else 0

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Targeted ZIP-Years", f"{target_count:,}")
col_m2.metric("Average Give Score (GI/PR)", f"{give_score:.2f}")
col_m3.metric("Average Target AGI", f"${target_avg_agi:,.0f}")

# set binned tooltips
tooltip_binned = create_tooltip([
    ('mean(participation_rate):Q', 'PR (%)', '.2%'),
    ('mean(generosity_index):Q', 'GI (%)', '.2%'),
    ('sum(N1):Q', 'Total Population', ','),
    ('count()', 'Number of ZIP Codes', None)
])

# define avg lines (static global references)
avg_df = pd.DataFrame({'generosity_index': [avg_gen], 'participation_rate': [avg_part]})
rule_x = alt.Chart(avg_df).mark_rule(color='red', strokeDash=[5, 5]).encode(x='participation_rate:Q')
rule_y = alt.Chart(avg_df).mark_rule(color='red', strokeDash=[5, 5]).encode(y='generosity_index:Q')

# create new target tracking lines (dynamic user input)
target_lines_df = pd.DataFrame({'generosity_index': [min_gen], 'participation_rate': [min_part]})
rule_x_target = alt.Chart(target_lines_df).mark_rule(color='#9ca3af', strokeDash=[2, 4], strokeWidth=1).encode(x='participation_rate:Q')
rule_y_target = alt.Chart(target_lines_df).mark_rule(color='#9ca3af', strokeDash=[2, 4], strokeWidth=1).encode(y='generosity_index:Q')

# render background binned scatter
chart_binned_scatter = alt.Chart(df).mark_circle(opacity=1.0).encode(
    x=alt.X('participation_rate:Q', bin=alt.Bin(maxbins=30), title='Charitable Participation Rate (%)', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.0f')")),
    y=alt.Y('generosity_index:Q', bin=alt.Bin(maxbins=30), title='Generosity Index (%)', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.0f')")),
    size=alt.Size('sum(N1):Q', title='Total Population', scale=alt.Scale(range=[10, 1000]), legend=None),
    color=alt.Color('sum(N1):Q', scale=THEME_PALETTE, legend=None),
    tooltip=tooltip_binned
).properties(height=400)

# color quadrants based on user targets
max_p = df['participation_rate'].max() * 1.05 if not df.empty else 1
max_g = df['generosity_index'].max() * 1.05 if not df.empty else 1

quads = pd.DataFrame([
    {'participation_rate': 0, 'participation_rate2': min_part, 'generosity_index': 0, 'generosity_index2': min_gen, 'color': '#ef4444'},       # Bottom Left (Red)
    {'participation_rate': min_part, 'participation_rate2': max_p, 'generosity_index': 0, 'generosity_index2': min_gen, 'color': '#eab308'},   # Bottom Right (Yellow)
    {'participation_rate': 0, 'participation_rate2': min_part, 'generosity_index': min_gen, 'generosity_index2': max_g, 'color': '#3b82f6'},   # Top Left (Blue)
    {'participation_rate': min_part, 'participation_rate2': max_p, 'generosity_index': min_gen, 'generosity_index2': max_g, 'color': '#10b981'}# Top Right (Green)
])
chart_quads = alt.Chart(quads).mark_rect(opacity=0.1).encode(
    x='participation_rate:Q',
    x2='participation_rate2:Q',
    y='generosity_index:Q',
    y2='generosity_index2:Q',
    color=alt.Color('color:N', scale=None)
)

# dark overlay for non-targeted quadrants
overlay_quads = pd.DataFrame([
    {'participation_rate': 0, 'participation_rate2': min_part, 'generosity_index': 0, 'generosity_index2': min_gen},       # Bottom Left (Red)
    {'participation_rate': min_part, 'participation_rate2': max_p, 'generosity_index': 0, 'generosity_index2': min_gen},   # Bottom Right (Yellow)
    {'participation_rate': 0, 'participation_rate2': min_part, 'generosity_index': min_gen, 'generosity_index2': max_g}    # Top Left (Blue)
])
chart_overlay = alt.Chart(overlay_quads).mark_rect(color='#1a1d24', opacity=0.45).encode(
    x='participation_rate:Q', x2='participation_rate2:Q', y='generosity_index:Q', y2='generosity_index2:Q'
)

# overlay layered scatter
layered_scatter = (chart_quads + rule_x + rule_y + chart_binned_scatter + chart_overlay + rule_x_target + rule_y_target).configure(background=chart_bg)
st.altair_chart(layered_scatter, width='stretch', theme=None)

# divider
st.divider()
st.subheader("State Averages")
with st.expander("How is this calculated?"):
    st.markdown(
        "State-level averages, computed as the mean across all ZIP codes within each state for the selected filters.\n\n"
        "- **Generosity Index** = A19700 / A00100 (Total Charitable Contributions / Total AGI)\n"
        "  - A19700 = Total Charitable Contributions (sum of itemized Schedule A deductions, in $thousands)\n"
        "  - A00100 = Total AGI - Adjusted Gross Income (sum of all filers' AGI, in $thousands)\n"
        "- **Participation Rate** = N19700 / N1 (Returns with Contributions / Total Returns)\n"
        "  - N19700 = Itemizing Donors - count of returns reporting a charitable deduction\n"
        "  - N1 = Total Returns - total number of tax returns filed"
    )

# setup map axes
if map_metric == 'Generosity Index':
    metric_col, metric_title, map_colors = 'generosity_index', 'Avg Generosity', ['#e9d5ff', '#4c1d95']
else:
    metric_col, metric_title, map_colors = 'participation_rate', 'Avg Participation', ['#fca5a5', '#7f1d1d']

# group state avgs
state_avg = df.groupby('STATE', as_index=False)[['generosity_index', 'participation_rate']].mean().round(4)

# build fips dict
state_fips = {
    'AL': 1, 'AK': 2, 'AZ': 4, 'AR': 5, 'CA': 6, 'CO': 8, 'CT': 9, 'DE': 10, 'FL': 12, 'GA': 13,
    'HI': 15, 'ID': 16, 'IL': 17, 'IN': 18, 'IA': 19, 'KS': 20, 'KY': 21, 'LA': 22, 'ME': 23, 'MD': 24,
    'MA': 25, 'MI': 26, 'MN': 27, 'MS': 28, 'MO': 29, 'MT': 30, 'NE': 31, 'NV': 32, 'NH': 33, 'NJ': 34,
    'NM': 35, 'NY': 36, 'NC': 37, 'ND': 38, 'OH': 39, 'OK': 40, 'OR': 41, 'PA': 42, 'RI': 44, 'SC': 45,
    'SD': 46, 'TN': 47, 'TX': 48, 'UT': 49, 'VT': 50, 'VA': 51, 'WA': 53, 'WV': 54, 'WI': 55, 'WY': 56,
    'DC': 11
}
# assign map lookups
state_avg['id'] = state_avg['STATE'].map(state_fips)
state_avg = state_avg.dropna(subset=['id'])

# load map json
url = 'https://cdn.jsdelivr.net/npm/vega-datasets@v1.29.0/data/us-10m.json'
states = alt.topo_feature(url, 'states')

# render map chart
chart_map = alt.Chart(states).mark_geoshape().encode(
    color=alt.Color(f'{metric_col}:Q', scale=alt.Scale(range=map_colors), title=metric_title, legend=alt.Legend(labelColor=axis_color, titleColor=axis_color, format='.1%')),
    tooltip=[
        alt.Tooltip('STATE:N', title='State'),
        alt.Tooltip(f'{metric_col}:Q', title=metric_title, format='.1%')
    ]
).transform_lookup(
    lookup='id',
    from_=alt.LookupData(state_avg, 'id', [metric_col, 'STATE'])
).project('albersUsa').properties(height=500).configure(background=chart_bg)

# embed map
st.altair_chart(chart_map, width='stretch', theme=None)

# divider
st.divider()
col_c2, col_c3 = st.columns(2)

# subchart 1
with col_c2:
    st.subheader("Total AGI vs. Total Charitable Contributions")
    with st.expander("How is this calculated?"):
        st.markdown(
            "Heatmap showing the concentration of ZIP codes across combinations of Total AGI and Total Charitable Contributions.\n\n"
            "- **A00100** = Total AGI - Adjusted Gross Income (sum of all filers' AGI within a ZIP code, in $thousands)\n"
            "- **A19700** = Total Charitable Contributions (sum of itemized Schedule A deductions within a ZIP code, in $thousands)\n\n"
            "Color intensity = count of ZIP codes falling within each AGI x Contributions bin.\n\n"
            "**Note:** Both axes are nominal dollars (not inflation-adjusted). Values are not directly comparable across years separated by significant inflation. See Data Limitations for details."
        )
    # render heatmap
    chart2 = alt.Chart(df).mark_rect().encode(
        x=alt.X('A00100:Q', bin=alt.Bin(maxbins=heatmap_bins), title='Total AGI ($ millions)', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value / 1000, ',.0f')")),
        y=alt.Y('A19700:Q', bin=alt.Bin(maxbins=heatmap_bins), title='Total Charitable Contributions ($ millions)', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value / 1000, ',.0f')")),
        color=alt.Color('count()', scale=THEME_PALETTE, title='Number of ZIP Codes', legend=alt.Legend(labelColor=axis_color, titleColor=axis_color)),
    ).properties(height=400).configure(background=chart_bg)
    st.altair_chart(chart2, width='stretch', theme=None)

# subchart 2
with col_c3:
    st.subheader("Generosity Distribution")
    with st.expander("How is this calculated?"):
        st.markdown(
            "Histogram of **Generosity Index** = A19700 / A00100 across all ZIP codes in the selected filter.\n\n"
            "- A19700 = Total Charitable Contributions (sum of itemized Schedule A deductions, in $thousands)\n"
            "- A00100 = Total AGI - Adjusted Gross Income (sum of all filers' AGI, in $thousands)\n\n"
            "Each bar counts the number of ZIP codes whose Generosity Index falls within that bin. The dashed red line marks the dataset average."
        )
    # render histogram
    chart3 = alt.Chart(df).mark_bar().encode(
        x=alt.X('generosity_index:Q', bin=alt.Bin(maxbins=hist_bins), title='Generosity Index (%)', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.0f')")),
        y=alt.Y('count()', title='Number of ZIP Codes', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color)),
        color=alt.Color('count()', scale=THEME_PALETTE, title='Count of ZIP Codes', legend=None),
        tooltip=[alt.Tooltip('count()', title='Count')]
    ).properties(height=400)

    # overlay histogram
    rule_hist = alt.Chart(avg_df).mark_rule(color='red', strokeDash=[5, 5]).encode(x='avg_gen:Q')
    layered_hist = (chart3 + rule_hist).configure(background=chart_bg)
    st.altair_chart(layered_hist, width='stretch', theme=None)

# divider
st.divider()
st.subheader("Reportable Giving ZIP Codes by Year")
with st.expander("How is this calculated?"):
    st.markdown(
        "Count of ZIP codes per tax year satisfying the reportability threshold: A19700 > 0 and N1 >= 100.\n\n"
        "- A19700 = Total Charitable Contributions (sum of itemized Schedule A deductions, in $thousands)\n"
        "- N1 = Total Returns - total number of tax returns filed"
    )

# load full unfiltered dataset for this chart so year/state filters don't distort the trend
@st.cache_data
def get_zip_counts(data):
    return data.groupby('year', as_index=False).size().rename(columns={'size': 'zip_count'})

zip_counts = get_zip_counts(full_df)
# filter to dynamic global year range
zip_counts = zip_counts[zip_counts['year'].isin(selected_years_range)].copy()

# add era column so line breaks over 2017-2018 TCJA gap
zip_counts['era'] = zip_counts['year'].apply(lambda y: 'pre' if y <= 2017 else 'post')

chart_floor = alt.Chart(zip_counts).mark_line(point=alt.OverlayMarkDef(opacity=1), color='#38bdf8').encode(
    x=alt.X('year:Q', title='Tax Year', scale=alt.Scale(domain=[min_year, max_year], zero=False, nice=False), axis=alt.Axis(format='d', tickMinStep=1, labelColor=axis_color, titleColor=axis_color)),
    y=alt.Y('zip_count:Q', title='ZIP Codes with Reportable Giving', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color)),
    detail='era:N',
    tooltip=[
        alt.Tooltip('year:Q', title='Year'),
        alt.Tooltip('zip_count:Q', title='ZIP Count', format=',')
    ]
).properties(height=350)

max_y = zip_counts['zip_count'].max() if not zip_counts.empty else 1000

if spans_tcja:
    tcja_rect = alt.Chart(pd.DataFrame({'start': [2017], 'end': [2018]})).mark_rect(
        color='#4b5563', opacity=0.3
    ).encode(x='start:Q', x2='end:Q')
    tcja_label = alt.Chart(pd.DataFrame({'x': [2017.5], 'y': [max_y * 0.95], 'label': ['TCJA Gap']})).mark_text(
        color='#9ca3af', fontSize=10, fontWeight='bold', dy=-5
    ).encode(x='x:Q', y='y:Q', text='label:N')
    layered_floor = (tcja_rect + chart_floor + tcja_label).configure(background=chart_bg)
else:
    layered_floor = chart_floor.configure(background=chart_bg)
st.altair_chart(layered_floor, width='stretch', theme=None)

# divider
st.divider()
st.subheader("Generosity Trend by State")
with st.expander("How is this calculated?"):
    st.markdown(
        "Average **Generosity Index** = A19700 / A00100 per state per tax year, computed across all ZIP codes within the state.\n\n"
        "- A19700 = Total Charitable Contributions (sum of itemized Schedule A deductions, in $thousands)\n"
        "- A00100 = Total AGI - Adjusted Gross Income (sum of all filers' AGI, in $thousands)\n\n"
        "Use the state filter to compare or isolate specific states."
    )

# only show trend if multiple years in the range
if len(selected_years_range) < 2:
    st.info("Select a range of two or more years in the sidebar to view the trend chart.")
else:
    # use dataframe already filtered by year range and states
    trend_df = df.groupby(['year', 'STATE'], as_index=False)['generosity_index'].mean().round(4)
    
    trend_df['era'] = trend_df['year'].apply(lambda y: 'pre' if y <= 2017 else 'post')

    highlight = alt.selection_point(on='pointerover', fields=['STATE'], nearest=True)
    
    base_line = alt.Chart(trend_df).encode(
        x=alt.X('year:Q', title='Tax Year', scale=alt.Scale(domain=[min_year, max_year], zero=False, nice=False), axis=alt.Axis(format='d', tickMinStep=1, labelColor=axis_color, titleColor=axis_color)),
        y=alt.Y('generosity_index:Q', title='Avg Generosity Index', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.0f')")),
        color=alt.Color('STATE:N', title='State', legend=None),
        detail='era:N',
        tooltip=[
            alt.Tooltip('STATE:N', title='State'),
            alt.Tooltip('year:Q', title='Year'),
            alt.Tooltip('generosity_index:Q', title='Avg Generosity Index', format='.2%')
        ]
    )

    lines = base_line.mark_line(point=alt.OverlayMarkDef(opacity=1)).encode(
        size=alt.condition(~highlight, alt.value(1), alt.value(3)),
        opacity=alt.condition(~highlight, alt.value(0.2), alt.value(1))
    )

    points = base_line.mark_circle().encode(
        opacity=alt.value(0)
    ).add_params(highlight)
    
    max_y_trend = trend_df['generosity_index'].max() if not trend_df.empty else 0.1

    if spans_tcja:
        tcja_rect_trend = alt.Chart(pd.DataFrame({'start': [2017], 'end': [2018]})).mark_rect(
            color='#4b5563', opacity=0.3
        ).encode(x='start:Q', x2='end:Q')
        tcja_label_trend = alt.Chart(pd.DataFrame({'x': [2017.5], 'y': [max_y_trend * 0.95], 'label': ['TCJA Gap']})).mark_text(
            color='#9ca3af', fontSize=10, fontWeight='bold', dy=-5
        ).encode(x='x:Q', y='y:Q', text='label:N')
        interactive_trend = (tcja_rect_trend + tcja_label_trend + lines + points).properties(height=400).configure(background=chart_bg)
    else:
        interactive_trend = (lines + points).properties(height=400).configure(background=chart_bg)

    st.altair_chart(interactive_trend, width='stretch', theme=None)
