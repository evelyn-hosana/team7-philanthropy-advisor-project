import streamlit as st
import pandas as pd
import altair as alt

# set layout
st.set_page_config(page_title="Generosity Intelligence", layout="wide")

# load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/22zpallagi_cleaned.csv', dtype={'zipcode': str})
    cols = ['generosity_index', 'participation_rate', 'A00100', 'A19700']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    return df

# setup titles
st.title("Philanthropy Advisor Project")
st.markdown("An analysis of charitable giving patterns across US ZIP codes. Metrics are based on IRS data: **Generosity Index** (Total Charitable Contributions / Total AGI) and **Participation Rate** (Returns with Charitable Contributions / Total Returns).")

# try loading data
try:
    df = load_data()
except FileNotFoundError:
    st.error("Cleaned data not found. Please run data_processing.py first.")
    st.stop()

# global filters section
st.sidebar.header("Global Filters")

# setup themes
alt.themes.enable('dark')
THEME_PALETTE = alt.Scale(range=['#38bdf8', '#1e3a8a'], type='symlog')
chart_bg = '#1a1d24'
axis_color = '#fafafa'

# setup state filter
all_states = sorted(df['STATE'].unique())
selected_states = st.sidebar.multiselect("Select State(s) to Filter:", options=all_states, default=[])

# apply filter
if selected_states:
    df = df[df['STATE'].isin(selected_states)]

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

# divider
st.divider()
st.subheader("Top N ZIP Codes")

# expander info
with st.expander("How is this calculated?"):
    st.markdown(f"Top ZIP codes ranked by {selected_metric_label}.")

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
st.altair_chart(chart1, use_container_width=True, theme=None)

# divider
st.divider()
st.subheader("Participation vs. Generosity")
with st.expander("How is this calculated?"):
    st.markdown("Bubble size indicates total population. Compares generosity index (Total Charitable Contributions / Total AGI) against participation rate (Returns with Charitable Contributions / Total Returns).")

# set binned tooltips
tooltip_binned = create_tooltip([
    ('sum(N1):Q', 'Total Population', ','),
    ('count()', 'Number of ZIP Codes', None)
])

# define avg lines
avg_df = pd.DataFrame({'avg_gen': [avg_gen], 'avg_part': [avg_part]})
rule_x = alt.Chart(avg_df).mark_rule(color='red', strokeDash=[5, 5]).encode(x='avg_part:Q')
rule_y = alt.Chart(avg_df).mark_rule(color='red', strokeDash=[5, 5]).encode(y='avg_gen:Q')

# render scatter
chart_binned_scatter = alt.Chart(df).mark_circle().encode(
    x=alt.X('participation_rate:Q', bin=alt.Bin(maxbins=30), title='Charitable Participation Rate (%)', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.0f')")),
    y=alt.Y('generosity_index:Q', bin=alt.Bin(maxbins=30), title='Generosity Index (%)', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.0f')")),
    size=alt.Size('sum(N1):Q', title='Total Population', scale=alt.Scale(range=[10, 1000])),
    color=alt.Color('sum(N1):Q', scale=THEME_PALETTE, legend=None),
    tooltip=tooltip_binned
).properties(height=400)

# overlay layered scatter
layered_scatter = (chart_binned_scatter + rule_x + rule_y).configure(background=chart_bg)
st.altair_chart(layered_scatter, use_container_width=True, theme=None)

# divider
st.divider()
col_c2, col_c3 = st.columns(2)

# subchart 1
with col_c2:
    st.subheader("Total AGI vs. Total Charitable Contributions")
    with st.expander("How is this calculated?"):
        st.markdown("ZIP code count grouped by Total AGI and Total Charitable Contributions.")
    # render heatmap
    chart2 = alt.Chart(df).mark_rect().encode(
        x=alt.X('A00100:Q', bin=alt.Bin(maxbins=heatmap_bins), title='Total AGI ($ millions)', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value / 1000, ',.0f')")),
        y=alt.Y('A19700:Q', bin=alt.Bin(maxbins=heatmap_bins), title='Total Charitable Contributions ($ millions)', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value / 1000, ',.0f')")),
        color=alt.Color('count()', scale=THEME_PALETTE, title='Number of ZIP Codes', legend=alt.Legend(labelColor=axis_color, titleColor=axis_color)),
    ).properties(height=400).configure(background=chart_bg)
    st.altair_chart(chart2, use_container_width=True, theme=None)

# subchart 2
with col_c3:
    st.subheader("Generosity Distribution")
    with st.expander("How is this calculated?"):
        st.markdown("Distribution of generosity index (Total Charitable Contributions / Total AGI) across ZIP codes.")
    # render histogram
    chart3 = alt.Chart(df).mark_bar().encode(
        x=alt.X('generosity_index:Q', bin=alt.Bin(maxbins=hist_bins), title='Generosity Index (%)', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.0f')"), scale=alt.Scale(domain=[0, 0.04], clamp=True)),
        y=alt.Y('count()', title='Number of ZIP Codes', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color)),
        color=alt.Color('count()', scale=THEME_PALETTE, title='Count of ZIP Codes', legend=None),
        tooltip=[alt.Tooltip('count()', title='Count')]
    ).properties(height=400)
    
    # overlay histogram
    rule_hist = alt.Chart(avg_df).mark_rule(color='red', strokeDash=[5, 5]).encode(x='avg_gen:Q')
    layered_hist = (chart3 + rule_hist).configure(background=chart_bg)
    st.altair_chart(layered_hist, use_container_width=True, theme=None)

# divider
st.divider()
st.subheader("State Averages")
with st.expander("How is this calculated?"):
    st.markdown("State-level averages for generosity index (Total Charitable Contributions / Total AGI) and participation rate (Returns with Charitable Contributions / Total Returns).")

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
    color=alt.Color(f'{metric_col}:Q', scale=alt.Scale(range=map_colors, type='symlog'), title=metric_title, legend=alt.Legend(labelColor=axis_color, titleColor=axis_color, format='.1%')),
    tooltip=[
        alt.Tooltip('STATE:N', title='State'),
        alt.Tooltip(f'{metric_col}:Q', title=metric_title, format='.1%')
    ]
).transform_lookup(
    lookup='id',
    from_=alt.LookupData(state_avg, 'id', [metric_col, 'STATE'])
).project('albersUsa').properties(height=500).configure(background=chart_bg)

# embed map
st.altair_chart(chart_map, use_container_width=True, theme=None)
