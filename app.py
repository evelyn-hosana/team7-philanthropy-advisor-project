import streamlit as st
import ai_assistant
import anomaly_detection

import pandas as pd
import altair as alt
import numpy as np
import html as _html
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# set layout
st.set_page_config(page_title="Generosity Intelligence", layout="wide")

st.title("Generosity Intelligence Dashboard")

# load data
@st.cache_data
def load_data(filepath='data/zpallagi_cleaned.csv'):
    df = pd.read_csv(filepath, dtype={'zipcode': str})
    df['year'] = df['year'].astype(int)
    return df

# detect anomalies in full dataset
@st.cache_data
def compute_anomalies(df):
    return anomaly_detection.detect_all_anomalies(df.copy())

# setup titles

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

def handle_year_selection():
    opts = st.session_state['year_filter']
    if "All Available Years" in opts and len(opts) > 1:
        st.session_state['year_filter'] = ["All Available Years"]

if 'year_filter' not in st.session_state:
    st.session_state['year_filter'] = [max(all_years)]

cy1, cy2 = st.sidebar.columns([1, 1.8])
cy1.markdown("<div style='margin-top: 5px;'>Select Year:</div>", unsafe_allow_html=True)
selected_option = cy2.multiselect("Select Year:", options=year_options, key='year_filter', on_change=handle_year_selection, label_visibility="collapsed")
st.sidebar.markdown("<p style='font-size: 0.75rem; color: gray; margin-top: -8px;'>Select one year for a snapshot. Select two (or more) years to analyze range. Selecting 'All Available Years' resets dropdown and uses full range dataset.</p>", unsafe_allow_html=True)

all_states = sorted(full_df['STATE'].unique())
cs1, cs2 = st.sidebar.columns([1, 1.8])
cs1.markdown("<div style='margin-top: 5px;'>Select State(s):</div>", unsafe_allow_html=True)
selected_states = cs2.multiselect("Select State(s):", options=all_states, default=[], label_visibility="collapsed")

if not selected_option or "All Available Years" in selected_option:
    selected_years = all_years
else:
    selected_years = [y for y in selected_option if y != "All Available Years"]

min_year = min(selected_years)
max_year = max(selected_years)
selected_years_range = [y for y in all_years if min_year <= y <= max_year]
display_year = selected_years[-1]

df = full_df[full_df['year'].isin(selected_years_range)]


# apply filter
if selected_states:
    df = df[df['STATE'].isin(selected_states)]

if df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# cross-era warning when selected range spans TCJA boundary
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


# calc averages
avg_gen = df['generosity_index'].mean()
med_gen = df['generosity_index'].median()
avg_part = df['participation_rate'].mean()
total_zips = len(df)

# calc 2022 baseline for input defaults
default_df = full_df[full_df['year'] == 2022] if 2022 in full_df['year'].values else full_df[full_df['year'] == max(full_df['year'])]
default_avg_gen = default_df['generosity_index'].mean()
default_avg_part = default_df['participation_rate'].mean()

avg_df = pd.DataFrame({'generosity_index': [avg_gen], 'participation_rate': [avg_part], 'avg_gen': [avg_gen]})


if 'ai_messages' not in st.session_state:
    st.session_state.ai_messages = []
if 'ai_text_input' not in st.session_state:
    st.session_state.ai_text_input = ""

@st.dialog("AI Advisor", width="large")
def ai_dialog():
    for msg in st.session_state.ai_messages:
        icon = ":material/smart_toy:" if msg["role"] == "assistant" else ":material/person:"
        with st.chat_message(msg["role"], avatar=icon):
            st.markdown(msg["content"])
            
    def handle_submit():
        val = st.session_state.get("ai_text_input", "")
        if val:
            st.session_state.ai_messages.append({"role": "user", "content": val})
            st.session_state.ai_text_input = ""
            
    if st.session_state.ai_messages and st.session_state.ai_messages[-1]["role"] == "user":
        # global context to pass to anthropic
        glb_top = df.nlargest(20, 'generosity_index')
        try:
            from sklearn.cluster import KMeans
            cs_df = df.dropna(subset=['generosity_index', 'participation_rate']).copy()
            if len(cs_df) > 3:
                cs_df['cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(cs_df[['generosity_index', 'participation_rate']])
                seg = cs_df.groupby('cluster').agg(ZIP_Codes=('zipcode', 'count'), Avg_GI=('generosity_index', 'mean'), Avg_PR=('participation_rate', 'mean')).reset_index()
                seg['Segment'] = 'Cluster ' + seg['cluster'].astype(str)
            else:
                seg = pd.DataFrame(columns=['Segment', 'ZIP_Codes', 'Avg_GI', 'Avg_PR'])
        except Exception:
            seg = pd.DataFrame(columns=['Segment', 'ZIP_Codes', 'Avg_GI', 'Avg_PR'])
            
        ctx = ai_assistant.build_context(glb_top, seg, pd.DataFrame(), selected_states, selected_years_range)
        with st.chat_message("assistant", avatar=":material/smart_toy:"):
            with st.spinner("Analyzing data..."):
                try:
                    resp = ai_assistant.ask_assistant(st.session_state.ai_messages[-1]["content"], ctx, st.session_state.ai_messages[:-1])
                    st.markdown(resp)
                    st.session_state.ai_messages.append({"role": "assistant", "content": resp})
                except Exception as e:
                    st.error(f"API Error: {e}")
        
    c1, c2, c3 = st.columns([7, 1, 1], vertical_alignment="bottom")
    
    c1.text_input("Ask the AI", key="ai_text_input", label_visibility="collapsed", placeholder="Ask a question about this data...", on_change=handle_submit)
    
    if c2.button(":material/send:", help="Send", use_container_width=True):
        handle_submit()
        
    if c3.button(":material/delete:", help="Clear Chat", use_container_width=True):
        st.session_state.ai_messages.clear()



# ai dialog rendering block
st.sidebar.markdown("<br>", unsafe_allow_html=True)
if st.sidebar.button(":material/smart_toy: Consult AI Advisor", use_container_width=True):
    ai_dialog()

selected_phase = st.segmented_control(
    "Phase Navigation:",
    options=["Phase 1", "Phase 2", "Additional Insights"],
    default="Phase 1",
    label_visibility="collapsed",
    key="phase_selector"
)

st.sidebar.divider()
st.sidebar.header("Chart Filters")

if selected_phase == "Phase 1":
    c1, c2 = st.sidebar.columns([1, 1.8])
    c1.markdown("<div style='margin-top: 10px;'>Primary Metric:</div>", unsafe_allow_html=True)
    selected_metric_label = c2.selectbox("Primary:", options=["Generosity", "Participation"], index=["Generosity", "Participation"].index(st.session_state.get("p1_primary_metric", "Generosity") or "Generosity"), key="p1_primary_metric", label_visibility="collapsed")
    selected_metric_label = selected_metric_label or "Generosity"
    
    c3, c4 = st.sidebar.columns([1, 1.8])
    c3.markdown("<div style='margin-top: 10px;'>Top N ZIPs:</div>", unsafe_allow_html=True)
    top_n = c4.number_input("Top N ZIPs:", min_value=1, value=st.session_state.get("p1_top_n", 20), step=1, key="p1_top_n", label_visibility="collapsed")
    
    c5, c6 = st.sidebar.columns([1, 1.8])
    c5.markdown("<div style='margin-top: 10px;'>Min GI (%):</div>", unsafe_allow_html=True)
    min_gen = c6.number_input("Min GI (%)", min_value=0.0, max_value=100.0, value=st.session_state.get("p1_min_gen", float(default_avg_gen * 100)), step=0.5, format="%.1f", key="p1_min_gen", label_visibility="collapsed") / 100
    
    c7, c8 = st.sidebar.columns([1, 1.8])
    c7.markdown("<div style='margin-top: 10px;'>Min PR (%):</div>", unsafe_allow_html=True)
    min_part = c8.number_input("Min PR (%)", min_value=0.0, max_value=100.0, value=st.session_state.get("p1_min_part", float(default_avg_part * 100)), step=1.0, format="%.1f", key="p1_min_part", label_visibility="collapsed") / 100

elif selected_phase == "Phase 2":
    c1, c2 = st.sidebar.columns([1, 1.8])
    c1.markdown("<div style='margin-top: 10px;'>Primary Metric:</div>", unsafe_allow_html=True)
    selected_metric_label = c2.selectbox("Primary:", options=["Generosity", "Participation"], index=["Generosity", "Participation"].index(st.session_state.get("p2_primary_metric", "Generosity") or "Generosity"), key="p2_primary_metric", label_visibility="collapsed")
    selected_metric_label = selected_metric_label or "Generosity"
    
    c3, c4 = st.sidebar.columns([1, 1.8])
    c3.markdown("<div style='margin-top: 10px;'>Top N ZIPs:</div>", unsafe_allow_html=True)
    top_n = c4.number_input("Top N ZIPs:", min_value=1, value=st.session_state.get("p2_top_n", 20), step=1, key="p2_top_n", label_visibility="collapsed")
    
    start_year = min(selected_years_range)
    end_year = max(selected_years_range)
        
    c5, c6 = st.sidebar.columns([1, 1.8])
    c5.markdown("<div style='margin-top: 10px;'>Clusters (k):</div>", unsafe_allow_html=True)
    k_clusters = c6.number_input("Clusters (k):", min_value=1, max_value=4, value=st.session_state.get("p2_k", 3), step=1, key="p2_k", label_visibility="collapsed")
    
    c7, c8 = st.sidebar.columns([1, 1.8])
    c7.markdown("<div style='margin-top: 10px;'>Min Filers (N1):</div>", unsafe_allow_html=True)
    min_filers = c8.number_input("Min Filers (N1):", min_value=0, max_value=1000, value=st.session_state.get("p2_min_filers", 100), step=50, key="p2_min_filers", label_visibility="collapsed")

elif selected_phase == "Additional Insights":
    c1, c2 = st.sidebar.columns([1, 1.8])
    c1.markdown("<div style='margin-top: 10px;'>Primary Metric:</div>", unsafe_allow_html=True)
    selected_metric_label = c2.selectbox("Primary:", options=["Generosity", "Participation"], index=["Generosity", "Participation"].index(st.session_state.get("p3_primary_metric", "Generosity") or "Generosity"), key="p3_primary_metric", label_visibility="collapsed")
    selected_metric_label = selected_metric_label or "Generosity"
    
    c3, c4 = st.sidebar.columns([1, 1.8])
    c3.markdown("<div style='margin-top: 10px;'>Hist Bins:</div>", unsafe_allow_html=True)
    hist_bins = c4.number_input("Hist Bins:", min_value=10, max_value=100, value=st.session_state.get("p3_hist_bins", 50), step=10, key="p3_hist_bins", label_visibility="collapsed")


if selected_phase == "Phase 1":
    # display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Generosity Index", f"{avg_gen:.2%}")
    col2.metric("Average Participation", f"{avg_part:.2%}")
    col3.metric("Total ZIP Codes Analyzed", f"{total_zips:,}")

    with st.expander("How are these metrics calculated?"):
        st.markdown("""- **Average Generosity Index** = mean(A19700 / A00100)
- **Average Participation** = mean(N19700 / N1)""")

    st.subheader("Top N ZIP Codes")
    if len(selected_years_range) > 1:
        st.caption(f"Showing data for **{display_year}** (last selected year). Multi-year data is not layered here to avoid duplicate ZIP entries.")


    metric_map = {"Generosity": "generosity_index", "Participation": "participation_rate"}
    selected_metric = metric_map[selected_metric_label]

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

    # get top n metrics - use most recent year when multiple years selected
    top_n_source = df[df['year'] == display_year] if len(selected_years_range) > 1 else df
    top_n_df = top_n_source.nlargest(top_n, selected_metric)
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
        color=alt.Color(f'{selected_metric}:Q', scale=alt.Scale(range=['#e9d5ff', '#4c1d95'] if selected_metric == 'generosity_index' else ['#fca5a5', '#7f1d1d']), legend=None),
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

    st.subheader("Participation vs. Generosity")
    if len(selected_years_range) > 1:
        st.caption(f"Showing data for **{display_year}** (last selected year).")
    with st.expander("How is this calculated?"):
        st.markdown("""Scatter plot comparing **Generosity Index** vs **Participation Rate**.

- **Generosity Index** = A19700 / A00100
- **Participation Rate** = N19700 / N1

Bubble size = sum(N1)""")



    st.markdown("##### Target Statistics")
    scatter_source = df[df['year'] == display_year].copy() if len(selected_years_range) > 1 else df.copy()
    target_df = scatter_source[
        (scatter_source['generosity_index'] >= min_gen) &
        (scatter_source['participation_rate'] >= min_part)
    ].copy()

    target_count = len(target_df)
    target_avg_gen = target_df['generosity_index'].mean() if target_count > 0 else 0
    target_avg_part = target_df['participation_rate'].mean() if target_count > 0 else 0

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Targeted ZIP Codes", f"{target_count:,}")
    col_m2.metric("Target Avg Generosity", f"{target_avg_gen:.2%}")
    col_m3.metric("Target Avg Participation", f"{target_avg_part:.2%}")

    # set binned tooltips
    tooltip_binned = create_tooltip([
        ('mean(participation_rate):Q', 'PR (%)', '.2%'),
        ('mean(generosity_index):Q', 'GI (%)', '.2%'),
        ('sum(N1):Q', 'Total Population', ','),
        ('count()', 'Number of ZIP Codes', None)
    ])

    # avg lines filtered to display year (matches scatter_source)
    scatter_avg_gen = scatter_source['generosity_index'].mean() if not scatter_source.empty else 0
    scatter_avg_part = scatter_source['participation_rate'].mean() if not scatter_source.empty else 0
    scatter_avg_df = pd.DataFrame({'generosity_index': [scatter_avg_gen], 'participation_rate': [scatter_avg_part]})
    rule_x = alt.Chart(scatter_avg_df).mark_rule(color='red', strokeDash=[5, 5]).encode(x='participation_rate:Q')
    rule_y = alt.Chart(scatter_avg_df).mark_rule(color='red', strokeDash=[5, 5]).encode(y='generosity_index:Q')

    # target tracking lines (dynamic user input)
    target_lines_df = pd.DataFrame({'generosity_index': [min_gen], 'participation_rate': [min_part]})
    rule_x_target = alt.Chart(target_lines_df).mark_rule(color='#9ca3af', strokeDash=[2, 4], strokeWidth=1).encode(x='participation_rate:Q')
    rule_y_target = alt.Chart(target_lines_df).mark_rule(color='#9ca3af', strokeDash=[2, 4], strokeWidth=1).encode(y='generosity_index:Q')

    # background binned scatter
    chart_binned_scatter = alt.Chart(scatter_source).mark_circle(opacity=1.0).encode(
        x=alt.X('participation_rate:Q', bin=alt.Bin(maxbins=30), title='Charitable Participation Rate (%)', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.0f')")),
        y=alt.Y('generosity_index:Q', bin=alt.Bin(maxbins=30), title='Generosity Index (%)', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.0f')")),
        size=alt.Size('sum(N1):Q', title='Total Population', scale=alt.Scale(range=[10, 1000]), legend=None),
        color=alt.Color('sum(N1):Q', scale=THEME_PALETTE, legend=None),
        tooltip=tooltip_binned
    ).properties(height=400)

    # color quadrants based on user targets
    max_p = scatter_source['participation_rate'].max() * 1.05 if not scatter_source.empty else 1
    max_g = scatter_source['generosity_index'].max() * 1.05 if not scatter_source.empty else 1

    quads = pd.DataFrame([
        {'participation_rate': 0, 'participation_rate2': min_part, 'generosity_index': 0, 'generosity_index2': min_gen, 'color': '#ef4444'},       # bottom left (red)
        {'participation_rate': min_part, 'participation_rate2': max_p, 'generosity_index': 0, 'generosity_index2': min_gen, 'color': '#eab308'},   # bottom right (yellow)
        {'participation_rate': 0, 'participation_rate2': min_part, 'generosity_index': min_gen, 'generosity_index2': max_g, 'color': '#3b82f6'},   # top left (blue)
        {'participation_rate': min_part, 'participation_rate2': max_p, 'generosity_index': min_gen, 'generosity_index2': max_g, 'color': '#10b981'}# top right (green)
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
        {'participation_rate': 0, 'participation_rate2': min_part, 'generosity_index': 0, 'generosity_index2': min_gen},       # bottom left (red)
        {'participation_rate': min_part, 'participation_rate2': max_p, 'generosity_index': 0, 'generosity_index2': min_gen},   # bottom right (yellow)
        {'participation_rate': 0, 'participation_rate2': min_part, 'generosity_index': min_gen, 'generosity_index2': max_g}    # top left (blue)
    ])
    chart_overlay = alt.Chart(overlay_quads).mark_rect(color='#1a1d24', opacity=0.45).encode(
        x='participation_rate:Q', x2='participation_rate2:Q', y='generosity_index:Q', y2='generosity_index2:Q'
    )

    # layered scatter
    layered_scatter = (chart_quads + rule_x + rule_y + chart_binned_scatter + chart_overlay + rule_x_target + rule_y_target).configure(background=chart_bg)
    st.altair_chart(layered_scatter, width='stretch', theme=None)

    # divider

elif selected_phase == "Phase 2":
    st.subheader("Donor Momentum")
    with st.expander("How is this calculated?"):
        st.markdown("""Dumbbell chart showing each ZIP code's **Generosity Index** at the start and end of the selected year range.

- **Generosity Index (GI)** = A19700 / A00100
- **Momentum** = end GI − start GI""")

    st.caption("Requires a multi-year range — select two or more years in the sidebar (e.g. 2019 and 2022 shows the full 2019–2022 range).")
    if len(selected_years_range) >= 2:

        gi_start = df[df['year'] == start_year][['zipcode', 'STATE', 'generosity_index']].rename(columns={'generosity_index': 'gi_start'})
        gi_end   = df[df['year'] == end_year  ][['zipcode', 'generosity_index']].rename(columns={'generosity_index': 'gi_end'})
        momentum_base = gi_start.merge(gi_end, on='zipcode')
        momentum_base['momentum'] = momentum_base['gi_end'] - momentum_base['gi_start']

        # top n by absolute momentum (biggest movers either direction)
        momentum_zips = momentum_base.reindex(momentum_base['momentum'].abs().nlargest(top_n).index)['zipcode'].tolist()

        dumbbell_df = momentum_base[momentum_base['zipcode'].isin(momentum_zips)].copy()
        dumbbell_df = dumbbell_df.sort_values('momentum', ascending=True)
        zip_order = dumbbell_df['zipcode'].tolist()
        chart_height = max(300, len(zip_order) * 24)

        shared_tooltip = [
            alt.Tooltip('zipcode:N', title='ZIP Code'),
            alt.Tooltip('STATE:N',   title='State'),
            alt.Tooltip('gi_start:Q', title=f'GI {start_year}', format='.2%'),
            alt.Tooltip('gi_end:Q',   title=f'GI {end_year}',   format='.2%'),
            alt.Tooltip('momentum:Q', title='GI Change',         format='+.2%'),
        ]

        # connecting bar - neutral mid tone
        bar_layer = alt.Chart(dumbbell_df).mark_rule(strokeWidth=2, color='#334155').encode(
            y=alt.Y('zipcode:N', sort=zip_order, title='ZIP Code',
                    axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelFontSize=11)),
            x=alt.X('gi_start:Q', title='Generosity Index (%)',
                    axis=alt.Axis(labelColor=axis_color, titleColor=axis_color,
                                  labelExpr="format(datum.value * 100, '.1f')", tickCount=6)),
            x2=alt.X2('gi_end:Q'),
            tooltip=shared_tooltip,
        )

        # start node - dark blue (#1e3a8a)
        dot_start = alt.Chart(dumbbell_df).mark_circle(size=70, color='#1e3a8a', opacity=1.0).encode(
            y=alt.Y('zipcode:N', sort=zip_order),
            x=alt.X('gi_start:Q'),
            tooltip=shared_tooltip,
        )

        # end node - light blue (#38bdf8)
        dot_end = alt.Chart(dumbbell_df).mark_circle(size=70, color='#38bdf8', opacity=1.0).encode(
            y=alt.Y('zipcode:N', sort=zip_order),
            x=alt.X('gi_end:Q'),
            tooltip=shared_tooltip,
        )

        mom_chart = (bar_layer + dot_start + dot_end).properties(height=chart_height).configure(background=chart_bg)
        st.altair_chart(mom_chart, width='stretch', theme=None)
        st.markdown(
            f"<div style='font-size:0.82rem; color:#9ca3af; margin-top:-8px;'>"
            f"<span style='color:#1e3a8a; font-size:1rem;'>&#9679;</span> {start_year} (start) &nbsp;&nbsp;"
            f"<span style='color:#38bdf8; font-size:1rem;'>&#9679;</span> {end_year} (end) &nbsp;&nbsp;"
            f"Bar direction shows change: right = growth, left = decline."
            f"</div>",
            unsafe_allow_html=True
        )

    st.subheader("ZIP Code Cluster Analysis")
    with st.expander("How is this calculated?"):
        st.markdown("""K-Means clustering into **k** scaled segments.

- **Generosity Index** = A19700 / A00100
- **Participation Rate** = N19700 / N1""")

    cluster_rank_by = selected_metric_label
    cluster_rank_col = {"Generosity": "generosity_index", "Participation": "participation_rate"}

    # label definitions per k - ordered lowest to highest engagement
    CLUSTER_LABELS = {
        1: [
            ("General Population",    "All ZIP codes fall into a single segment. Increase k to reveal sub-groups.")
        ],
        2: [
            ("Low Engagement",        "Below-average generosity and participation. Fewer itemizing donors relative to total filers."),
            ("High Engagement",       "Above-average generosity and participation. Strong donor base relative to total filers."),
        ],
        3: [
            ("Emerging Donors",       "Low generosity and participation. Likely standard-deduction-dominated or lower-income ZIP codes with limited itemizing activity."),
            ("Moderate Givers",       "Mid-range generosity and participation. Solid donor presence but not yet standout giving depth."),
            ("High-Value Donors",     "High generosity and participation. Core target audience — deep giving culture with broad filer involvement."),
        ],
        4: [
            ("Low Engagement",        "Low generosity and participation with modest household incomes. Limited itemizing donor activity."),
            ("Broad but Shallow",     "Higher participation but lower generosity per filer. Many donors, but smaller average gift relative to income."),
            ("Deep but Narrow",       "High generosity index among a smaller share of filers. Concentrated giving — fewer but high-value donors."),
            ("High-Value Donors",     "Top generosity and participation with higher household incomes. Strongest prospect segment for major gift outreach."),
        ],
    }

    # aggregate per zip across years before clustering
    cluster_input = df.groupby('zipcode', as_index=False).agg(
        generosity_index=('generosity_index', 'mean'),
        participation_rate=('participation_rate', 'mean'),
        N1=('N1', 'mean')
    ).copy()
    cluster_input = cluster_input.dropna(subset=['generosity_index', 'participation_rate'])

    # apply filters
    cluster_input = cluster_input[cluster_input['N1'] >= min_filers]

    rank_col = cluster_rank_col[cluster_rank_by]

    st.caption(f"Clustering **{len(cluster_input):,}** ZIP codes — filtered to N1 ≥ {min_filers:,}, ranked by {cluster_rank_by}.")

    if cluster_input.empty:
        st.warning("No ZIP codes match the current filters. Lower the minimum filers threshold.")
        st.stop()

    features = cluster_input[['generosity_index', 'participation_rate']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init='auto')
    cluster_input['cluster_id'] = kmeans.fit_predict(features_scaled)

    # rank clusters by user-selected metric low to high, map to labels
    cluster_stats = cluster_input.groupby('cluster_id')[['generosity_index', 'participation_rate']].mean()
    rank_order = cluster_stats[rank_col].rank(method='first').astype(int) - 1  # 0-indexed rank
    label_list = CLUSTER_LABELS[k_clusters]
    cluster_input['cluster_label'] = cluster_input['cluster_id'].map(
        lambda cid: label_list[rank_order[cid]][0]
    )

    # dark to light blue palette using app colors (lowest to highest engagement)
    all_cluster_colors = ['#1e3a8a', '#1d4ed8', '#38bdf8', '#bae6fd']
    ordered_labels = [label_list[i][0] for i in range(k_clusters)]
    cluster_scale = alt.Scale(domain=ordered_labels, range=all_cluster_colors[:k_clusters])

    cluster_chart = alt.Chart(cluster_input).mark_circle(opacity=0.55, size=25).encode(
        x=alt.X('participation_rate:Q', title='Participation Rate (%)',
                axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.0f')")),
        y=alt.Y('generosity_index:Q', title='Generosity Index (%)',
                axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.0f')")),
        color=alt.Color('cluster_label:N', scale=cluster_scale, title='Cluster',
                        legend=alt.Legend(labelColor=axis_color, titleColor=axis_color)),
        tooltip=[
            alt.Tooltip('zipcode:N', title='ZIP Code'),
            alt.Tooltip('cluster_label:N', title='Cluster'),
            alt.Tooltip('participation_rate:Q', title='Participation Rate', format='.2%'),
            alt.Tooltip('generosity_index:Q', title='Generosity Index', format='.2%'),
        ]
    ).properties(height=450).configure(background=chart_bg)

    st.altair_chart(cluster_chart, width='stretch', theme=None)
    

    # cluster legend with descriptions
    st.markdown("**Cluster Descriptions**")
    legend_cols = st.columns(k_clusters)
    for i, col in enumerate(legend_cols):
        label, desc = label_list[i]
        color = all_cluster_colors[i]
        text_color = '#111827' if color in ('#38bdf8', '#bae6fd') else '#ffffff'
        col.markdown(
            f"<div style='background:{color}; color:{text_color}; padding:10px 12px; border-radius:6px;'>"
            f"<strong>{label}</strong><br><span style='font-size:0.82rem'>{desc}</span></div>",
            unsafe_allow_html=True
        )

    # ANOMALY DETECTION SECTION
    st.markdown("---")
    st.subheader("🔍 Anomaly Detection - Irregular Giving Patterns")
    with st.expander("What are anomalies? How are they detected?"):
        st.markdown("""
        **Anomalies** are ZIP codes with unusual charitable giving patterns:

        - **Community Driven**: Areas where strong community culture drives unexpectedly high generosity
        - **Rising Stars**: ZIP codes showing consistent upward trends in giving
        - **Declining**: ZIP codes showing consistent downward trends in giving
        - **Wealth Issues**: Concentration risk and non-itemizer dominated areas
        - **Peer Outliers**: Communities outperforming/underperforming similar areas
        - **Lifecycle Patterns**: Emerging prosperity, young professionals, mature givers
        """)

    # Compute anomalies lazily on the display-year slice only (fast, cached)
    raw_anomaly_input = df[df['year'] == display_year].copy() if len(selected_years_range) > 1 else df.copy()
    with st.spinner(f"Running anomaly detection for {display_year}…"):
        display_anomaly_df = compute_anomalies(raw_anomaly_input)

    all_anomalies_df = display_anomaly_df[display_anomaly_df['is_any_anomaly'] == True].copy()
    all_anomalies_df = all_anomalies_df.dropna(subset=['anomaly_score']).sort_values('anomaly_score', ascending=False)
    
    if all_anomalies_df.empty:
        st.info("No anomalies detected for current year selection.")
    else:
        st.write(f"### 📊 Detected {len(all_anomalies_df)} anomalies across selected periods")
        
        # Create comprehensive visualization
        st.markdown("#### Anomaly Landscape Visualization")
        
        # Prepare data for visualization
        viz_df = all_anomalies_df[['zipcode', 'STATE', 'generosity_index', 'participation_rate', 
                                    'A00100', 'N1', 'anomaly_category', 'anomaly_score', 'fundraising_priority_score']].copy()
        viz_df = viz_df.dropna(subset=['generosity_index', 'participation_rate'])
        
        # Create scatter plot: Generosity vs Participation, colored by anomaly category, sized by AGI
        color_map = {
            'Community Driven': '#8b5cf6',
            'Hidden Gem Community': '#06b6d4',
            'Philanthropic Powerhouse': '#10b981',
            'Wealth Concentration Zone': '#f97316',
        }
        
        viz_df['category_color'] = viz_df['anomaly_category'].map(color_map).fillna('#6b7280')
        
        scatter = alt.Chart(viz_df).mark_circle(opacity=0.7, size=100).encode(
            x=alt.X('participation_rate:Q', title='Participation Rate (%)',
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.0f')")),
            y=alt.Y('generosity_index:Q', title='Generosity Index (%)',
                    scale=alt.Scale(domain=[0, viz_df['generosity_index'].quantile(0.99) * 1.1]),
                    axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.2f')")),
            size=alt.Size('A00100:Q', title='Total AGI ($k)', scale=alt.Scale(range=[50, 400])),
            color=alt.Color('anomaly_category:N', scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
                           title='Anomaly Category', legend=alt.Legend(labelColor=axis_color, titleColor=axis_color)),
            tooltip=[
                alt.Tooltip('zipcode:N', title='ZIP Code'),
                alt.Tooltip('STATE:N', title='State'),
                alt.Tooltip('generosity_index:Q', title='Generosity Index', format='.2%'),
                alt.Tooltip('participation_rate:Q', title='Participation Rate', format='.2%'),
                alt.Tooltip('A00100:Q', title='Total AGI ($k)', format=',.0f'),
                alt.Tooltip('N1:Q', title='Total Filers', format=',.0f'),
                alt.Tooltip('anomaly_score:Q', title='Anomaly Score', format='.3f'),
                alt.Tooltip('anomaly_category:N', title='Category'),
            ]
        ).properties(height=500, width=900).interactive().configure(background=chart_bg)
        
        st.altair_chart(scatter, use_container_width=True)
        
        # Group anomalies by category
        st.markdown("#### Anomalies by Category")
        
        category_groups = all_anomalies_df.groupby('anomaly_category')
        
        for category, group in category_groups:
            with st.expander(f"**{category}** — Top 5 of {len(group)} detected", expanded=False):
                # Summary stats for category
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Count", len(group))
                col2.metric("Avg Generosity", f"{group['generosity_index'].mean():.2%}")
                col3.metric("Avg Participation", f"{group['participation_rate'].mean():.2%}")
                col4.metric("Avg AGI", f"${group['A00100'].mean():,.0f}k")
                
                st.divider()
                
                # Show only top 5 highest-confidence anomalies per category
                st.write("**Top 5 ZIP Codes by Anomaly Score:**")

                zips_in_category = group.sort_values('anomaly_score', ascending=False).head(5)

                for idx, (_, zip_row) in enumerate(zips_in_category.iterrows()):
                    with st.container(border=True):
                        col_zip, col_score, col_gi, col_pr = st.columns([1, 1, 1, 1])
                        
                        with col_zip:
                            st.write(f"**{zip_row['zipcode']} ({zip_row['STATE']})**")
                        
                        with col_score:
                            st.metric("Score", f"{zip_row['anomaly_score']:.3f}")
                        
                        with col_gi:
                            st.metric("GI", f"{zip_row['generosity_index']:.2%}")
                        
                        with col_pr:
                            st.metric("PR", f"{zip_row['participation_rate']:.2%}")
                        
                        # Show explanation when expanded
                        if st.button(f"📋 Why is {zip_row['zipcode']} anomalous?", key=f"explain_{zip_row['zipcode']}_{category}"):
                            anom_detail = anomaly_detection.get_anomaly_detail(display_anomaly_df, str(zip_row['zipcode']), str(zip_row['STATE']))
                            
                            st.markdown(f"##### Why {zip_row['zipcode']} is Flagged as {category}")
                            
                            # Key metrics
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            metric_col1.metric("Total AGI", f"${zip_row['A00100']:,.0f}k")
                            metric_col2.metric("Total Filers", f"{int(zip_row['N1']):,}")
                            metric_col3.metric("Itemizing Donors", f"{int(zip_row['N19700']):,}")
                            metric_col4.metric("Priority Score", f"{zip_row.get('fundraising_priority_score', 0):.0f}/100")
                            
                            st.divider()
                            
                            # Why it's unusual
                            st.markdown("**Real-World Interpretation:**")
                            st.info(anom_detail.get('real_world_reason', 'Pattern analysis pending'))
                            
                            # Anomaly details
                            if anom_detail.get('anomaly_types'):
                                st.markdown(f"**Detected Patterns:** {', '.join(anom_detail['anomaly_types'])}")
                            
                            # State rankings
                            st.markdown("**State Performance Ranking:**")
                            rank_col1, rank_col2, rank_col3 = st.columns(3)
                            rank_col1.metric("GI Rank", anom_detail['state_ranking']['generosity_percentile'])
                            rank_col2.metric("PR Rank", anom_detail['state_ranking']['participation_percentile'])
                            rank_col3.metric("Wealth Rank", anom_detail['state_ranking']['wealth_percentile'])
                            
                            # Fundraising strategy
                            st.markdown("**Recommended Strategy:**")
                            if anom_detail.get('strategy'):
                                for strategy_point in anom_detail['strategy']:
                                    st.write(f"• {strategy_point}")
                            
                            # Similar ZIPs
                            st.markdown("**Peer ZIP Codes (for comparison):**")
                            similar_zips = anomaly_detection.get_similar_zips(display_anomaly_df, str(zip_row['zipcode']), str(zip_row['STATE']), n=3)
                            if not similar_zips.empty:
                                similar_display = similar_zips.copy()
                                similar_display.columns = ['ZIP', 'State', 'GI %', 'PR %', 'AGI ($k)', 'Distance']
                                similar_display['GI %'] = (similar_display['GI %'] * 100).round(2)
                                similar_display['PR %'] = (similar_display['PR %'] * 100).round(2)
                                similar_display['AGI ($k)'] = similar_display['AGI ($k)'].round(0)
                                similar_display['Distance'] = similar_display['Distance'].round(3)
                                st.dataframe(similar_display, hide_index=True, use_container_width=True)
    
    st.subheader("AI ZIP Brief Generator")
    st.caption("Pick one ZIP code and generate a short fundraising brief for outreach.")

    brief_source = df[df['year'] == display_year].copy() if len(selected_years_range) > 1 else df.copy()
    brief_source = brief_source[brief_source['N1'] >= min_filers].dropna(
        subset=['zipcode', 'STATE', 'generosity_index', 'participation_rate', 'N19700', 'A00100', 'N1']
    )

    if brief_source.empty:
        st.info("No ZIP codes available for brief generation under current filters.")
    else:
        brief_options = brief_source.sort_values('generosity_index', ascending=False).copy()
        brief_options["zip_label"] = brief_options.apply(
            lambda r: f"{r['zipcode']} ({r['STATE']}) | GI {r['generosity_index']:.2%} | PR {r['participation_rate']:.2%}",
            axis=1
        )

        selected_zip_label = st.selectbox(
            "Choose a ZIP code",
            brief_options["zip_label"].tolist(),
            key="zip_brief_select"
        )

        if st.button("Generate ZIP Brief", key="zip_brief_btn"):
            selected_zip_row = brief_options.loc[
                brief_options["zip_label"] == selected_zip_label
            ].iloc[0]

            seg_summary = cluster_input.groupby('cluster_label', as_index=False).agg(
                ZIP_Codes=('zipcode', 'count'),
                Avg_GI=('generosity_index', 'mean'),
                Avg_PR=('participation_rate', 'mean')
            ).rename(columns={'cluster_label': 'Segment'})

            if len(selected_years_range) >= 2:
                rising_df = momentum_base[momentum_base['momentum'] > 0].copy()
                rising_df = rising_df.merge(
                    brief_source[['zipcode', 'generosity_index']].drop_duplicates('zipcode'),
                    on='zipcode',
                    how='left'
                )
                rising_df = rising_df.rename(columns={'momentum': 'momentum_score'})
                rising_df = rising_df.sort_values('momentum_score', ascending=False)
            else:
                rising_df = pd.DataFrame()

            glb_top = brief_source.nlargest(20, 'generosity_index')
            ctx = ai_assistant.build_context(
                glb_top,
                seg_summary,
                rising_df,
                selected_states,
                selected_years_range
            )

            try:
                zip_brief = ai_assistant.generate_zip_report(selected_zip_row, ctx)
                st.markdown(zip_brief)
            except Exception as e:
                st.error(f"ZIP Brief Error: {e}")    




elif selected_phase == "Additional Insights":
    st.subheader("State Averages")
    if len(selected_years_range) > 1:
        st.caption(f"Showing data for **{display_year}** (last selected year).")
    with st.expander("How is this calculated?"):
        st.markdown("""State-level averages computed as the mean across all ZIP codes within each state.

- **Generosity Index** = A19700 / A00100
- **Participation Rate** = N19700 / N1""")

    map_metric_map = {"Generosity": "Generosity Index", "Participation": "Participation Rate"}
    map_metric = map_metric_map.get(selected_metric_label, "Generosity Index")

    # setup map axes
    if map_metric == 'Generosity Index':
        metric_col, metric_title, map_colors = 'generosity_index', 'Avg Generosity', ['#e9d5ff', '#4c1d95']
    else:
        metric_col, metric_title, map_colors = 'participation_rate', 'Avg Participation', ['#fca5a5', '#7f1d1d']

    # group state avgs - use single display year when multiple years selected
    state_source = df[df['year'] == display_year] if len(selected_years_range) > 1 else df
    state_avg = state_source.groupby('STATE', as_index=False)[['generosity_index', 'participation_rate']].mean().round(4)

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

    # map chart
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


    col_dist, col_zip = st.columns(2)

    with col_dist:
        st.subheader("Generosity Distribution")
        if len(selected_years_range) > 1:
            st.caption(f"Showing data for **{display_year}** (last selected year).")
        with st.expander("How is this calculated?"):
            st.markdown("""Histogram of **Generosity Index** across all selected ZIP codes.

- **Generosity Index** = A19700 / A00100""")

        hist_source = df[df['year'] == display_year] if len(selected_years_range) > 1 else df
        # histogram
        chart3 = alt.Chart(hist_source).mark_bar().encode(
            x=alt.X('generosity_index:Q', bin=alt.Bin(maxbins=hist_bins), title='Generosity Index (%)', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.0f')")),
            y=alt.Y('count()', title='Number of ZIP Codes', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color)),
            color=alt.Color('count()', scale=THEME_PALETTE, title='Count of ZIP Codes', legend=None),
            tooltip=[alt.Tooltip('count()', title='Count')]
        ).properties(height=400)

        # overlay histogram
        rule_hist = alt.Chart(avg_df).mark_rule(color='red', strokeDash=[5, 5]).encode(x='avg_gen:Q')
        layered_hist = (chart3 + rule_hist).configure(background=chart_bg)
        st.altair_chart(layered_hist, width='stretch', theme=None)

    with col_zip:
        st.subheader("Reportable Giving ZIP Codes by Year")
        with st.expander("How is this calculated?"):
            st.markdown("""Count of ZIP codes per tax year where:

- A19700 > 0
- N1 >= 100""")
        if len(selected_years_range) < 2:
            st.info("Select a range of two or more years in the sidebar to view the trend chart.")

        # load full unfiltered dataset - year/state filters would distort this trend
        @st.cache_data
        def get_zip_counts(data):
            return data.groupby('year', as_index=False).size().rename(columns={'size': 'zip_count'})

        zip_counts = get_zip_counts(full_df)
        # filter to dynamic global year range
        zip_counts = zip_counts[zip_counts['year'].isin(selected_years_range)].copy()

        # era column so line breaks over 2017-2018 tcja gap
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

    st.subheader("Generosity Trend by State")
    with st.expander("How is this calculated?"):
        st.markdown("""Average **Generosity Index** per state per tax year.

- **Generosity Index** = A19700 / A00100""")

    # only show trend if multiple years in range
    if len(selected_years_range) < 2:
        st.info("Select a range of two or more years in the sidebar to view the trend chart.")
    else:
        # already filtered by year range and states
        trend_df = df.groupby(['year', 'STATE'], as_index=False)['generosity_index'].mean().round(4)
        
        trend_df['era'] = trend_df['year'].apply(lambda y: 'pre' if y <= 2017 else 'post')

        highlight = alt.selection_point(on='pointerover', fields=['STATE'], nearest=True)

        base_line = alt.Chart(trend_df).encode(
            x=alt.X('year:Q', title='Tax Year', scale=alt.Scale(domain=[min_year, max_year], zero=False, nice=False), axis=alt.Axis(format='d', tickMinStep=1, labelColor=axis_color, titleColor=axis_color)),
            y=alt.Y('generosity_index:Q', title='Avg Generosity Index', axis=alt.Axis(labelColor=axis_color, titleColor=axis_color, labelExpr="format(datum.value * 100, '.0f')")),
            detail=alt.Detail(['STATE:N', 'era:N']),
            tooltip=[
                alt.Tooltip('STATE:N', title='State'),
                alt.Tooltip('year:Q', title='Year'),
                alt.Tooltip('generosity_index:Q', title='Avg Generosity Index', format='.2%')
            ]
        )

        lines = base_line.mark_line(point=alt.OverlayMarkDef(opacity=1)).encode(
            color=alt.condition(highlight, alt.value('#38bdf8'), alt.value('#2563eb')),
            size=alt.condition(~highlight, alt.value(1), alt.value(3)),
            opacity=alt.condition(~highlight, alt.value(0.7), alt.value(1))
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

